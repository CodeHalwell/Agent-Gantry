"""OpenAI Agents SDK adapter: Gantry tools as native ``FunctionTool``s.

Gantry plugs into the SDK's own mechanisms rather than handing it opaque
callables:

- **Errors the model can act on** -- a failed execution comes back to the model
  as the tool's output (``error_type: message``, capped, optionally rewritten by
  ``error_formatter``) instead of raising ``UserError`` and ending the run. The
  SDK's default would hide the reason ("An error occurred while running the
  tool"), so the model could not correct a bad argument.
- **Human approval** -- Gantry's confirmation gate (the tool's
  ``requires_confirmation`` flag or a ``SecurityPolicy`` pattern) sets the SDK's
  ``needs_approval``, so the run pauses with ``result.interruptions``; approving
  with ``state.approve(item)`` replays the call once with the gate cleared.
- **Run context** -- the call's ``ToolContext`` (``context``, ``tool_call_id``,
  ``tool_name``) is visible to Gantry tool handlers via
  :func:`current_run_context` / :func:`agent_gantry.current_tool_invocation`.
- **Tracing** -- the SDK's current span (or an OpenTelemetry span) becomes the
  Gantry ``ToolCall``'s trace parent.
- **Per-turn selection** -- :meth:`OpenAIAgentsAdapter.dynamic_tools` gives the
  agent the catalogue with ``is_enabled`` callbacks, so each turn the model is
  offered only the tools Gantry selects for the newest prompt or tool result,
  with per-run state (no shared ``agent.tools`` mutation).

The ``agents`` import is lazy so ``import agent_gantry`` never requires the SDK.
Supported: ``openai-agents`` >= 0.23.1 for everything above. Older SDKs still
get static tools, readable errors and run context, but no ``needs_approval``
(a gated call is refused, not paused) and no :meth:`~OpenAIAgentsAdapter.dynamic_tools`.

Public entry point: :class:`OpenAIAgentsAdapter`.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import TYPE_CHECKING, Any

from agent_gantry.adapters.tool_spec.schema_utils import (
    strict_json_schema,
    unsupported_strict_paths,
)
from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    BaseFrameworkAdapter,
    GantryToolset,
    ToolConfirmationRequiredError,
    ToolExecutionError,
)
from agent_gantry.integrations.frameworks.native import (
    ErrorFormatter,
    ErrorMode,
    check_error_mode,
    execute_spec,
    failure_text,
    otel_ids,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.integrations.frameworks.base import ToolSpec

FRAMEWORK_NAME = "openai_agents"

_INSTALL_HINT = (
    "The OpenAI Agents SDK is required for OpenAIAgentsAdapter; install it with "
    "`pip install 'agent-gantry[openai-agents]'` (or `uv add 'agent-gantry[openai-agents]'`)."
)

_PENDING_APPROVAL_TEXT = (
    "This tool call requires human approval and was not run. Ask the user to "
    "approve it, or continue without it."
)

_logger = logging.getLogger(__name__)


def _strict_schema(params: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return ``params`` reshaped for OpenAI Agents strict-mode function tools.

    ``FunctionTool.strict_json_schema`` defaults to ``True``, and the SDK then
    runs ``ensure_strict_json_schema``, which rewrites ``required`` to list
    *every* property. Setting only a top-level ``additionalProperties: False``
    therefore silently promoted every optional Gantry parameter to mandatory
    (no ``null`` union was added), raised ``UserError`` on any nested
    ``additionalProperties: true``, and on SDK versions predating that
    transform sent a non-strict schema with ``strict=true`` — a 400.

    Delegating to the shared transform applies the constraints recursively and
    keeps optionality by widening those properties to admit ``null``. It is
    idempotent, so the SDK's own pass over the result is a no-op.

    Returns:
        A tuple of the schema to publish and whether strict mode is usable.
        An object with arbitrary keys (a ``dict[str, int]`` parameter, an
        untyped ``dict``) has no strict-mode representation, and
        ``strict_json_schema`` deliberately leaves it alone rather than
        forcing it closed — so the caller must turn ``strict_json_schema``
        *off* for that tool, or the SDK's own ``ensure_strict_json_schema``
        rejects it with ``UserError``.
    """
    unsupported = unsupported_strict_paths(params)
    if unsupported:
        _logger.warning(
            "Tool cannot use OpenAI Agents strict mode: %s describes an object "
            "with arbitrary keys, which strict mode cannot express. Emitting "
            "the tool with strict_json_schema=False so the SDK still accepts "
            "it — declare the object's properties explicitly to make it "
            "strict-compatible.",
            ", ".join(unsupported),
        )
        return copy.deepcopy(params), False
    return strict_json_schema(params), True


def _function_tool_fields(function_tool: Any) -> frozenset[str]:
    """Constructor fields of the installed ``FunctionTool`` (older SDKs lack some)."""
    import dataclasses

    try:
        return frozenset(f.name for f in dataclasses.fields(function_tool))
    except TypeError:  # pragma: no cover - not a dataclass (test doubles)
        import inspect

        return frozenset(inspect.signature(function_tool).parameters)


def current_run_context() -> Any:
    """The SDK ``ToolContext`` of the tool call executing now, or ``None``.

    Call from inside a Gantry tool handler to reach the run's context object
    (``Runner.run(..., context=...)``) and the call's id::

        @gantry.register
        async def get_balance(account: str) -> str:
            \"\"\"Fetch an account balance.\"\"\"
            ctx = current_run_context()
            return await ctx.context.bank.balance(account)
    """
    from agent_gantry.core.invocation import current_tool_invocation

    invocation = current_tool_invocation()
    if invocation is None or invocation.framework != FRAMEWORK_NAME:
        return None
    return invocation.native_context


def _trace_ids() -> tuple[str | None, str | None]:
    """The SDK's current span when it is tracing, else the OpenTelemetry span."""
    try:
        from agents.tracing import get_current_span

        span = get_current_span()
    except Exception:
        span = None
    trace_id = getattr(span, "trace_id", None)
    span_id = getattr(span, "span_id", None)
    if isinstance(trace_id, str) and trace_id and trace_id != "no-op":
        return trace_id, span_id if isinstance(span_id, str) else None
    return otel_ids()


def _approved(ctx: Any) -> bool:
    """Whether the SDK's approval flow approved this exact call."""
    check = getattr(ctx, "is_tool_approved", None)
    if not callable(check):
        return False
    try:
        return check(getattr(ctx, "tool_name", ""), getattr(ctx, "tool_call_id", "")) is True
    except Exception:
        return False


async def execute_for_openai_agents(
    spec: ToolSpec,
    arguments: Any,
    ctx: Any,
    *,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
) -> Any:
    """Run ``spec`` through ``gantry.execute`` for an SDK tool call.

    In ``"model"`` mode a failure becomes text the model reads (see
    :func:`~agent_gantry.integrations.frameworks.native.failure_text`); a call
    stopped at the confirmation gate without native approval (a custom policy
    the pre-check could not predict) is reported as not run. In ``"raise"`` mode
    the :class:`ToolExecutionError` family propagates.
    """
    approved = _approved(ctx)
    try:
        return await execute_spec(
            spec,
            arguments,
            framework=FRAMEWORK_NAME,
            native_context=ctx,
            deps=getattr(ctx, "context", None),
            tool_call_id=getattr(ctx, "tool_call_id", None),
            metadata={"tool_name": getattr(ctx, "tool_name", None), "approved": approved},
            approved=approved,
            trace_ids=_trace_ids(),
        )
    except ToolExecutionError as exc:
        if error_mode == "raise":
            raise
        if isinstance(exc, ToolConfirmationRequiredError):
            return _PENDING_APPROVAL_TEXT
        return f"Error: {failure_text(exc, error_formatter)}"


def _spec_to_openai_agents(
    spec: ToolSpec,
    *,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
    is_enabled: Any = True,
    on_output: Callable[[Any, str, str], None] | None = None,
) -> Any:
    """Convert a :class:`ToolSpec` into an OpenAI Agents SDK ``FunctionTool``.

    Args:
        spec: The selected tool.
        error_mode: See :data:`~agent_gantry.integrations.frameworks.native.ErrorMode`.
        error_formatter: Rewrites failure text before the model sees it.
        is_enabled: Passed through to ``FunctionTool.is_enabled`` (per-turn selection).
        on_output: Called with ``(tool_context, tool_name, output_text)`` after
            each call, so a per-turn provider can select from the result. The
            text is ``""`` when the call failed or awaits approval, so error
            text never feeds retrieval.

    Raises:
        ImportError: If ``openai-agents`` is not installed.
    """
    try:
        from agents import FunctionTool
    except ImportError as exc:  # pragma: no cover - exercised via fake module
        raise ImportError(_INSTALL_HINT) from exc
    from agent_gantry.adapters.tool_spec.providers import _result_text

    async def _on_invoke_tool(ctx: Any, args: Any) -> str:
        data = json.loads(args) if isinstance(args, str) else dict(args or {})
        result = await execute_for_openai_agents(
            spec, data, ctx, error_mode=error_mode, error_formatter=error_formatter
        )
        # ``str()`` on a dict yields Python repr; serialize structured results
        # as JSON, matching the Agent Framework bridge.
        text = _result_text(result)
        if on_output is not None and ctx is not None:
            failed = text == _PENDING_APPROVAL_TEXT or text.startswith("Error: ")
            on_output(ctx, spec.native_name, "" if failed else text)
        return text

    fields = _function_tool_fields(FunctionTool)
    kwargs: dict[str, Any] = {}
    if is_enabled is not True:
        if "is_enabled" not in fields:
            raise ImportError(
                "Per-turn Gantry selection needs openai-agents>=0.23.1 "
                "(FunctionTool.is_enabled). " + _INSTALL_HINT
            )
        kwargs["is_enabled"] = is_enabled
    # SDKs without ``needs_approval`` still never run a gated call: it reaches
    # the confirmation gate and the model is told so (warned above).
    if (
        error_mode == "model"
        and "needs_approval" not in fields
        and spec._gantry.requires_confirmation(spec.tool or spec.name, spec._namespace)
    ):
        _logger.warning(
            "Tool %r needs confirmation but this openai-agents has no needs_approval; "
            "calls will be refused instead of paused for approval. Upgrade to "
            "openai-agents>=0.23.1.",
            spec.native_name,
        )
    if error_mode == "model" and "needs_approval" in fields:
        gantry = spec._gantry
        tool_def = spec.tool if spec.tool is not None else spec.name

        async def _needs_approval(ctx: Any, params: dict[str, Any], call_id: str) -> bool:
            return gantry.requires_confirmation(tool_def, spec._namespace)

        kwargs["needs_approval"] = _needs_approval

    params, use_strict = _strict_schema(spec.parameters)
    return FunctionTool(
        name=spec.native_name,
        description=spec.description,
        params_json_schema=params,
        strict_json_schema=use_strict,
        on_invoke_tool=_on_invoke_tool,
        **kwargs,
    )


class OpenAIAgentsAdapter(BaseFrameworkAdapter):
    """Route Gantry tools into the OpenAI Agents SDK through its native mechanisms.

    Static slices (:meth:`select`), per-turn selection without touching shared
    state (:meth:`dynamic_tools`, recommended for live use), and the earlier
    per-run helpers (:meth:`session`, :meth:`run`, :meth:`run_hooks`, which
    rewrite ``agent.tools`` and so suit one run at a time per agent). Every
    tool routes through ``gantry.execute``; see the module docs for errors,
    approval, run context and tracing.

    Args:
        gantry: The registry to select from and execute against.
        default_limit: Default selection budget.
        error_mode: Default ``ErrorMode`` for tools built by :meth:`select`,
            :meth:`convert` and :meth:`dynamic_tools`.
        error_formatter: Rewrites failure text before the model sees it.
    """

    live_tier = "per-turn"
    _live_delegate = "session"
    _live_required_kwargs = ("agent",)

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        default_limit: int = DEFAULT_TOOL_LIMIT,
        error_mode: ErrorMode = "model",
        error_formatter: ErrorFormatter | None = None,
    ) -> None:
        super().__init__(gantry, default_limit=default_limit)
        self._error_mode = check_error_mode(error_mode)
        self._error_formatter = error_formatter

    def _tool_kwargs(self) -> dict[str, Any]:
        return {"error_mode": self._error_mode, "error_formatter": self._error_formatter}

    def _mode(self, error_mode: ErrorMode | None) -> ErrorMode:
        return self._error_mode if error_mode is None else check_error_mode(error_mode)

    @staticmethod
    def convert(
        spec: ToolSpec,
        *,
        error_mode: ErrorMode = "model",
        error_formatter: ErrorFormatter | None = None,
    ) -> Any:
        """Wrap a single :class:`ToolSpec` as an OpenAI Agents ``FunctionTool``."""
        return _spec_to_openai_agents(
            spec, error_mode=check_error_mode(error_mode), error_formatter=error_formatter
        )

    async def select(
        self,
        query: str,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        tools_already_used: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        error_mode: ErrorMode | None = None,
    ) -> list[Any]:
        """Select tools for ``query`` once, as ``FunctionTool`` objects."""
        mode = self._mode(error_mode)
        specs = await GantryToolset(self._gantry).select(
            query,
            tools_already_used=tools_already_used,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
        )
        return [
            self.convert(s, error_mode=mode, error_formatter=self._error_formatter) for s in specs
        ]

    async def dynamic_tools(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        error_mode: ErrorMode | None = None,
        sticky: bool = True,
        max_offered: int | None = None,
        query_from_results: bool = True,
    ) -> list[Any]:
        """The catalogue as ``FunctionTool``s that Gantry enables per turn.

        ``Agent(tools=await adapter.dynamic_tools())``. Each tool's
        ``is_enabled`` asks one shared selector which tools to offer this turn:
        the newest user message, or after a Gantry tool call the run's prompt
        plus that tool's output, so a chained task surfaces the next tool it
        needs. Selection is cached per run and state is keyed to the run, so
        concurrent runs of one agent are independent and ``agent.tools`` is never
        rewritten. Tools registered after this call are not in the catalogue;
        call it again to pick them up.

        Args:
            sticky: Keep tools offered earlier in the run, up to ``max_offered``
                per turn (default ``3 * limit``; current selection and pins
                always stay, older tools drop least recently selected first).
            query_from_results: Let tool output (its first 2,000 characters)
                drive the next selection; it is sent to the embedder. Set
                ``False`` to keep tool output out of retrieval.
        """
        from agent_gantry.integrations.frameworks.openai_agents_live import (
            GantryDynamicTools,
        )

        provider = GantryDynamicTools(
            self._gantry,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            error_mode=self._mode(error_mode),
            error_formatter=self._error_formatter,
            sticky=sticky,
            max_offered=max_offered,
            query_from_results=query_from_results,
        )
        return await provider.build()

    async def run(
        self,
        agent: Any,
        run_input: Any,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **run_kwargs: Any,
    ) -> Any:
        """Re-select ``agent``'s tools for ``run_input`` and run it once via Gantry (one-shot live)."""
        from agent_gantry.integrations.frameworks.openai_agents_live import (
            _run_with_gantry,
        )

        return await _run_with_gantry(
            agent,
            self._gantry,
            run_input,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            **self._tool_kwargs(),
            **run_kwargs,
        )

    def session(
        self,
        agent: Any,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
    ) -> Any:
        """Return a live session that re-selects ``agent``'s tools each run."""
        from agent_gantry.integrations.frameworks.openai_agents_live import (
            GantryAgentSession,
        )

        return GantryAgentSession(
            agent,
            self._gantry,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            **self._tool_kwargs(),
        )

    def run_hooks(
        self,
        agent: Any,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
    ) -> Any:
        """Build ``agents.RunHooks`` that re-select ``agent.tools`` before each model call."""
        from agent_gantry.integrations.frameworks.openai_agents_live import (
            _gantry_run_hooks,
        )

        return _gantry_run_hooks(
            self._gantry,
            agent,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            **self._tool_kwargs(),
        )

    async def refresh(
        self,
        agent: Any,
        query_or_input: Any,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
    ) -> list[Any]:
        """Re-select tools and rewrite ``agent.tools`` in place; return the new tools."""
        from agent_gantry.integrations.frameworks.openai_agents_live import (
            _refresh_agent_tools,
        )

        return await _refresh_agent_tools(
            agent,
            self._gantry,
            query_or_input,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            **self._tool_kwargs(),
        )

    async def select_function_tools(
        self,
        query_or_input: Any,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
    ) -> list[Any]:
        """Re-select tools for a query string OR a run-input/message list; return ``FunctionTool``s."""
        from agent_gantry.integrations.frameworks.openai_agents_live import (
            _select_function_tools,
        )

        return await _select_function_tools(
            self._gantry,
            query_or_input,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            **self._tool_kwargs(),
        )
