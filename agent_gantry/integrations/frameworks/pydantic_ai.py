"""Pydantic AI native tool adapter for Agent-Gantry.

Pydantic AI is a first-class target: Gantry plugs into each of its native
extension points rather than handing it opaque callables.

- **Static tools** -- :meth:`PydanticAIAdapter.select` / :meth:`~PydanticAIAdapter.convert`
  build ``pydantic_ai.Tool`` objects from Gantry's JSON schema verbatim.
- **Per-turn toolset** -- :meth:`PydanticAIAdapter.toolset` returns an
  ``AbstractToolset`` that re-selects on every model step from the newest user
  prompt or tool result (``Agent(model, toolsets=[...])``).
- **Tool search** -- :meth:`PydanticAIAdapter.deferred_toolset` exposes the
  catalogue as ``defer_loading`` tools and :meth:`PydanticAIAdapter.tool_search`
  backs Pydantic AI's ``ToolSearch`` capability with Gantry's semantic router,
  so the model discovers tools by meaning while the tool list (and the provider
  prompt cache) stays stable.
- **Capability** -- :meth:`PydanticAIAdapter.capability` bundles either mode
  with per-turn Gantry skills as instructions (``Agent(model, capabilities=[...])``).

Every path executes through ``gantry.execute`` (security policy, retries,
timeouts, circuit breakers, telemetry) and speaks Pydantic AI's own control
flow: argument errors become ``ModelRetry`` so the model can correct itself,
terminal failures become ``ToolFailed``, and Gantry's confirmation gate becomes
``ApprovalRequired`` -- Pydantic AI's human-in-the-loop flow -- with an
approved replay executing exactly once. The run's ``RunContext`` (``deps``,
``run_id``, ``tool_call_id``) is visible to Gantry tool handlers through
:func:`agent_gantry.current_tool_invocation`, and the active OpenTelemetry span
is propagated into Gantry's execution telemetry.

The ``pydantic_ai`` import is lazy so ``import agent_gantry`` never requires
Pydantic AI to be installed. Supported: ``pydantic-ai`` / ``pydantic-ai-slim``
2.x (``ToolFailed`` is used where available; on earlier 2.x releases a
terminal failure is returned as the tool's result instead).

Public entry point: :class:`PydanticAIAdapter`.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Literal

from agent_gantry.core.invocation import tool_invocation
from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    BaseFrameworkAdapter,
    GantryToolset,
    ToolConfirmationRequiredError,
    ToolExecutionError,
    ToolPermissionDeniedError,
    ToolSpec,
    _coerce_arguments,
)
from agent_gantry.schema.execution import ToolInvocationContext

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

#: How a failed Gantry execution surfaces inside a Pydantic AI run.
#:
#: - ``"model"`` (default) -- Pydantic AI's own control flow: argument
#:   validation errors raise ``ModelRetry`` (the model sees the error and
#:   retries within the tool's retry budget), other failures raise
#:   ``ToolFailed`` (the model sees a terminal failure and adapts), and a
#:   confirmation-gated call raises ``ApprovalRequired``.
#:
#:   The model sees the failure's text (``error_type: message``), as it would
#:   from a hand-written tool -- so an exception message carrying internals
#:   (paths, connection strings) reaches the model and its logs. Raise
#:   model-safe messages from handlers, or use ``"raise"``.
#: - ``"raise"`` -- the cross-framework default contract: the
#:   :class:`~agent_gantry.integrations.frameworks.base.ToolExecutionError`
#:   family propagates and ends the run.
ErrorMode = Literal["model", "raise"]

FRAMEWORK_NAME = "pydantic_ai"

_INSTALL_HINT = (
    "Pydantic AI support requires `pydantic-ai` (2.x). Install it with "
    "`pip install 'agent-gantry[pydantic-ai]'` (or `uv add 'agent-gantry[pydantic-ai]'`)."
)

#: Executor ``error_type`` values the model can fix by changing its call.
_MODEL_CORRECTABLE_ERRORS = frozenset({"ValidationError", "ToolNotFound"})

#: Longest failure text shown to the model. Exception messages can carry
#: stack-like detail or echoed payloads; the head is what the model needs.
_MAX_ERROR_CHARS = 2000

#: Key under ``ToolDefinition.metadata`` carrying Gantry's identity for a tool.
METADATA_KEY = "agent_gantry"


def registry_token(gantry: AgentGantry) -> str:
    """An in-process identity for ``gantry``, stored in tool metadata.

    Lets a search strategy tell its own registry's tools from those of another
    ``AgentGantry`` whose deferred toolset sits on the same agent.
    """
    return f"{id(gantry):x}"


def check_error_mode(error_mode: str) -> ErrorMode:
    """Validate an ``error_mode`` value early, at construction time."""
    if error_mode not in ("model", "raise"):
        raise ValueError(f"error_mode must be 'model' or 'raise', not {error_mode!r}")
    return error_mode  # type: ignore[return-value]


def _otel_ids() -> tuple[str | None, str | None]:
    """The active OpenTelemetry span's trace/span ids, when one is recording.

    Pydantic AI opens a span per tool call when instrumented; passing its ids
    to ``gantry.execute`` puts Gantry's execution telemetry on the same trace.
    """
    try:
        from opentelemetry import trace
    except ImportError:
        return None, None
    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return None, None
    return format(span_context.trace_id, "032x"), format(span_context.span_id, "016x")


def invocation_context(ctx: Any) -> ToolInvocationContext:
    """Describe a Pydantic AI ``RunContext`` for Gantry tool handlers."""
    return ToolInvocationContext(
        framework=FRAMEWORK_NAME,
        native_context=ctx,
        deps=getattr(ctx, "deps", None),
        run_id=getattr(ctx, "run_id", None),
        tool_call_id=getattr(ctx, "tool_call_id", None),
        metadata={
            "retry": getattr(ctx, "retry", 0),
            "run_step": getattr(ctx, "run_step", None),
            "tool_call_approved": getattr(ctx, "tool_call_approved", False) is True,
        },
    )


def current_run_context() -> Any:
    """The Pydantic AI ``RunContext`` of the tool call executing now, or ``None``.

    Call from inside a Gantry tool handler to reach ``ctx.deps`` and friends::

        @gantry.register
        async def get_balance(account: str) -> str:
            \"\"\"Fetch an account balance.\"\"\"
            ctx = current_run_context()
            return await ctx.deps.bank.balance(account)
    """
    from agent_gantry.core.invocation import current_tool_invocation

    invocation = current_tool_invocation()
    if invocation is None or invocation.framework != FRAMEWORK_NAME:
        return None
    return invocation.native_context


def _capped(message: str) -> str:
    if len(message) <= _MAX_ERROR_CHARS:
        return message
    return message[:_MAX_ERROR_CHARS] + " ... [truncated]"


def failure_text(exc: ToolExecutionError) -> str:
    """The model-facing text for a failed execution (``error_type: message``, capped)."""
    message = exc.error or "tool execution failed (no error message)"
    if isinstance(exc, ToolPermissionDeniedError):
        message = f"Permission denied: {message}"
    elif (
        exc.error_type
        and exc.error_type not in message
        and (exc.error_type not in _MODEL_CORRECTABLE_ERRORS)
    ):
        message = f"{exc.error_type}: {message}"
    return _capped(message)


def to_pydantic_ai_exception(spec: ToolSpec, exc: ToolExecutionError) -> Exception | None:
    """Translate a failed Gantry execution into Pydantic AI's control flow.

    Returns ``None`` for a terminal failure on a Pydantic AI release without
    ``ToolFailed``: :func:`execute_for_pydantic_ai` then returns the failure
    text as the tool's result, which is terminal (no retry loop) on every
    release. ``ModelRetry`` would make the model repeat a call that cannot
    succeed until the retry budget ran out and the run failed.
    """
    try:
        from pydantic_ai.exceptions import ApprovalRequired, ModelRetry
    except ImportError as import_exc:  # pragma: no cover - adapter needs pydantic_ai
        raise ImportError(_INSTALL_HINT) from import_exc

    if isinstance(exc, ToolConfirmationRequiredError):
        return ApprovalRequired(
            metadata={
                METADATA_KEY: {
                    "qualified_name": spec.qualified_name,
                    "reason": exc.error,
                }
            }
        )
    message = failure_text(exc)
    if exc.error_type in _MODEL_CORRECTABLE_ERRORS:
        return ModelRetry(message)
    try:
        from pydantic_ai.exceptions import ToolFailed
    except ImportError:  # 2.x releases before ToolFailed existed
        return None
    return ToolFailed(message)


async def execute_for_pydantic_ai(
    spec: ToolSpec,
    arguments: Any,
    ctx: Any,
    *,
    error_mode: ErrorMode = "model",
) -> Any:
    """Run ``spec`` through ``gantry.execute`` on behalf of a Pydantic AI tool call.

    - ``ctx.tool_call_approved`` (set by Pydantic AI only when the host
      approved a deferred call via ``DeferredToolResults``) clears Gantry's
      confirmation gate for this one call; every denial check still runs.
    - ``ctx`` is exposed to the handler via :func:`current_run_context`.
    - The active OpenTelemetry span is passed on as the call's trace parent.
    """
    approved = ctx is not None and getattr(ctx, "tool_call_approved", False) is True
    trace_id, parent_span_id = _otel_ids()
    args = _coerce_arguments((), dict(arguments or {}))
    scope = tool_invocation(invocation_context(ctx)) if ctx is not None else nullcontext()
    try:
        with scope:
            return await spec._ainvoke(
                args,
                confirmation_approved=approved,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
            )
    except ToolExecutionError as exc:
        if error_mode == "raise":
            raise
        mapped = to_pydantic_ai_exception(spec, exc)
        if mapped is None:
            return f"Error: {failure_text(exc)}"
        raise mapped from exc


def json_schema_for(spec: ToolSpec) -> dict[str, Any]:
    """The tool's JSON schema, copied so nothing downstream mutates the registry's."""
    return copy.deepcopy(spec.parameters) or {"type": "object", "properties": {}}


def _spec_to_pydantic_ai(spec: ToolSpec, *, error_mode: ErrorMode = "model") -> Any:
    """Wrap a :class:`ToolSpec` as a Pydantic AI ``Tool``.

    Uses ``Tool.from_schema`` so the JSON schema Gantry already produced is sent
    verbatim (enums, descriptions, nested objects), with ``takes_ctx=True`` so
    the ``RunContext`` reaches :func:`execute_for_pydantic_ai`. The context is
    positional-only, so a schema property called ``ctx`` cannot collide.

    The ``pydantic_ai`` import happens here, lazily, so callers without Pydantic
    AI installed only hit the error when they actually export a tool.
    """
    try:
        from pydantic_ai.tools import Tool
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    async def _call(run_context: Any, /, **kwargs: Any) -> Any:
        return await execute_for_pydantic_ai(spec, kwargs, run_context, error_mode=error_mode)

    _call.__name__ = _call.__qualname__ = spec.native_name
    _call.__doc__ = spec.description
    return Tool.from_schema(
        function=_call,
        name=spec.native_name,
        description=spec.description,
        json_schema=json_schema_for(spec),
        takes_ctx=True,
    )


class PydanticAIAdapter(BaseFrameworkAdapter):
    """Route Gantry tools into Pydantic AI through its native extension points.

    Static ``Tool`` slices (:meth:`select`), a per-turn ``AbstractToolset``
    (:meth:`toolset`, also what :meth:`live` returns), semantic tool search
    (:meth:`deferred_toolset` + :meth:`tool_search`) and a bundled
    ``Capability`` (:meth:`capability`). All route execution through
    ``gantry.execute`` and map outcomes to ``ModelRetry`` / ``ToolFailed`` /
    ``ApprovalRequired`` (see :data:`ErrorMode`).

    Args:
        gantry: The registry to select from and execute against.
        default_limit: Default selection budget.
        error_mode: Default :data:`ErrorMode` for every tool this adapter builds.
    """

    live_tier = "per-turn"
    _live_delegate = "toolset"

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        default_limit: int = DEFAULT_TOOL_LIMIT,
        error_mode: ErrorMode = "model",
    ) -> None:
        super().__init__(gantry, default_limit=default_limit)
        self._error_mode = check_error_mode(error_mode)

    def _mode(self, error_mode: ErrorMode | None) -> ErrorMode:
        return self._error_mode if error_mode is None else check_error_mode(error_mode)

    @staticmethod
    def convert(spec: ToolSpec, *, error_mode: ErrorMode = "model") -> Any:
        """Wrap a single :class:`ToolSpec` as a Pydantic AI ``Tool``."""
        return _spec_to_pydantic_ai(spec, error_mode=check_error_mode(error_mode))

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
        """Select tools for ``query`` once, as Pydantic AI ``Tool`` objects.

        Hand the result to ``Agent(model, tools=[...])``. See
        :meth:`GantryToolset.select` for the ``required``/``always_include``
        pinning contract.
        """
        mode = self._mode(error_mode)
        specs = await GantryToolset(self._gantry).select(
            query,
            tools_already_used=tools_already_used,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
        )
        return [self.convert(s, error_mode=mode) for s in specs]

    def toolset(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        error_mode: ErrorMode | None = None,
        id: str | None = None,  # noqa: A002 - mirrors Pydantic AI's toolset ``id``
        sticky: bool = True,
        max_retries: int = 1,
        query_from_results: bool = True,
    ) -> Any:
        """A live ``AbstractToolset`` that re-selects tools on every model step.

        Plug into ``Agent(model, toolsets=[...])``. Each step's query is the
        newest user prompt, or -- after a tool call -- the run's prompt plus
        the newest tool result, so a chained task surfaces the next tool it
        needs. Selections are cached per run, and with ``sticky`` (default)
        tools offered earlier in a run stay available for the rest of it.

        Args:
            id: Toolset id, unique within an agent (default ``"agent-gantry"``);
                required to differ when an agent uses several Gantry toolsets,
                and stable ids are what durable execution keys on.
            sticky: Keep earlier selections within a run.
            max_retries: ``ModelRetry`` budget per tool (Pydantic AI's default is 1).
            query_from_results: Let tool output (its first 2,000 characters)
                drive the next step's selection. That text is sent to the
                embedder -- a remote service for OpenAI-style embedders -- so
                set ``False`` when tool results may hold data that must not
                leave the process; selection then uses the run's prompt.
        """
        from agent_gantry.integrations.frameworks.pydantic_ai_live import _gantry_toolset

        return _gantry_toolset(
            self._gantry,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            error_mode=self._mode(error_mode),
            id=id,
            sticky=sticky,
            max_retries=max_retries,
            query_from_results=query_from_results,
        )

    def deferred_toolset(
        self,
        *,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        error_mode: ErrorMode | None = None,
        id: str | None = None,  # noqa: A002 - mirrors Pydantic AI's toolset ``id``
        max_retries: int = 1,
    ) -> Any:
        """The whole catalogue as ``defer_loading`` tools, for ``ToolSearch``.

        The model sees none of these tools until it discovers them through
        Pydantic AI's ``search_tools`` (or the provider's native tool search);
        pair with :meth:`tool_search` so discovery uses Gantry's semantic
        router. ``required``/``always_include`` tools are left visible.
        """
        from agent_gantry.integrations.frameworks.pydantic_ai_live import _gantry_deferred_toolset

        return _gantry_deferred_toolset(
            self._gantry,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
            error_mode=self._mode(error_mode),
            id=id,
            max_retries=max_retries,
        )

    def search_strategy(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
    ) -> Any:
        """A ``ToolSearch`` strategy function backed by Gantry's semantic router.

        ``ToolSearch(strategy=adapter.search_strategy())``. Gantry-managed tools
        are ranked by semantic retrieval; any other deferred tools on the agent
        fall back to keyword matching so they remain discoverable.
        """
        from agent_gantry.integrations.frameworks.pydantic_ai_capability import (
            gantry_search_strategy,
        )

        return gantry_search_strategy(
            self._gantry,
            **self._selection_kwargs(limit, score_threshold, namespaces, None, None),
        )

    def tool_search(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
    ) -> Any:
        """Pydantic AI's ``ToolSearch`` capability using :meth:`search_strategy`."""
        from pydantic_ai.capabilities import ToolSearch

        resolved = self._default_limit if limit is None else limit
        return ToolSearch(
            strategy=self.search_strategy(
                limit=resolved, score_threshold=score_threshold, namespaces=namespaces
            ),
            max_results=resolved,
        )

    def capability(
        self,
        *,
        mode: Literal["per-turn", "search"] = "per-turn",
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        error_mode: ErrorMode | None = None,
        skills: int = 0,
        skill_namespaces: list[str] | None = None,
        id: str | None = None,  # noqa: A002 - mirrors Pydantic AI's capability ``id``
        sticky: bool = True,
        max_retries: int = 1,
        query_from_results: bool = True,
    ) -> Any:
        """One ``Capability`` wiring Gantry into an agent: ``Agent(model, capabilities=[...])``.

        Args:
            mode: ``"per-turn"`` pushes a fresh top-k each step
                (:meth:`toolset`); ``"search"`` defers the whole catalogue and
                lets the model pull tools in via Gantry-backed ``ToolSearch``
                (:meth:`deferred_toolset` + :meth:`tool_search`) -- the tool
                list stays stable, so provider prompt caching survives.
            skills: When > 0, inject up to this many Gantry skills relevant to
                the current turn as instructions (requires a skills store).
            skill_namespaces: Restrict injected skills to these namespaces.

        The remaining arguments are as for :meth:`toolset`.
        """
        from agent_gantry.integrations.frameworks.pydantic_ai_capability import (
            _gantry_capability,
        )

        if mode not in ("per-turn", "search"):
            raise ValueError(f"mode must be 'per-turn' or 'search', not {mode!r}")
        if skills < 0:
            raise ValueError("skills must be >= 0")
        kwargs = self._selection_kwargs(
            limit, score_threshold, namespaces, required, always_include
        )
        mode_error = self._mode(error_mode)
        if mode == "per-turn":
            toolset = self.toolset(
                **kwargs,
                error_mode=mode_error,
                id=id,
                sticky=sticky,
                max_retries=max_retries,
                query_from_results=query_from_results,
            )
            search = None
        else:
            toolset = self.deferred_toolset(
                namespaces=namespaces,
                required=required,
                always_include=always_include,
                error_mode=mode_error,
                id=id,
                max_retries=max_retries,
            )
            search = self.tool_search(
                limit=kwargs["limit"], score_threshold=score_threshold, namespaces=namespaces
            )
        return _gantry_capability(
            self._gantry,
            toolset,
            tool_search=search,
            skills=skills,
            skill_namespaces=skill_namespaces,
            id=id,
            query_from_results=query_from_results,
        )
