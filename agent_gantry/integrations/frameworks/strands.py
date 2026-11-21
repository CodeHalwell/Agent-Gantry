"""AWS Strands Agents native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a Strands
``DecoratedFunctionTool`` — the native tool object a Strands ``Agent``
introspects (name / description / JSON-Schema input) and invokes. The
``strands`` import is lazy so ``import agent_gantry`` never requires Strands
Agents to be installed.

Public entry point: :class:`StrandsAdapter`.

Every call routes through ``gantry.execute`` and uses Strands' own mechanisms:

- **Errors** -- a failed call returns an error ``ToolResult``
  (``{"status": "error", "content": [{"text": "<type>: <message>"}]}``, capped,
  redactable with ``error_formatter``), so the model reads it and the run
  continues. ``error_mode="raise"`` raises
  :class:`~agent_gantry.integrations.frameworks.base.ToolExecutionError` instead
  (Strands then reports its own ``Error: ...`` text).
- **Approval** -- a confirmation-gated call (``requires_confirmation=True`` or a
  ``SecurityPolicy(require_confirmation=[...])`` pattern) raises a Strands
  interrupt from inside the tool (``tool_context.interrupt("gantry-approval",
  reason=...)``), so the run stops with ``result.stop_reason == "interrupt"``.
  Resume with ``agent([{"interruptResponse": {"interruptId": ..., "response":
  "yes"}}])`` to run the call exactly once; anything else rejects it.
- **Run context** -- the Strands ``ToolContext`` (``tool_use``, ``agent``,
  ``invocation_state``) is injected and readable with
  :func:`current_tool_context` or :func:`agent_gantry.current_tool_invocation`.
- **Tracing** -- Strands traces with OpenTelemetry; its active span becomes the
  Gantry ``ToolCall``'s trace and parent span.

Strands genuinely supports **per-turn** dynamic tool re-selection: it fires a
``BeforeModelCallEvent`` hook immediately before every model call, and only
reads ``agent.tool_registry`` for the tool specs sent to the model *after*
that hook runs (see
:mod:`agent_gantry.integrations.frameworks.strands_live`). That is deeper than
the per-top-level-call rebuild used for frameworks that fix their tool list at
agent construction (CrewAI, Agno, Haystack) — it mirrors the depth
of Google ADK's ``before_model_callback``.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from agent_gantry.core.invocation import current_tool_invocation
from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    BaseFrameworkAdapter,
    ToolConfirmationRequiredError,
    ToolExecutionError,
    ToolSpec,
)
from agent_gantry.integrations.frameworks.native import (
    ErrorFormatter,
    ErrorMode,
    check_error_mode,
    execute_spec,
    failure_text,
)

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

FRAMEWORK_NAME = "strands"

_INSTALL_HINT = (
    "Strands Agents support requires `strands-agents`. Install it with "
    "`pip install 'agent-gantry[strands]'` (or `uv add 'agent-gantry[strands]'`)."
)

_PENDING_APPROVAL_TEXT = (
    "This tool call requires human approval and was not run. Ask the user to "
    "approve it, or continue without it."
)
_REJECTED_TEXT = "The user rejected this tool call, so it was not run."

#: Interrupt responses (strings) that approve a gated call.
_APPROVALS = frozenset({"y", "yes", "approve", "approved"})

#: The interrupt name Gantry raises for confirmation-gated calls.
APPROVAL_INTERRUPT = "gantry-approval"


def current_tool_context() -> Any:
    """The Strands ``ToolContext`` of the Gantry tool currently executing.

    Carries ``tool_use``, ``agent`` and ``invocation_state`` (the keyword
    arguments passed to ``agent(...)``). ``None`` outside an agent run.
    """
    invocation = current_tool_invocation()
    if invocation is None or invocation.framework != FRAMEWORK_NAME:
        return None
    return invocation.native_context


def _is_approval(answer: Any) -> bool:
    if answer is True:
        return True
    if isinstance(answer, Mapping):
        return answer.get("approved") is True
    return isinstance(answer, str) and answer.strip().lower() in _APPROVALS


def _error_result(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": text}]}


def _spec_to_strands(
    spec: ToolSpec,
    *,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
) -> Any:
    """Wrap a :class:`ToolSpec` as a Strands ``DecoratedFunctionTool``.

    Builds an async callable with a real ``__signature__`` via
    :meth:`ToolSpec.callable_for_signature` (so Strands' ``@tool`` decorator
    validates and advertises the actual parameters, not a bare ``**kwargs``
    no-argument tool), then decorates it with explicit
    ``name``/``description``/``inputSchema`` overrides — all three are
    supported directly by ``strands.tool()`` — so Gantry's own metadata,
    including per-parameter descriptions, wins over whatever the decorator
    would otherwise infer from the wrapper's docstring and its own
    Pydantic-derived schema. The advertised schema uses the signature's
    parameter aliases, since Strands validates the model's input against a
    model built from that signature.

    The ``strands`` import happens here, lazily, so callers without Strands
    Agents installed only hit the error when they actually export a tool.

    Raises:
        ImportError: If ``strands-agents`` is not installed.
    """
    try:
        from strands import tool as strands_tool
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(_INSTALL_HINT) from exc

    error_mode = check_error_mode(error_mode)
    model_mode = error_mode == "model"
    signature_fn = spec.callable_for_signature()
    aliases = spec.parameter_aliases()

    def _needs_approval() -> bool:
        gantry = spec._gantry
        return (
            model_mode
            and gantry is not None
            and gantry.requires_confirmation(spec.tool or spec.name, spec._namespace)
        )

    async def _call(tool_context: Any = None, **kwargs: Any) -> Any:
        if aliases:
            kwargs = {aliases.get(key, key): value for key, value in kwargs.items()}
        approved = False
        if tool_context is not None and _needs_approval():
            # The tool re-runs from the top when the run resumes, at which
            # point interrupt() returns the human's response instead.
            answer = tool_context.interrupt(
                APPROVAL_INTERRUPT,
                reason={
                    "type": "gantry_approval",
                    "tool": spec.native_name,
                    "description": spec.description,
                    "arguments": dict(kwargs),
                },
            )
            if not _is_approval(answer):
                return _error_result(_REJECTED_TEXT)
            approved = True
        tool_use = getattr(tool_context, "tool_use", None) or {}
        agent = getattr(tool_context, "agent", None)
        name = getattr(agent, "name", None)
        try:
            return await execute_spec(
                spec,
                kwargs,
                framework=FRAMEWORK_NAME,
                native_context=tool_context,
                deps=getattr(tool_context, "invocation_state", None),
                tool_call_id=tool_use.get("toolUseId") if isinstance(tool_use, Mapping) else None,
                metadata={"agent_name": name} if isinstance(name, str) else None,
                approved=approved,
            )
        except ToolExecutionError as exc:
            if not model_mode:
                raise
            if isinstance(exc, ToolConfirmationRequiredError):
                return _error_result(_PENDING_APPROVAL_TEXT)
            return _error_result(failure_text(exc, error_formatter))

    signature = signature_fn.__signature__  # type: ignore[attr-defined]
    context_param = inspect.Parameter("tool_context", inspect.Parameter.KEYWORD_ONLY, default=None)
    params = list(signature.parameters.values())
    positional = [p for p in params if p.kind is not inspect.Parameter.KEYWORD_ONLY]
    keyword = [p for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY]
    _call.__name__ = spec.native_name
    _call.__doc__ = spec.description
    _call.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=[*positional, context_param, *keyword]
    )
    _call.__annotations__ = dict(getattr(signature_fn, "__annotations__", {}))
    return strands_tool(
        name=spec.native_name,
        description=spec.description,
        inputSchema={"json": spec.aliased_parameters()},
        context=True,
    )(_call)


class StrandsAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into AWS Strands Agents.

    Static slice (``DecoratedFunctionTool`` objects for ``Agent(tools=[...])``)
    plus a genuine per-turn live hook — Strands fires a ``BeforeModelCallEvent``
    before every model call and only reads the tool registry afterward, so a
    hook that swaps the registry during that event changes what the model sees
    on the very call about to happen (see
    :mod:`agent_gantry.integrations.frameworks.strands_live`). Every call routes
    through ``gantry.execute`` so retries, timeouts, circuit breakers, and the
    security policy all apply::

        from agent_gantry import AgentGantry
        from agent_gantry.strands import StrandsAdapter
        from strands import Agent

        gantry = AgentGantry()
        # ... register tools, await gantry.sync() ...

        tools = await StrandsAdapter(gantry).select("email the quarterly report", limit=3)
        agent = Agent(tools=tools)

    :meth:`live` returns the :meth:`tool_hook` — plug it into
    ``Agent(tools=[], hooks=[<result>])`` or ``agent.add_hook(<result>)``.
    """

    live_tier = "per-turn"
    _live_delegate = "tool_hook"

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

    def _mode(self, error_mode: ErrorMode | None) -> ErrorMode:
        return self._error_mode if error_mode is None else check_error_mode(error_mode)

    @staticmethod
    def convert(
        spec: ToolSpec,
        *,
        error_mode: ErrorMode = "model",
        error_formatter: ErrorFormatter | None = None,
    ) -> Any:
        """Wrap a single :class:`ToolSpec` as a Strands ``DecoratedFunctionTool``."""
        return _spec_to_strands(spec, error_mode=error_mode, error_formatter=error_formatter)

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
        """Select tools for ``query`` once, as Strands ``DecoratedFunctionTool`` objects."""
        from agent_gantry.integrations.frameworks.base import GantryToolset

        mode = self._mode(error_mode)
        specs = await GantryToolset(self._gantry).select(
            query,
            tools_already_used=tools_already_used,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
        )
        return [
            _spec_to_strands(s, error_mode=mode, error_formatter=self._error_formatter)
            for s in specs
        ]

    def tool_hook(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
    ) -> Any:
        """Build a ``HookProvider`` that re-selects Gantry tools before every model call.

        Pass the result straight to ``Agent(hooks=[...])`` (or
        ``agent.add_hook(hook)`` on an already-built agent). Each turn it
        re-runs semantic selection against the latest user message and swaps
        the agent's tool registry in place — the deepest re-selection tier
        Strands supports.
        """
        from agent_gantry.integrations.frameworks.strands_live import (
            GantryStrandsToolHook,
        )

        return GantryStrandsToolHook(
            self._gantry,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            error_mode=self._error_mode,
            error_formatter=self._error_formatter,
        )

    def agent(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **agent_kwargs: Any,
    ) -> Any:
        """Build a ``strands.Agent`` wired for per-turn dynamic tool selection.

        Constructs ``Agent(tools=[], hooks=[<tool_hook>], **agent_kwargs)`` —
        the agent ships with no statically registered tools; the hook injects
        the relevant slice before every model call (see :meth:`tool_hook`).
        ``agent_kwargs`` (``model``, ``system_prompt``, ...) are forwarded.
        """
        from agent_gantry.integrations.frameworks.strands_live import (
            _gantry_strands_agent,
        )

        return _gantry_strands_agent(
            self._gantry,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            error_mode=self._error_mode,
            error_formatter=self._error_formatter,
            **agent_kwargs,
        )


__all__ = ["APPROVAL_INTERRUPT", "StrandsAdapter", "current_tool_context"]
