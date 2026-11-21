"""CrewAI native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a CrewAI
``BaseTool``, routed through ``gantry.execute`` and using CrewAI's own
mechanisms:

- **Errors** -- a failed call returns CrewAI's ``ToolFailure`` (message, error
  ``code``, ``retryable``), so the agent reads the failure and CrewAI records it
  on ``TaskOutput.tool_failures``; the agent's or task's ``tool_failure_policy``
  decides whether the run carries on or aborts. ``error_formatter`` redacts the
  text; ``error_mode="raise"`` lets
  :class:`~agent_gantry.integrations.frameworks.base.ToolExecutionError` propagate.
- **Approval** -- a confirmation-gated tool (``requires_confirmation=True`` or a
  ``SecurityPolicy(require_confirmation=[...])`` pattern) is checked in a
  CrewAI ``before_tool_call`` hook. With an approver (``approve=`` a callable,
  or ``"console"`` for CrewAI's ``request_human_input`` prompt) an approved
  call runs once with only Gantry's confirmation gate cleared and a refused one
  is blocked; without one the agent is told the call needs approval.
- **Run context** -- a Gantry tool reads the CrewAI agent, task and crew of the
  call with :func:`current_crew_context`, or through
  :func:`agent_gantry.current_tool_invocation`.
- **Tracing** -- an active OpenTelemetry span becomes the Gantry ``ToolCall``'s
  trace and parent span.

CrewAI fixes an agent's tools when it is built, so the live path
(:meth:`CrewAIAdapter.agent_builder`) re-selects per call. The ``crewai``
import is lazy so ``import agent_gantry`` never requires CrewAI.

Public entry point: :class:`CrewAIAdapter`.
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from agent_gantry.core.invocation import current_tool_invocation
from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    BaseFrameworkAdapter,
    ToolConfirmationRequiredError,
    ToolExecutionError,
    ToolSpec,
    _run_coroutine_sync,
)
from agent_gantry.integrations.frameworks.native import (
    MODEL_CORRECTABLE_ERRORS,
    ErrorFormatter,
    ErrorMode,
    check_error_mode,
    execute_spec,
    failure_text,
)

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

logger = logging.getLogger(__name__)

FRAMEWORK_NAME = "crewai"

_INSTALL_HINT = (
    "CrewAI support requires `crewai` on Python 3.10–3.13 (its SDK requires Python <3.14). "
    "Install it with `pip install 'agent-gantry[crewai]'` (or `uv add 'agent-gantry[crewai]'`)."
)

_PENDING_APPROVAL_TEXT = (
    "This tool call requires human approval and was not run. Ask the user to "
    "approve it, or continue without it."
)


@dataclass(frozen=True)
class ApprovalRequest:
    """A confirmation-gated Gantry tool call waiting for a decision.

    Attributes:
        tool_name: The tool's name as the agent sees it.
        arguments: The arguments the agent called it with.
        description: The tool's description.
        agent: The CrewAI agent making the call, when known.
        task: The CrewAI task being worked on, when known.
        crew: The crew, when known.
    """

    tool_name: str
    arguments: dict[str, Any]
    description: str = ""
    agent: Any = None
    task: Any = None
    crew: Any = None
    hook_context: Any = field(default=None, repr=False)


#: Decides an :class:`ApprovalRequest`: ``True`` runs the call, anything else blocks it.
Approver = Callable[[ApprovalRequest], bool]

#: The CrewAI ``ToolCallHookContext`` of the call about to run (set by Gantry's hook).
_HOOK_CONTEXT: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "agent_gantry_crewai_hook_context", default=None
)
#: Name of the tool whose next call the approver just allowed. CrewAI runs the
#: hook and then the tool in the same context, so this is consumed by that call.
_APPROVED: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agent_gantry_crewai_approved", default=None
)

#: native tool name -> (gated check, approver) for every Gantry CrewAI tool built.
_GATES: dict[str, tuple[Callable[[], bool], Approver | None, str]] = {}
_HOOK_LOCK = threading.Lock()
_HOOK_INSTALLED = False


def current_crew_context() -> Any:
    """The CrewAI ``ToolCallHookContext`` of the Gantry tool currently executing.

    Its ``agent``, ``task`` and ``crew`` attributes are the CrewAI objects
    running the call. ``None`` outside a CrewAI tool call, or when CrewAI's
    hooks are unavailable.
    """
    invocation = current_tool_invocation()
    if invocation is None or invocation.framework != FRAMEWORK_NAME:
        return None
    return invocation.native_context


def _console_approver(request: ApprovalRequest) -> bool:
    """CrewAI's own ``request_human_input`` prompt; only "y"/"yes" approves."""
    context = request.hook_context
    prompt = (
        f"Agent-Gantry: allow tool {request.tool_name!r} with arguments "
        f"{json.dumps(request.arguments, default=str)}? [y/N]"
    )
    answer = context.request_human_input(prompt) if context is not None else ""
    return str(answer).strip().lower() in {"y", "yes"}


def _gantry_before_tool_call(context: Any) -> bool | None:
    """CrewAI ``before_tool_call`` hook: run context and the approval gate."""
    _HOOK_CONTEXT.set(context)
    _APPROVED.set(None)
    name = getattr(context, "tool_name", None)
    gate = _GATES.get(name) if isinstance(name, str) else None
    if gate is None:
        return None
    needs_approval, approver, description = gate
    if approver is None or not needs_approval():
        return None
    arguments = dict(getattr(context, "tool_input", None) or {})
    request = ApprovalRequest(
        tool_name=name,
        arguments=arguments,
        description=description,
        agent=getattr(context, "agent", None),
        task=getattr(context, "task", None),
        crew=getattr(context, "crew", None),
        hook_context=context,
    )
    try:
        approved = approver(request) is True
    except Exception:
        logger.warning("CrewAI approver for %r raised; blocking the call.", name, exc_info=True)
        approved = False
    if not approved:
        return False  # CrewAI blocks the call and tells the agent
    _APPROVED.set(name)
    return None


def _install_hook() -> None:
    """Register Gantry's ``before_tool_call`` hook with CrewAI once per process."""
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    with _HOOK_LOCK:
        if _HOOK_INSTALLED:
            return
        try:
            from crewai.hooks import register_before_tool_call_hook
        except ImportError:  # CrewAI without hooks: no run context or approver
            logger.debug("CrewAI has no tool hooks; Gantry approval and run context are off.")
            _HOOK_INSTALLED = True
            return
        register_before_tool_call_hook(_gantry_before_tool_call)
        _HOOK_INSTALLED = True


def _failure(exc: ToolExecutionError, formatter: ErrorFormatter | None) -> Any:
    """The failure as CrewAI's ``ToolFailure``, or as text on older CrewAI."""
    if isinstance(exc, ToolConfirmationRequiredError):
        message = _PENDING_APPROVAL_TEXT
    else:
        message = failure_text(exc, formatter)
    try:
        from crewai.tools import ToolFailure
    except ImportError:
        return f"Error: {message}"
    return ToolFailure(
        message=message,
        code=exc.error_type or None,
        retryable=(exc.error_type or "") in MODEL_CORRECTABLE_ERRORS,
        details={"gantry_status": exc.status} if exc.status else {},
    )


def _spec_to_crewai(
    spec: ToolSpec,
    *,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
    approve: Approver | Literal["console"] | None = None,
) -> Any:
    """Wrap a :class:`ToolSpec` as a CrewAI ``BaseTool``.

    A ``BaseTool`` subclass is built whose ``name`` / ``description`` come from
    the spec, whose ``args_schema`` is generated from its JSON schema, and whose
    ``_run`` routes through ``gantry.execute``.

    Args:
        spec: The tool to wrap.
        error_mode: ``"model"`` (failures become ``ToolFailure``, gated calls go
            through the approver) or ``"raise"``.
        error_formatter: Rewrites failure text before the agent sees it.
        approve: Decides confirmation-gated calls (a callable taking an
            :class:`ApprovalRequest`, or ``"console"``); ``None`` reports them
            as needing approval.

    Raises:
        ImportError: If ``crewai`` is not installed.
    """
    try:
        from crewai.tools import BaseTool
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(_INSTALL_HINT) from exc

    error_mode = check_error_mode(error_mode)
    model_mode = error_mode == "model"
    approver: Approver | None = _console_approver if approve == "console" else approve  # type: ignore[assignment]
    gantry = spec._gantry
    if model_mode and gantry is not None:
        tool_ref = spec.tool or spec.name
        namespace = spec._namespace

        def _needs_approval() -> bool:
            return gantry.requires_confirmation(tool_ref, namespace)

        _GATES[spec.native_name] = (_needs_approval, approver, spec.description)
        _install_hook()

    def _execute(kwargs: dict[str, Any]) -> Any:
        context = _HOOK_CONTEXT.get()
        if context is not None and getattr(context, "tool_name", None) != spec.native_name:
            context = None  # a stale context from another tool's call
        approved = model_mode and _APPROVED.get() == spec.native_name
        coro = execute_spec(
            spec,
            kwargs,
            framework=FRAMEWORK_NAME,
            native_context=context,
            deps=getattr(context, "crew", None) if context is not None else None,
            metadata=_metadata(context),
            approved=approved,
        )
        try:
            return _run_coroutine_sync(coro)
        except ToolExecutionError as exc:
            if not model_mode:
                raise
            return _failure(exc, error_formatter)
        finally:
            _APPROVED.set(None)

    # CrewAI's BaseTool is a Pydantic v2 model: ``name`` / ``description`` are
    # declared fields and MUST be set via the constructor. ``_run`` closes over
    # ``spec``; the generated ``args_schema`` surfaces the real parameters.
    class GantryCrewAITool(BaseTool):  # type: ignore[misc, valid-type]
        def _run(self, **kwargs: Any) -> Any:
            return _execute(kwargs)

    kwargs: dict[str, Any] = {"name": spec.native_name, "description": spec.description}
    args_schema = _build_args_schema(spec)
    if args_schema is not None:
        kwargs["args_schema"] = args_schema
    return GantryCrewAITool(**kwargs)


def _metadata(context: Any) -> dict[str, Any]:
    if context is None:
        return {}
    extras: dict[str, Any] = {}
    role = getattr(getattr(context, "agent", None), "role", None)
    if isinstance(role, str):
        extras["agent_role"] = role
    task = getattr(context, "task", None)
    for key, attr in (("task_name", "name"), ("task_id", "id")):
        value = getattr(task, attr, None)
        if value is not None:
            extras[key] = str(value)
    return extras


def _build_args_schema(spec: ToolSpec) -> Any:
    """Build a Pydantic args model from the spec's JSON-Schema parameters.

    Delegates to the shared
    :func:`~agent_gantry.integrations.frameworks.schema_bridge.pydantic_model_from_schema`
    bridge, so per-parameter descriptions, enums (as ``Literal``), typed array
    items, nested objects, and defaults all survive into the model CrewAI
    advertises to the LLM. Returns ``None`` when there are no properties
    (CrewAI then uses its own default empty schema) or when the schema can't
    be expressed — the tool is still usable, sans typed args.
    """
    properties = spec.parameters.get("properties") or {}
    if not properties:
        return None
    from agent_gantry.integrations.frameworks.schema_bridge import (
        pydantic_model_from_schema,
    )

    return pydantic_model_from_schema(f"{spec.native_name}_Args", spec.parameters)


class CrewAIAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into CrewAI through its native mechanisms.

    Static slice (``crewai.tools.BaseTool`` objects from :meth:`select`) plus a
    per-call live builder (CrewAI fixes an agent's tools at construction, so
    :meth:`agent_builder` rebuilds a fresh agent per call). Every call routes
    through ``gantry.execute``; see the module docs for errors, approval, run
    context and tracing.

    :meth:`live` returns the :meth:`agent_builder` builder; call
    ``await builder.build(query)`` before each new task.

    Args:
        gantry: The registry to select from and execute against.
        default_limit: Default selection budget.
        error_mode: Default ``ErrorMode`` for the tools this adapter builds.
        error_formatter: Rewrites failure text before the agent sees it.
        approve: Decides confirmation-gated calls: a callable taking an
            :class:`ApprovalRequest` and returning ``True`` to allow, or
            ``"console"`` for CrewAI's own ``request_human_input`` prompt.
            ``None`` (default) tells the agent the call needs approval.
    """

    live_tier = "per-call"
    _live_delegate = "agent_builder"

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        default_limit: int = DEFAULT_TOOL_LIMIT,
        error_mode: ErrorMode = "model",
        error_formatter: ErrorFormatter | None = None,
        approve: Approver | Literal["console"] | None = None,
    ) -> None:
        super().__init__(gantry, default_limit=default_limit)
        self._error_mode = check_error_mode(error_mode)
        self._error_formatter = error_formatter
        if approve is not None and approve != "console" and not callable(approve):
            raise ValueError("approve must be a callable, 'console', or None")
        self._approve = approve

    def _mode(self, error_mode: ErrorMode | None) -> ErrorMode:
        return self._error_mode if error_mode is None else check_error_mode(error_mode)

    @staticmethod
    def convert(
        spec: ToolSpec,
        *,
        error_mode: ErrorMode = "model",
        error_formatter: ErrorFormatter | None = None,
        approve: Approver | Literal["console"] | None = None,
    ) -> Any:
        """Wrap a single :class:`ToolSpec` as a CrewAI ``BaseTool``."""
        return _spec_to_crewai(
            spec, error_mode=error_mode, error_formatter=error_formatter, approve=approve
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
        """Select tools for ``query`` once, as CrewAI ``BaseTool`` objects."""
        from agent_gantry.integrations.frameworks.base import GantryToolset

        mode = self._mode(error_mode)
        specs = await GantryToolset(self._gantry).select(
            query,
            tools_already_used=tools_already_used,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
        )
        return [
            _spec_to_crewai(
                s, error_mode=mode, error_formatter=self._error_formatter, approve=self._approve
            )
            for s in specs
        ]

    async def live_tools(
        self, query: str, *, limit: int | None = None, **select_kwargs: Any
    ) -> list[Any]:
        """Re-select CrewAI ``BaseTool``s for THIS call's ``query`` (per-call selection)."""
        return await self.select(query, limit=limit, **select_kwargs)

    def agent_builder(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **agent_kwargs: Any,
    ) -> Any:
        """Return a builder that rebuilds a fresh ``crewai.Agent`` per call with re-selected tools.

        ``agent_kwargs`` (role/goal/backstory/llm/...) are forwarded to ``crewai.Agent``
        on every rebuild. Call ``await builder.build(query)`` per task. The
        builder uses this adapter's ``error_mode``, ``error_formatter`` and
        ``approve``.
        """
        from agent_gantry.integrations.frameworks.live_wrappers import GantryLiveCrewAgent

        builder = GantryLiveCrewAgent(
            self._gantry,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            **agent_kwargs,
        )
        builder._adapter = self
        return builder


__all__ = ["ApprovalRequest", "Approver", "CrewAIAdapter", "current_crew_context"]
