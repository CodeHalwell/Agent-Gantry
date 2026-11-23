"""LangChain native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a LangChain
``StructuredTool``, the native tool object LangChain chat models, ``create_agent``
and LangGraph's ``ToolNode`` all consume. Every call routes through
``gantry.execute``, and the tools use LangChain's and LangGraph's own mechanisms:

- **Errors** -- a failed call raises LangChain's ``ToolException`` and the tool
  has ``handle_tool_error=True``, so the model receives an error ``ToolMessage``
  (``status="error"``) and can correct itself instead of the run ending.
  ``error_formatter`` redacts the text; ``error_mode="raise"`` lets
  :class:`~agent_gantry.integrations.frameworks.base.ToolExecutionError`
  propagate instead.
- **Approval** -- a confirmation-gated call (``requires_confirmation=True`` or a
  ``SecurityPolicy(require_confirmation=[...])`` pattern) pauses the graph with
  LangGraph's ``interrupt()``. Resume with ``Command(resume=True)`` (or
  ``{"approved": True}``) to run it exactly once; anything else rejects it.
  This needs a LangGraph run with a checkpointer; elsewhere the model is told
  the call needs approval and it is not run.
- **Run context** -- a Gantry tool reads the LangGraph run (``context=``,
  ``store``, ``config``, ``tool_call_id``) with :func:`current_tool_runtime`, or
  through :func:`agent_gantry.current_tool_invocation`.
- **Tracing** -- an active OpenTelemetry span, else the current LangSmith run,
  becomes the Gantry ``ToolCall``'s trace and parent span.

Per-turn selection lives in :class:`~agent_gantry.langgraph.LangGraphAdapter`
(``create_agent`` middleware). The ``langchain-core`` import is lazy so
``import agent_gantry`` never requires LangChain.

Public entry point: :class:`LangChainAdapter`.
"""

from __future__ import annotations

import contextvars
import functools
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_gantry.core.invocation import current_tool_invocation
from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    PER_RUN_TIER,
    BaseFrameworkAdapter,
    ToolConfirmationRequiredError,
    ToolExecutionError,
    ToolSpec,
    _run_coroutine_sync,
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
    from agent_gantry.core.gantry import AgentGantry

logger = logging.getLogger(__name__)

FRAMEWORK_NAME = "langchain"

_INSTALL_HINT = (
    "LangChain support requires `langchain-core`. Install it with "
    "`pip install 'agent-gantry[langchain]'` (or `uv add 'agent-gantry[langchain]'`)."
)

_PENDING_APPROVAL_TEXT = (
    "This tool call requires human approval and was not run. Ask the user to "
    "approve it, or continue without it."
)
_REJECTED_TEXT = "The user rejected this tool call, so it was not run."

#: The id of the tool call being run, set by the tool's ``invoke``/``ainvoke``.
_CALL_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agent_gantry_langchain_call_id", default=None
)


@dataclass(frozen=True)
class GantryToolRuntime:
    """The LangChain / LangGraph run a Gantry tool is executing in.

    Read inside a Gantry tool handler with :func:`current_tool_runtime`.

    Attributes:
        config: The ``RunnableConfig`` for this call (``configurable``,
            ``metadata``, ``tags``, callbacks), or ``None`` outside a run.
        tool_call_id: The model-issued id of this call, when the tool was
            invoked with a ``ToolCall`` (as ``ToolNode`` and agents do).
        context: The LangGraph run's ``context=`` object (the static runtime
            context), or ``None``.
        store: The LangGraph ``BaseStore`` attached to the graph, or ``None``.
    """

    config: Any = None
    tool_call_id: str | None = None
    context: Any = None
    store: Any = None
    extras: Mapping[str, Any] = field(default_factory=dict)


def current_tool_runtime() -> GantryToolRuntime | None:
    """The LangChain/LangGraph run of the Gantry tool currently executing.

    ``None`` outside a tool call driven by this adapter.
    """
    invocation = current_tool_invocation()
    if invocation is None or not isinstance(invocation.native_context, GantryToolRuntime):
        return None
    return invocation.native_context


def _call_id(tool_input: Any) -> str | None:
    if isinstance(tool_input, Mapping) and tool_input.get("type") == "tool_call":
        call_id = tool_input.get("id")
        return str(call_id) if call_id is not None else None
    return None


def _runtime() -> GantryToolRuntime:
    """Capture the current run's config and LangGraph runtime, best effort."""
    config = None
    context = None
    store = None
    try:
        from langchain_core.runnables.config import var_child_runnable_config

        config = var_child_runnable_config.get()
    except Exception:  # pragma: no cover - langchain-core internals moved
        logger.debug("Could not read the LangChain RunnableConfig.", exc_info=True)
    try:
        from langgraph.runtime import get_runtime

        runtime = get_runtime()
        context = getattr(runtime, "context", None)
        store = getattr(runtime, "store", None)
    except Exception:  # not in a LangGraph run, or LangGraph not installed
        pass
    return GantryToolRuntime(
        config=config, tool_call_id=_CALL_ID.get(), context=context, store=store
    )


def _metadata(runtime: GantryToolRuntime) -> dict[str, Any]:
    config = runtime.config or {}
    configurable = config.get("configurable") or {}
    metadata = config.get("metadata") or {}
    extras: dict[str, Any] = {}
    if configurable.get("thread_id") is not None:
        extras["thread_id"] = configurable["thread_id"]
    for key in ("langgraph_node", "langgraph_step"):
        if key in metadata:
            extras[key] = metadata[key]
    return extras


def _trace_ids() -> tuple[str | None, str | None]:
    """The active OTel span, else the current LangSmith run tree."""
    ids = otel_ids()
    if ids != (None, None):
        return ids
    try:
        from langsmith.run_helpers import get_current_run_tree

        run = get_current_run_tree()
    except Exception:
        return None, None
    if run is None:
        return None, None
    trace_id = getattr(run, "trace_id", None)
    span_id = getattr(run, "id", None)
    return (str(trace_id) if trace_id else None, str(span_id) if span_id else None)


def _is_approval(answer: Any) -> bool:
    """Whether a ``Command(resume=...)`` value approves the call.

    Only ``True`` or a mapping with ``"approved": True`` approves; anything
    else (strings included) rejects, so a loosely worded answer never runs a
    gated tool.
    """
    if answer is True:
        return True
    return isinstance(answer, Mapping) and answer.get("approved") is True


def _ask_approval(spec: ToolSpec, arguments: Mapping[str, Any]) -> bool | None:
    """Pause the LangGraph run for a human decision on this call.

    Returns ``True``/``False`` once the run is resumed with a decision, or
    ``None`` when there is no resumable LangGraph run to pause (no graph, or no
    checkpointer). Raises LangGraph's ``GraphInterrupt`` to pause the run.
    """
    try:
        from langgraph.config import get_config
        from langgraph.types import interrupt
    except ImportError:
        return None
    try:
        configurable = get_config().get("configurable") or {}
    except RuntimeError:  # not inside a graph run
        return None
    if configurable.get("__pregel_checkpointer") is None:
        # An interrupt could never be resumed, so the model is told instead.
        logger.debug(
            "Tool %r needs approval but this LangGraph run has no checkpointer; "
            "reporting it as not run.",
            spec.native_name,
        )
        return None
    answer = interrupt(
        {
            "type": "gantry_approval",
            "tool": spec.native_name,
            "description": spec.description,
            "arguments": dict(arguments),
            "tool_call_id": _CALL_ID.get(),
        }
    )
    return _is_approval(answer)


def _needs_approval(spec: ToolSpec) -> bool:
    gantry = spec._gantry
    if gantry is None:
        return False
    return gantry.requires_confirmation(spec.tool or spec.name, spec._namespace)


async def _execute(
    spec: ToolSpec,
    arguments: dict[str, Any],
    runtime: GantryToolRuntime,
    *,
    approved: bool,
    framework: str,
    trace_ids: tuple[str | None, str | None],
) -> Any:
    return await execute_spec(
        spec,
        arguments,
        framework=framework,
        native_context=runtime,
        deps=runtime.context,
        tool_call_id=runtime.tool_call_id,
        metadata=_metadata(runtime),
        approved=approved,
        trace_ids=trace_ids,
    )


def _tool_exception(exc: ToolExecutionError, formatter: ErrorFormatter | None) -> Exception:
    from langchain_core.tools import ToolException

    if isinstance(exc, ToolConfirmationRequiredError):
        return ToolException(_PENDING_APPROVAL_TEXT)
    return ToolException(failure_text(exc, formatter))


_TOOL_CLASSES: dict[Any, Any] = {}


def _tool_class() -> Any:
    """``StructuredTool`` subclass that records the ``ToolCall`` id while it runs."""
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    if not isinstance(StructuredTool, type):  # a test stub
        return StructuredTool
    cached = _TOOL_CLASSES.get(StructuredTool)
    if cached is not None:
        return cached

    class GantryStructuredTool(StructuredTool):  # type: ignore[misc, valid-type]
        """A Gantry tool as a LangChain ``StructuredTool``."""

        def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            token = _CALL_ID.set(_call_id(input))
            try:
                return super().invoke(input, config, **kwargs)
            finally:
                _CALL_ID.reset(token)

        async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            token = _CALL_ID.set(_call_id(input))
            try:
                return await super().ainvoke(input, config, **kwargs)
            finally:
                _CALL_ID.reset(token)

    _TOOL_CLASSES[StructuredTool] = GantryStructuredTool
    return GantryStructuredTool


def _spec_to_langchain(
    spec: ToolSpec,
    *,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
    framework: str = FRAMEWORK_NAME,
) -> Any:
    """Wrap a :class:`ToolSpec` as a LangChain ``StructuredTool``.

    Args:
        spec: The tool to wrap.
        error_mode: ``"model"`` (errors become an error ``ToolMessage``,
            confirmation-gated calls interrupt for approval) or ``"raise"``.
        error_formatter: Rewrites failure text before the model sees it.
        framework: Name recorded on the :class:`ToolInvocationContext`.

    Raises:
        ImportError: If ``langchain-core`` is not installed.
    """
    error_mode = check_error_mode(error_mode)
    tool_cls = _tool_class()
    model_mode = error_mode == "model"

    def _decide(arguments: dict[str, Any]) -> bool | None:
        """``True`` to run approved, ``False`` if rejected, ``None`` to just run."""
        if not model_mode or not _needs_approval(spec):
            return None
        return _ask_approval(spec, arguments)

    async def _arun(**kwargs: Any) -> Any:
        runtime = _runtime()
        decision = _decide(kwargs)
        if decision is False:
            from langchain_core.tools import ToolException

            raise ToolException(_REJECTED_TEXT)
        try:
            return await _execute(
                spec,
                kwargs,
                runtime,
                approved=decision is True,
                framework=framework,
                trace_ids=_trace_ids(),
            )
        except ToolExecutionError as exc:
            if not model_mode:
                raise
            raise _tool_exception(exc, error_formatter) from exc

    def _run(**kwargs: Any) -> Any:
        # Read the run state and ask for approval on this thread: LangGraph's
        # interrupt() and config live in this thread's context, not the bridge's.
        runtime = _runtime()
        decision = _decide(kwargs)
        if decision is False:
            from langchain_core.tools import ToolException

            raise ToolException(_REJECTED_TEXT)
        trace_ids = _trace_ids()
        try:
            return _run_coroutine_sync(
                _execute(
                    spec,
                    kwargs,
                    runtime,
                    approved=decision is True,
                    framework=framework,
                    trace_ids=trace_ids,
                )
            )
        except ToolExecutionError as exc:
            if not model_mode:
                raise
            raise _tool_exception(exc, error_formatter) from exc

    return tool_cls.from_function(
        func=_run,
        coroutine=_arun,
        name=spec.native_name,
        description=spec.description,
        args_schema=spec.parameters,
        handle_tool_error=model_mode,
    )


class LangChainAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into LangChain.

    :meth:`select` returns native ``StructuredTool`` objects for
    ``model.bind_tools(...)`` or ``create_agent(tools=...)``; each call still
    goes through ``gantry.execute`` (retries, timeouts, circuit breakers and the
    security policy apply)::

        from agent_gantry.langchain import LangChainAdapter

        adapter = LangChainAdapter(gantry)
        tools = await adapter.select("email the quarterly report", limit=3)
        llm = ChatOpenAI(model="gpt-5.5").bind_tools(tools)

    For selection that changes every model turn inside an agent, use
    :meth:`middleware` with ``langchain.agents.create_agent`` (or
    :class:`~agent_gantry.langgraph.LangGraphAdapter`).

    Args:
        gantry: The registry to select from and execute against.
        default_limit: Default selection budget.
        error_mode: Default ``ErrorMode`` for the tools this adapter builds.
        error_formatter: Rewrites failure text before the model sees it.
    """

    live_tier = PER_RUN_TIER
    _live_delegate = "_bound_select"
    _framework_name = FRAMEWORK_NAME

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
        """Wrap a single :class:`ToolSpec` as a LangChain ``StructuredTool``."""
        return _spec_to_langchain(spec, error_mode=error_mode, error_formatter=error_formatter)

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
        """Select tools for ``query`` once, as ``StructuredTool`` objects."""
        from agent_gantry.integrations.frameworks.base import GantryToolset

        mode = self._mode(error_mode)
        specs = await GantryToolset(self._gantry).select(
            query,
            tools_already_used=tools_already_used,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
        )
        return [
            _spec_to_langchain(
                s,
                error_mode=mode,
                error_formatter=self._error_formatter,
                framework=self._framework_name,
            )
            for s in specs
        ]

    async def middleware(
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
    ) -> Any:
        """A ``create_agent`` middleware that re-selects Gantry tools every model turn.

        The middleware carries the whole in-scope catalogue as its ``tools``, so
        pass it alone: ``create_agent(model, middleware=[await adapter.middleware()])``.
        See :class:`~agent_gantry.integrations.frameworks.langgraph_live.GantryToolMiddleware`.
        """
        from agent_gantry.integrations.frameworks.langgraph_live import build_middleware

        return await build_middleware(
            self._gantry,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            error_mode=self._mode(error_mode),
            error_formatter=self._error_formatter,
            sticky=sticky,
            max_offered=max_offered,
            query_from_results=query_from_results,
            framework=self._framework_name,
        )

    def _bound_select(self, **select_kwargs: Any) -> Callable[[str], Awaitable[list[Any]]]:
        """The per-run live object: :meth:`select` bound to the selection keywords."""
        return functools.partial(self.select, **select_kwargs)


__all__ = ["GantryToolRuntime", "LangChainAdapter", "current_tool_runtime"]
