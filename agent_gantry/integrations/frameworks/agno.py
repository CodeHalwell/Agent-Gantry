"""Agno (formerly Phidata) native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as an Agno
``Function`` — the native tool object an Agno agent introspects (name /
description / JSON-schema parameters) and invokes. The ``agno`` import is lazy
so ``import agent_gantry`` never requires Agno to be installed.

Public entry point: :class:`AgnoAdapter`.

Every call routes through ``gantry.execute`` and uses Agno's own mechanisms:

- **Errors** -- a failed call raises :class:`AgnoToolError`, whose text
  (``"<type>: <message>"``, capped, redactable with ``error_formatter``) Agno
  records as the tool's failed result (``tool_call_error=True``) and shows the
  model, so the run continues. ``error_mode="raise"`` raises the plain
  :class:`~agent_gantry.integrations.frameworks.base.ToolExecutionError`
  instead.
- **Approval** -- a confirmation-gated tool (``requires_confirmation=True`` or a
  ``SecurityPolicy(require_confirmation=[...])`` pattern) is a ``Function`` with
  ``requires_confirmation=True``, so the run pauses (``run.is_paused``) with a
  requirement in ``run.active_requirements``. ``requirement.confirm()`` then
  ``agent.continue_run(run_response=run)`` runs the call exactly once;
  ``requirement.reject()`` leaves it unrun.
- **Run context** -- Agno's ``RunContext`` (``run_id``, ``session_id``,
  ``user_id``, ``session_state``, ``dependencies``, ``metadata``) is injected
  and readable with :func:`current_run_context` or
  :func:`agent_gantry.current_tool_invocation` (whose ``deps`` is
  ``run_context.dependencies``).
- **Tracing** -- the active OpenTelemetry span (Agno's own instrumentation)
  becomes the Gantry ``ToolCall``'s trace and parent span.

:meth:`AgnoAdapter.agent` builds one ``Agent`` whose tools are a callable
factory, so Agno re-selects them for each run's input; the selection is kept
for that run, so ``continue_run`` after an approval sees the same tools.
"""

from __future__ import annotations

import inspect
import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from agent_gantry.core.invocation import current_tool_invocation
from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    BaseFrameworkAdapter,
    GantryToolset,
    ToolConfirmationRequiredError,
    ToolExecutionError,
    ToolSpec,
    _resolve_pins,
    _run_coroutine_sync,
    _with_native_names,
    check_query_bounds,
)
from agent_gantry.integrations.frameworks.errors import MissingRequiredToolError
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

FRAMEWORK_NAME = "agno"

_INSTALL_HINT = (
    "Agno support requires `agno`. Install it with `pip install 'agent-gantry[agno]'` "
    "(or `uv add 'agent-gantry[agno]'`)."
)

_PENDING_APPROVAL_TEXT = (
    "This tool call requires human approval and was not run. Ask the user to "
    "approve it, or continue without it."
)

#: Unfinished runs whose selection :meth:`AgnoAdapter.agent` remembers (for
#: ``continue_run``). Finished runs are forgotten by a post-hook; past this
#: count the oldest run that cannot be awaiting approval goes first.
_MAX_REMEMBERED_RUNS = 256

#: Hard bound, reached only when every remembered run could be awaiting approval.
_MAX_REMEMBERED_GATED_RUNS = 4 * _MAX_REMEMBERED_RUNS

#: Parameters Agno fills in itself: the run context, the agent, the call.
_INJECTED = ("_agno_run_context", "_agno_agent", "fc")


class AgnoToolError(ToolExecutionError):
    """A failed Gantry call as Agno shows it to the model.

    ``str()`` is the model-facing failure text, which Agno stores as the tool's
    failed result; the attributes are those of the original
    :class:`~agent_gantry.integrations.frameworks.base.ToolExecutionError`.
    """

    def __init__(self, source: ToolExecutionError, text: str) -> None:
        super().__init__(source.tool_name, source.status, source.error, source.error_type)
        self.text = text

    def __str__(self) -> str:
        return self.text


def current_run_context() -> Any:
    """The Agno ``RunContext`` of the Gantry tool currently executing.

    ``None`` outside an Agno run (or when the tool is called directly).
    """
    invocation = current_tool_invocation()
    if invocation is None or invocation.framework != FRAMEWORK_NAME:
        return None
    return invocation.native_context


def _metadata(run_context: Any, agent: Any) -> dict[str, Any]:
    values = {
        "agent_name": getattr(agent, "name", None),
        "session_id": getattr(run_context, "session_id", None),
        "user_id": getattr(run_context, "user_id", None),
    }
    return {key: value for key, value in values.items() if isinstance(value, str)}


def _spec_to_agno(
    spec: ToolSpec,
    *,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
) -> Any:
    """Wrap a :class:`ToolSpec` as an Agno ``Function``.

    The entrypoint is synchronous, so the tool works under both ``agent.run()``
    and ``agent.arun()`` (Agno refuses async tools in a sync run); it routes
    through ``gantry.execute`` on Gantry's sync bridge. Its signature carries
    the tool's real parameters plus Agno's injected ``_agno_run_context``,
    ``_agno_agent`` and ``fc``, which never reach the model's schema. The
    ``agno`` import happens here, lazily, so callers without Agno installed
    only hit the error when they actually export a tool.

    Raises:
        ImportError: If ``agno`` is not installed.
    """
    try:
        from agno.tools.function import Function
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(_INSTALL_HINT) from exc

    error_mode = check_error_mode(error_mode)
    model_mode = error_mode == "model"
    signature_fn = spec.callable_for_signature()
    signature = signature_fn.__signature__  # type: ignore[attr-defined]
    aliases = spec.parameter_aliases()
    injected = [name for name in _INJECTED if name not in signature.parameters]
    gantry = spec._gantry
    gated = (
        model_mode
        and gantry is not None
        and gantry.requires_confirmation(spec.tool or spec.name, spec._namespace)
    )
    holder: dict[str, Any] = {}

    def _entrypoint(**kwargs: Any) -> Any:
        run_context = (
            kwargs.pop("_agno_run_context", None) if "_agno_run_context" in injected else None
        )
        agent = kwargs.pop("_agno_agent", None) if "_agno_agent" in injected else None
        call = kwargs.pop("fc", None) if "fc" in injected else None
        if aliases:
            kwargs = {aliases.get(key, key): value for key, value in kwargs.items()}
        # Agno runs a requires_confirmation Function only once the call is
        # confirmed; a direct call (no run context) is never treated as approved.
        function = holder.get("function")
        approved = (
            run_context is not None
            and function is not None
            and getattr(function, "requires_confirmation", False) is True
        )
        try:
            return _run_coroutine_sync(
                execute_spec(
                    spec,
                    kwargs,
                    framework=FRAMEWORK_NAME,
                    native_context=run_context,
                    deps=getattr(run_context, "dependencies", None),
                    run_id=getattr(run_context, "run_id", None),
                    tool_call_id=getattr(call, "call_id", None),
                    metadata=_metadata(run_context, agent),
                    approved=approved,
                    # Read here: the bridge may run the call on another thread.
                    trace_ids=otel_ids(),
                )
            )
        except ToolExecutionError as exc:
            if not model_mode:
                raise
            if isinstance(exc, ToolConfirmationRequiredError):
                raise AgnoToolError(exc, _PENDING_APPROVAL_TEXT) from None
            raise AgnoToolError(exc, failure_text(exc, error_formatter)) from None

    params = list(signature.parameters.values())
    extra = [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None) for name in injected
    ]
    positional = [p for p in params if p.kind is not inspect.Parameter.KEYWORD_ONLY]
    keyword = [p for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY]
    _entrypoint.__name__ = spec.native_name
    _entrypoint.__doc__ = spec.description
    # The real parameters, so Agno introspection sees them rather than a bare
    # **kwargs (no-argument) tool.
    _entrypoint.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=[*positional, *extra, *keyword]
    )
    _entrypoint.__annotations__ = dict(getattr(signature_fn, "__annotations__", {}))

    kwargs: dict[str, Any] = {
        "name": spec.native_name,
        "description": spec.description,
        "parameters": spec.aliased_parameters(),
        "entrypoint": _entrypoint,
    }
    if gated:
        kwargs["requires_confirmation"] = True
    function = Function(**kwargs)
    holder["function"] = function
    return function


class _RunSelector:
    """Agno pre-hook plus tools factory: select once per run, keep it for the run.

    Agno runs pre-hooks (which see the run input) before it resolves a callable
    ``tools`` factory, and resolves the factory again under the same ``run_id``
    on ``continue_run``. The hook records each run's input; the factory selects
    for it and remembers the result, so a run resumed after an approval is
    offered the same tools. A post-hook forgets each run when it finishes
    (Agno skips post-hooks while a run is paused), and eviction spares runs
    offered a confirmation-gated tool, which may still be paused. One lock
    guards both stores (never held while selecting), so one agent can serve
    concurrent runs from several threads.
    """

    def __init__(self, adapter: AgnoAdapter, select_kwargs: dict[str, Any]) -> None:
        self._adapter = adapter
        self._select_kwargs = select_kwargs
        self._queries: OrderedDict[str, str] = OrderedDict()
        self._tools: OrderedDict[str, list[Any]] = OrderedDict()
        self._lock = threading.Lock()

    def _may_be_paused(self, run_id: str) -> bool:
        tools = self._tools.get(run_id)
        return tools is not None and any(
            getattr(tool, "requires_confirmation", False) is True for tool in tools
        )

    def _remember(self, store: OrderedDict[str, Any], key: str, value: Any) -> None:
        """Store ``value`` (call with the lock held)."""
        store[key] = value
        store.move_to_end(key)
        while len(store) > _MAX_REMEMBERED_RUNS:
            victim = next((k for k in store if k != key and not self._may_be_paused(k)), None)
            if victim is None:
                if len(store) <= _MAX_REMEMBERED_GATED_RUNS:
                    return
                victim = next(iter(store))
                logger.warning(
                    "Agno: over %d runs may be awaiting approval; forgetting the "
                    "oldest (%s), which will re-select its tools if continued.",
                    _MAX_REMEMBERED_GATED_RUNS,
                    victim,
                )
            del store[victim]

    def forget(self, run_context: Any = None) -> None:
        """Agno post-hook: the run finished, so it will not be continued."""
        run_id = getattr(run_context, "run_id", None)
        with self._lock:
            self._queries.pop(run_id, None)
            self._tools.pop(run_id, None)

    def record_input(self, run_input: Any = None, run_context: Any = None) -> None:
        """Agno pre-hook: remember this run's input as its selection query."""
        run_id = getattr(run_context, "run_id", None)
        if run_id is None or run_input is None:
            return
        reader = getattr(run_input, "input_content_string", None)
        query = reader() if callable(reader) else str(getattr(run_input, "input_content", ""))
        with self._lock:
            self._remember(self._queries, run_id, query or "")

    def tools(self, agent: Any = None, run_context: Any = None) -> list[Any]:
        """Agno tools factory: the tools selected for this run."""
        run_id = getattr(run_context, "run_id", None)
        with self._lock:
            if run_id is not None and run_id in self._tools:
                return list(self._tools[run_id])
            query = self._queries.get(run_id, "") if run_id is not None else ""
        if not query:
            query = _stored_input(agent, run_context)
        tools = _run_coroutine_sync(self._select(query))
        if run_id is not None:
            with self._lock:
                self._remember(self._tools, run_id, tools)
                self._queries.pop(run_id, None)  # only needed until the tools are chosen
        return list(tools)

    async def _select(self, query: str) -> list[Any]:
        adapter = self._adapter
        kwargs = self._select_kwargs
        try:
            specs = await GantryToolset(adapter._gantry).select_or_empty(query, **kwargs)
        except MissingRequiredToolError:
            raise
        except Exception:
            # A broken retrieval must not drop the pinned tools too.
            logger.warning(
                "Agno: semantic retrieval failed; offering pinned tools only", exc_info=True
            )
            specs = _with_native_names(
                adapter._gantry,
                _resolve_pins(
                    adapter._gantry,
                    [],
                    required=kwargs.get("required"),
                    always_include=kwargs.get("always_include"),
                    namespaces=kwargs.get("namespaces"),
                ),
            )
        return [adapter._convert(spec) for spec in specs]


def _stored_input(agent: Any, run_context: Any) -> str:
    """A resumed run's input from the agent's storage (another process paused it)."""
    reader = getattr(agent, "get_run_output", None)
    if not callable(reader):
        return ""
    try:
        run = reader(
            run_id=getattr(run_context, "run_id", None),
            session_id=getattr(run_context, "session_id", None),
        )
        run_input = getattr(run, "input", None)
        return run_input.input_content_string() if run_input is not None else ""
    except Exception:
        logger.warning(
            "Agno: could not read run %s's input from the agent's db; "
            "selecting its tools without a query",
            getattr(run_context, "run_id", None),
            exc_info=True,
        )
        return ""


class AgnoAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into Agno.

    Static slice (``agno.tools.function.Function`` objects for
    ``Agent(tools=[...])``), a per-run :meth:`agent` (one Agent that re-selects
    for every run) and the per-call :meth:`agent_builder`. Every call routes
    through ``gantry.execute`` so retries, timeouts, circuit breakers, and the
    security policy all apply::

        from agent_gantry import AgentGantry
        from agent_gantry.agno import AgnoAdapter

        gantry = AgentGantry()
        # ... register tools, await gantry.sync() ...

        agent = AgnoAdapter(gantry).agent(model=model, limit=4)
        run = await agent.arun("email the quarterly report to Sam")

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` to select from.
        default_limit: Tools per selection when a method gets no ``limit``.
        error_mode: ``"model"`` (default) shows a failed call to the model as an
            :class:`AgnoToolError`; ``"raise"`` raises the plain
            :class:`~agent_gantry.integrations.frameworks.base.ToolExecutionError`
            and turns off the native approval pause (gated calls report as not
            run).
        error_formatter: Rewrites the failure text the model sees (fail-closed,
            capped).
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
    ) -> None:
        super().__init__(gantry, default_limit=default_limit)
        self._error_mode = check_error_mode(error_mode)
        self._error_formatter = error_formatter

    @staticmethod
    def convert(
        spec: ToolSpec,
        *,
        error_mode: ErrorMode = "model",
        error_formatter: ErrorFormatter | None = None,
    ) -> Any:
        """Wrap a single :class:`ToolSpec` as an Agno ``Function``."""
        return _spec_to_agno(spec, error_mode=error_mode, error_formatter=error_formatter)

    def _convert(self, spec: ToolSpec, error_mode: ErrorMode | None = None) -> Any:
        mode = self._error_mode if error_mode is None else check_error_mode(error_mode)
        return _spec_to_agno(spec, error_mode=mode, error_formatter=self._error_formatter)

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
        """Select tools for ``query`` once, as Agno ``Function`` objects."""
        specs = await GantryToolset(self._gantry).select(
            query,
            tools_already_used=tools_already_used,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
        )
        return [self._convert(spec, error_mode) for spec in specs]

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
        """Build one ``agno.agent.Agent`` that re-selects Gantry tools for every run.

        ``agent_kwargs`` (``model``, ``name``, ``instructions``, ``db``, ...) are
        forwarded to ``Agent``; any ``pre_hooks``/``post_hooks`` you pass still run. The tools
        are an Agno callable factory (with ``cache_callables=False``), selected
        for each run's input and kept for that run so ``continue_run`` after an
        approval is offered the same tools. A run resumed by a different agent
        object reads its input back from the agent's ``db``.

        Raises:
            ImportError: If ``agno`` is not installed.
        """
        try:
            from agno.agent import Agent
        except ImportError as exc:  # pragma: no cover - exercised via importorskip
            raise ImportError(_INSTALL_HINT) from exc

        if "tools" in agent_kwargs:
            raise TypeError("AgnoAdapter.agent() selects the tools; pass no tools=...")
        selection = self._selection_kwargs(
            limit, score_threshold, namespaces, required, always_include
        )
        check_query_bounds(
            limit=selection["limit"],
            score_threshold=score_threshold,
            owner="AgnoAdapter.agent",
        )
        selector = _RunSelector(self, selection)
        pre_hooks = list(agent_kwargs.pop("pre_hooks", None) or [])
        post_hooks = list(agent_kwargs.pop("post_hooks", None) or [])
        agent_kwargs.setdefault("cache_callables", False)
        return Agent(
            tools=selector.tools,
            pre_hooks=[selector.record_input, *pre_hooks],
            post_hooks=[selector.forget, *post_hooks],
            **agent_kwargs,
        )

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
        """Return a builder that rebuilds a fresh ``agno.agent.Agent`` per call with re-selected tools.

        ``agent_kwargs`` (model/...) are forwarded. Call ``await builder.build(query)`` per run.
        Prefer :meth:`agent`, which keeps one Agent and re-selects inside Agno.
        """
        from agent_gantry.integrations.frameworks.live_wrappers import (
            GantryLiveAgnoAgent,
        )

        builder = GantryLiveAgnoAgent(
            self._gantry,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            **agent_kwargs,
        )
        builder._adapter = self  # build with this adapter's error settings
        return builder
