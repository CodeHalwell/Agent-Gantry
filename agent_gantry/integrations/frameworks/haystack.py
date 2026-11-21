"""Haystack native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a Haystack
``haystack.tools.Tool`` — the native tool object Haystack components introspect
(name / description / JSON-schema parameters) and invoke. The ``haystack``
import is lazy so ``import agent_gantry`` never requires Haystack to be
installed.

Public entry point: :class:`HaystackAdapter`.

Every call routes through ``gantry.execute`` and uses Haystack's own mechanisms
(haystack-ai >= 3.0):

- **Errors** -- a failed call raises :class:`HaystackToolError`, whose text
  (``"<type>: <message>"``, capped, redactable with ``error_formatter``) the
  Agent turns into an error tool result for the model, so the run continues.
  ``error_mode="raise"`` raises the plain
  :class:`~agent_gantry.integrations.frameworks.base.ToolExecutionError`.
- **Approval** -- :meth:`HaystackAdapter.confirmation_hook` is a ``before_tool``
  hook built on Haystack's ``ConfirmationHook`` and ``ConfirmationUI``: it asks
  about confirmation-gated Gantry tools (``requires_confirmation=True`` or a
  ``SecurityPolicy(require_confirmation=[...])`` pattern) and nothing else. A
  confirmed (or modified) call runs once; a rejected one never runs and the
  model is told.
- **Run context** -- the Agent's live ``State`` is injected into each call and
  readable with :func:`current_agent_state` or
  :func:`agent_gantry.current_tool_invocation` (``deps`` is ``state.data``).
- **Tracing** -- the active OpenTelemetry span becomes the Gantry ``ToolCall``'s
  trace and parent span.

:meth:`HaystackAdapter.agent` builds an ``Agent`` whose tools Gantry re-selects
before every step (see :mod:`agent_gantry.integrations.frameworks.haystack_live`).
"""

from __future__ import annotations

import inspect
import json
import logging
from collections import Counter
from typing import TYPE_CHECKING, Any

from agent_gantry.core.invocation import current_tool_invocation
from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    BaseFrameworkAdapter,
    GantryToolset,
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

logger = logging.getLogger(__name__)

#: Logged when a confirmation UI is given to an adapter in ``error_mode="raise"``.
RAISE_MODE_UI_WARNING = (
    "Haystack: error_mode='raise' turns off Gantry's confirmation hook, so the "
    "confirmation UI is never asked and confirmation-gated tools report that they "
    "were not run. Use error_mode='model' to confirm them."
)

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

FRAMEWORK_NAME = "haystack"

_INSTALL_HINT = (
    "Haystack support requires `haystack-ai`. Install it with "
    "`pip install 'agent-gantry[haystack]'` (or `uv add 'agent-gantry[haystack]'`)."
)

_PENDING_APPROVAL_TEXT = (
    "This tool call requires human approval and was not run. Ask the user to "
    "approve it, or continue without it."
)

#: Attribute on the Agent ``State`` holding this run's approved calls.
_APPROVALS_ATTR = "_gantry_approvals"
#: Attribute on a Gantry-built ``Tool``: whether Gantry gates it.
GATED_ATTR = "_gantry_requires_confirmation"


class HaystackToolError(ToolExecutionError):
    """A failed Gantry call as Haystack shows it to the model.

    ``str()`` is the model-facing failure text, which the Agent puts in an
    error tool result; the attributes are those of the original
    :class:`~agent_gantry.integrations.frameworks.base.ToolExecutionError`.
    """

    def __init__(self, source: ToolExecutionError, text: str) -> None:
        super().__init__(source.tool_name, source.status, source.error, source.error_type)
        self.text = text

    def __str__(self) -> str:
        return self.text


def current_agent_state() -> Any:
    """The Haystack Agent ``State`` of the Gantry tool currently executing.

    ``None`` outside an Agent run (or when the tool is called directly).
    """
    invocation = current_tool_invocation()
    if invocation is None or invocation.framework != FRAMEWORK_NAME:
        return None
    return invocation.native_context


def _args_key(name: str, arguments: Any) -> str:
    return json.dumps([name, arguments], sort_keys=True, default=str)


def _approvals(state: Any) -> Counter[str]:
    approvals = getattr(state, _APPROVALS_ATTR, None)
    if approvals is None:
        approvals = Counter()
        setattr(state, _APPROVALS_ATTR, approvals)
    return approvals


def _record_approval(state: Any, name: str, arguments: Any) -> None:
    _approvals(state)[_args_key(name, arguments)] += 1


def _take_approval(state: Any, name: str, arguments: Any) -> bool:
    """Consume one approval of exactly this call, recorded by the confirmation hook."""
    approvals = getattr(state, _APPROVALS_ATTR, None)
    key = _args_key(name, arguments)
    if not approvals or approvals[key] <= 0:
        return False
    approvals[key] -= 1
    return True


_TOOL_CLASS: Any = None


def _gantry_tool_class(tool_cls: Any) -> Any:
    """A Haystack ``Tool`` whose failures carry only Gantry's failure text.

    ``Tool.invoke`` would otherwise wrap a failure as ``Failed to invoke Tool
    ... with parameters {...}``, echoing every argument (the injected ``State``
    object included) back to the model. Falls back to ``tool_cls`` itself when
    Haystack's error type is unavailable (partial installs, test doubles).
    """
    global _TOOL_CLASS
    if _TOOL_CLASS is not None and _TOOL_CLASS.__bases__ == (tool_cls,):
        return _TOOL_CLASS
    try:
        from haystack.tools.errors import ToolInvocationError

        def _clean(exc: Any, name: str) -> Any:
            cause = exc.__cause__
            if isinstance(cause, HaystackToolError):
                return ToolInvocationError(str(cause), tool_name=name)
            return exc

        class GantryHaystackTool(tool_cls):  # type: ignore[misc, valid-type]
            def invoke(self, **kwargs: Any) -> Any:
                try:
                    return super().invoke(**kwargs)
                except ToolInvocationError as exc:
                    raise _clean(exc, self.name) from exc.__cause__

            async def invoke_async(self, **kwargs: Any) -> Any:
                try:
                    return await super().invoke_async(**kwargs)
                except ToolInvocationError as exc:
                    raise _clean(exc, self.name) from exc.__cause__

    except (ImportError, TypeError):
        return tool_cls
    _TOOL_CLASS = GantryHaystackTool
    return _TOOL_CLASS


def _state_class() -> Any:
    try:
        from haystack.components.agents.state import State
    except ImportError:  # pragma: no cover - haystack < 3 or partial installs
        return None
    return State


def _spec_to_haystack(
    spec: ToolSpec,
    *,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
) -> Any:
    """Wrap a :class:`ToolSpec` as a Haystack ``Tool``.

    The tool has a synchronous ``function`` (Gantry's sync bridge) and, where
    Haystack supports it, a native ``async_function`` for ``Agent.run_async``.
    Both take the Agent's ``State`` through a ``State``-annotated parameter
    (``gantry_state``, renamed if a tool argument uses that name) that
    Haystack injects and the model's schema never shows. The ``haystack``
    import happens here, lazily, so callers without Haystack installed only hit
    the error when they actually export a tool.

    Raises:
        ImportError: If ``haystack-ai`` is not installed.
    """
    try:
        from haystack.tools import Tool
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(_INSTALL_HINT) from exc

    error_mode = check_error_mode(error_mode)
    model_mode = error_mode == "model"
    parameters = spec.aliased_parameters()
    aliases = spec.parameter_aliases()
    state_cls = _state_class()
    inject_state = state_cls is not None
    # Haystack injects State into any parameter annotated State, whatever its
    # name; pick one no tool argument uses, so a tool's own ``state`` argument
    # (e.g. a US state) still gets the Agent State and its approvals.
    state_param = "gantry_state"
    while state_param in (parameters.get("properties") or {}):
        state_param = f"_{state_param}"
    gantry = spec._gantry
    gated = (
        model_mode
        and gantry is not None
        and gantry.requires_confirmation(spec.tool or spec.name, spec._namespace)
    )

    def _prepare(kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any], bool]:
        state = kwargs.pop(state_param, None) if inject_state else None
        approved = gated and state is not None and _take_approval(state, spec.native_name, kwargs)
        if aliases:
            kwargs = {aliases.get(key, key): value for key, value in kwargs.items()}
        return state, kwargs, approved

    async def _execute(
        state: Any,
        kwargs: dict[str, Any],
        approved: bool,
        trace_ids: tuple[str | None, str | None],
    ) -> Any:
        try:
            return await execute_spec(
                spec,
                kwargs,
                framework=FRAMEWORK_NAME,
                native_context=state,
                deps=getattr(state, "data", None),
                approved=approved,
                trace_ids=trace_ids,
            )
        except ToolExecutionError as exc:
            if not model_mode:
                raise
            if isinstance(exc, ToolConfirmationRequiredError):
                raise HaystackToolError(exc, _PENDING_APPROVAL_TEXT) from None
            raise HaystackToolError(exc, failure_text(exc, error_formatter)) from None

    def _function(**kwargs: Any) -> Any:
        state, kwargs, approved = _prepare(kwargs)
        # Read here: the bridge may run the call on another thread.
        return _run_coroutine_sync(_execute(state, kwargs, approved, otel_ids()))

    async def _async_function(**kwargs: Any) -> Any:
        state, kwargs, approved = _prepare(kwargs)
        return await _execute(state, kwargs, approved, otel_ids())

    for fn in (_function, _async_function):
        fn.__name__ = spec.native_name
        fn.__doc__ = spec.description
        if inject_state:
            fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
                [
                    inspect.Parameter(
                        state_param,
                        inspect.Parameter.KEYWORD_ONLY,
                        default=None,
                        annotation=state_cls,
                    ),
                    inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
                ]
            )
            fn.__annotations__ = {state_param: state_cls}

    kwargs: dict[str, Any] = {
        "name": spec.native_name,
        "description": spec.description,
        "parameters": parameters,
        "function": _function,
    }
    if "async_function" in inspect.signature(Tool.__init__).parameters:
        kwargs["async_function"] = _async_function
    tool_cls = _gantry_tool_class(Tool) if model_mode else Tool
    tool = tool_cls(**kwargs)
    setattr(tool, GATED_ATTR, gated)
    return tool


class _GantryConfirmationStrategy:
    """Ask the UI about a gated call and record what it approves on the run's State."""

    def __init__(self, ui: Any, state: Any) -> None:
        from haystack.hooks.human_in_the_loop import AlwaysAskPolicy, BlockingConfirmationStrategy

        self._inner = BlockingConfirmationStrategy(
            confirmation_policy=AlwaysAskPolicy(), confirmation_ui=ui
        )
        self._state = state

    def _record(self, decision: Any) -> Any:
        if decision.execute:
            params = decision.final_tool_params
            _record_approval(self._state, decision.tool_name, dict(params or {}))
        return decision

    def run(self, **kwargs: Any) -> Any:
        return self._record(self._inner.run(**kwargs))

    async def run_async(self, **kwargs: Any) -> Any:
        return self._record(await self._inner.run_async(**kwargs))

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - not serialisable
        return {"type": "agent_gantry.haystack.confirmation_strategy"}


class GantryConfirmationHook:
    """A ``before_tool`` hook asking a ``ConfirmationUI`` about gated Gantry tools.

    Wraps Haystack's ``ConfirmationHook``, with a strategy for each Gantry tool
    on offer this step that Gantry gates; other tools run without asking.
    """

    allowed_hook_points = ("before_tool",)

    def __init__(self, confirmation_ui: Any) -> None:
        self.confirmation_ui = confirmation_ui

    def _hook(self, state: Any) -> Any:
        from haystack.hooks.human_in_the_loop import ConfirmationHook

        strategy = _GantryConfirmationStrategy(self.confirmation_ui, state)
        gated = tuple(
            tool.name for tool in state.data.get("tools") or [] if getattr(tool, GATED_ATTR, False)
        )
        return ConfirmationHook(confirmation_strategies={gated: strategy} if gated else {})

    def run(self, state: Any) -> None:
        self._hook(state).run(state)

    async def run_async(self, state: Any) -> None:
        await self._hook(state).run_async(state)

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - not serialisable
        return {"type": "agent_gantry.haystack.GantryConfirmationHook"}


class HaystackAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into Haystack.

    Static slice (``haystack.tools.Tool`` objects), a per-step :meth:`agent`
    and the per-call :meth:`tool_invoker_builder`. Every call routes through
    ``gantry.execute`` so retries, timeouts, circuit breakers, and the security
    policy all apply::

        from agent_gantry import AgentGantry
        from agent_gantry.haystack import HaystackAdapter
        from haystack.hooks.human_in_the_loop import SimpleConsoleUI

        gantry = AgentGantry()
        # ... register tools, await gantry.sync() ...

        agent = HaystackAdapter(gantry).agent(
            chat_generator=generator, limit=4, confirmation_ui=SimpleConsoleUI()
        )
        result = agent.run(messages=[ChatMessage.from_user("email Sam the report")])

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` to select from.
        default_limit: Tools per selection when a method gets no ``limit``.
        error_mode: ``"model"`` (default) shows a failed call to the model as a
            :class:`HaystackToolError`; ``"raise"`` raises the plain
            :class:`~agent_gantry.integrations.frameworks.base.ToolExecutionError`
            and turns off Gantry's confirmation hook (gated calls report as not
            run).
        error_formatter: Rewrites the failure text the model sees (fail-closed,
            capped).
    """

    live_tier = "per-call"
    _live_delegate = "tool_invoker_builder"

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
        """Wrap a single :class:`ToolSpec` as a Haystack ``Tool``."""
        return _spec_to_haystack(spec, error_mode=error_mode, error_formatter=error_formatter)

    def _convert(self, spec: ToolSpec, error_mode: ErrorMode | None = None) -> Any:
        mode = self._error_mode if error_mode is None else check_error_mode(error_mode)
        return _spec_to_haystack(spec, error_mode=mode, error_formatter=self._error_formatter)

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
        """Select tools for ``query`` once, as Haystack ``Tool`` objects."""
        specs = await GantryToolset(self._gantry).select(
            query,
            tools_already_used=tools_already_used,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
        )
        return [self._convert(spec, error_mode) for spec in specs]

    async def live_tools(
        self, query: str, *, limit: int | None = None, **select_kwargs: Any
    ) -> list[Any]:
        """Re-select Haystack ``Tool``s for THIS call's ``query`` (per-call selection)."""
        return await self.select(query, limit=limit, **select_kwargs)

    def confirmation_hook(self, confirmation_ui: Any) -> Any:
        """A ``before_tool`` hook that asks ``confirmation_ui`` about gated Gantry tools.

        ``confirmation_ui`` is any Haystack ``ConfirmationUI`` (``SimpleConsoleUI()``,
        ``RichConsoleUI()`` or your own). Register it with
        ``Agent(hooks={"before_tool": [hook]})``; :meth:`agent` does that for you
        when given ``confirmation_ui=``. With ``error_mode="raise"`` no tool is
        gated through it (a ``WARNING`` says so).
        """
        if self._error_mode == "raise":
            logger.warning(RAISE_MODE_UI_WARNING)
        return GantryConfirmationHook(confirmation_ui)

    def agent(
        self,
        *,
        chat_generator: Any,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        confirmation_ui: Any = None,
        sticky: bool = True,
        max_offered: int | None = None,
        query_from_results: bool = True,
        **agent_kwargs: Any,
    ) -> Any:
        """Build a ``haystack.components.agents.Agent`` that re-selects tools every step.

        Gantry selects tools for the run's latest user message before the
        first step and again after every tool step (joining the latest
        successful tool result to the query when ``query_from_results``).
        ``sticky`` keeps tools offered earlier in the run, up to
        ``max_offered`` (default ``3 * limit``). If retrieval fails the run
        keeps its previous tools plus the pins. ``confirmation_ui`` adds
        :meth:`confirmation_hook`; without it, gated calls are reported to the
        model as not run. ``agent_kwargs`` go to ``Agent`` (your own ``hooks``
        still run).

        Raises:
            ImportError: If ``haystack-ai`` >= 3.0 is not installed.
        """
        from agent_gantry.integrations.frameworks.haystack_live import _gantry_haystack_agent

        selection = self._selection_kwargs(
            limit, score_threshold, namespaces, required, always_include
        )
        return _gantry_haystack_agent(
            self,
            chat_generator=chat_generator,
            selection=selection,
            confirmation_ui=confirmation_ui,
            sticky=sticky,
            max_offered=max_offered,
            query_from_results=query_from_results,
            **agent_kwargs,
        )

    def tool_invoker_builder(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **invoker_kwargs: Any,
    ) -> Any:
        """Return a builder that rebuilds a fresh tool-execution component per call.

        On haystack 2.x each ``await builder.build(query)`` returns a
        ``ToolInvoker``; on haystack >= 3.0 (which removed ``ToolInvoker``)
        it returns a ``haystack.components.agents.Agent`` and
        ``invoker_kwargs`` must include ``chat_generator=...``.
        ``invoker_kwargs`` are forwarded to the built component. Prefer
        :meth:`agent`, which re-selects inside one Agent before every step.
        """
        from agent_gantry.integrations.frameworks.live_wrappers import (
            GantryLiveHaystackToolInvoker,
        )

        builder = GantryLiveHaystackToolInvoker(
            self._gantry,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            **invoker_kwargs,
        )
        builder._adapter = self  # build with this adapter's error settings
        return builder
