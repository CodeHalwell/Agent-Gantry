"""DEEP per-turn dynamic-tool provider for LangGraph.

Where :meth:`~agent_gantry.integrations.frameworks.langgraph.LangGraphAdapter.select`
is the *static* path — you select a slice of tools once and hand the resulting
LangChain ``StructuredTool`` list to a graph at construction time — this module
wires Agent-Gantry into LangGraph as a **live, per-turn** tool source, matching
the depth of the Microsoft Agent Framework ``GantryContextProvider``: the set of
tools the model can call is **re-selected by Gantry on every model turn**.

The native hook
---------------
This module builds the agent with :func:`langchain.agents.create_agent`, the
LangChain/LangGraph-recommended agent constructor (``langchain>=1.0``) that
replaces the deprecated :func:`langgraph.prebuilt.create_react_agent` — the
latter is removed outright in LangGraph 2.0, and ``langchain>=1.3.4`` /
``langgraph>=1.2.4`` (this project's floors, pinned together in the
``agent-frameworks`` extra) both post-date the migration, so ``create_agent``
is unconditionally available and no runtime fallback to the deprecated symbol
is kept (see :func:`_import_create_agent`).

``create_agent`` has no dynamic-``model`` callable like the old
``create_react_agent`` did. Its per-turn hook is **middleware**: an
``AgentMiddleware`` subclass implementing ``awrap_model_call(request, handler)``
is invoked on *every* model turn with a ``ModelRequest`` carrying the current
``state`` and ``tools``, and can call ``handler(request.override(tools=...))``
to change the tools advertised to the model for that turn only. This is the
same mechanism ``langchain.agents.middleware.LLMToolSelectorMiddleware`` (an
LLM-based tool selector shipped in ``langchain``) uses, so it is the documented,
supported way to reproduce ``create_react_agent``'s dynamic-model tool rebinding
under the new API.

LangGraph still imposes one rule: tools placed in ``request.tools`` **must be a
subset (by name) of the static ``tools=`` argument** passed to ``create_agent``,
because the agent's ``ToolNode`` can only execute tools it was constructed with.
This module therefore:

1. builds the **full superset** of Gantry tools once (every registered tool,
   wrapped as a LangChain ``StructuredTool`` that routes execution through
   ``gantry.execute``) and passes it as ``tools=`` so the ``ToolNode`` can run
   *any* tool Gantry might pick;
2. installs a ``wrap_model_call`` middleware that, on each turn, derives a
   retrieval query from the current ``state["messages"]`` via
   :func:`~agent_gantry.query.latest_activity`, runs Gantry semantic selection,
   and overrides ``request.tools`` with just the selected subset.

Because tool names (not identity) are what the ``ToolNode`` subset check and
dispatch rely on, the per-turn subset is always valid and executes correctly.
The LLM on each turn sees exactly the tools Gantry chose for the *current*
conversation — stale selections never accumulate.

Selection state
---------------
:class:`GantryToolMiddleware` keeps per-run state keyed by the thread and the
newest user message, so concurrent runs are independent and a run resumed after
an approval ``interrupt()`` carries on where it was. Within a run it caches
selections, keeps earlier tools offered up to ``max_offered`` (``sticky``), and
on a retrieval failure keeps the previous turn's tools plus the pins. Failed
tool results (``status="error"``) never feed the next query, and
``query_from_results=False`` keeps tool output out of retrieval entirely.
Tools the caller passed to ``create_agent`` themselves are never removed.

Usage
-----
.. code-block:: python

    from langchain.agents import create_agent
    from agent_gantry.langgraph import LangGraphAdapter

    adapter = LangGraphAdapter(gantry)
    agent = create_agent(chat_model, middleware=[await adapter.middleware(limit=5)])
    result = await agent.ainvoke({"messages": [("user", "what's the weather?")]})

The ``langchain`` / ``langgraph`` / ``langchain-core`` imports are lazy so
``import agent_gantry`` never requires them; the helpful
``pip install langchain langgraph langchain-core`` (or ``uv add langchain langgraph langchain-core``) hint is raised only when the
live agent is actually built.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    GantryToolset,
    QueryBoundsError,
    ToolSpec,
    _resolve_pins,
    _with_native_names,
    check_query_bounds,
    spec_from_tool,
)
from agent_gantry.integrations.frameworks.errors import MissingRequiredToolError
from agent_gantry.integrations.frameworks.langchain import FRAMEWORK_NAME, _spec_to_langchain
from agent_gantry.integrations.frameworks.native import (
    ErrorFormatter,
    ErrorMode,
    carry_over,
    check_error_mode,
)
from agent_gantry.query import latest_activity

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

logger = logging.getLogger(__name__)

LANGGRAPH_FRAMEWORK_NAME = "langgraph"

#: Runs whose selection state is kept at once (oldest dropped first).
_MAX_RUNS = 256
#: Distinct (query, used tools) selections cached per run.
_MAX_CACHED_SELECTIONS = 32


def _import_create_agent() -> tuple[Any, Any]:
    """Lazily import ``langchain.agents.create_agent`` and ``AgentMiddleware``.

    Deferred to call time so the module stays importable without LangChain /
    LangGraph installed. No fallback to the deprecated
    :func:`langgraph.prebuilt.create_react_agent` is kept: every supported
    ``langchain`` (>= 1.0) has ``create_agent``, and LangGraph 2.0 removes the
    old symbol.

    Raises:
        ImportError: If ``langchain`` / ``langgraph`` are not installed.
    """
    try:
        from langchain.agents import create_agent
        from langchain.agents.middleware import AgentMiddleware
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(
            "LangGraph live support requires `langchain` (>=1.0, for "
            "`langchain.agents.create_agent`) and `langgraph`. Install them with "
            "`pip install 'agent-gantry[langgraph]'` (or `uv add 'agent-gantry[langgraph]'`)."
        ) from exc
    return create_agent, AgentMiddleware


def _messages(state: Any) -> list[Any]:
    if isinstance(state, dict):
        messages = state.get("messages")
    else:  # pragma: no cover - defensive for non-dict state schemas
        messages = getattr(state, "messages", None)
    return list(messages or [])


def _role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "")
    return str(getattr(message, "type", "") or "")


def _query_messages(messages: Iterable[Any], *, from_results: bool) -> list[Any]:
    """The messages that may drive retrieval: never a failed tool result."""
    kept = []
    for message in messages:
        if _role(message) == "tool":
            status = (
                message.get("status")
                if isinstance(message, dict)
                else getattr(message, "status", None)
            )
            if not from_results or status == "error":
                continue
        kept.append(message)
    return kept


def _query_from_state(state: Any, *, from_results: bool = True) -> str:
    """Derive the per-turn retrieval query from a LangGraph agent state.

    The newest *activity* in ``state["messages"]`` (a new user request, or the
    last successful tool result while the agent chains tools) via
    :func:`~agent_gantry.query.latest_activity`. Failed tool results are
    skipped; ``from_results=False`` skips every tool result.
    """
    return latest_activity(_query_messages(_messages(state), from_results=from_results)) or ""


def _run_key(state: Any, config: Any) -> tuple[Any, Any]:
    """(thread id, newest user message id): one run of the agent."""
    thread_id = None
    if isinstance(config, dict):
        thread_id = (config.get("configurable") or {}).get("thread_id")
    newest = None
    for index, message in enumerate(_messages(state)):
        if _role(message) in ("human", "user"):
            newest = getattr(message, "id", None) or index
    return thread_id, newest


def _used_tools(state: Any) -> list[str]:
    """Tools the model has called since the newest user message."""
    used: list[str] = []
    for message in _messages(state):
        if _role(message) in ("human", "user"):
            used = []
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name and name not in used:
                used.append(name)
    return used


async def _select_tools_for_state(
    gantry: AgentGantry,
    state: Any,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    score_threshold: float = 0.0,
    namespaces: list[str] | None = None,
    required: list[str] | None = None,
    always_include: list[str] | None = None,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
) -> list[Any]:
    """Select Gantry tools for a LangGraph ``state`` and wrap them for LangChain.

    A stateless one-shot of what :class:`GantryToolMiddleware` does each turn.
    Never raises on a transient retrieval failure: it logs a WARNING and
    returns no tools. :class:`MissingRequiredToolError` and
    :class:`QueryBoundsError` are configuration errors and propagate.
    """
    query = _query_from_state(state)
    try:
        specs = await GantryToolset(gantry).select_or_empty(
            query,
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
        )
    except (MissingRequiredToolError, QueryBoundsError):
        raise
    except Exception:
        logger.warning(
            "_select_tools_for_state: semantic retrieval failed; "
            "continuing with no tools this turn.",
            exc_info=True,
        )
        return []
    return [
        _spec_to_langchain(s, error_mode=error_mode, error_formatter=error_formatter) for s in specs
    ]


async def _catalogue(
    gantry: AgentGantry,
    *,
    namespaces: list[str] | None = None,
    required: list[str] | None = None,
    always_include: list[str] | None = None,
) -> tuple[list[ToolSpec], frozenset[str]]:
    """In-scope tool specs (plus out-of-scope pins) and the pinned names."""
    definitions = await gantry.list_tools()
    specs = [
        spec_from_tool(gantry, d)
        for d in definitions
        if not namespaces or d.namespace in namespaces
    ]
    specs += _resolve_pins(gantry, specs, required=required, always_include=always_include)
    specs = _with_native_names(gantry, specs)
    pins = _resolve_pins(gantry, [], required=required, always_include=always_include)
    pin_keys = {(pin._namespace, pin.name) for pin in pins}
    pinned = frozenset(s.native_name for s in specs if (s._namespace, s.name) in pin_keys)
    return specs, pinned


async def _all_tools(
    gantry: AgentGantry,
    *,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
) -> list[Any]:
    """Every registered Gantry tool as a LangChain ``StructuredTool``."""
    specs, _ = await _catalogue(gantry)
    return [
        _spec_to_langchain(s, error_mode=error_mode, error_formatter=error_formatter) for s in specs
    ]


class _RunState:
    """One run's selection state."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.cache: OrderedDict[tuple[str, tuple[str, ...]], list[ToolSpec]] = OrderedDict()
        self.offered: dict[str, ToolSpec] = {}
        self.last_selected: dict[str, int] = {}
        self.step = 0
        self.enabled: frozenset[str] = frozenset()


class _Selector:
    """Per-turn selection shared by every run of one middleware."""

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        limit: int,
        score_threshold: float,
        namespaces: list[str] | None,
        required: list[str] | None,
        always_include: list[str] | None,
        sticky: bool,
        max_offered: int,
        query_from_results: bool,
        pinned: frozenset[str],
    ) -> None:
        self._toolset = GantryToolset(gantry)
        self._limit = limit
        self._score_threshold = score_threshold
        self._namespaces = namespaces
        self._required = required
        self._always_include = always_include
        self._sticky = sticky
        self._max_offered = max_offered
        self._query_from_results = query_from_results
        self._pinned = pinned
        self._runs: OrderedDict[tuple[Any, Any], _RunState] = OrderedDict()

    def _state(self, key: tuple[Any, Any]) -> _RunState:
        state = self._runs.get(key)
        if state is None:
            state = self._runs[key] = _RunState()
            while len(self._runs) > _MAX_RUNS:
                self._runs.popitem(last=False)
        else:
            self._runs.move_to_end(key)
        return state

    async def enabled(self, state: Any, config: Any) -> frozenset[str]:
        """The Gantry tool names to offer this turn."""
        run = self._state(_run_key(state, config))
        query = _query_from_state(state, from_results=self._query_from_results)
        used = tuple(_used_tools(state))
        async with run.lock:
            specs = await self._select(run, query, used)
            if specs is None:
                # Retrieval failed: keep the previous tools plus the pins.
                run.enabled = run.enabled | self._pinned
                return run.enabled
            if self._sticky:
                run.step += 1
                specs = carry_over(
                    run.offered, run.last_selected, run.step, specs, self._max_offered
                )
            run.enabled = frozenset(s.native_name for s in specs)
            return run.enabled

    async def _select(
        self, run: _RunState, query: str, used: tuple[str, ...]
    ) -> list[ToolSpec] | None:
        key = (query, used)
        if key in run.cache:
            run.cache.move_to_end(key)
            return run.cache[key]
        try:
            specs = await self._toolset.select_or_empty(
                query,
                limit=self._limit,
                score_threshold=self._score_threshold,
                namespaces=self._namespaces,
                tools_already_used=list(used),
                required=self._required,
                always_include=self._always_include,
            )
        except (MissingRequiredToolError, QueryBoundsError):
            raise
        except Exception:
            logger.warning(
                "GantryToolMiddleware: semantic retrieval failed; offering this run's "
                "previous tools and the pins this turn.",
                exc_info=True,
            )
            return None
        run.cache[key] = specs
        while len(run.cache) > _MAX_CACHED_SELECTIONS:
            run.cache.popitem(last=False)
        return specs


def _middleware_class() -> type:
    """``AgentMiddleware`` subclass, built lazily so LangChain stays optional."""
    _, AgentMiddleware = _import_create_agent()  # noqa: N806

    class GantryToolMiddleware(AgentMiddleware):  # type: ignore[misc, valid-type]
        """Re-selects Gantry tools for every model turn of a ``create_agent`` agent.

        Carries the in-scope Gantry catalogue as :attr:`tools`, so the agent's
        ``ToolNode`` can run any of them, and on each model call narrows the
        Gantry tools in ``request.tools`` to this turn's selection. Tools the
        caller passed to ``create_agent`` directly are left alone.
        """

        def __init__(self, tools: list[Any], selector: _Selector) -> None:
            super().__init__()
            self.tools = tools
            self._selector = selector
            self._names = frozenset(t.name for t in tools)

        def _narrow(self, request: Any, enabled: frozenset[str]) -> Any:
            tools = [
                t
                for t in request.tools
                if getattr(t, "name", None) not in self._names
                or getattr(t, "name", None) in enabled
            ]
            return request.override(tools=tools)

        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            config = _current_config()
            enabled = await self._selector.enabled(request.state, config)
            return await handler(self._narrow(request, enabled))

        def wrap_model_call(self, request: Any, handler: Any) -> Any:
            from agent_gantry.integrations.frameworks.base import _run_coroutine_sync

            config = _current_config()
            enabled = _run_coroutine_sync(self._selector.enabled(request.state, config))
            return handler(self._narrow(request, enabled))

    return GantryToolMiddleware


def _current_config() -> Any:
    try:
        from langgraph.config import get_config

        return get_config()
    except Exception:
        return None


async def build_middleware(
    gantry: AgentGantry,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    score_threshold: float = 0.0,
    namespaces: list[str] | None = None,
    required: list[str] | None = None,
    always_include: list[str] | None = None,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
    sticky: bool = True,
    max_offered: int | None = None,
    query_from_results: bool = True,
    framework: str = FRAMEWORK_NAME,
) -> Any:
    """Build a ``GantryToolMiddleware`` (see the module docs).

    Args:
        gantry: The registry to select from and execute against.
        limit: Tools selected per turn (pins are extra).
        score_threshold: Minimum relevance score.
        namespaces: Restrict selection (and the catalogue) to these namespaces.
        required: Tools offered every turn; a missing one raises.
        always_include: Tools offered every turn when they exist.
        error_mode: See :data:`~agent_gantry.integrations.frameworks.native.ErrorMode`.
        error_formatter: Rewrites failure text before the model sees it.
        sticky: Keep tools offered earlier in the run, up to ``max_offered``.
        max_offered: Cap on sticky tools; defaults to ``3 * limit``.
        query_from_results: Let successful tool output drive the next selection.
        framework: Name recorded on each call's ``ToolInvocationContext``.

    Raises:
        ImportError: If ``langchain`` / ``langgraph`` are not installed.
        ValueError: If ``max_offered`` is smaller than ``limit``.
    """
    check_query_bounds(limit=limit, score_threshold=score_threshold, owner="GantryToolMiddleware")
    error_mode = check_error_mode(error_mode)
    cap = 3 * limit if max_offered is None else max_offered
    if cap < limit:
        raise ValueError(f"max_offered ({cap}) must be at least limit ({limit})")
    middleware_cls = _middleware_class()
    try:
        await gantry.ensure_synced()
    except Exception:
        logger.warning(
            "GantryToolMiddleware: sync failed; building from the tools already registered.",
            exc_info=True,
        )
    specs, pinned = await _catalogue(
        gantry, namespaces=namespaces, required=required, always_include=always_include
    )
    tools = [
        _spec_to_langchain(
            s, error_mode=error_mode, error_formatter=error_formatter, framework=framework
        )
        for s in specs
    ]
    selector = _Selector(
        gantry,
        limit=limit,
        score_threshold=score_threshold,
        namespaces=namespaces,
        required=required,
        always_include=always_include,
        sticky=sticky,
        max_offered=cap,
        query_from_results=query_from_results,
        pinned=pinned,
    )
    return middleware_cls(tools, selector)


def _create_gantry_react_agent(model: Any, gantry: AgentGantry, **kwargs: Any) -> Any:
    """Build a ``create_agent`` agent whose Gantry tools are re-selected every turn.

    Synchronous: resolves the catalogue on Gantry's sync bridge, so it blocks
    the calling thread. In async code prefer :func:`_acreate_gantry_react_agent`.
    Selection keywords go to :func:`build_middleware`; everything else
    (``system_prompt``, ``checkpointer``, ``middleware``, ...) to ``create_agent``.
    """
    from agent_gantry.integrations.frameworks.base import _run_coroutine_sync

    return _run_coroutine_sync(_acreate_gantry_react_agent(model, gantry, **kwargs))


_MIDDLEWARE_KWARGS = frozenset(
    {
        "limit",
        "score_threshold",
        "namespaces",
        "required",
        "always_include",
        "error_mode",
        "error_formatter",
        "sticky",
        "max_offered",
        "query_from_results",
        "framework",
    }
)


async def _acreate_gantry_react_agent(model: Any, gantry: AgentGantry, **kwargs: Any) -> Any:
    """Async :func:`_create_gantry_react_agent`."""
    create_agent, _ = _import_create_agent()
    selection = {k: kwargs.pop(k) for k in list(kwargs) if k in _MIDDLEWARE_KWARGS}
    selection.setdefault("framework", LANGGRAPH_FRAMEWORK_NAME)
    middleware = await build_middleware(gantry, **selection)
    extra = list(kwargs.pop("middleware", ()) or ())
    tools = list(kwargs.pop("tools", ()) or ())
    return create_agent(model=model, tools=tools, middleware=[middleware, *extra], **kwargs)


__all__ = ["build_middleware"]
