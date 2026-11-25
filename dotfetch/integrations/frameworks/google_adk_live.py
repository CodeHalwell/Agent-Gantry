"""Per-turn dynamic-tool provider for Google ADK (the *deep* integration).

Where :meth:`~dotfetch.integrations.frameworks.google_adk.GoogleADKAdapter.select`
is the *static* path — a fixed slice of tools selected once and handed to
``Agent(tools=[...])`` — this module wires dotfetch into ADK's native
per-request hook so the tool surface is re-selected **before every model
request**. Each reasoning turn, the latest user content is matched against
dotfetch's semantic router and only the top-k relevant tools are injected into the
``LlmRequest`` the model sees. This minimises the tool-definition payload sent to
the LLM — the original design goal of dotfetch — while keeping the agent's
*registered* tool list empty.

The native hook is ``google.adk.agents.Agent(before_model_callback=...)``. ADK
invokes the callback with ``(callback_context, llm_request)`` before each model
request and lets the callback mutate ``llm_request`` in place. The canonical way
to add tools to a request is :meth:`LlmRequest.append_tools` — it builds the
``FunctionDeclaration`` for each ``BaseTool`` (the new experimental JSON-schema
path), appends them to ``llm_request.config.tools``, **and** registers each tool
in ``llm_request.tools_dict`` so ADK's flow can execute the resulting function
calls. That dual responsibility is exactly why the recommended setup is
``tools=[]`` on the agent plus callback injection: the callback both shows the
model the relevant tools *and* wires their execution for that turn — no static
registration needed.

.. code-block:: python

    from dotfetch import DotFetch
    from dotfetch.integrations.frameworks.google_adk_live import (
        _fetcher_adk_agent,
    )

    fetcher = DotFetch()
    # ... register tools, await fetcher.sync() ...

    agent = _fetcher_adk_agent(
        fetcher,
        model="gemini-2.0-flash",
        name="assistant",
        instruction="You are a helpful assistant.",
        limit=5,
    )

Or attach the callback to a hand-built agent:

.. code-block:: python

    from google.adk.agents import Agent
    from dotfetch.integrations.frameworks.google_adk_live import (
        _fetcher_before_model_callback,
    )

    agent = Agent(
        model="gemini-2.0-flash",
        name="assistant",
        tools=[],
        before_model_callback=_fetcher_before_model_callback(fetcher, limit=5),
    )

Per-run state
-------------
State is keyed by the ADK invocation (one ``Runner.run_async`` call), so
concurrent runs are independent. Within a run selections are cached, tools
offered earlier stay offered up to ``max_offered`` (``sticky``), and a
retrieval failure keeps the previous turn's tools plus the pins. The query is
the newest user message, plus the latest successful tool result after it
(``query_from_results=False`` keeps tool output out); a failed result, an
``{"error": ...}`` response, never feeds retrieval.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from dotfetch.adapters.selectors.base import SelectorFallbackError
from dotfetch.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    DotFetchToolset,
    QueryBoundsError,
    ToolSpec,
    _resolve_pins,
    _with_native_names,
    check_query_bounds,
    spec_from_tool,
)
from dotfetch.integrations.frameworks.errors import MissingRequiredToolError
from dotfetch.integrations.frameworks.google_adk import _INSTALL_HINT, _spec_to_google_adk
from dotfetch.integrations.frameworks.native import (
    MAX_TOOL_RESULT_QUERY_CHARS,
    ErrorFormatter,
    ErrorMode,
    carry_over,
    check_error_mode,
)

if TYPE_CHECKING:
    from dotfetch.core.fetcher import DotFetch

logger = logging.getLogger(__name__)

#: Runs whose selection state is kept at once (oldest dropped first).
_MAX_RUNS = 256
#: Distinct (query, used tools) selections cached per run.
_MAX_CACHED_SELECTIONS = 32
#: Recent session events scanned for the query.
_EVENT_WINDOW = 40


def _import_adk() -> Any:
    """Lazily import the ADK ``Agent`` class with a helpful error.

    Deferred so ``import dotfetch`` never requires ``google-adk``.
    """
    try:
        from google.adk.agents import Agent
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError(_INSTALL_HINT) from exc
    return Agent


def _content_text(content: Any) -> str:
    """Extract the concatenated text of a ``google.genai.types.Content``.

    Tolerates plain strings (and string parts) as well as the structured
    ``Content`` / ``Part`` objects, so simplified test payloads don't raise.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    parts = getattr(content, "parts", None) or []
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, str):
            if part.strip():
                chunks.append(part)
            continue
        text = getattr(part, "text", None)
        if text:
            chunks.append(text)
    return " ".join(chunks).strip()


def _function_responses(content: Any) -> list[Any]:
    parts = getattr(content, "parts", None) or []
    return [p.function_response for p in parts if getattr(p, "function_response", None)]


def _function_calls(content: Any) -> list[Any]:
    parts = getattr(content, "parts", None) or []
    return [p.function_call for p in parts if getattr(p, "function_call", None)]


def _response_text(response: Any) -> str | None:
    """A successful function response as text, or ``None`` for an error response."""
    payload = getattr(response, "response", None)
    if isinstance(payload, dict):
        if payload.get("error"):
            return None
        payload = payload.get("result", payload)
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, default=str)
    except (TypeError, ValueError):
        return str(payload)


def _recent_events(callback_context: Any) -> list[Any]:
    session = getattr(callback_context, "session", None)
    events = getattr(session, "events", None) or []
    return list(events)[-_EVENT_WINDOW:]


def _query_from_callback_context(callback_context: Any, *, from_results: bool = True) -> str:
    """Derive this turn's retrieval query from the ADK callback context.

    The newest user message in the session (falling back to the invocation's
    ``user_content``), plus the latest successful tool result after it when
    ``from_results``.
    """
    events = _recent_events(callback_context)
    prompt = ""
    output: str | None = None
    for event in reversed(events):
        content = getattr(event, "content", None)
        if output is None and from_results:
            for response in reversed(_function_responses(content)):
                output = _response_text(response)
                if output is not None:
                    break
        if getattr(event, "author", None) == "user":
            text = _content_text(content)
            if text:
                prompt = text
                break
    if not prompt:
        prompt = _content_text(getattr(callback_context, "user_content", None))
    if output:
        return f"{prompt}\n{output[:MAX_TOOL_RESULT_QUERY_CHARS]}".strip()
    return prompt


def _used_tools(callback_context: Any) -> list[str]:
    """Tools the model has called since the newest user message."""
    used: list[str] = []
    for event in reversed(_recent_events(callback_context)):
        content = getattr(event, "content", None)
        if getattr(event, "author", None) == "user" and _content_text(content):
            break
        for call in _function_calls(content):
            name = getattr(call, "name", None)
            if name and name not in used and name != "adk_request_confirmation":
                used.append(name)
    return used


class _RunState:
    """One invocation's selection state."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.cache: OrderedDict[tuple[str, tuple[str, ...]], list[ToolSpec]] = OrderedDict()
        self.offered: dict[str, ToolSpec] = {}
        self.last_selected: dict[str, int] = {}
        self.step = 0
        self.current: list[ToolSpec] = []


class _Selector:
    """Per-turn selection shared by every run of one callback."""

    def __init__(
        self,
        fetcher: DotFetch,
        *,
        limit: int,
        score_threshold: float,
        namespaces: list[str] | None,
        required: list[str] | None,
        always_include: list[str] | None,
        sticky: bool,
        max_offered: int,
        query_from_results: bool,
    ) -> None:
        self._fetcher = fetcher
        self._toolset = DotFetchToolset(fetcher)
        self._limit = limit
        self._score_threshold = score_threshold
        self._namespaces = namespaces
        self._required = required
        self._always_include = always_include
        self._sticky = sticky
        self._max_offered = max_offered
        self._query_from_results = query_from_results
        self._runs: OrderedDict[Any, _RunState] = OrderedDict()

    def _state(self, key: Any) -> _RunState:
        state = self._runs.get(key)
        if state is None:
            state = self._runs[key] = _RunState()
            while len(self._runs) > _MAX_RUNS:
                self._runs.popitem(last=False)
        else:
            self._runs.move_to_end(key)
        return state

    def _pins(self) -> list[ToolSpec]:
        try:
            # Native names as selection assigns them, so same-named pins from
            # different namespaces stay distinct and keep their names.
            return _with_native_names(
                self._fetcher,
                _resolve_pins(
                    self._fetcher,
                    [],
                    required=self._required,
                    always_include=self._always_include,
                    namespaces=self._namespaces,
                ),
            )
        except MissingRequiredToolError:
            raise
        except Exception:  # pragma: no cover - registry lookup failing too
            logger.warning("fetcher_before_model_callback: could not resolve pins.", exc_info=True)
            return []

    async def select(self, callback_context: Any) -> list[ToolSpec]:
        """The dotfetch tools to offer this turn."""
        key = getattr(callback_context, "invocation_id", None)
        # No invocation id (not a real ADK run): nothing to keep state against.
        run = self._state(key) if isinstance(key, str) and key else _RunState()
        query = _query_from_callback_context(
            callback_context, from_results=self._query_from_results
        )
        used = tuple(_used_tools(callback_context))
        async with run.lock:
            specs = await self._select(run, query, used)
            if specs is None:
                # Retrieval failed: keep the previous tools plus the pins.
                current = {s.native_name: s for s in run.current}
                for pin in self._pins():
                    current.setdefault(pin.native_name, pin)
                run.current = list(current.values())
                return run.current
            if self._sticky:
                run.step += 1
                specs = carry_over(
                    run.offered, run.last_selected, run.step, specs, self._max_offered
                )
            run.current = list(specs)
            return run.current

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
        except (MissingRequiredToolError, QueryBoundsError, SelectorFallbackError):
            raise
        except Exception:
            logger.warning(
                "fetcher_before_model_callback: semantic retrieval failed; offering this "
                "run's previous tools and the pins this turn.",
                exc_info=True,
            )
            return None
        run.cache[key] = specs
        while len(run.cache) > _MAX_CACHED_SELECTIONS:
            run.cache.popitem(last=False)
        return specs


async def _inject_selected_tools(
    fetcher: DotFetch,
    query: str,
    llm_request: Any,
    *,
    limit: int,
    score_threshold: float,
    namespaces: list[str] | None = None,
    required: list[str] | None = None,
    always_include: list[str] | None = None,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
) -> list[str]:
    """Select tools for ``query`` once and inject them into ``llm_request``.

    The stateless one-shot of what the callback does each turn. Never raises
    on a transient retrieval failure (logs a WARNING and injects nothing);
    :class:`MissingRequiredToolError` and :class:`QueryBoundsError` propagate.
    """
    try:
        specs = await DotFetchToolset(fetcher).select_or_empty(
            query,
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
        )
    except (MissingRequiredToolError, QueryBoundsError, SelectorFallbackError):
        raise
    except Exception:
        logger.warning(
            "fetcher_before_model_callback: semantic retrieval failed; "
            "continuing without dynamic tools.",
            exc_info=True,
        )
        return []
    return _append(llm_request, specs, error_mode=error_mode, error_formatter=error_formatter)


#: Marks tools the confirmation toolset supplies only so a confirmed call can resume.
_CONFIRMATION_ONLY = "_fetcher_confirmation_only"


def _hide_confirmation_only_tools(llm_request: Any, keep: set[str]) -> None:
    """Drop confirmation-toolset tools this turn did not select from the request.

    ADK resumes a confirmed call through ``agent.canonical_tools()``, so the
    toolset's tools need not reach the model; advertising them would offer the
    whole gated catalogue and bypass ``limit`` / ``max_offered``.
    """
    tools_dict = getattr(llm_request, "tools_dict", None) or {}
    hidden = {
        name
        for name, tool in tools_dict.items()
        if getattr(tool, _CONFIRMATION_ONLY, False) and name not in keep
    }
    if not hidden:
        return
    for name in hidden:
        del tools_dict[name]
    config = getattr(llm_request, "config", None)
    for tool in getattr(config, "tools", None) or []:
        declarations = getattr(tool, "function_declarations", None)
        if declarations:
            tool.function_declarations = [d for d in declarations if d.name not in hidden]
    if config is not None and getattr(config, "tools", None):
        config.tools = [
            tool
            for tool in config.tools
            if getattr(tool, "function_declarations", None) is None or tool.function_declarations
        ]


def _append(
    llm_request: Any,
    specs: list[ToolSpec],
    *,
    error_mode: ErrorMode,
    error_formatter: ErrorFormatter | None,
) -> list[str]:
    existing = set(getattr(llm_request, "tools_dict", None) or {})
    tools = [
        _spec_to_google_adk(spec, error_mode=error_mode, error_formatter=error_formatter)
        for spec in specs
        if spec.native_name not in existing
    ]
    if tools:
        llm_request.append_tools(tools)
    injected = [getattr(t, "name", "") for t in tools]
    logger.debug("fetcher_before_model_callback: injected %d tools: %s", len(injected), injected)
    return injected


def _fetcher_before_model_callback(
    fetcher: DotFetch,
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
) -> Any:
    """Build an ADK ``before_model_callback`` that injects dotfetch tools per turn.

    ADK calls it before **each** model request with ``(callback_context,
    llm_request)``. It selects this turn's tools (see the module docs for the
    query and per-run state) and injects them with
    :meth:`LlmRequest.append_tools`, which also registers them in
    ``llm_request.tools_dict`` so ADK can execute them this turn. Tools the
    agent already has are left alone. Recommended setup:
    ``Agent(tools=[], before_model_callback=...)``.

    Args:
        fetcher: The registry to select from and execute against.
        limit: Tools selected per turn (pins are extra).
        score_threshold: Minimum relevance score.
        namespaces: Restrict selection to these namespaces.
        required: Tools offered every turn; a missing one raises.
        always_include: Tools offered every turn when they exist.
        error_mode: See :data:`~dotfetch.integrations.frameworks.native.ErrorMode`.
        error_formatter: Rewrites failure text before the model sees it.
        sticky: Keep tools offered earlier in the run, up to ``max_offered``.
        max_offered: Cap on sticky tools; defaults to ``3 * limit``.
        query_from_results: Let successful tool output drive the next selection.

    Returns:
        An ``async`` callback ``(callback_context, llm_request) -> None``.
    """
    check_query_bounds(limit=limit, score_threshold=score_threshold, owner="GoogleADKAdapter")
    error_mode = check_error_mode(error_mode)
    cap = 3 * limit if max_offered is None else max_offered
    if cap < limit:
        raise ValueError(f"max_offered ({cap}) must be at least limit ({limit})")
    selector = _Selector(
        fetcher,
        limit=limit,
        score_threshold=score_threshold,
        namespaces=namespaces,
        required=required,
        always_include=always_include,
        sticky=sticky,
        max_offered=cap,
        query_from_results=query_from_results,
    )

    async def _callback(callback_context: Any, llm_request: Any) -> None:
        specs = await selector.select(callback_context)
        _hide_confirmation_only_tools(llm_request, {spec.native_name for spec in specs})
        _append(llm_request, specs, error_mode=error_mode, error_formatter=error_formatter)
        return None  # ADK proceeds with the mutated llm_request

    return _callback


_CONFIRMATION_CALL = "adk_request_confirmation"


def _answers_confirmation(readonly_context: Any) -> bool:
    """Whether the newest user event answers an ``adk_request_confirmation``.

    ADK resumes a confirmed call inside the original invocation, so the
    context's ``user_content`` is still the first message; the answer is the
    latest user-authored event in the session.
    """
    session = getattr(readonly_context, "session", None)
    for event in reversed(list(getattr(session, "events", None) or [])):
        if getattr(event, "author", None) != "user":
            continue
        return any(
            getattr(response, "name", None) == _CONFIRMATION_CALL
            for response in _function_responses(getattr(event, "content", None))
        )
    return False


def _confirmation_toolset(
    fetcher: DotFetch,
    *,
    namespaces: list[str] | None = None,
    pins: list[str] | None = None,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
) -> Any:
    """An ADK ``BaseToolset`` that lets a confirmed per-turn call resume.

    ADK resumes a confirmed call by looking the tool up in
    ``agent.canonical_tools()``, which per-turn injection never touches. This
    toolset returns the confirmation-gated dotfetch tools, but only on an
    invocation whose user message answers an ``adk_request_confirmation`` --
    every other turn it returns nothing, so the model is never shown the whole
    catalogue. ``pins`` (the agent's ``required`` / ``always_include``) are
    looked up too, since a qualified pin can sit outside ``namespaces``.
    """
    try:
        from google.adk.tools.base_toolset import BaseToolset
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError(_INSTALL_HINT) from exc

    class DotFetchConfirmationToolset(BaseToolset):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            # The answer arrives inside the same invocation, so a per-invocation
            # cache would keep serving the earlier, empty result.
            self._use_invocation_cache = False

        async def get_tools(self, readonly_context: Any = None) -> list[Any]:
            if readonly_context is None or not _answers_confirmation(readonly_context):
                return []
            definitions = await fetcher.list_tools()
            in_scope = [
                spec_from_tool(fetcher, d)
                for d in definitions
                if not namespaces or d.namespace in namespaces
            ]
            # Pins are offered even outside the namespace filter, so they must
            # resolve here too (always_include: a missing pin is not an error).
            pinned = _resolve_pins(
                fetcher, in_scope, required=None, always_include=pins, namespaces=namespaces
            )
            specs = _with_native_names(fetcher, in_scope + pinned)
            tools = [
                _spec_to_google_adk(s, error_mode=error_mode, error_formatter=error_formatter)
                for s in specs
                if fetcher.requires_confirmation(s.tool or s.name, s._namespace)
            ]
            for tool in tools:
                # The per-turn callback keeps these out of the model's request.
                setattr(tool, _CONFIRMATION_ONLY, True)
            return tools

        async def close(self) -> None:
            return None

    return DotFetchConfirmationToolset()


def _fetcher_adk_agent(
    fetcher: DotFetch,
    *,
    model: Any,
    name: str,
    instruction: str = "",
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
    **agent_kwargs: Any,
) -> Any:
    """Build an ADK ``Agent`` wired for per-turn dynamic tool selection.

    ``Agent(model=model, name=name, instruction=instruction, tools=[],
    before_model_callback=_fetcher_before_model_callback(...))``; extra keyword
    arguments go to ``Agent``.

    Raises:
        ImportError: If ``google-adk`` is not installed.
    """
    agent_cls = _import_adk()
    tools = list(agent_kwargs.pop("tools", []) or [])
    # Lets a confirmed call resume: ADK looks it up in the agent's own tools.
    tools.append(
        _confirmation_toolset(
            fetcher,
            namespaces=namespaces,
            pins=[*(required or []), *(always_include or [])],
            error_mode=error_mode,
            error_formatter=error_formatter,
        )
    )
    return agent_cls(
        model=model,
        name=name,
        instruction=instruction,
        tools=tools,
        before_model_callback=_fetcher_before_model_callback(
            fetcher,
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
            error_mode=error_mode,
            error_formatter=error_formatter,
            sticky=sticky,
            max_offered=max_offered,
            query_from_results=query_from_results,
        ),
        **agent_kwargs,
    )
