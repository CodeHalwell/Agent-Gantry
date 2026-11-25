"""Per-step dotfetch tool selection for the Haystack ``Agent`` (haystack-ai >= 3.0).

Haystack re-flattens an Agent's toolsets at the start of every step, and runs
``before_run`` / ``after_tool`` hooks against the live ``State``. A
:class:`DotFetchHaystackSelector` uses both: its hooks select dotfetch tools for
the latest user message (and, after a tool step, the latest successful tool
result) and store them for the run; its dynamic ``Toolset`` yields that
selection when the Agent flattens it. So every model call after a tool step
sees freshly selected tools.

Per-run state travels in a context variable (Haystack copies the context into
its tool threads and async tasks), so concurrent runs of one Agent never see
each other's tools.
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dotfetch.adapters.selectors.base import SelectorFallbackError
from dotfetch.integrations.frameworks.base import (
    DotFetchToolset,
    QueryBoundsError,
    ToolSpec,
    _resolve_pins,
    _run_coroutine_sync,
    _with_native_names,
    check_query_bounds,
)
from dotfetch.integrations.frameworks.errors import MissingRequiredToolError
from dotfetch.integrations.frameworks.haystack import _INSTALL_HINT
from dotfetch.integrations.frameworks.native import (
    MAX_TOOL_RESULT_QUERY_CHARS,
    carry_over,
)

if TYPE_CHECKING:
    from dotfetch.integrations.frameworks.haystack import HaystackAdapter

logger = logging.getLogger(__name__)


@dataclass
class _Run:
    """One Agent run's selection state."""

    tools: list[Any] = field(default_factory=list)
    specs: list[ToolSpec] = field(default_factory=list)
    offered: dict[str, ToolSpec] = field(default_factory=dict)
    last_selected: dict[str, int] = field(default_factory=dict)
    step: int = 0
    last_query: tuple[str, tuple[str, ...]] | None = None


def _role(message: Any) -> str:
    role = getattr(message, "role", None)
    return str(getattr(role, "value", role) or "")


def _query(messages: list[Any], *, from_results: bool) -> tuple[str, tuple[str, ...]]:
    """The newest user text, plus the latest successful tool result after it; tools used since."""
    prompt = ""
    output: str | None = None
    used: list[str] = []
    for message in reversed(messages or []):
        role = _role(message)
        if role == "user" and (getattr(message, "text", None) or "").strip():
            prompt = message.text
            break
        if role == "tool" and output is None and from_results:
            for result in reversed(getattr(message, "tool_call_results", None) or []):
                if not getattr(result, "error", False) and getattr(result, "result", None):
                    output = str(result.result)
                    break
        for call in getattr(message, "tool_calls", None) or []:
            name = getattr(call, "tool_name", None)
            if name and name not in used:
                used.append(name)
    if output:
        prompt = f"{prompt}\n{output[:MAX_TOOL_RESULT_QUERY_CHARS]}".strip()
    return prompt, tuple(used)


class DotFetchHaystackSelector:
    """Hooks plus a dynamic ``Toolset`` that re-select dotfetch tools every Agent step."""

    def __init__(
        self,
        adapter: HaystackAdapter,
        selection: dict[str, Any],
        *,
        sticky: bool = True,
        max_offered: int | None = None,
        query_from_results: bool = True,
    ) -> None:
        limit = selection["limit"]
        check_query_bounds(
            limit=limit, score_threshold=selection["score_threshold"], owner="HaystackAdapter.agent"
        )
        cap = 3 * limit if max_offered is None else max_offered
        if cap < limit:
            raise ValueError(f"max_offered ({cap}) must be at least limit ({limit})")
        self._adapter = adapter
        self._selection = selection
        self._sticky = sticky
        self._max_offered = cap
        self._query_from_results = query_from_results
        self._run: contextvars.ContextVar[_Run | None] = contextvars.ContextVar(
            "fetcher_haystack_run", default=None
        )

    # -- hooks ---------------------------------------------------------------

    def start(self, state: Any) -> None:
        """``before_run``: a fresh run, with tools selected for its input."""
        run = _Run()
        self._run.set(run)
        _run_coroutine_sync(self._refresh(run, state))

    async def start_async(self, state: Any) -> None:
        run = _Run()
        self._run.set(run)
        await self._refresh(run, state)

    def refresh(self, state: Any) -> None:
        """``after_tool``: re-select for the next step."""
        run = self._run.get()
        if run is not None:
            _run_coroutine_sync(self._refresh(run, state))

    async def refresh_async(self, state: Any) -> None:
        run = self._run.get()
        if run is not None:
            await self._refresh(run, state)

    def current_tools(self) -> list[Any]:
        """The tools selected for the current run's next step."""
        run = self._run.get()
        return list(run.tools) if run is not None else []

    # -- selection -----------------------------------------------------------

    def _pins(self) -> list[ToolSpec]:
        try:
            return _with_native_names(
                self._adapter._fetcher,
                _resolve_pins(
                    self._adapter._fetcher,
                    [],
                    required=self._selection.get("required"),
                    always_include=self._selection.get("always_include"),
                    namespaces=self._selection.get("namespaces"),
                ),
            )
        except MissingRequiredToolError:
            raise
        except Exception:  # pragma: no cover - registry lookup failing too
            return []

    async def _refresh(self, run: _Run, state: Any) -> None:
        messages = getattr(state, "data", {}).get("messages") or []
        key = _query(messages, from_results=self._query_from_results)
        if key == run.last_query:
            return
        query, used = key
        try:
            specs = await DotFetchToolset(self._adapter._fetcher).select_or_empty(
                query, tools_already_used=list(used), **self._selection
            )
        except (MissingRequiredToolError, QueryBoundsError, SelectorFallbackError):
            raise
        except Exception:
            logger.warning(
                "Haystack: semantic retrieval failed; keeping this run's previous tools and the pins.",
                exc_info=True,
            )
            current = {spec.native_name: spec for spec in run.specs}
            for pin in self._pins():
                current.setdefault(pin.native_name, pin)
            self._publish(run, list(current.values()))
            return
        run.last_query = key
        if self._sticky:
            run.step += 1
            specs = carry_over(run.offered, run.last_selected, run.step, specs, self._max_offered)
        self._publish(run, specs)

    def _publish(self, run: _Run, specs: list[ToolSpec]) -> None:
        run.specs = list(specs)
        run.tools = [self._adapter._convert(spec) for spec in specs]

    # -- haystack objects ----------------------------------------------------

    def toolset(self) -> Any:
        """A Haystack ``Toolset`` yielding the current run's selection."""
        from haystack.tools import Toolset

        selector = self

        class DotFetchDynamicToolset(Toolset):  # type: ignore[misc, valid-type]
            """Yields the tools dotfetch selected for this run's next step."""

            def __init__(self) -> None:
                super().__init__(tools=[])

            def __iter__(self) -> Any:
                return iter(selector.current_tools())

            def __len__(self) -> int:
                return len(selector.current_tools())

            def __contains__(self, item: Any) -> bool:
                name = item if isinstance(item, str) else getattr(item, "name", None)
                return any(tool.name == name for tool in selector.current_tools())

            def to_dict(self) -> dict[str, Any]:  # pragma: no cover - not serialisable
                return {"type": "dotfetch.haystack.DotFetchDynamicToolset"}

        return DotFetchDynamicToolset()

    def hooks(self) -> dict[str, list[Any]]:
        """The ``before_run`` / ``after_tool`` hooks that keep the selection current."""
        selector = self

        class _Hook:
            def __init__(self, sync: Any, async_: Any) -> None:
                self._sync = sync
                self._async = async_

            def run(self, state: Any) -> None:
                self._sync(state)

            async def run_async(self, state: Any) -> None:
                await self._async(state)

            def to_dict(self) -> dict[str, Any]:  # pragma: no cover - not serialisable
                return {"type": "dotfetch.haystack.DotFetchHaystackSelector"}

        return {
            "before_run": [_Hook(selector.start, selector.start_async)],
            "after_tool": [_Hook(selector.refresh, selector.refresh_async)],
        }


def _fetcher_haystack_agent(
    adapter: HaystackAdapter,
    *,
    chat_generator: Any,
    selection: dict[str, Any],
    confirmation_ui: Any = None,
    sticky: bool = True,
    max_offered: int | None = None,
    query_from_results: bool = True,
    **agent_kwargs: Any,
) -> Any:
    """Build an ``Agent`` with a :class:`DotFetchHaystackSelector` (and confirmation hook)."""
    try:
        from haystack.components.agents import Agent
        from haystack.hooks.protocol import VALID_HOOK_POINTS  # noqa: F401 - hooks need >= 3.0
    except ImportError as exc:
        raise ImportError(
            f"{_INSTALL_HINT} (HaystackAdapter.agent needs haystack-ai>=3.0)"
        ) from exc

    if "tools" in agent_kwargs:
        raise TypeError("HaystackAdapter.agent() selects the tools; pass no tools=...")
    selector = DotFetchHaystackSelector(
        adapter,
        selection,
        sticky=sticky,
        max_offered=max_offered,
        query_from_results=query_from_results,
    )
    hooks: dict[str, list[Any]] = {
        point: list(entries) for point, entries in (agent_kwargs.pop("hooks", None) or {}).items()
    }
    for point, entries in selector.hooks().items():
        hooks[point] = [*entries, *hooks.get(point, [])]
    if confirmation_ui is not None and adapter._error_mode == "raise":
        from dotfetch.integrations.frameworks.haystack import RAISE_MODE_UI_WARNING

        logger.warning(RAISE_MODE_UI_WARNING)
    if confirmation_ui is not None and adapter._error_mode == "model":
        hooks["before_tool"] = [
            adapter.confirmation_hook(confirmation_ui),
            *hooks.get("before_tool", []),
        ]
    agent = Agent(
        chat_generator=chat_generator, tools=selector.toolset(), hooks=hooks, **agent_kwargs
    )
    agent._fetcher_selector = selector  # type: ignore[attr-defined]
    return agent
