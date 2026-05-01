"""
Microsoft Agent Framework context provider for Agent-Gantry.

``GantryContextProvider`` is the idiomatic, AF-native way to attach
Agent-Gantry to an :class:`agent_framework.Agent`. It plugs into the AF 1.2
context-engineering pipeline by subclassing
:class:`agent_framework.ContextProvider`, which means it co-operates by
construction with every other provider — most importantly with
:class:`agent_framework.SkillsProvider`. Each provider contributes its own
tools and instructions via ``SessionContext.extend_tools`` /
``extend_instructions`` keyed by a unique ``source_id``; AF merges them
without conflict.

Two retrieval modes are supported:

- ``query_strategy="per_run"`` (default, back-compatible): retrieval happens
  once at the start of each ``agent.run(...)``. Tool selection is fixed for
  the whole reasoning loop.
- ``query_strategy="per_call"``: retrieval re-runs on every chat-completion
  round. Use this for multi-step agents whose tool needs evolve as they
  reason. The provider exposes a chat middleware via
  :meth:`GantryContextProvider.as_chat_middleware` that the integrator
  attaches to the same agent — AF context providers cannot themselves hook
  per-round events, so the middleware is the canonical extension point.

Why a context provider rather than (only) the existing
:class:`~agent_gantry.integrations.agent_framework_bridge.GantryToolBridge`?
The bridge is excellent for *static* tool sets — pre-bake once, pass to
``Agent(tools=[...])``. The provider is for *dynamic* per-turn tool
selection: every ``agent.run(...)`` call, the latest user message is
matched against Gantry's semantic router and only the top-k tools are
injected. This minimises the tool-definition payload sent to the LLM,
which is the original design goal of Agent-Gantry.

Both APIs coexist intentionally:

.. code-block:: python

    # Static — known tool set, baked at construction:
    bridge = GantryToolBridge(gantry)
    tools = bridge.wrap_tools(my_defs)
    agent = Agent(client, "...", tools=tools)

    # Dynamic per-run — top-k driven by the initial user message:
    provider = GantryContextProvider(gantry, top_k=5)
    agent = Agent(client, "...", context_providers=[provider])

    # Dynamic per-call — top-k re-selected every chat round:
    from agent_gantry.query import last_tool_result, fallback_chain, last_user_text
    provider = GantryContextProvider(
        gantry,
        top_k=3,
        query_strategy="per_call",
        query_generator=fallback_chain(last_tool_result, last_user_text),
    )
    agent = Agent(
        client,
        "...",
        context_providers=[provider],
        middleware=[provider.as_chat_middleware()],
    )

The provider also flows transparently through every AF orchestration
primitive (``WorkflowBuilder``, ``SequentialBuilder``, ``HandoffBuilder``,
``AgentExecutor``, ``WorkflowAgent``) because they all dispatch through
``agent.run()``, which fires the provider pipeline.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable

from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge
from agent_gantry.query import last_user_text as _default_query_generator

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.integrations.anthropic_skills import SkillRegistry
    from agent_gantry.schema.tool import ToolDefinition

logger = logging.getLogger(__name__)


class MissingRequiredToolError(LookupError):
    """Raised when a tool listed in ``required=[...]`` is not present in the gantry."""


def _import_context_provider() -> type:
    """Lazily import :class:`agent_framework.ContextProvider`.

    Import is deferred to instantiation so the module remains importable
    in environments without ``agent-framework`` installed (e.g. unit
    tests for the rest of Gantry).
    """
    try:
        from agent_framework import ContextProvider
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError(
            "GantryContextProvider requires the 'agent-framework' package. "
            "Install with: pip install 'agent-gantry[agent-frameworks]'"
        ) from exc
    return ContextProvider


def _import_chat_middleware() -> Any:
    """Lazily import :func:`agent_framework.chat_middleware`."""
    try:
        from agent_framework import chat_middleware
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError(
            "GantryContextProvider.as_chat_middleware() requires the "
            "'agent-framework' package. Install with: "
            "pip install 'agent-gantry[agent-frameworks]'"
        ) from exc
    return chat_middleware


def _tool_name(tool: Any) -> str:
    return (
        getattr(tool, "name", None)
        or getattr(tool, "__name__", None)
        or ""
    )


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it's awaitable, else return it as-is."""
    if inspect.isawaitable(value):
        return await value
    return value


class GantryContextProvider:
    """AF context provider that injects semantically-selected Gantry tools.

    Constructing the provider returns an instance of the dynamically
    generated ``agent_framework.ContextProvider`` subclass. The factory
    pattern keeps the import of ``agent_framework`` lazy so importing
    ``agent_gantry`` doesn't require AF to be installed.

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` instance
            providing semantic retrieval and execution.
        top_k: Maximum number of dynamically retrieved tools per chat-completion
            round (or per ``agent.run`` call in ``per_run`` mode). Defaults
            to ``5``.
        score_threshold: Minimum semantic relevance score for retrieved
            tools. Defaults to ``0.3``.
        query_strategy: ``"per_run"`` (default) refreshes the tool selection
            once at the start of ``agent.run()``. ``"per_call"`` re-runs
            retrieval on every chat-completion round; pair this mode with
            :meth:`as_chat_middleware` to install the per-round refresher.
        query_generator: Callable used to derive the retrieval query string
            from the conversation messages. Defaults to
            :func:`agent_gantry.query.last_user_text`. Sync or async are both
            accepted. See :mod:`agent_gantry.query` for built-in alternatives
            (``last_tool_result``, ``last_assistant_text``, ``concatenate_recent``,
            ``fallback_chain``).
        skills: When ``True``, every invocation also injects the union of
            tools bound to all registered Gantry skills (from the supplied
            ``skill_registry`` if any, else the registry attached to the
            gantry instance via :class:`SkillsClient`). These tools are
            *always* included regardless of the semantic match. Defaults
            to ``False``.
        skill_registry: Optional explicit
            :class:`~agent_gantry.integrations.anthropic_skills.SkillRegistry`.
            When ``None`` and ``skills=True``, the provider attempts to use
            ``gantry._skill_registry`` if present; otherwise it logs a
            warning and skips skill tools.
        always_include: Optional list of Gantry tool names to inject on
            every invocation (in addition to the dynamic top-k). Useful
            for pinning utility tools the LLM should always see. Missing
            names log a WARNING and are skipped.
        required: Optional list of Gantry tool names that **must** be
            available in the registry. The provider validates these at
            construction time and raises :class:`MissingRequiredToolError`
            if any are missing. They are also injected on every invocation
            (treated as ``always_include`` once present). Use this when a
            workflow is broken if the tool isn't surfaced (typo / dropped
            registration).
        as_function_tool: Forwarded to the underlying
            :class:`GantryToolBridge`. ``None`` (default) auto-detects;
            ``True`` forces ``agent_framework.FunctionTool`` wrapping;
            ``False`` returns bare callables (still accepted by AF).
        source_id: Provider identifier used by AF for tool/message
            attribution. Defaults to ``"agent_gantry"``. Override when
            running multiple Gantry instances side-by-side.
        bridge: Optional pre-built :class:`GantryToolBridge` to reuse its
            wrapper cache across multiple providers. When ``None`` a new
            bridge is built from the supplied ``gantry``.
        **query_kwargs: Additional keyword arguments forwarded to
            :meth:`GantryToolBridge.get_tools` (e.g. ``namespaces``,
            ``required_capabilities``, ``enable_reranking``).

    Example:
        .. code-block:: python

            from agent_framework import Agent, SkillsProvider
            from agent_framework.openai import OpenAIChatClient
            from agent_gantry import AgentGantry, GantryContextProvider
            from agent_gantry.query import last_tool_result

            gantry = AgentGantry()
            # ... register tools, sync ...

            provider = GantryContextProvider(
                gantry,
                top_k=3,
                query_strategy="per_call",
                query_generator=last_tool_result,
                required=["validate_boundaries"],
            )

            agent = Agent(
                OpenAIChatClient(),
                "You are a helpful assistant.",
                context_providers=[provider, SkillsProvider(skill_paths="./skills")],
                middleware=[provider.as_chat_middleware()],
            )
    """

    def __new__(
        cls,
        gantry: AgentGantry,
        *,
        top_k: int = 5,
        score_threshold: float = 0.3,
        query_strategy: str = "per_run",
        query_generator: Callable[..., Any] | None = None,
        skills: bool = False,
        skill_registry: SkillRegistry | None = None,
        always_include: list[str] | None = None,
        required: list[str] | None = None,
        as_function_tool: bool | None = None,
        source_id: str = "agent_gantry",
        bridge: GantryToolBridge | None = None,
        **query_kwargs: Any,
    ) -> Any:
        if query_strategy not in ("per_run", "per_call"):
            raise ValueError(
                f"query_strategy must be 'per_run' or 'per_call', got {query_strategy!r}"
            )

        context_provider_cls = _import_context_provider()

        bridge = bridge or GantryToolBridge(
            gantry,
            score_threshold=score_threshold,
            as_function_tool=as_function_tool,
        )
        always_include = list(always_include or [])
        required = list(required or [])
        query_generator = query_generator or _default_query_generator

        if required:
            registry = getattr(gantry, "_registry", None)
            missing = []
            for name in required:
                tool_def = None
                if registry is not None:
                    lookup = getattr(registry, "get_tool_by_name", None)
                    if callable(lookup):
                        tool_def = lookup(name)
                    if tool_def is None:
                        lookup = getattr(registry, "get_tool", None)
                        if callable(lookup):
                            tool_def = lookup(name)
                if tool_def is None:
                    missing.append(name)
            if missing:
                raise MissingRequiredToolError(
                    f"GantryContextProvider: required tool(s) not found in gantry: "
                    f"{missing}. Did you forget to register them, or is there a typo?"
                )

        # Combine for "always inject" set; required is a strict superset.
        always_include_effective: list[str] = []
        seen_ai: set[str] = set()
        for n in (*required, *always_include):
            if n not in seen_ai:
                seen_ai.add(n)
                always_include_effective.append(n)

        class _GantryContextProviderImpl(context_provider_cls):  # type: ignore[misc,valid-type]
            """Concrete ContextProvider subclass; see :class:`GantryContextProvider`."""

            def __init__(self) -> None:
                super().__init__(source_id=source_id)
                self._gantry = gantry
                self._bridge = bridge
                self._top_k = top_k
                self._score_threshold = score_threshold
                self._query_strategy = query_strategy
                self._query_generator = query_generator
                self._skills_enabled = skills
                self._skill_registry = skill_registry
                self._always_include = always_include_effective
                self._declared_always_include = list(always_include)
                self._required = list(required)
                self._query_kwargs = dict(query_kwargs)
                self._source_id = source_id

            # ----- Public, read-only configuration accessors --------------
            @property
            def top_k(self) -> int:
                return self._top_k

            @property
            def score_threshold(self) -> float:
                return self._score_threshold

            @property
            def query_strategy(self) -> str:
                return self._query_strategy

            @property
            def always_include(self) -> tuple[str, ...]:
                return tuple(self._declared_always_include)

            @property
            def required(self) -> tuple[str, ...]:
                return tuple(self._required)

            @property
            def gantry(self) -> AgentGantry:
                return self._gantry

            @property
            def bridge(self) -> GantryToolBridge:
                return self._bridge

            # ----- AF lifecycle ------------------------------------------
            async def before_run(
                self,
                *,
                agent: Any,
                session: Any,
                context: Any,
                state: dict[str, Any],
            ) -> None:
                # In per_call mode, the chat_middleware does the dynamic
                # retrieval per LLM round; we still inject always_include /
                # skill tools at run-start so they are present even if the
                # middleware is not attached.
                tools, seen = await self._collect_tools(
                    context.input_messages,
                    include_dynamic=(self._query_strategy == "per_run"),
                )
                if tools:
                    context.extend_tools(self._source_id, tools)
                    logger.debug(
                        "GantryContextProvider: injected %d tools at run start "
                        "(strategy=%s)",
                        len(tools),
                        self._query_strategy,
                    )

            # ----- Per-call middleware factory ---------------------------
            def as_chat_middleware(self) -> Any:
                """Return an AF chat_middleware that refreshes tools each round.

                Use with ``Agent(middleware=[provider.as_chat_middleware()])``
                in conjunction with ``query_strategy="per_call"``. The
                returned middleware reads ``context.messages`` (or
                ``context.options`` as appropriate), runs the configured
                query generator, retrieves a fresh top-k from the gantry,
                and updates the tool list before the LLM call. If retrieval
                fails it logs the exception and leaves the existing tools
                unchanged — retrieval must never break the agent run.
                """
                chat_middleware_decorator = _import_chat_middleware()
                provider = self

                @chat_middleware_decorator
                async def _per_call_retrieval(context: Any, call_next: Any) -> None:
                    try:
                        await provider._refresh_tools_on_chat_context(context)
                    except Exception:
                        logger.exception(
                            "GantryContextProvider: per-call retrieval failed; "
                            "continuing with existing tools."
                        )
                    await call_next()

                return _per_call_retrieval

            async def _refresh_tools_on_chat_context(self, context: Any) -> None:
                """Mutate ``context.options['tools']`` for the current round."""
                options = getattr(context, "options", None)
                existing = []
                if isinstance(options, dict):
                    existing = list(options.get("tools") or [])

                # Preserve tools contributed by other providers / static tools.
                # We only swap out the slice we previously injected, recognised
                # by name match against everything we'd produce now.
                messages = (
                    getattr(context, "messages", None)
                    or getattr(context, "input_messages", None)
                    or []
                )
                fresh, seen = await self._collect_tools(messages, include_dynamic=True)
                fresh_names = {_tool_name(t) for t in fresh if _tool_name(t)}

                # Preserve everything that wasn't ours; replace ours with fresh.
                preserved = []
                for t in existing:
                    name = _tool_name(t)
                    if name and name in fresh_names:
                        continue
                    preserved.append(t)

                combined: list[Any] = []
                seen_combined: set[str] = set()
                for t in preserved + fresh:
                    name = _tool_name(t)
                    if name and name in seen_combined:
                        continue
                    if name:
                        seen_combined.add(name)
                    combined.append(t)

                if isinstance(options, dict):
                    new_options = dict(options)
                    new_options["tools"] = combined
                    try:
                        context.options = new_options
                    except AttributeError:
                        # Some AF versions expose options as a read-only attr;
                        # mutate in place as a fallback.
                        options.clear()
                        options.update(new_options)

                logger.debug(
                    "GantryContextProvider: refreshed tools per-call "
                    "(%d preserved + %d gantry = %d total)",
                    len(preserved),
                    len(fresh),
                    len(combined),
                )

            # ----- Tool assembly -----------------------------------------
            async def _collect_tools(
                self,
                messages: Any,
                *,
                include_dynamic: bool,
            ) -> tuple[list[Any], set[str]]:
                tools: list[Any] = []
                seen: set[str] = set()

                if include_dynamic:
                    query = await self._build_query(messages)
                    if query:
                        try:
                            retrieved = await self._bridge.get_tools(
                                query,
                                limit=self._top_k,
                                score_threshold=self._score_threshold,
                                **self._query_kwargs,
                            )
                        except Exception:
                            logger.exception(
                                "GantryContextProvider: semantic retrieval failed; "
                                "continuing without dynamic tools."
                            )
                            retrieved = []
                        for t in retrieved:
                            name = _tool_name(t)
                            if name and name in seen:
                                continue
                            if name:
                                seen.add(name)
                            tools.append(t)

                # Skills: always-on tools bound to registered Gantry skills.
                if self._skills_enabled:
                    skill_tool_names = self._collect_skill_tool_names()
                    extra = self._wrap_named(skill_tool_names, seen, source="skill")
                    tools.extend(extra)

                # Explicit always_include / required pins.
                if self._always_include:
                    extra = self._wrap_named(
                        self._always_include, seen, source="always_include"
                    )
                    tools.extend(extra)

                return tools, seen

            async def _build_query(self, messages: Any) -> str:
                try:
                    result = self._query_generator(messages)
                    return await _maybe_await(result) or ""
                except Exception:
                    logger.exception(
                        "GantryContextProvider: query_generator raised; falling "
                        "back to last_user_text."
                    )
                    return _default_query_generator(messages)

            def _resolve_skill_registry(self) -> SkillRegistry | None:
                if self._skill_registry is not None:
                    return self._skill_registry
                # Best-effort: pick up a registry attached to the gantry
                # instance (e.g. via SkillsClient).
                return getattr(self._gantry, "_skill_registry", None) or getattr(
                    self._gantry, "skill_registry", None
                )

            def _collect_skill_tool_names(self) -> list[str]:
                registry = self._resolve_skill_registry()
                if registry is None:
                    logger.debug(
                        "GantryContextProvider: skills=True but no SkillRegistry "
                        "available on gantry — no skill tools injected."
                    )
                    return []
                names: list[str] = []
                seen_names: set[str] = set()
                for skill in registry.list_skills():
                    for tool_name in skill.tools or []:
                        if tool_name not in seen_names:
                            seen_names.add(tool_name)
                            names.append(tool_name)
                return names

            def _wrap_named(
                self, names: list[str], seen: set[str], *, source: str
            ) -> list[Any]:
                wrapped: list[Any] = []
                missing: list[str] = []
                for name in names:
                    if name in seen:
                        continue
                    tool_def = self._lookup_tool_def(name)
                    if tool_def is None:
                        missing.append(name)
                        continue
                    wrapped.append(self._bridge.wrap_single(tool_def))
                    seen.add(name)
                if missing:
                    logger.warning(
                        "GantryContextProvider: %s tool(s) not found in registry "
                        "and will be skipped: %s",
                        source,
                        missing,
                    )
                return wrapped

            def _lookup_tool_def(self, name: str) -> ToolDefinition | None:
                registry = getattr(self._gantry, "_registry", None)
                if registry is None:
                    return None
                lookup = getattr(registry, "get_tool_by_name", None)
                if callable(lookup):
                    found = lookup(name)
                    if found is not None:
                        return found
                lookup = getattr(registry, "get_tool", None)
                if callable(lookup):
                    return lookup(name)
                return None

        return _GantryContextProviderImpl()


__all__ = ["GantryContextProvider", "MissingRequiredToolError"]
