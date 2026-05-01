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

    # Dynamic — per-turn semantic retrieval:
    provider = GantryContextProvider(gantry, top_k=5)
    agent = Agent(client, "...", context_providers=[provider])

    # Mixed — pinned static + dynamic top-up + AF skills, all in one agent:
    agent = Agent(
        client,
        "...",
        tools=bridge.wrap_tools(pinned_defs),
        context_providers=[
            GantryContextProvider(gantry, top_k=3, skills=True),
            SkillsProvider(skills=[my_af_skill]),
        ],
    )

The provider also flows transparently through every AF orchestration
primitive (``WorkflowBuilder``, ``SequentialBuilder``, ``HandoffBuilder``,
``AgentExecutor``, ``WorkflowAgent``) because they all dispatch through
``agent.run()``, which fires the provider pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.integrations.anthropic_skills import SkillRegistry
    from agent_gantry.schema.tool import ToolDefinition

logger = logging.getLogger(__name__)


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


def _last_user_text(messages: list[Any]) -> str:
    """Return the text of the most recent user message, or empty string.

    AF stores incoming user turns in ``SessionContext.input_messages``.
    We pick the most recent ``user``-role message — that's the query
    Gantry should retrieve tools against.
    """
    for msg in reversed(messages or []):
        role = getattr(msg, "role", None)
        if role is None or str(role) == "user":
            text = getattr(msg, "text", None)
            if text:
                return text
    return ""


class GantryContextProvider:
    """AF context provider that injects semantically-selected Gantry tools.

    Constructing the provider returns an instance of the dynamically
    generated ``agent_framework.ContextProvider`` subclass. The factory
    pattern keeps the import of ``agent_framework`` lazy so importing
    ``agent_gantry`` doesn't require AF to be installed.

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` instance
            providing semantic retrieval and execution.
        top_k: Maximum number of dynamically retrieved tools per ``agent.run``
            call. Defaults to ``5``.
        score_threshold: Minimum semantic relevance score for retrieved
            tools. Defaults to ``0.3``.
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
            for pinning utility tools the LLM should always see.
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
            from agent_gantry import AgentGantry
            from agent_gantry.integrations import GantryContextProvider

            gantry = AgentGantry()
            # ... register tools, sync ...

            agent = Agent(
                OpenAIChatClient(),
                "You are a helpful assistant.",
                context_providers=[
                    GantryContextProvider(gantry, top_k=5, skills=True),
                    SkillsProvider(skill_paths="./skills"),
                ],
            )
            await agent.run("Book a flight to Tokyo")  # flight tools auto-injected
    """

    def __new__(
        cls,
        gantry: AgentGantry,
        *,
        top_k: int = 5,
        score_threshold: float = 0.3,
        skills: bool = False,
        skill_registry: SkillRegistry | None = None,
        always_include: list[str] | None = None,
        as_function_tool: bool | None = None,
        source_id: str = "agent_gantry",
        bridge: GantryToolBridge | None = None,
        **query_kwargs: Any,
    ) -> Any:
        context_provider_cls = _import_context_provider()

        bridge = bridge or GantryToolBridge(
            gantry,
            score_threshold=score_threshold,
            as_function_tool=as_function_tool,
        )
        always_include = list(always_include or [])

        class _GantryContextProviderImpl(context_provider_cls):  # type: ignore[misc,valid-type]
            """Concrete ContextProvider subclass; see :class:`GantryContextProvider`."""

            def __init__(self) -> None:
                super().__init__(source_id=source_id)
                self._gantry = gantry
                self._bridge = bridge
                self._top_k = top_k
                self._score_threshold = score_threshold
                self._skills_enabled = skills
                self._skill_registry = skill_registry
                self._always_include = always_include
                self._query_kwargs = dict(query_kwargs)

            async def before_run(
                self,
                *,
                agent: Any,
                session: Any,
                context: Any,
                state: dict[str, Any],
            ) -> None:
                query = _last_user_text(context.input_messages)
                tools: list[Any] = []
                seen: set[str] = set()

                # 1) Dynamic semantic retrieval against the latest user turn.
                if query:
                    try:
                        retrieved = await self._bridge.get_tools(
                            query,
                            limit=self._top_k,
                            score_threshold=self._score_threshold,
                            **self._query_kwargs,
                        )
                    except Exception:
                        # Tool retrieval should never break the agent run.
                        logger.exception(
                            "GantryContextProvider: semantic retrieval failed; "
                            "continuing without dynamic tools."
                        )
                        retrieved = []
                    for t in retrieved:
                        name = getattr(t, "name", None) or getattr(t, "__name__", None)
                        if name and name in seen:
                            continue
                        if name:
                            seen.add(name)
                        tools.append(t)

                # 2) Skills: always-on tools bound to registered Gantry skills.
                if self._skills_enabled:
                    skill_tool_names = self._collect_skill_tool_names()
                    extra = self._wrap_named(skill_tool_names, seen)
                    tools.extend(extra)

                # 3) Explicit always_include pins.
                if self._always_include:
                    extra = self._wrap_named(self._always_include, seen)
                    tools.extend(extra)

                if tools:
                    context.extend_tools(self.source_id, tools)
                    logger.debug(
                        "GantryContextProvider: injected %d tools (query=%r)",
                        len(tools),
                        query[:60],
                    )

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
                self, names: list[str], seen: set[str]
            ) -> list[Any]:
                wrapped: list[Any] = []
                for name in names:
                    if name in seen:
                        continue
                    tool_def = self._lookup_tool_def(name)
                    if tool_def is None:
                        logger.warning(
                            "GantryContextProvider: tool %r not found in "
                            "registry; skipping.",
                            name,
                        )
                        continue
                    wrapped.append(self._bridge.wrap_single(tool_def))
                    seen.add(name)
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


__all__ = ["GantryContextProvider"]
