"""A2A agents can be registered, routed to and searched within, as MCP servers can.

Before this the only way in was ``add_a2a_agent()``: fetch the card and turn
every skill into a tool. Nothing indexed the agent itself, so there was no way
to register one without contacting it, or to ask which agent a request is for.
"""

from __future__ import annotations

from typing import Any

import pytest

from dotfetch import DotFetch
from dotfetch.adapters.embedders.simple import SimpleEmbedder
from dotfetch.providers.a2a_client import A2AClient
from dotfetch.schema.a2a import AgentCard, AgentSkill, a2a_agent_tag
from dotfetch.schema.config import A2AAgentConfig
from dotfetch.schema.query import ConversationContext, ToolQuery
from dotfetch.schema.selection import SelectionCandidate, SelectionResult
from dotfetch.schema.tool import A2A_AGENT_NAMESPACE, is_reserved_namespace

_CARDS = {
    "travel": AgentCard(
        name="Travel Desk",
        description="Plans trips.",
        url="https://travel.example",
        skills=[
            AgentSkill(id="book_flight", name="Book flight", description="Book a flight between two cities."),
            AgentSkill(id="find_hotel", name="Find hotel", description="Find a hotel room for given nights."),
        ],
    ),
    "ledger": AgentCard(
        name="Ledger",
        description="Bookkeeping help.",
        url="https://ledger.example",
        skills=[
            AgentSkill(id="reconcile", name="Reconcile", description="Reconcile a bank statement against the books."),
            AgentSkill(id="file_vat", name="File VAT", description="Prepare a quarterly VAT return."),
        ],
    ),
}


@pytest.fixture
def cards(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Serve ``_CARDS`` to every ``A2AClient`` without a network, counting fetches."""
    fetched: dict[str, int] = {}

    async def discover(self: A2AClient) -> AgentCard:
        fetched[self.config.name] = fetched.get(self.config.name, 0) + 1
        self._agent_card = _CARDS[self.config.name]
        return self._agent_card

    async def close(self: A2AClient) -> None:
        return None

    monkeypatch.setattr(A2AClient, "discover", discover)
    monkeypatch.setattr(A2AClient, "close", close)
    return fetched


def _fetcher(**kwargs: Any) -> DotFetch:
    fetcher = DotFetch(embedder=SimpleEmbedder(), **kwargs)
    fetcher.register_a2a_agent(
        "travel", "https://travel.example", description="Plans trips and books travel."
    )
    fetcher.register_a2a_agent(
        "ledger", "https://ledger.example", description="Bookkeeping and tax returns."
    )
    return fetcher


def _query(text: str, **fields: Any) -> ToolQuery:
    return ToolQuery(context=ConversationContext(query=text), score_threshold=0.0, **fields)


async def test_an_agent_is_registered_and_routed_to_without_being_contacted(
    cards: dict[str, int],
) -> None:
    fetcher = _fetcher()
    try:
        agents = await fetcher.retrieve_a2a_agents("bookkeeping and tax returns", limit=1)

        assert [a.name for a in agents] == ["ledger"]
        assert cards == {}  # nothing was fetched
        assert await fetcher.list_tools() == []  # and no skill became a tool
        assert [a.name for a in fetcher.list_a2a_agents()] == ["travel", "ledger"]
        assert await fetcher.retrieve_a2a_agents("   ") == []
    finally:
        await fetcher.close()


def test_registration_checks_the_url_and_the_description() -> None:
    fetcher = DotFetch(embedder=SimpleEmbedder())
    with pytest.raises(ValueError, match="http"):
        fetcher.register_a2a_agent("bad", "ftp://agent.example", description="Plans trips for people.")
    with pytest.raises(ValueError):
        fetcher.register_a2a_agent("bad", "https://agent.example", description="short")


async def test_discovering_a_registered_agent_makes_its_skills_tools_and_findable_by(
    cards: dict[str, int],
) -> None:
    fetcher = _fetcher()
    try:
        # Nothing in either registration mentions hotels.
        before = await fetcher.retrieve_a2a_agents("find a hotel room for given nights", limit=1)

        assert await fetcher.discover_skills_from_agent("travel") == 2
        assert cards == {"travel": 1}

        after = await fetcher.retrieve_a2a_agents("find a hotel room for given nights", limit=1)
        assert [a.name for a in after] == ["travel"]
        agent = fetcher.list_a2a_agents()[0]
        assert agent.description == "Plans trips and books travel."  # the registration's wording stays
        assert agent.skills == [
            "Book flight: Book a flight between two cities.",
            "Find hotel: Find a hotel room for given nights.",
        ]
        assert sorted(t.name for t in await fetcher.list_tools()) == sorted(
            t.name for t in await fetcher.list_tools() if t.metadata["a2a_agent"] == "travel"
        )
        assert before is not None

        with pytest.raises(ValueError, match="not found"):
            await fetcher.discover_skills_from_agent("nobody")
    finally:
        await fetcher.close()


async def test_an_agent_added_directly_becomes_routable_from_its_card(cards: dict[str, int]) -> None:
    fetcher = DotFetch(embedder=SimpleEmbedder())
    try:
        await fetcher.add_a2a_agent(A2AAgentConfig(name="ledger", url="https://ledger.example"))

        agents = await fetcher.retrieve_a2a_agents("prepare a quarterly VAT return", limit=1)
        assert [a.name for a in agents] == ["ledger"]
        # "Bookkeeping help." is the card's own description, kept as written.
        assert agents[0].description == "Bookkeeping help."
    finally:
        await fetcher.close()


async def test_a_converted_skill_is_tagged_and_grouped_by_its_agent(cards: dict[str, int]) -> None:
    fetcher = _fetcher()
    try:
        await fetcher.discover_skills_from_agent("travel")
        await fetcher.discover_skills_from_agent("ledger")
        tools = await fetcher.list_tools()

        for tool in tools:
            agent = tool.metadata["a2a_agent"]
            assert tool.tags == [a2a_agent_tag(agent)] == [f"a2a:{agent}"]
            assert agent in tool.to_searchable_text()
            assert SelectionCandidate.from_tool(tool).group == agent
    finally:
        await fetcher.close()


async def test_a_search_can_be_scoped_to_one_agent(cards: dict[str, int]) -> None:
    fetcher = _fetcher()
    try:
        await fetcher.discover_skills_from_agent("travel")
        await fetcher.discover_skills_from_agent("ledger")

        scoped = await fetcher.retrieve(_query("book", limit=10, a2a_agents=["ledger"]))
        assert len(scoped.tools) == 2
        assert {t.tool.metadata["a2a_agent"] for t in scoped.tools} == {"ledger"}

        assert (await fetcher.retrieve(_query("book", limit=10, a2a_agents=[]))).tools == []
    finally:
        await fetcher.close()


async def test_agent_entries_are_never_returned_as_tools_and_go_with_the_agent(
    cards: dict[str, int],
) -> None:
    fetcher = _fetcher()
    try:
        await fetcher.discover_skills_from_agent("travel")
        await fetcher.retrieve_a2a_agents("plans trips", limit=2)  # syncs the agent entries
        rows = await fetcher.list_tools(namespace=A2A_AGENT_NAMESPACE)
        assert len(rows) >= 3  # two agents, and travel's skills

        result = await fetcher.retrieve(_query("plans trips and books travel", limit=10))
        assert result.tools
        assert not [t for t in result.tools if is_reserved_namespace(t.tool.namespace)]

        # Pruning tools leaves the agents' entries alone.
        await fetcher.sync(prune=True)
        assert len(await fetcher.list_tools(namespace=A2A_AGENT_NAMESPACE)) == len(rows)

        assert fetcher._a2a_manager.remove("travel")
        await fetcher.retrieve_a2a_agents("plans trips", limit=2)
        remaining = await fetcher.list_tools(namespace=A2A_AGENT_NAMESPACE)
        assert {r.metadata["agent_name"] for r in remaining} == {"ledger"}
    finally:
        await fetcher.close()


async def test_a_selector_chooses_among_agents_and_is_shown_their_skills(
    cards: dict[str, int],
) -> None:
    seen: list[Any] = []

    class _Selector:
        async def select(self, query: str, candidates: Any, limit: int, *, kind: str = "tool"):
            if kind == "a2a_agent":
                seen.extend(candidates)
            chosen = [c.id for c in candidates if "ledger" in c.id][:limit]
            return SelectionResult(selected=chosen, scores={cid: 0.9 for cid in chosen})

    fetcher = _fetcher(selector=_Selector())
    try:
        await fetcher.discover_skills_from_agent("ledger")
        agents = await fetcher.retrieve_a2a_agents("file my VAT", limit=2)

        assert [a.name for a in agents] == ["ledger"]
        by_id = {c.id: c for c in seen}
        assert "Prepare a quarterly VAT return." in by_id["default.ledger"].description
        assert by_id["default.travel"].description == "Plans trips and books travel."
    finally:
        await fetcher.close()


async def test_agents_join_the_combined_view(cards: dict[str, int]) -> None:
    fetcher = _fetcher()
    try:
        await fetcher.discover_skills_from_agent("travel")
        result = await fetcher.retrieve_context(
            "book a flight between two cities", tools=1, skills=0, servers=0, agents=1
        )
        assert [(e.kind, e.rank) for e in result.entries] == [("tool", 1), ("a2a_agent", 1)]
        assert [a.name for a in result.a2a_agents] == ["travel"]
    finally:
        await fetcher.close()


async def test_a_namespace_scope_is_not_lost_among_other_namespaces_agents() -> None:
    """Seventy better-matching agents elsewhere; the one in scope is still found.

    The scope is applied to what the store returns, and the search used to
    stop at 64 rows for one agent asked for.
    """
    fetcher = DotFetch(embedder=SimpleEmbedder())
    try:
        for i in range(70):
            fetcher.register_a2a_agent(
                f"payroll_{i}",
                f"https://payroll-{i}.example",
                description="Payroll payroll payroll queries about payroll.",
                namespace="finance",
            )
        fetcher.register_a2a_agent(
            "wages", "https://wages.example", description="Answers payroll queries.", namespace="ops"
        )

        agents = await fetcher.retrieve_a2a_agents("payroll", limit=1, namespaces=["ops"])
        assert [a.name for a in agents] == ["wages"]
        # Unscoped, one of the seventy is the better match: the scope did the work.
        assert (await fetcher.retrieve_a2a_agents("payroll", limit=1))[0].namespace == "finance"
    finally:
        await fetcher.close()


async def test_an_agent_registered_while_a_sync_is_in_flight_is_not_lost() -> None:
    """The sync's snapshot does not hold it, so the sync must not mark it synced."""
    fetcher = DotFetch(embedder=SimpleEmbedder())

    class _Interrupting(SimpleEmbedder):
        registered = False

        async def embed_batch(self, texts: list[str], batch_size: int | None = None) -> Any:
            if not self.registered:
                self.registered = True
                fetcher.register_a2a_agent(
                    "latecomer", "https://late.example", description="Handles payroll queries."
                )
            return await super().embed_batch(texts, batch_size)

    manager = fetcher._a2a_manager
    manager._embedder = _Interrupting()
    try:
        fetcher.register_a2a_agent(
            "travel", "https://travel.example", description="Plans trips and books travel."
        )
        await fetcher._ensure_initialized()
        await manager.sync()

        assert manager.needs_sync  # the latecomer is still owed a sync
        agents = await fetcher.retrieve_a2a_agents("payroll queries", limit=1)
        assert [a.name for a in agents] == ["latecomer"]
        assert not manager.needs_sync
    finally:
        await fetcher.close()


async def test_agent_entries_are_re_embedded_when_the_embedding_model_changes() -> None:
    """Even when a tool sync has already told the store about the new model."""

    class _Counting(SimpleEmbedder):
        def __init__(self, name: str) -> None:
            super().__init__()
            self._name = name
            self.embedded: list[str] = []

        @property
        def model_name(self) -> str:
            return self._name

        async def embed_batch(self, texts: list[str], batch_size: int | None = None) -> Any:
            self.embedded.extend(texts)
            return await super().embed_batch(texts, batch_size)

    def tool() -> Any:
        from dotfetch.schema.tool import ToolDefinition

        return ToolDefinition(
            name="add_numbers",
            description="Add two numbers together.",
            parameters_schema={"type": "object", "properties": {}},
        )

    first = DotFetch(embedder=_Counting("model-a"))
    first.register_a2a_agent("travel", "https://travel.example", description="Plans trips and books travel.")
    await first.add_tool(tool())
    await first.retrieve_a2a_agents("plans trips", limit=1)

    newer = _Counting("model-b")
    second = DotFetch(embedder=newer, vector_store=first._vector_store)
    try:
        second.register_a2a_agent("travel", "https://travel.example", description="Plans trips and books travel.")
        await second.add_tool(tool())  # the tool sync marks the store as model-b
        newer.embedded.clear()

        await second.retrieve_a2a_agents("plans trips", limit=1)
        assert any(text.startswith("Agent: travel") for text in newer.embedded)

        newer.embedded.clear()
        second.register_a2a_agent("ledger", "https://ledger.example", description="Bookkeeping and tax returns.")
        await second.retrieve_a2a_agents("plans trips", limit=1)
        assert [t for t in newer.embedded if t.startswith("Agent: ")] == [
            "Agent: ledger Description: Bookkeeping and tax returns."
        ]
    finally:
        await first.close()


async def test_many_agent_entries_do_not_hide_a_tool_from_a_tool_search() -> None:
    """Agents share the tools index; a hundred of them ranked first hid the one tool.

    Three searches of 32, 64 and 96 rows held nothing but agents, and the
    search stopped there with no tool, at a threshold of zero.
    """
    fetcher = DotFetch(embedder=SimpleEmbedder())
    try:
        for i in range(110):
            fetcher.register_a2a_agent(
                f"payroll_{i}",
                f"https://payroll-{i}.example",
                description="Payroll payroll payroll queries about payroll.",
            )

        @fetcher.register
        def run_report(month: str) -> str:
            """Run the monthly payroll report for the finance team."""
            return month

        await fetcher.retrieve_a2a_agents("payroll", limit=1)  # indexes the agents
        result = await fetcher.retrieve(_query("payroll", limit=1))
        assert [scored.tool.name for scored in result.tools] == ["run_report"]
    finally:
        await fetcher.close()
