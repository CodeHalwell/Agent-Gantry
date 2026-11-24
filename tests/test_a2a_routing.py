"""A2A agents can be registered, routed to and searched within, as MCP servers can.

Before this the only way in was ``add_a2a_agent()``: fetch the card and turn
every skill into a tool. Nothing indexed the agent itself, so there was no way
to register one without contacting it, or to ask which agent a request is for.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.providers.a2a_client import A2AClient
from agent_gantry.schema.a2a import AgentCard, AgentSkill, a2a_agent_tag
from agent_gantry.schema.config import A2AAgentConfig
from agent_gantry.schema.query import ConversationContext, ToolQuery
from agent_gantry.schema.selection import SelectionCandidate, SelectionResult
from agent_gantry.schema.tool import A2A_AGENT_NAMESPACE, RESERVED_NAMESPACES

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


def _gantry(**kwargs: Any) -> AgentGantry:
    gantry = AgentGantry(embedder=SimpleEmbedder(), **kwargs)
    gantry.register_a2a_agent(
        "travel", "https://travel.example", description="Plans trips and books travel."
    )
    gantry.register_a2a_agent(
        "ledger", "https://ledger.example", description="Bookkeeping and tax returns."
    )
    return gantry


def _query(text: str, **fields: Any) -> ToolQuery:
    return ToolQuery(context=ConversationContext(query=text), score_threshold=0.0, **fields)


async def test_an_agent_is_registered_and_routed_to_without_being_contacted(
    cards: dict[str, int],
) -> None:
    gantry = _gantry()
    try:
        agents = await gantry.retrieve_a2a_agents("bookkeeping and tax returns", limit=1)

        assert [a.name for a in agents] == ["ledger"]
        assert cards == {}  # nothing was fetched
        assert await gantry.list_tools() == []  # and no skill became a tool
        assert [a.name for a in gantry.list_a2a_agents()] == ["travel", "ledger"]
        assert await gantry.retrieve_a2a_agents("   ") == []
    finally:
        await gantry.close()


def test_registration_checks_the_url_and_the_description() -> None:
    gantry = AgentGantry(embedder=SimpleEmbedder())
    with pytest.raises(ValueError, match="http"):
        gantry.register_a2a_agent("bad", "ftp://agent.example", description="Plans trips for people.")
    with pytest.raises(ValueError):
        gantry.register_a2a_agent("bad", "https://agent.example", description="short")


async def test_discovering_a_registered_agent_makes_its_skills_tools_and_findable_by(
    cards: dict[str, int],
) -> None:
    gantry = _gantry()
    try:
        # Nothing in either registration mentions hotels.
        before = await gantry.retrieve_a2a_agents("find a hotel room for given nights", limit=1)

        assert await gantry.discover_skills_from_agent("travel") == 2
        assert cards == {"travel": 1}

        after = await gantry.retrieve_a2a_agents("find a hotel room for given nights", limit=1)
        assert [a.name for a in after] == ["travel"]
        agent = gantry.list_a2a_agents()[0]
        assert agent.description == "Plans trips and books travel."  # the registration's wording stays
        assert agent.skills == [
            "Book flight: Book a flight between two cities.",
            "Find hotel: Find a hotel room for given nights.",
        ]
        assert sorted(t.name for t in await gantry.list_tools()) == sorted(
            t.name for t in await gantry.list_tools() if t.metadata["a2a_agent"] == "travel"
        )
        assert before is not None

        with pytest.raises(ValueError, match="not found"):
            await gantry.discover_skills_from_agent("nobody")
    finally:
        await gantry.close()


async def test_an_agent_added_directly_becomes_routable_from_its_card(cards: dict[str, int]) -> None:
    gantry = AgentGantry(embedder=SimpleEmbedder())
    try:
        await gantry.add_a2a_agent(A2AAgentConfig(name="ledger", url="https://ledger.example"))

        agents = await gantry.retrieve_a2a_agents("prepare a quarterly VAT return", limit=1)
        assert [a.name for a in agents] == ["ledger"]
        # "Bookkeeping help." is the card's own description, kept as written.
        assert agents[0].description == "Bookkeeping help."
    finally:
        await gantry.close()


async def test_a_converted_skill_is_tagged_and_grouped_by_its_agent(cards: dict[str, int]) -> None:
    gantry = _gantry()
    try:
        await gantry.discover_skills_from_agent("travel")
        await gantry.discover_skills_from_agent("ledger")
        tools = await gantry.list_tools()

        for tool in tools:
            agent = tool.metadata["a2a_agent"]
            assert tool.tags == [a2a_agent_tag(agent)] == [f"a2a:{agent}"]
            assert agent in tool.to_searchable_text()
            assert SelectionCandidate.from_tool(tool).group == agent
    finally:
        await gantry.close()


async def test_a_search_can_be_scoped_to_one_agent(cards: dict[str, int]) -> None:
    gantry = _gantry()
    try:
        await gantry.discover_skills_from_agent("travel")
        await gantry.discover_skills_from_agent("ledger")

        scoped = await gantry.retrieve(_query("book", limit=10, a2a_agents=["ledger"]))
        assert len(scoped.tools) == 2
        assert {t.tool.metadata["a2a_agent"] for t in scoped.tools} == {"ledger"}

        assert (await gantry.retrieve(_query("book", limit=10, a2a_agents=[]))).tools == []
    finally:
        await gantry.close()


async def test_agent_entries_are_never_returned_as_tools_and_go_with_the_agent(
    cards: dict[str, int],
) -> None:
    gantry = _gantry()
    try:
        await gantry.discover_skills_from_agent("travel")
        await gantry.retrieve_a2a_agents("plans trips", limit=2)  # syncs the agent entries
        rows = await gantry.list_tools(namespace=A2A_AGENT_NAMESPACE)
        assert len(rows) >= 3  # two agents, and travel's skills

        result = await gantry.retrieve(_query("plans trips and books travel", limit=10))
        assert result.tools
        assert not [t for t in result.tools if t.tool.namespace in RESERVED_NAMESPACES]

        # Pruning tools leaves the agents' entries alone.
        await gantry.sync(prune=True)
        assert len(await gantry.list_tools(namespace=A2A_AGENT_NAMESPACE)) == len(rows)

        assert gantry._a2a_manager.remove("travel")
        await gantry.retrieve_a2a_agents("plans trips", limit=2)
        remaining = await gantry.list_tools(namespace=A2A_AGENT_NAMESPACE)
        assert {r.metadata["agent_name"] for r in remaining} == {"ledger"}
    finally:
        await gantry.close()


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

    gantry = _gantry(selector=_Selector())
    try:
        await gantry.discover_skills_from_agent("ledger")
        agents = await gantry.retrieve_a2a_agents("file my VAT", limit=2)

        assert [a.name for a in agents] == ["ledger"]
        by_id = {c.id: c for c in seen}
        assert "Prepare a quarterly VAT return." in by_id["default.ledger"].description
        assert by_id["default.travel"].description == "Plans trips and books travel."
    finally:
        await gantry.close()


async def test_agents_join_the_combined_view(cards: dict[str, int]) -> None:
    gantry = _gantry()
    try:
        await gantry.discover_skills_from_agent("travel")
        result = await gantry.retrieve_context(
            "book a flight between two cities", tools=1, skills=0, servers=0, agents=1
        )
        assert [(e.kind, e.rank) for e in result.entries] == [("tool", 1), ("a2a_agent", 1)]
        assert [a.name for a in result.a2a_agents] == ["travel"]
    finally:
        await gantry.close()
