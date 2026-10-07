"""Regression tests for the October 2026 whole-library review.

Each test pins one defect that the review reproduced, named for the behaviour
rather than the finding, and each fails on the code as it stood before.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_gantry import AgentGantry, ConversationContext, ToolCall, ToolQuery
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore
from agent_gantry.core.router import SemanticRouter
from agent_gantry.schema.config import AgentGantryConfig, ExecutionConfig
from agent_gantry.schema.execution import ExecutionStatus
from agent_gantry.schema.mcp import PSEUDO_NAMESPACE
from agent_gantry.schema.tool import ToolDefinition


def _tool(name: str, description: str, namespace: str = "default") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        namespace=namespace,
        parameters_schema={"type": "object", "properties": {}},
    )


def _query(text: str, **kwargs: Any) -> ToolQuery:
    kwargs.setdefault("score_threshold", 0.0)
    return ToolQuery(context=ConversationContext(query=text), **kwargs)


# --------------------------------------------------------------------------
# MCP server pseudo-tools are stored beside real tools but are not tools
# --------------------------------------------------------------------------


async def test_the_router_never_returns_an_mcp_server_pseudo_tool() -> None:
    store = InMemoryVectorStore()
    embedder = SimpleEmbedder(dimension=64)
    real = _tool("read_file", "Read a file from disk")
    # The pseudo-tool's text is the server's description, so for this query it
    # embeds at least as close as the real tool.
    pseudo = _tool("mcp_server_default_fs_1f9c8dcf", "Read a file from disk", PSEUDO_NAMESPACE)
    tools = [real, pseudo]
    await store.initialize()
    await store.add_tools(tools, await embedder.embed_batch([t.description for t in tools]))

    router = SemanticRouter(vector_store=store, embedder=embedder)
    query = _query("Read a file from disk", limit=5)

    routed = await router.route(query)
    assert [tool.name for tool, _ in routed.tools] == ["read_file"]
    # The selector path filters through the same rules.
    assert router.filter_tools(tools, query) == [real]


async def test_listing_and_retrieval_leave_registered_mcp_servers_out() -> None:
    gantry = AgentGantry(vector_store=InMemoryVectorStore(), embedder=SimpleEmbedder(dimension=64))
    try:

        @gantry.register()
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        gantry.register_mcp_server(
            name="filesystem",
            command=["echo"],
            description="Read and write files on the local filesystem",
        )
        assert await gantry.sync_mcp_servers() == 1

        listed = await gantry.list_tools()
        assert [tool.name for tool in listed] == ["add_numbers"]

        retrieved = await gantry.retrieve(_query("read a file from the filesystem", limit=5))
        assert [scored.tool.name for scored in retrieved.tools] == ["add_numbers"]

        # The servers are still there for the code that does want them.
        pseudo = await gantry.list_tools(namespace=PSEUDO_NAMESPACE)
        assert len(pseudo) == 1
        assert [s.name for s in await gantry.retrieve_mcp_servers("files", limit=3)] == [
            "filesystem"
        ]
    finally:
        await gantry.close()


# --------------------------------------------------------------------------
# Health-aware routing has to work on stores that return copies
# --------------------------------------------------------------------------


class _CopyingStore(InMemoryVectorStore):
    """Snapshots each tool when it is written, the way a persistent store does.

    LanceDB, Qdrant, Chroma and pgvector serialise a tool at sync time and
    deserialise a fresh copy for every candidate, so the object routing sees is
    never the one the executor updates afterwards. The in-memory store keeps
    the registry's own object, which is why the defect hid there.
    """

    async def add_tools(self, tools: Any, *args: Any, **kwargs: Any) -> int:
        snapshots = [tool.model_copy(deep=True) for tool in tools]
        return await super().add_tools(snapshots, *args, **kwargs)


async def test_an_open_circuit_breaker_hides_a_tool_on_a_store_that_returns_copies() -> None:
    config = AgentGantryConfig(
        execution=ExecutionConfig(circuit_breaker_threshold=2, max_retries=0)
    )
    gantry = AgentGantry(
        config=config, vector_store=_CopyingStore(), embedder=SimpleEmbedder(dimension=64)
    )
    try:

        @gantry.register()
        def fragile_lookup(query: str) -> str:
            """Look a record up by query string."""
            raise RuntimeError("backend down")

        @gantry.register()
        def robust_lookup(query: str) -> str:
            """Look a record up by query string."""
            return "found"

        for _ in range(2):
            failed = await gantry.execute(
                ToolCall(tool_name="fragile_lookup", arguments={"query": "x"})
            )
            assert failed.status == ExecutionStatus.FAILURE
        assert gantry._registry.get_tool("fragile_lookup").health.circuit_breaker_open

        found = await gantry.retrieve(_query("Look a record up by query string", limit=5))
        names = [scored.tool.name for scored in found.tools]
        assert names == ["robust_lookup"]

        # ...and the opt-out still returns it, so the filter is what hid it.
        everything = await gantry.retrieve(
            _query("Look a record up by query string", limit=5, exclude_unhealthy=False)
        )
        assert {scored.tool.name for scored in everything.tools} == {
            "fragile_lookup",
            "robust_lookup",
        }
    finally:
        await gantry.close()


@pytest.mark.parametrize("live", [True, False])
async def test_the_health_score_reads_live_health_when_given_a_source(live: bool) -> None:
    from agent_gantry.schema.tool import ToolHealth

    store = _CopyingStore()
    embedder = SimpleEmbedder(dimension=64)
    tool = _tool("lookup", "Look a record up")
    await store.initialize()
    await store.add_tools([tool], await embedder.embed_batch([tool.description]))

    degraded = ToolHealth(success_rate=0.0)
    router = SemanticRouter(
        vector_store=store,
        embedder=embedder,
        health_for=(lambda namespace, name: degraded) if live else None,
    )
    routed = await router.route(_query("Look a record up", limit=3))
    (_, score) = routed.tools[0]

    healthy = SemanticRouter(vector_store=store, embedder=embedder)
    (_, baseline) = (await healthy.route(_query("Look a record up", limit=3))).tools[0]
    if live:
        assert score < baseline  # the health term (weight 0.1) dropped out
    else:
        assert score == pytest.approx(baseline)
