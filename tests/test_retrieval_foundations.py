"""What a retriever is handed, and what it says about how it answered.

Four behaviours, each of which used to fail without a sound:

* a deployment without sentence-transformers retrieved on a hash of tokens;
* a tool discovered from an MCP server carried its server's name nowhere a
  retriever could see it;
* a tool's parameter names were not searchable;
* the pseudo-entry an MCP server is indexed as could be returned as a tool.
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.core import factories
from agent_gantry.schema.config import EmbedderConfig
from agent_gantry.schema.mcp import mcp_server_tag
from agent_gantry.schema.query import ConversationContext, ToolQuery
from agent_gantry.schema.selection import SelectionCandidate, SelectionResult
from agent_gantry.schema.tool import RESERVED_NAMESPACES, ToolDefinition

FACTORIES_LOG = "agent_gantry.core.factories"


def _tool(name: str, description: str, **fields: Any) -> ToolDefinition:
    fields.setdefault("parameters_schema", {"type": "object", "properties": {}})
    return ToolDefinition(name=name, description=description, **fields)


def _query(text: str, **fields: Any) -> ToolQuery:
    return ToolQuery(context=ConversationContext(query=text), score_threshold=0.0, **fields)


# --------------------------------------------------------------------------- #
# The embedder downgrade is reported
# --------------------------------------------------------------------------- #


@pytest.fixture
def no_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import sentence_transformers`` fail, and forget any earlier warning."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setattr(factories, "_simple_fallback_warned", False)


def test_the_downgrade_is_a_warning_that_names_the_fix(
    no_sentence_transformers: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG, logger=FACTORIES_LOG):
        embedder = factories.build_embedder(EmbedderConfig())

    assert isinstance(embedder, SimpleEmbedder)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "SimpleEmbedder" in message
    assert "agent-gantry[embeddings]" in message
    assert "RetrievalResult.embedder" in message


def test_the_downgrade_is_reported_once_per_process(
    no_sentence_transformers: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG, logger=FACTORIES_LOG):
        for _ in range(3):
            factories.build_embedder(EmbedderConfig())

    levels = [r.levelno for r in caplog.records]
    assert levels.count(logging.WARNING) == 1
    assert levels.count(logging.DEBUG) == 2


def test_strict_mode_refuses_the_downgrade(no_sentence_transformers: None) -> None:
    with pytest.raises(ImportError, match=r"agent-gantry\[embeddings\]"):
        factories.build_embedder(EmbedderConfig(on_unavailable="raise"))


def test_an_explicit_embedder_never_warns(
    no_sentence_transformers: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Choosing the hash embedder (as tests do) is not a downgrade."""
    with caplog.at_level(logging.WARNING, logger=FACTORIES_LOG):
        AgentGantry(embedder=SimpleEmbedder())
    assert not caplog.records


async def test_a_result_names_the_embedder_that_searched() -> None:
    gantry = AgentGantry(embedder=SimpleEmbedder())
    try:
        await gantry.add_tool(_tool("add_numbers", "Add two numbers together."))
        result = await gantry.retrieve(_query("add two numbers"))
        assert result.embedder == "simple-hash-embedder"
        assert result.backend == "semantic"
    finally:
        await gantry.close()


async def test_a_selector_answer_names_no_embedder() -> None:
    """None was used, and saying otherwise would misattribute the result."""

    class _Selector:
        async def select(self, query: str, candidates: Any, limit: int, *, kind: str = "tool"):
            return SelectionResult(
                selected=[c.id for c in candidates][:limit],
                scores={c.id: 0.9 for c in candidates},
            )

    gantry = AgentGantry(embedder=SimpleEmbedder(), selector=_Selector())
    try:
        await gantry.add_tool(_tool("add_numbers", "Add two numbers together."))
        result = await gantry.retrieve(_query("add two numbers"))
        assert result.backend == "selector"
        assert result.embedder is None
    finally:
        await gantry.close()


async def test_an_embedder_that_does_not_name_itself_is_not_an_error() -> None:
    class _Anonymous(SimpleEmbedder):
        model_name = property(lambda self: None)  # type: ignore[assignment]

    gantry = AgentGantry(embedder=_Anonymous())
    try:
        await gantry.add_tool(_tool("add_numbers", "Add two numbers together."))
        result = await gantry.retrieve(_query("add two numbers"))
        assert result.tools and result.embedder is None
    finally:
        await gantry.close()


async def test_the_retrieval_decision_carries_the_embedder() -> None:
    pytest.importorskip("agent_framework")
    from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

    gantry = AgentGantry(embedder=SimpleEmbedder())
    try:

        @gantry.register
        async def add_numbers(a: int, b: int) -> int:
            """Add two numbers together for the caller."""
            return a + b

        _, decision = await GantryToolBridge(gantry).get_tools_with_decision("add", limit=2)
        assert decision.embedder == "simple-hash-embedder"
        assert decision.as_span_attributes()["embedder"] == "simple-hash-embedder"
    finally:
        await gantry.close()


# --------------------------------------------------------------------------- #
# An MCP tool carries its server
# --------------------------------------------------------------------------- #


def _mcp_tool(server: str, name: str, description: str) -> ToolDefinition:
    pytest.importorskip("mcp")
    from agent_gantry.adapters.executors.mcp_client import MCPClient
    from agent_gantry.schema.config import MCPServerConfig

    client = MCPClient(MCPServerConfig(name=server, url=f"https://{server}.example/mcp"))
    return client._convert_tool(SimpleNamespace(name=name, description=description, inputSchema={}))


def test_a_discovered_tool_is_tagged_with_its_server() -> None:
    # Neither the name nor the description mentions the server, so the tag is
    # the only way its name can reach the embedding.
    tool = _mcp_tool("fathom-meetings", "list_transcripts", "List all available transcripts.")

    assert tool.tags == [mcp_server_tag("fathom-meetings")] == ["mcp:fathom-meetings"]
    assert "fathom-meetings" in tool.to_searchable_text()


def test_a_selector_groups_mcp_tools_by_server_and_native_tools_by_namespace() -> None:
    logs = _mcp_tool("datadog", "search_logs", "Search the application logs.")
    storage = _mcp_tool("azure", "get_storage_details", "Get storage account details.")
    native = _tool("add_numbers", "Add two numbers together.", namespace="maths")

    assert logs.namespace == storage.namespace == "default"
    assert SelectionCandidate.from_tool(logs).group == "datadog"
    assert SelectionCandidate.from_tool(storage).group == "azure"
    assert SelectionCandidate.from_tool(native).group == "maths"
    assert "mcp:datadog" in SelectionCandidate.from_tool(logs).tags


async def test_a_store_search_can_be_scoped_to_one_server() -> None:
    from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore

    embedder = SimpleEmbedder()
    tools = [
        _mcp_tool("datadog", "search_logs", "Search the application logs."),
        _mcp_tool("datadog", "list_monitors", "List the alerting monitors."),
        _mcp_tool("azure", "search_resources", "Search the cloud resources."),
        _tool("search_notes", "Search the user's notes."),
    ]
    store = InMemoryVectorStore()
    await store.initialize()
    await store.add_tools(
        tools, await embedder.embed_batch([t.to_searchable_text() for t in tools])
    )

    hits = await store.search(
        await embedder.embed_text("search"),
        limit=10,
        filters={"tags": [mcp_server_tag("datadog")]},
    )
    assert {tool.name for tool, _ in hits} == {"search_logs", "list_monitors"}


# --------------------------------------------------------------------------- #
# Parameter names are searchable
# --------------------------------------------------------------------------- #


def _with_parameters(*names: str, description: str = "Search the logs.") -> ToolDefinition:
    return _tool(
        "search_logs",
        description,
        tags=["observability"],
        parameters_schema={"type": "object", "properties": {n: {"type": "string"} for n in names}},
    )


def test_distinctive_parameter_names_are_indexed_and_generic_ones_are_not() -> None:
    text = _with_parameters("facet_query", "query", "limit", "Page", "time_range").to_searchable_text()

    assert "facet_query" in text and "time_range" in text
    assert text.split().count("query") == 0
    assert "limit" not in text.split() and "Page" not in text.split()


def test_short_fields_come_before_the_description() -> None:
    """Whatever follows a long description is cut off by the embedder."""
    tool = _with_parameters("facet_query", description="Search the logs. " * 100)
    text = tool.to_searchable_text()

    start = text.index("Search the logs.")
    assert text.index("observability") < start
    assert text.index("facet_query") < start
    assert text.startswith("search_logs default ")


def test_a_wide_schema_cannot_crowd_out_the_description() -> None:
    text = _with_parameters(*(f"option_{i:03d}" for i in range(200))).to_searchable_text()

    indexed = [word for word in text.split() if word.startswith("option_")]
    assert len(indexed) == 16
    assert text.endswith("Search the logs.")


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object"},
        {"type": "object", "properties": None},
        {"type": "object", "properties": ["not", "a", "mapping"]},
        {"type": "object", "properties": {"": {}, 7: {}}},
    ],
)
def test_an_odd_schema_indexes_no_parameters(schema: dict[str, Any]) -> None:
    tool = _tool("ping_host", "Check whether a host answers.", parameters_schema=schema)
    assert tool.to_searchable_text() == "ping_host default Check whether a host answers."


def test_a_tool_with_nothing_optional_has_no_stray_spaces() -> None:
    assert _tool("ping_host", "Check whether a host answers.").to_searchable_text() == (
        "ping_host default Check whether a host answers."
    )


async def test_a_query_naming_a_parameter_finds_the_tool() -> None:
    gantry = AgentGantry(embedder=SimpleEmbedder())
    try:
        await gantry.add_tool(
            _with_parameters("facet_query", description="Look through application output.")
        )
        for i in range(12):
            await gantry.add_tool(_tool(f"read_report_{i}", f"Read report number {i} aloud."))
        result = await gantry.retrieve(_query("run a facet_query for checkout", limit=3))
        assert result.tools[0].tool.name == "search_logs"
    finally:
        await gantry.close()


# --------------------------------------------------------------------------- #
# Index bookkeeping is never returned as a tool
# --------------------------------------------------------------------------- #


async def _gantry_with_servers(count: int) -> AgentGantry:
    pytest.importorskip("mcp")
    gantry = AgentGantry(embedder=SimpleEmbedder())
    for i in range(6):
        await gantry.add_tool(_tool(f"restart_service_{i}", f"Restart service number {i}."))
    for i in range(count):
        gantry.register_mcp_server(
            f"restarter_{i}",
            url=f"https://restarter-{i}.example/mcp",
            # Worded to out-score every real tool for the query below.
            description="restart restart restart the service",
        )
    await gantry.sync()
    await gantry.sync_mcp_servers()
    return gantry


async def test_an_mcp_server_entry_is_not_returned_as_a_tool() -> None:
    gantry = await _gantry_with_servers(3)
    try:
        result = await gantry.retrieve(_query("restart the service", limit=5))
        assert result.tools
        assert not [t for t in result.tools if t.tool.namespace in RESERVED_NAMESPACES]
        # The servers themselves are still routable, through their own call.
        assert len(await gantry.retrieve_mcp_servers("restart the service", limit=3)) == 3
    finally:
        await gantry.close()


async def test_many_server_entries_do_not_crowd_the_tools_out() -> None:
    """Forty entries that out-score the tools more than fill the first pool of 32."""
    gantry = await _gantry_with_servers(40)
    try:
        result = await gantry.retrieve(_query("restart the service", limit=5))
        assert len(result.tools) == 5
        assert all(t.tool.name.startswith("restart_service_") for t in result.tools)
    finally:
        await gantry.close()


async def test_asking_for_the_reserved_namespace_by_name_returns_nothing() -> None:
    gantry = await _gantry_with_servers(3)
    try:
        result = await gantry.retrieve(
            _query("restart the service", limit=5, namespaces=sorted(RESERVED_NAMESPACES))
        )
        assert result.tools == []
    finally:
        await gantry.close()


# --------------------------------------------------------------------------- #
# A long description is not an intent match
# --------------------------------------------------------------------------- #


def _intent_score(tool: ToolDefinition) -> float:
    from agent_gantry.core.router import INTENT_TAG_PATTERNS, TaskIntent, intent_match_score

    return intent_match_score(tool, INTENT_TAG_PATTERNS[TaskIntent.DATA_QUERY])


def test_a_keyword_in_the_name_or_tags_is_a_full_match() -> None:
    assert _intent_score(_tool("search_logs", "Look through application output.")) == 1.0
    assert _intent_score(_tool("log_viewer", "Look through application output.", tags=["search"])) == 1.0
    assert _intent_score(_tool("log_viewer", "Look through application output.")) == 0.0


def test_a_keyword_only_in_the_description_counts_for_less_the_longer_it_is() -> None:
    short = _tool("log_viewer", "Lets you read application output.")
    medium = _tool("log_viewer", "Lets you read application output. " + "Then it rests. " * 20)
    long = _tool("log_viewer", "Lets you read application output. " + "Then it rests. " * 120)

    assert _intent_score(short) == 0.5
    assert 0.0 < _intent_score(long) < _intent_score(medium) < _intent_score(short)
    assert _intent_score(long) < 0.06  # a passing mention in 1800 characters


async def test_a_verbose_tool_no_longer_outranks_the_concise_right_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``read`` appears once, in passing, in 1250 characters about something else.

    Under the old rule that passing mention was worth a flat 0.15 of the final
    score: a quarter of a cosine point, which is more than separates the two
    tools here. The second half of the test puts the old rule back to show that
    this ranking is what the change is responsible for.
    """
    from agent_gantry.core import router

    right = _tool("tail_logs", "Stream the newest log lines from a service as they arrive.")
    verbose = _tool(
        "archive_logs",
        "Move the oldest log lines from a service to cold storage. " * 12
        + "Nobody can read an archive without a restore. "
        + "Archived log lines are billed as they age. " * 12,
    )

    async def top(query: str) -> str:
        gantry = AgentGantry(embedder=SimpleEmbedder())
        try:
            await gantry.add_tool(right)
            await gantry.add_tool(verbose)
            return (await gantry.retrieve(_query(query, limit=2))).tools[0].tool.name
        finally:
            await gantry.close()

    question = "read the newest log lines as they arrive"  # ``read``: a data-query intent
    assert await top(question) == "tail_logs"

    def anywhere(tool: ToolDefinition, pattern: Any) -> float:
        text = f"{tool.name.lower()} {tool.description.lower()} {' '.join(tool.tags).lower()}"
        return 1.0 if pattern.search(text) else 0.0

    monkeypatch.setattr(router, "intent_match_score", anywhere)
    assert await top(question) == "archive_logs"
