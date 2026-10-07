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


# --------------------------------------------------------------------------
# sync(): updates are not lost, first use syncs once, the embedder is recorded
# --------------------------------------------------------------------------


class _CountingEmbedder(SimpleEmbedder):
    """A hash embedder that remembers every batch it was asked to embed."""

    def __init__(self, dimension: int = 64) -> None:
        super().__init__(dimension=dimension)
        self.batches: list[list[str]] = []

    async def embed_batch(self, texts: Any) -> Any:
        import asyncio

        self.batches.append(list(texts))
        # A real embedder awaits the network. Without a yield here the
        # coroutines never interleave and a missing lock goes unnoticed.
        await asyncio.sleep(0)
        return await super().embed_batch(texts)

    @property
    def embedded(self) -> list[str]:
        return [text for batch in self.batches for text in batch]


class _OtherEmbedder(_CountingEmbedder):
    """Same vectors, different identity: what a switch of model looks like to a store."""


async def test_redefining_a_tool_after_a_sync_reaches_the_store() -> None:
    store = InMemoryVectorStore()
    gantry = AgentGantry(vector_store=store, embedder=SimpleEmbedder(dimension=64))
    try:
        await gantry.add_tool(_tool("weather", "Old description about cats."))
        await gantry.sync()
        await gantry.add_tool(_tool("weather", "New description about dogs."))
        await gantry.sync()

        stored = await store.get_by_name("weather", "default")
        assert stored is not None and stored.description == "New description about dogs."
        assert [t.description for t in gantry.list_tools_sync()] == ["New description about dogs."]
        assert gantry._registry.get_tool("weather").description == "New description about dogs."
    finally:
        await gantry.close()


async def test_concurrent_first_use_embeds_the_registry_once() -> None:
    import asyncio

    embedder = _CountingEmbedder()
    gantry = AgentGantry(vector_store=InMemoryVectorStore(), embedder=embedder)
    try:

        @gantry.register()
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        await asyncio.gather(*(gantry.retrieve(_query("add two numbers")) for _ in range(4)))
        tool_texts = [t for t in embedder.embedded if "Add two numbers together" in t]
        assert len(tool_texts) == 1  # was one per waiting coroutine
    finally:
        await gantry.close()


async def test_importing_tools_from_modules_records_the_embedder() -> None:
    store = InMemoryVectorStore()
    first = await AgentGantry.from_modules(
        ["tests.test_modules.module_a"], vector_store=store, embedder=_CountingEmbedder()
    )
    assert await store.get_metadata("embedder_id") is not None

    # A restart with another model now notices, and re-embeds, instead of
    # searching the old model's vectors with the new model's queries.
    other = _OtherEmbedder()
    second = AgentGantry(vector_store=store, embedder=other)
    try:
        await second.collect_tools_from_modules(["tests.test_modules.module_a"], persist=False)
        assert await second.sync() == 2
        assert len(other.embedded) == 2
    finally:
        await second.close()
        await first.close()


@pytest.mark.parametrize("mcp_first", [False, True], ids=["tools-first", "mcp-first"])
async def test_an_embedder_change_re_embeds_tools_and_servers_whichever_syncs_first(
    mcp_first: bool,
) -> None:
    store = InMemoryVectorStore()

    def build(embedder: SimpleEmbedder) -> AgentGantry:
        gantry = AgentGantry(vector_store=store, embedder=embedder)

        @gantry.register()
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        gantry.register_mcp_server(
            name="filesystem", command=["echo"], description="Read and write local files"
        )
        return gantry

    first = build(_CountingEmbedder())
    await first.sync()
    await first.sync_mcp_servers()

    # Same store, new embedder: a process restarted with a different model.
    embedder = _OtherEmbedder()
    second = build(embedder)
    try:
        steps = [second.sync, second.sync_mcp_servers]
        for step in reversed(steps) if mcp_first else steps:
            await step()

        embedded = " | ".join(embedder.embedded)
        assert "Add two numbers together" in embedded  # the tool was re-embedded
        assert "filesystem" in embedded.lower()  # ...and so was the server (it was skipped)
    finally:
        await second.close()
        await first.close()


# --------------------------------------------------------------------------
# Public surface
# --------------------------------------------------------------------------


def test_every_name_in_all_resolves_without_the_mcp_extra() -> None:
    """``from agent_gantry import *`` reads each name in ``__all__``.

    ``reset_sse_shutdown_latch`` needs ``mcp`` and raises AttributeError
    without it (so ``hasattr`` works), which made the star-import itself fail on
    a base install. Run in a child interpreter that blocks ``mcp`` so the
    result does not depend on what the test environment happens to have.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        sys.modules["mcp"] = None  # makes ``import mcp`` raise ImportError
        namespace = {}
        exec("from agent_gantry import *", namespace)
        import agent_gantry
        missing = [n for n in agent_gantry.__all__ if n not in namespace]
        assert not missing, missing
        assert not hasattr(agent_gantry, "reset_sse_shutdown_latch")
        """
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------
# Execution: limits that can be configured, errors attributed correctly
# --------------------------------------------------------------------------


def _gantry(**execution: Any) -> AgentGantry:
    return AgentGantry(
        config=AgentGantryConfig(execution=ExecutionConfig(**execution)),
        vector_store=InMemoryVectorStore(),
        embedder=SimpleEmbedder(dimension=64),
    )


async def test_the_configured_default_timeout_applies_to_a_call_that_sets_none() -> None:
    import asyncio

    gantry = _gantry(default_timeout_ms=100, max_retries=0)
    try:

        @gantry.register()
        async def slow_report() -> str:
            """Take far too long to produce a report."""
            await asyncio.sleep(2)
            return "late"

        result = await gantry.execute(ToolCall(tool_name="slow_report", arguments={}))
        assert result.status == ExecutionStatus.TIMEOUT
        assert result.error == "Execution timed out"
        # The slow calls used to report no start time, hence 0 ms.
        assert result.started_at is not None
        assert result.latency_ms >= 90
    finally:
        await gantry.close()


async def test_retry_count_zero_means_no_retries_and_unset_takes_the_engine_default() -> None:
    gantry = _gantry(max_retries=1)
    attempts = 0
    try:

        @gantry.register()
        def send_email() -> str:
            """Send an email, failing after the message has gone out."""
            nonlocal attempts
            attempts += 1
            raise ValueError("smtp: connection reset")

        explicit = await gantry.execute(
            ToolCall(tool_name="send_email", arguments={}, retry_count=0)
        )
        assert (attempts, explicit.attempt_number) == (1, 1)  # was 4 attempts

        attempts = 0
        unset = await gantry.execute(ToolCall(tool_name="send_email", arguments={}))
        assert (attempts, unset.attempt_number) == (2, 2)  # the engine's max_retries=1
    finally:
        await gantry.close()


async def test_a_timeout_error_raised_by_the_handler_keeps_its_message() -> None:
    gantry = _gantry(max_retries=0)
    try:

        @gantry.register()
        def run_query() -> str:
            """Run a database query."""
            raise TimeoutError("postgres statement_timeout (2s)")

        result = await gantry.execute(ToolCall(tool_name="run_query", arguments={}))
        # Not Gantry's own deadline, so not TIMEOUT and not "Execution timed out".
        assert result.status == ExecutionStatus.FAILURE
        assert result.error == "postgres statement_timeout (2s)"
        assert result.error_type == "TimeoutError"
        assert result.started_at is not None
    finally:
        await gantry.close()


async def test_a_permission_denied_result_carries_its_start_time() -> None:
    from agent_gantry.core.security import PermissionDeniedError

    gantry = _gantry(max_retries=0)
    try:

        @gantry.register()
        def purge_cache_entries() -> str:
            """Purge every cached entry."""
            raise PermissionDeniedError("not today")

        result = await gantry.execute(ToolCall(tool_name="purge_cache_entries", arguments={}))
        assert result.status == ExecutionStatus.PERMISSION_DENIED
        assert result.started_at is not None
    finally:
        await gantry.close()


async def test_a_malformed_root_schema_is_a_validation_result_not_a_crash() -> None:
    gantry = _gantry(max_retries=0)
    try:
        odd = _tool("odd_schema", "A tool imported with a null properties map")
        odd.parameters_schema = {"type": "object", "properties": None, "required": None}
        gantry._registry.register_tool(odd, lambda **kwargs: "ok")

        result = await gantry._executor.execute(
            ToolCall(tool_name="odd_schema", arguments={"unexpected": 1})
        )
        assert result.status == ExecutionStatus.FAILURE  # was: TypeError out of execute()
        assert "unexpected" in (result.error or "")
    finally:
        await gantry.close()


async def test_fail_fast_stops_an_adaptive_batch_and_only_then() -> None:
    from agent_gantry.schema.execution import BatchToolCall

    gantry = _gantry(max_retries=0)
    ran: list[str] = []
    try:

        @gantry.register()
        def failing_step() -> str:
            """Always fail with an error."""
            ran.append("failing_step")
            raise RuntimeError("no")

        @gantry.register()
        def later_step() -> str:
            """Always succeed with a result."""
            ran.append("later_step")
            return "ok"

        calls = [
            ToolCall(tool_name="failing_step", arguments={}),
            ToolCall(tool_name="later_step", arguments={}),
            ToolCall(tool_name="later_step", arguments={}),
        ]
        stopped = await gantry.execute_batch(BatchToolCall(calls=calls, fail_fast=True))
        assert [r.tool_name for r in stopped.results] == ["failing_step"]
        assert ran == ["failing_step"]

        ran.clear()
        everything = await gantry.execute_batch(BatchToolCall(calls=calls))
        assert len(everything.results) == 3
        assert sorted(ran) == ["failing_step", "later_step", "later_step"]
    finally:
        await gantry.close()


async def test_one_calls_exception_does_not_discard_the_rest_of_a_parallel_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_gantry.schema.execution import BatchToolCall

    gantry = _gantry(max_retries=0)
    try:

        @gantry.register()
        def fine_step() -> str:
            """Always succeed with a result."""
            return "ok"

        engine = gantry._executor
        real_execute = engine.execute

        async def execute(call: ToolCall) -> Any:
            if call.tool_name == "defective_step":
                raise RuntimeError("a defect in one call")
            return await real_execute(call)

        monkeypatch.setattr(engine, "execute", execute)
        batch = BatchToolCall(
            calls=[
                ToolCall(tool_name="defective_step", arguments={}),
                ToolCall(tool_name="fine_step", arguments={}),
            ],
            execution_strategy="parallel",
        )
        out = await engine.execute_batch(batch)
        assert [r.status for r in out.results] == [ExecutionStatus.FAILURE, ExecutionStatus.SUCCESS]
        assert out.results[0].error == "a defect in one call"
        assert (out.successful_count, out.failed_count) == (1, 1)
    finally:
        await gantry.close()


def test_a_not_keyword_that_forbids_null_stops_it_validating() -> None:
    from agent_gantry.schema.base import null_validates_against

    assert not null_validates_against({"type": ["string", "null"], "not": {"type": "null"}})
    assert not null_validates_against({"not": {}})  # nothing satisfies "not anything"
    assert null_validates_against({"type": ["string", "null"], "not": {"type": "string"}})
    assert null_validates_against({"not": False})


async def test_a_call_waiting_for_approval_is_not_logged_as_an_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging
    from datetime import datetime, timezone

    from agent_gantry.observability.console import ConsoleTelemetryAdapter
    from agent_gantry.schema.execution import ToolResult

    adapter = ConsoleTelemetryAdapter(log_level=logging.INFO)
    caplog.set_level(logging.INFO, logger="agent_gantry")
    now = datetime.now(timezone.utc)

    def result(status: ExecutionStatus) -> ToolResult:
        return ToolResult(
            tool_name="refund_order",
            status=status,
            queued_at=now,
            completed_at=now,
            trace_id="t",
            span_id="s",
            error="Re-issue the call once a human approves it.",
        )

    call = ToolCall(tool_name="refund_order", arguments={})
    await adapter.record_execution(call, result(ExecutionStatus.PENDING_CONFIRMATION))
    assert caplog.records[-1].levelno == logging.INFO
    await adapter.record_execution(call, result(ExecutionStatus.FAILURE))
    assert caplog.records[-1].levelno == logging.ERROR


# --------------------------------------------------------------------------
# Provider layer
# --------------------------------------------------------------------------


def test_usage_from_a_dumped_anthropic_message_with_no_cache_activity() -> None:
    from anthropic.types import Message, TextBlock, Usage

    from agent_gantry.metrics.token_usage import ProviderUsage

    message = Message(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-test",
        content=[TextBlock(type="text", text="hi")],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    for dumped in (message.model_dump(), message.model_dump(mode="json")):
        # cache_creation_input_tokens / cache_read_input_tokens are None here, and
        # ``None`` used to be coerced with int() and raise.
        usage = ProviderUsage.from_response_usage(dumped)
        assert usage == ProviderUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def test_gemini_usage_counts_thinking_and_tool_use_tokens() -> None:
    from google.genai import types

    from agent_gantry.metrics.token_usage import ProviderUsage

    metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=100,
        candidates_token_count=30,
        thoughts_token_count=250,
        tool_use_prompt_token_count=40,
        total_token_count=420,
    )
    response = types.GenerateContentResponse(usage_metadata=metadata)
    for source in (response, {"usage_metadata": metadata.model_dump()}):
        usage = ProviderUsage.from_response_usage(source)
        assert usage is not None
        # google-genai defines the total as the sum of all four, so the two
        # figures must add up to it; the thinking tokens are billed output.
        assert (usage.prompt_tokens, usage.completion_tokens) == (140, 280)
        assert usage.prompt_tokens + usage.completion_tokens == usage.total_tokens == 420


def test_openai_and_anthropic_usage_is_unchanged() -> None:
    from agent_gantry.metrics.token_usage import ProviderUsage

    assert ProviderUsage.from_usage(
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    ) == ProviderUsage(10, 5, 15)
    cached = ProviderUsage.from_usage(
        {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 7}
    )
    assert (cached.prompt_tokens, cached.cached_prompt_tokens) == (17, 7)


def test_a_dataclass_result_holding_an_enum_or_tuple_keyed_dict_formats_for_every_dialect() -> None:
    import dataclasses
    import enum
    import json

    from agent_gantry.adapters.tool_spec.registry import DialectRegistry

    class Color(enum.Enum):
        RED = "red"

    @dataclasses.dataclass
    class Report:
        counts: dict[Any, int]

    results = [
        Report(counts={Color.RED: 1}),
        Report(counts={(1, 2): 3}),
        [Report(counts={Color.RED: 1})],
        {"report": Report(counts={Color.RED: 1})},
    ]
    registry = DialectRegistry.default()
    for dialect in registry.list_dialects():
        adapter = registry.get(dialect)
        for result in results:
            # Used to raise TypeError ("keys must be str, int, float, bool or None").
            formatted = adapter.format_tool_result("report", result, "call_1")
            text = json.dumps(formatted)
            assert "Color.RED" in text or "(1, 2)" in text


def test_a_call_with_no_arguments_is_not_logged_as_malformed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from agent_gantry.adapters.tool_spec.registry import get_adapter

    caplog.set_level(logging.WARNING)
    payload = get_adapter("gemini").from_provider_payload(
        {"name": "no_arguments", "args": None, "id": "call_1"}
    )
    assert payload.arguments == {}
    assert not [r for r in caplog.records if "decoded to" in r.getMessage()]


def test_the_anthropic_clients_read_a_block_shaped_latest_user_turn() -> None:
    from agent_gantry.integrations.anthropic_features import AnthropicClient

    messages = [
        {"role": "user", "content": "an older question about the weather"},
        {"role": "assistant", "content": "It is sunny."},
        {"role": "user", "content": [{"type": "text", "text": "now what about stocks?"}]},
    ]
    assert AnthropicClient._last_user_query(messages) == "now what about stocks?"  # was the older one
    assert AnthropicClient._last_user_query(messages[-1:]) == "now what about stocks?"  # was None
    only_results = [
        {"role": "user", "content": "check stocks"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "42"}]},
    ]
    assert AnthropicClient._last_user_query(only_results) == "check stocks"


def test_an_over_long_tool_name_is_flagged_once_when_a_provider_schema_is_built(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    caplog.set_level(logging.WARNING)
    long_name = "fetch_" + "x" * 70  # valid for ToolDefinition (<=128), too long for providers
    tool = _tool(long_name, "Fetch something with a very long name")
    short = _tool("fetch_short", "Fetch something with a short name")

    tool.to_dialect("openai")
    tool.to_dialect("gemini")  # once per name, not once per call
    short.to_dialect("openai")
    flagged = [r for r in caplog.records if "accept at most 64" in r.getMessage()]
    assert len(flagged) == 1 and long_name in flagged[0].getMessage()


# --------------------------------------------------------------------------
# Retrieval path
# --------------------------------------------------------------------------


async def test_a_diversity_factor_still_applies_when_a_reranker_is_configured() -> None:
    """Rerankers keep ``top_k``; MMR must be handed more than ``limit`` to choose from."""
    embedder = SimpleEmbedder(dimension=256)
    store = InMemoryVectorStore()
    tools = [
        _tool("send_email", "Send an email message to a person"),
        _tool("send_email_copy", "Send an email message copy to a person"),
        _tool("convert_units", "Convert a temperature between units"),
    ]
    await store.initialize()
    await store.add_tools(tools, await embedder.embed_batch([t.to_searchable_text() for t in tools]))

    class OrderedReranker:
        """Ranks the near-duplicates above the different tool, keeping ``top_k``."""

        order = ["send_email", "send_email_copy", "convert_units"]

        async def rerank(self, query: str, tools: Any, top_k: int) -> Any:
            return sorted(tools, key=lambda pair: self.order.index(pair[0].name))[:top_k]

    router = SemanticRouter(vector_store=store, embedder=embedder, reranker=OrderedReranker())
    query = _query(
        "send an email message", limit=2, diversity_factor=0.9, enable_reranking=True
    )
    routed = await router.route(query)
    names = [tool.name for tool, _ in routed.tools]
    assert names == ["send_email", "convert_units"]  # was the two near-duplicates

    # Without diversity the reranker's order stands.
    plain = await router.route(_query("send an email message", limit=2, enable_reranking=True))
    assert [t.name for t, _ in plain.tools] == ["send_email", "send_email_copy"]


async def test_lancedb_and_memory_stores_agree_on_a_zero_threshold(tmp_path: Any) -> None:
    pytest.importorskip("lancedb")
    from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

    tools = [_tool("aligned", "Points the same way"), _tool("orthogonal", "Points sideways"),
             _tool("opposed", "Points the other way")]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]
    lance = LanceDBVectorStore(db_path=str(tmp_path), dimension=4)
    memory = InMemoryVectorStore()
    found = {}
    for label, store in (("lancedb", lance), ("memory", memory)):
        await store.initialize()
        await store.add_tools(tools, vectors)
        hits = await store.search([1.0, 0.0, 0.0, 0.0], limit=10, score_threshold=0.0)
        found[label] = sorted(tool.name for tool, _score in hits)
    # An opposed vector has cosine -1; LanceDB clamped it to 0 and kept it.
    assert found["memory"] == ["aligned", "orthogonal"]
    assert found["lancedb"] == found["memory"]


async def test_lancedb_upsert_with_a_repeated_id_keeps_one_row(tmp_path: Any) -> None:
    pytest.importorskip("lancedb")
    from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

    store = LanceDBVectorStore(db_path=str(tmp_path), dimension=4)
    await store.initialize()
    first, second = _tool("report", "First wording of the report tool"), _tool(
        "report", "Second wording of the report tool"
    )
    other = _tool("other", "Some other tool entirely here")
    written = await store.add_tools(
        [first, other, second], [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [1.0, 0, 0, 0]], upsert=True
    )
    assert written == 2 and await store.count() == 2  # was 3 rows, 'report' twice
    hits = await store.search([1.0, 0.0, 0.0, 0.0], limit=10, score_threshold=0.0)
    assert sorted(t.name for t, _ in hits) == ["other", "report"]
    kept = await store.get_by_name("report")
    assert kept is not None and kept.description == second.description  # the last one wins


class _FailingChroma:
    """Stands in for a Chroma collection whose server is down."""

    def get(self, **kwargs: Any) -> Any:
        raise ConnectionError("chroma is down")

    def count(self) -> int:
        raise ConnectionError("chroma is down")

    def delete(self, **kwargs: Any) -> Any:
        raise ConnectionError("chroma is down")


async def test_chroma_reports_an_outage_instead_of_an_empty_store() -> None:
    from agent_gantry.adapters.vector_stores.remote import ChromaVectorStore

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store._initialized = True  # type: ignore[attr-defined]
    store._collection = _FailingChroma()  # type: ignore[attr-defined]
    for call in (
        store.get_by_name("x"),
        store.delete("x"),
        store.list_all(),
        store.count(),
    ):
        with pytest.raises(ConnectionError, match="chroma is down"):
            await call  # each used to return None / False / [] / 0


async def test_remote_stores_refuse_a_short_embedding_batch() -> None:
    from agent_gantry.adapters.vector_stores.remote import (
        ChromaVectorStore,
        PGVectorStore,
        QdrantVectorStore,
    )

    tools = [_tool("one", "The first tool in the batch"), _tool("two", "The second tool in it")]
    for cls in (QdrantVectorStore, ChromaVectorStore, PGVectorStore):
        store = cls.__new__(cls)  # a mismatch is refused before any connection is made
        with pytest.raises(ValueError, match="length mismatch"):
            await store.add_tools(tools, [[0.0, 1.0]])


async def test_closing_a_gantry_closes_an_openai_embedders_http_client() -> None:
    pytest.importorskip("openai")
    from agent_gantry.adapters.embedders.openai import OpenAIEmbedder
    from agent_gantry.schema.config import EmbedderConfig

    embedder = OpenAIEmbedder(EmbedderConfig(type="openai", api_key="sk-test-not-a-real-key"))
    closed: list[bool] = []

    class FakeClient:
        async def close(self) -> None:
            closed.append(True)

    embedder._client = FakeClient()  # type: ignore[assignment]
    gantry = AgentGantry(vector_store=InMemoryVectorStore(), embedder=embedder)
    await gantry.close()
    assert closed == [True]  # the embedder had no close, so the pool leaked


async def test_lancedb_scores_are_true_cosines_whatever_the_vector_length(tmp_path: Any) -> None:
    pytest.importorskip("lancedb")
    from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

    store = LanceDBVectorStore(db_path=str(tmp_path), dimension=4)
    await store.initialize()
    tools = [_tool("long_same", "Same direction, longer"), _tool("diagonal", "Halfway round")]
    await store.add_tools(tools, [[3.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    hits = dict(
        (tool.name, score)
        for tool, score in await store.search([2.0, 0.0, 0.0, 0.0], limit=5, score_threshold=0.0)
    )
    # 1 - d/2 on squared-L2 gave -0.5 and -1.25 here (clamped to 0 before).
    assert hits["long_same"] == pytest.approx(1.0)
    assert hits["diagonal"] == pytest.approx(2**-0.5, abs=1e-6)


# --------------------------------------------------------------------------
# Framework integrations
# --------------------------------------------------------------------------


async def test_the_agent_framework_provider_rejects_out_of_range_bounds_at_construction() -> None:
    """The six other live providers raise QueryBoundsError; this one logged and ran tool-less.

    Checked before the ``agent-framework`` import, so it holds whether or not the
    package is installed.
    """
    from agent_gantry.integrations.agent_framework_provider import GantryContextProvider
    from agent_gantry.integrations.frameworks.base import QueryBoundsError

    gantry = AgentGantry(vector_store=InMemoryVectorStore(), embedder=SimpleEmbedder(dimension=64))
    try:
        with pytest.raises(QueryBoundsError, match="limit"):
            GantryContextProvider(gantry, top_k=60)
        with pytest.raises(QueryBoundsError, match="score_threshold"):
            GantryContextProvider(gantry, score_threshold=2.0)
        # A relative threshold is a string the bridge parses, not a ToolQuery bound,
        # so it must get past the check (and only then meets the missing package).
        try:
            GantryContextProvider(gantry, score_threshold="relative:0.8")
        except ImportError:
            pass
    finally:
        await gantry.close()


async def test_the_agent_framework_bridge_reports_a_bad_limit_as_a_bounds_error() -> None:
    from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge
    from agent_gantry.integrations.frameworks.base import QueryBoundsError

    gantry = AgentGantry(vector_store=InMemoryVectorStore(), embedder=SimpleEmbedder(dimension=64))
    try:
        with pytest.raises(QueryBoundsError):
            await GantryToolBridge(gantry).get_tools("add two numbers", limit=60)  # was ValidationError
    finally:
        await gantry.close()


def _renamed_property_tool() -> ToolDefinition:
    """A tool whose only property cannot be a Python parameter name."""
    tool = _tool("lookup_user", "Look a user up by their identifier")
    tool.parameters_schema = {
        "type": "object",
        "properties": {"user-id": {"type": "string", "description": "Identifier"}},
        "required": ["user-id"],
    }
    return tool


async def _gantry_with_renamed_property() -> tuple[AgentGantry, Any]:
    from agent_gantry.integrations.frameworks.base import spec_from_tool

    gantry = AgentGantry(vector_store=InMemoryVectorStore(), embedder=SimpleEmbedder(dimension=64))
    tool = _renamed_property_tool()

    def lookup_user(**arguments: Any) -> dict[str, Any]:
        return {"called_with": sorted(arguments)}

    await gantry.add_tool(tool, lookup_user)
    return gantry, spec_from_tool(gantry, tool)


async def test_agno_dispatches_a_renamed_parameter_and_advertises_the_same_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import sys
    import types

    class Function:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    function_module = types.ModuleType("agno.tools.function")
    function_module.Function = Function  # type: ignore[attr-defined]
    for name, module in (
        ("agno", types.ModuleType("agno")),
        ("agno.tools", types.ModuleType("agno.tools")),
        ("agno.tools.function", function_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    from agent_gantry.integrations.frameworks.agno import AgnoAdapter

    gantry, spec = await _gantry_with_renamed_property()
    try:
        function = AgnoAdapter.convert(spec)
        # Schema and signature name the parameter the same way...
        assert list(function.parameters["properties"]) == ["user_id"]
        # ...and a call made with that name reaches the tool under the schema's.
        result = await asyncio.to_thread(function.entrypoint, user_id="u-1")
        assert result == {"called_with": ["user-id"]}  # was: "Missing required parameter"
    finally:
        await gantry.close()


async def test_llamaindex_dispatches_a_renamed_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    class FunctionTool:
        @classmethod
        def from_defaults(cls, **kwargs: Any) -> Any:
            tool = cls()
            tool.__dict__.update(kwargs)
            return tool

    tools_module = types.ModuleType("llama_index.core.tools")
    tools_module.FunctionTool = FunctionTool  # type: ignore[attr-defined]
    core = types.ModuleType("llama_index.core")
    pkg = types.ModuleType("llama_index")
    for name, module in (
        ("llama_index", pkg),
        ("llama_index.core", core),
        ("llama_index.core.tools", tools_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    from agent_gantry.integrations.frameworks.llamaindex import LlamaIndexAdapter

    gantry, spec = await _gantry_with_renamed_property()
    try:
        tool = LlamaIndexAdapter.convert(spec)
        # No fn_schema (the property is not a valid field name), so LlamaIndex
        # advertises the signature, which carries the renamed parameter.
        assert "user_id" in tool.async_fn.__signature__.parameters
        assert await tool.async_fn(user_id="u-1") == {"called_with": ["user-id"]}
    finally:
        await gantry.close()


async def test_the_agent_framework_wrapper_serialises_results_json_rejects() -> None:
    import datetime
    import json

    from agent_gantry.integrations.agent_framework_bridge import _build_callable_for_tool

    gantry = AgentGantry(vector_store=InMemoryVectorStore(), embedder=SimpleEmbedder(dimension=64))
    try:

        @gantry.register()
        def when_is_it(city: str) -> dict[Any, Any]:
            """Say when something happens in a city."""
            return {"at": datetime.datetime(2026, 1, 2, 3, 4, tzinfo=datetime.timezone.utc),
                    (1, 2): "a tuple key"}

        tool = gantry._registry.get_tool("when_is_it")
        wrapper = _build_callable_for_tool(tool, gantry, as_function_tool=False)
        out = json.loads(await wrapper(city="Leeds"))  # was TypeError, outside the error guard
        assert out["at"].startswith("2026-01-02T03:04")
        assert out["(1, 2)"] == "a tuple key"

        # A failure inside the tool is still reported as an error object.
        @gantry.register()
        def always_fails(city: str) -> str:
            """Fail whenever it is asked."""
            raise RuntimeError("down")

        failing = _build_callable_for_tool(
            gantry._registry.get_tool("always_fails"), gantry, as_function_tool=False
        )
        assert "RuntimeError" in json.loads(await failing(city="Leeds"))["error"]
    finally:
        await gantry.close()


def test_the_aggregate_integrations_package_exports_every_adapter() -> None:
    import agent_gantry.integrations as integrations
    from agent_gantry.integrations import frameworks

    missing = [name for name in frameworks.__all__ if name.endswith("Adapter")
               and name not in integrations.__all__]
    assert not missing, missing  # StrandsAdapter and DSPyAdapter were absent


# --------------------------------------------------------------------------
# Configuration that must not silently do nothing
# --------------------------------------------------------------------------


def test_asking_for_a_sandbox_is_refused_because_none_exists() -> None:
    from pydantic import ValidationError

    ExecutionConfig()  # the defaults are fine
    ExecutionConfig(enable_sandbox=False, sandbox_type="none")
    for kwargs in ({"enable_sandbox": True}, {"sandbox_type": "docker"}, {"sandbox_type": "subprocess"}):
        with pytest.raises(ValidationError, match="not implemented"):
            ExecutionConfig(**kwargs)


def test_an_old_yaml_with_the_removed_inert_fields_still_loads(tmp_path: Any) -> None:
    path = tmp_path / "gantry.yaml"
    path.write_text(
        "routing:\n  enable_mmr: false\n  mmr_lambda: 0.5\n  enable_intent_classification: true\n"
        "sync_on_register: true\n"
    )
    config = AgentGantryConfig.from_yaml(str(path))
    assert not hasattr(config.routing, "enable_mmr") and not hasattr(config, "sync_on_register")


def test_the_config_class_is_importable_from_the_package_root() -> None:
    import agent_gantry

    assert agent_gantry.AgentGantryConfig is AgentGantryConfig
    assert "AgentGantryConfig" in agent_gantry.__all__


async def test_get_tool_health_reads_the_live_record() -> None:
    gantry = AgentGantry(
        config=AgentGantryConfig(execution=ExecutionConfig(circuit_breaker_threshold=2, max_retries=0)),
        vector_store=_CopyingStore(),
        embedder=SimpleEmbedder(dimension=64),
    )
    try:

        @gantry.register()
        def flaky_service() -> str:
            """Call a service that is down."""
            raise RuntimeError("down")

        assert gantry.get_tool_health("flaky_service") is not None
        assert gantry.get_tool_health("never_registered") is None
        for _ in range(2):
            await gantry.execute(ToolCall(tool_name="flaky_service", arguments={}))
        health = gantry.get_tool_health("flaky_service")
        assert health is not None and health.circuit_breaker_open and health.total_calls == 2
        # ...which the store's own copy (what get_tool returns) does not show.
        stored = await gantry.get_tool("flaky_service")
        assert stored is not None and not stored.health.circuit_breaker_open
    finally:
        await gantry.close()


async def test_lancedb_never_scores_a_vector_above_one(tmp_path: Any) -> None:
    """float32 rounding gave an identical vector a cosine distance of about -1e-7."""
    pytest.importorskip("lancedb")
    import numpy as np

    from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

    rng = np.random.default_rng(0)  # seeded: 95 of 400 such vectors rounded over 1.0
    vectors = [rng.random(8).astype(np.float32).tolist() for _ in range(120)]
    store = LanceDBVectorStore(db_path=str(tmp_path), dimension=8)
    await store.initialize()
    tools = [_tool(f"tool_{i}", f"Tool number {i} doing something") for i in range(len(vectors))]
    await store.add_tools(tools, vectors)
    for vector in vectors:
        hits = await store.search(vector, limit=1, score_threshold=0.0)
        assert hits and hits[0][1] <= 1.0
