"""
Tests for critical bug fixes identified in the code review.

Covers:
- Phase 1: MMR with diversity_factor > 0 (math import fix + numpy migration)
- Phase 2: Rate limiter wiring, namespace resolution, sliding window fix
- Phase 3: DRY adapters, lifecycle management, AsyncNoopContext utility
- Phase 4: Expanded domain validation
"""

from __future__ import annotations

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.adapters.tool_spec.providers import (
    GroqAdapter,
    MistralAdapter,
    OpenAIAdapter,
)
from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore
from agent_gantry.core.executor import ExecutionEngine
from agent_gantry.core.rate_limiter import RateLimiter, RateLimitExceeded
from agent_gantry.core.registry import ToolRegistry
from agent_gantry.core.security import PermissionDeniedError, SecurityPolicy
from agent_gantry.core.sync_manager import SyncManager
from agent_gantry.schema.config import RateLimitConfig
from agent_gantry.schema.execution import ToolCall
from agent_gantry.schema.query import ConversationContext, ToolQuery
from agent_gantry.schema.tool import ToolDefinition
from agent_gantry.utils.async_utils import AsyncNoopContext

# ---------------------------------------------------------------------------
# Phase 1: MMR with diversity_factor > 0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mmr_does_not_crash_with_diversity() -> None:
    """Phase 1.1+1.3: MMR should not crash when diversity_factor > 0."""
    gantry = AgentGantry()

    @gantry.register(tags=["math"])
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @gantry.register(tags=["math"])
    def subtract(a: int, b: int) -> int:
        """Subtract two numbers."""
        return a - b

    @gantry.register(tags=["text"])
    def uppercase(text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()

    await gantry.sync()

    # This should NOT crash (previously raised NameError: math not defined)
    context = ConversationContext(query="do some math")
    # Use score_threshold=0.0 because SimpleEmbedder (hash-based) produces
    # poor similarity scores. The point of this test is to verify MMR doesn't
    # crash with diversity_factor > 0, not to test semantic accuracy.
    query = ToolQuery(context=context, limit=2, diversity_factor=0.3, score_threshold=0.0)
    result = await gantry.retrieve(query)

    assert len(result.tools) > 0


# ---------------------------------------------------------------------------
# Phase 2.1: Rate limiter wired into ExecutionEngine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limiter_blocks_execution() -> None:
    """Phase 2.1: ExecutionEngine should enforce rate limits."""
    registry = ToolRegistry()

    tool = ToolDefinition(
        name="test_tool",
        description="A test tool",
        parameters_schema={"type": "object", "properties": {}},
    )
    registry.register_tool(tool)
    registry.register_handler("default.test_tool", lambda: "ok")

    # Create a rate limiter that allows only 1 call per minute
    config = RateLimitConfig(
        enabled=True,
        max_calls_per_minute=1,
        max_calls_per_hour=100,
        strategy="sliding_window",
    )
    limiter = RateLimiter(config)

    engine = ExecutionEngine(
        registry=registry,
        rate_limiter=limiter,
    )

    # First call should succeed
    result1 = await engine.execute(ToolCall(tool_name="test_tool", arguments={}))
    assert result1.status.value == "success"

    # Second call should be rate-limited
    result2 = await engine.execute(ToolCall(tool_name="test_tool", arguments={}))
    assert result2.status.value == "permission_denied"
    assert result2.error_type == "RateLimitExceeded"


# ---------------------------------------------------------------------------
# Phase 2.2: Namespace-aware tool resolution
# ---------------------------------------------------------------------------


def test_get_tool_by_name_across_namespaces() -> None:
    """Phase 2.2: Registry should find tools across namespaces."""
    registry = ToolRegistry()

    tool = ToolDefinition(
        name="my_tool",
        namespace="custom_ns",
        description="A tool in a custom namespace",
        parameters_schema={"type": "object", "properties": {}},
    )
    registry.register_tool(tool)

    # Should NOT find via get_tool with wrong namespace
    assert registry.get_tool("my_tool", "default") is None

    # Should find via get_tool_by_name (searches all namespaces)
    found = registry.get_tool_by_name("my_tool")
    assert found is not None
    assert found.name == "my_tool"
    assert found.namespace == "custom_ns"


def test_get_tool_by_name_prefers_default_namespace() -> None:
    """Phase 2.2: get_tool_by_name should prefer default namespace."""
    registry = ToolRegistry()

    tool_default = ToolDefinition(
        name="shared_tool",
        namespace="default",
        description="Default namespace version",
        parameters_schema={"type": "object", "properties": {}},
    )
    tool_custom = ToolDefinition(
        name="shared_tool",
        namespace="custom",
        description="Custom namespace version",
        parameters_schema={"type": "object", "properties": {}},
    )
    registry.register_tool(tool_default)
    registry.register_tool(tool_custom)

    found = registry.get_tool_by_name("shared_tool")
    assert found is not None
    assert found.namespace == "default"


# ---------------------------------------------------------------------------
# Phase 2.3: Sliding window bug fix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sliding_window_hour_limit_uses_full_history() -> None:
    """Phase 2.3: Hour limit should count all calls, not just recent-minute ones."""
    config = RateLimitConfig(
        enabled=True,
        max_calls_per_minute=100,  # High minute limit
        max_calls_per_hour=3,  # Low hour limit
        strategy="sliding_window",
    )
    limiter = RateLimiter(config)

    # Simulate 3 calls (just under the hour limit)
    for _ in range(3):
        await limiter.acquire("test_tool")

    # 4th call should be blocked by hour limit
    with pytest.raises(RateLimitExceeded, match="calls per hour"):
        await limiter.acquire("test_tool")


# ---------------------------------------------------------------------------
# Phase 3.2: DRY tool spec adapters
# ---------------------------------------------------------------------------


def test_mistral_adapter_inherits_from_openai() -> None:
    """Phase 3.2: MistralAdapter should inherit from OpenAIAdapter."""
    assert issubclass(MistralAdapter, OpenAIAdapter)

    adapter = MistralAdapter()
    assert adapter.dialect_name == "mistral"

    # Verify it produces OpenAI-compatible schema
    tool = ToolDefinition(
        name="test_tool",
        description="A test tool for validation purposes",
        parameters_schema={"type": "object", "properties": {}},
    )
    schema = adapter.to_provider_schema(tool)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "test_tool"


def test_groq_adapter_inherits_from_openai() -> None:
    """Phase 3.2: GroqAdapter should inherit from OpenAIAdapter."""
    assert issubclass(GroqAdapter, OpenAIAdapter)

    adapter = GroqAdapter()
    assert adapter.dialect_name == "groq"


# ---------------------------------------------------------------------------
# Phase 3.3: Resource lifecycle management
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_context_manager() -> None:
    """Phase 3.3: AgentGantry should support async context manager."""
    async with AgentGantry() as gantry:
        @gantry.register(tags=["test"])
        def hello() -> str:
            """Say hello."""
            return "hello"

        assert gantry.tool_count == 1

    # After exiting, gantry should be cleaned up
    assert gantry._initialized is False


@pytest.mark.asyncio
async def test_close_is_idempotent() -> None:
    """Phase 3.3: close() should be safe to call multiple times."""
    gantry = AgentGantry()
    await gantry.close()
    await gantry.close()  # Should not raise


# ---------------------------------------------------------------------------
# Phase 3.5: AsyncNoopContext utility
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_noop_context() -> None:
    """Phase 3.5: AsyncNoopContext should work as an async context manager."""
    async with AsyncNoopContext() as ctx:
        assert ctx is not None
    # Should complete without any error


# ---------------------------------------------------------------------------
# Phase 4.3: Expanded domain validation
# ---------------------------------------------------------------------------


def test_security_policy_blocks_ftp_urls() -> None:
    """Phase 4.3: SecurityPolicy should detect FTP URLs."""
    policy = SecurityPolicy(allowed_domains=["safe.example.com"])

    with pytest.raises(PermissionDeniedError, match="not in allowed_domains"):
        policy.check_permission(
            "fetch_data",
            {"url": "ftp://evil.example.com/file.txt"},
        )


def test_security_policy_blocks_protocol_relative_urls() -> None:
    """Phase 4.3: SecurityPolicy should detect protocol-relative URLs."""
    policy = SecurityPolicy(allowed_domains=["safe.example.com"])

    with pytest.raises(PermissionDeniedError, match="not in allowed_domains"):
        policy.check_permission(
            "fetch_data",
            {"url": "//evil.example.com/path"},
        )


def test_security_policy_allows_valid_ftp_domain() -> None:
    """Phase 4.3: SecurityPolicy should allow FTP URLs to allowed domains."""
    policy = SecurityPolicy(allowed_domains=["files.example.com"])

    # Should NOT raise
    policy.check_permission(
        "fetch_data",
        {"url": "ftp://files.example.com/data.csv"},
    )


# ---------------------------------------------------------------------------
# SyncManager unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_manager_get_embedder_id() -> None:
    """SyncManager should generate consistent embedder IDs."""
    embedder = SimpleEmbedder()
    store = InMemoryVectorStore()
    registry = ToolRegistry()

    manager = SyncManager(
        vector_store=store,
        embedder=embedder,
        registry=registry,
    )

    eid = manager.get_embedder_id()
    assert "SimpleEmbedder" in eid
    # Same call should return the same ID
    assert manager.get_embedder_id() == eid
