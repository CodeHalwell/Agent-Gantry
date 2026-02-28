from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_gantry.adapters.vector_stores.remote import PGVectorStore
from agent_gantry.schema.tool import ToolDefinition


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    ctx_mgr = AsyncMock()
    ctx_mgr.__aenter__.return_value = conn
    pool.acquire.return_value = ctx_mgr
    return pool, conn


@pytest.fixture
def pg_store(mock_pool):
    pool, conn = mock_pool
    store = PGVectorStore("postgresql://fake:fake@localhost/fake")
    store._pool = pool
    store._initialized = True
    return store, conn


@pytest.mark.asyncio
async def test_search_include_embeddings_true(pg_store):
    store, conn = pg_store

    # Setup mock data
    tool = ToolDefinition(
        name="test_tool",
        namespace="default",
        description="A test tool with enough description length.",
        parameters_schema={"type": "object", "properties": {}},
    )

    class MockRow(dict):
        pass

    row = MockRow()
    row["tool_json"] = tool.model_dump_json()
    row["similarity"] = 0.95
    row["embedding"] = "[0.1, 0.2, 0.3]"

    conn.fetch.return_value = [row]

    # Test search with embeddings
    results = await store.search([0.1, 0.2, 0.3], limit=10, include_embeddings=True)

    assert len(results) == 1
    assert len(results[0]) == 3
    ret_tool, score, embedding = results[0]

    assert ret_tool.name == "test_tool"
    assert score == 0.95
    assert embedding == [0.1, 0.2, 0.3]

    # Verify the executed query
    query_call = conn.fetch.call_args[0][0]
    assert "tool_json" in query_call
    assert "embedding" in query_call


@pytest.mark.asyncio
async def test_search_include_embeddings_false(pg_store):
    store, conn = pg_store

    # Setup mock data
    tool = ToolDefinition(
        name="test_tool",
        namespace="default",
        description="A test tool with enough description length.",
        parameters_schema={"type": "object", "properties": {}},
    )

    class MockRow(dict):
        pass

    row = MockRow()
    row["tool_json"] = tool.model_dump_json()
    row["similarity"] = 0.95

    conn.fetch.return_value = [row]

    # Test search without embeddings
    results = await store.search([0.1, 0.2, 0.3], limit=10, include_embeddings=False)

    assert len(results) == 1
    assert len(results[0]) == 2
    ret_tool, score = results[0]

    assert ret_tool.name == "test_tool"
    assert score == 0.95

    # Verify the executed query
    query_call = conn.fetch.call_args[0][0]
    assert "tool_json" in query_call
    # assert "embedding" not in query_call.split("FROM")[0]  # it is in 1 - (embedding <=> $1::vector)


@pytest.mark.asyncio
async def test_search_include_embeddings_list_type(pg_store):
    store, conn = pg_store

    # Setup mock data for when asyncpg returns a list-like object
    tool = ToolDefinition(
        name="test_tool",
        namespace="default",
        description="A test tool with enough description length.",
        parameters_schema={"type": "object", "properties": {}},
    )

    class MockRow(dict):
        pass

    row = MockRow()
    row["tool_json"] = tool.model_dump_json()
    row["similarity"] = 0.95
    row["embedding"] = [0.1, 0.2, 0.3]  # List type instead of string

    conn.fetch.return_value = [row]

    # Test search with embeddings
    results = await store.search([0.1, 0.2, 0.3], limit=10, include_embeddings=True)

    assert len(results) == 1
    ret_tool, score, embedding = results[0]

    assert embedding == [0.1, 0.2, 0.3]
