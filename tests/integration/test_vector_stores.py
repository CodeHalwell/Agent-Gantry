"""
Integration tests for vector store backends.

Requires running instances of vector stores. Run with:
    pytest -m integration tests/integration/test_vector_stores.py

For local testing, use docker-compose:
    docker compose -f docker-compose.test.yml up -d
"""

from __future__ import annotations

import pytest

from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.schema.tool import ToolDefinition


def _make_tools(count: int = 5) -> list[ToolDefinition]:
    """Create sample tool definitions for testing."""
    return [
        ToolDefinition(
            name=f"tool_{i}",
            description=f"Test tool number {i} that performs operation {i}",
            parameters_schema={
                "type": "object",
                "properties": {"input": {"type": "string"}},
            },
            tags=[f"tag_{i}", "test"],
        )
        for i in range(count)
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_basic_flow(qdrant_url: str) -> None:
    """Test basic add/search/delete flow with Qdrant."""
    pytest.importorskip("qdrant_client")

    from agent_gantry.adapters.vector_stores.remote import QdrantVectorStore

    store = QdrantVectorStore(
        url=qdrant_url,
        collection_name="test_integration",
        dimension=128,
    )
    embedder = SimpleEmbedder()

    await store.initialize()

    tools = _make_tools(3)
    texts = [t.to_searchable_text() for t in tools]
    embeddings = await embedder.embed_batch(texts)

    count = await store.add_tools(tools, embeddings, upsert=True)
    assert count == 3

    query_embedding = await embedder.embed_text("test tool 1")
    results = await store.search(query_vector=query_embedding, limit=2)
    assert len(results) > 0

    assert await store.health_check()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pgvector_basic_flow(pgvector_url: str) -> None:
    """Test basic add/search/delete flow with PGVector."""
    pytest.importorskip("asyncpg")

    from agent_gantry.adapters.vector_stores.remote import PGVectorStore

    store = PGVectorStore(
        url=pgvector_url,
        table_name="test_integration",
        dimension=128,
    )
    embedder = SimpleEmbedder()

    await store.initialize()

    tools = _make_tools(3)
    texts = [t.to_searchable_text() for t in tools]
    embeddings = await embedder.embed_batch(texts)

    count = await store.add_tools(tools, embeddings, upsert=True)
    assert count == 3

    query_embedding = await embedder.embed_text("test tool 1")
    results = await store.search(query_vector=query_embedding, limit=2)
    assert len(results) > 0

    assert await store.health_check()
