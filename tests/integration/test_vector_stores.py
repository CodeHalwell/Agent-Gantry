"""
Integration tests for vector store backends.

Requires running instances of vector stores. Run with:
    pytest -m integration tests/integration/test_vector_stores.py

For local testing, use docker-compose:
    docker compose -f tests/docker-compose.test.yml up -d
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


def test_pgvector_meta_table_name_bounded() -> None:
    """The derived metadata table name never truncates onto the tools table.

    PostgreSQL silently truncates identifiers to 63 bytes, so for a
    63-character table_name a naive "<table>__meta" would collapse back to
    the tools table's own identifier. No database needed — this is pure
    construction logic.
    """
    pytest.importorskip("asyncpg")

    from agent_gantry.adapters.vector_stores.remote import PGVectorStore

    short = PGVectorStore(url="postgresql://unused", table_name="tools", dimension=8)
    assert short._meta_table_name == "tools__meta"

    long_name = "t" * 63
    long = PGVectorStore(url="postgresql://unused", table_name=long_name, dimension=8)
    assert len(long._meta_table_name) <= 63
    assert long._meta_table_name != long_name
    assert long._meta_table_name.endswith("__meta")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_fingerprint_keys_align_with_sync_manager(qdrant_url: str) -> None:
    """get_stored_fingerprints() must key by 'namespace.name' with
    compute_tool_fingerprint values — the contract SyncManager.detect_changes
    relies on. A mismatch silently degrades incremental sync to a full
    re-embed every time (the exact bug the in-memory store had)."""
    pytest.importorskip("qdrant_client")

    from agent_gantry.adapters.vector_stores.remote import QdrantVectorStore
    from agent_gantry.utils.fingerprint import compute_tool_fingerprint

    store = QdrantVectorStore(
        url=qdrant_url,
        collection_name="test_fingerprint_keys",
        dimension=128,
    )
    embedder = SimpleEmbedder()
    await store.initialize()

    tools = _make_tools(3)
    embeddings = await embedder.embed_batch([t.to_searchable_text() for t in tools])
    await store.add_tools(tools, embeddings, upsert=True)

    stored = await store.get_stored_fingerprints()
    expected = {f"{t.namespace}.{t.name}": compute_tool_fingerprint(t) for t in tools}
    assert stored == expected


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pgvector_fingerprint_keys_align_with_sync_manager(pgvector_url: str) -> None:
    """Same contract check as the Qdrant variant, for PGVector."""
    pytest.importorskip("asyncpg")

    from agent_gantry.adapters.vector_stores.remote import PGVectorStore
    from agent_gantry.utils.fingerprint import compute_tool_fingerprint

    store = PGVectorStore(
        url=pgvector_url,
        table_name="test_fingerprint_keys",
        dimension=128,
    )
    embedder = SimpleEmbedder()
    await store.initialize()

    tools = _make_tools(3)
    embeddings = await embedder.embed_batch([t.to_searchable_text() for t in tools])
    await store.add_tools(tools, embeddings, upsert=True)

    stored = await store.get_stored_fingerprints()
    expected = {f"{t.namespace}.{t.name}": compute_tool_fingerprint(t) for t in tools}
    assert stored == expected


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_quantized_search(qdrant_url: str) -> None:
    """Scalar-quantized collections search correctly with exact rescoring."""
    pytest.importorskip("qdrant_client")

    from agent_gantry.adapters.vector_stores.remote import QdrantVectorStore

    store = QdrantVectorStore(
        url=qdrant_url,
        collection_name="test_quantized",
        dimension=128,
        quantization="scalar",
    )
    embedder = SimpleEmbedder()
    await store.initialize()

    tools = _make_tools(5)
    embeddings = await embedder.embed_batch([t.to_searchable_text() for t in tools])
    assert await store.add_tools(tools, embeddings, upsert=True) == 5

    query = await embedder.embed_text(tools[1].to_searchable_text())
    results = await store.search(query_vector=query, limit=3)
    assert results
    # Rescoring against original vectors keeps the exact match on top
    assert results[0][0].name == tools[1].name


def test_qdrant_quantization_mode_validated() -> None:
    """Bad quantization modes fail fast at construction."""
    pytest.importorskip("qdrant_client")

    from agent_gantry.adapters.vector_stores.remote import QdrantVectorStore

    with pytest.raises(ValueError, match="Unsupported quantization mode"):
        QdrantVectorStore(url="http://localhost:6333", quantization="turbo")
