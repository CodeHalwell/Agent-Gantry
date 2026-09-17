"""
Regression tests for the vector-store and embedder audit.

Covers the store and embedder defects found by sweeping the adapters: filters
the router actually sends, escaping that made rows invisible, upsert and delete
contracts that differed between backends, and embedder output that downstream
cosine scoring assumes is unit-length. The remote backends (Qdrant, Chroma,
pgvector) need a live server, so their predicate builders — where every one of
those bugs lived — are tested directly.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from agent_gantry.adapters.embedders.cached import CachedEmbedder
from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore
from agent_gantry.schema.tool import ToolDefinition


def _tool(name: str, namespace: str = "default", tags: list[str] | None = None) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        namespace=namespace,
        description=f"Tool {name} does something useful for tests",
        parameters_schema={"type": "object", "properties": {}},
        tags=tags or [],
    )


# --------------------------------------------------------------------------- #
# Remote adapters: the namespace filter the router actually sends is a LIST
# --------------------------------------------------------------------------- #


class TestRemoteNamespaceFilters:
    """``SemanticRouter`` always passes ``{"namespace": [...]}`` and the MCP
    router hard-codes ``["__mcp_servers__"]``, so a scalar-only predicate made
    every remote store raise — MCP server retrieval unconditionally."""

    def test_pgvector_binds_a_list_as_an_array(self) -> None:
        from agent_gantry.adapters.vector_stores.remote import _pg_namespace_clause

        clause, value = _pg_namespace_clause(["a", "b"], 3)
        # ``namespace = $3`` with a list bound to it raises asyncpg DataError
        assert clause == "WHERE namespace = ANY($3::text[])"
        assert value == ["a", "b"]

        clause, value = _pg_namespace_clause("a", 3)
        assert clause == "WHERE namespace = $3"
        assert value == "a"

    def test_chroma_uses_in_for_a_list(self) -> None:
        from agent_gantry.adapters.vector_stores.remote import ChromaVectorStore

        # ``{"namespace": ["a", "b"]}`` fails Chroma's own where-validation
        assert ChromaVectorStore._build_namespace_where(["a", "b"]) == {
            "namespace": {"$in": ["a", "b"]}
        }
        assert ChromaVectorStore._build_namespace_where("a") == {"namespace": "a"}

    def test_qdrant_uses_matchany_for_a_list(self) -> None:
        pytest.importorskip("qdrant_client")
        from qdrant_client.models import MatchAny, MatchValue

        from agent_gantry.adapters.vector_stores.remote import QdrantVectorStore

        # MatchValue(value=[...]) fails pydantic validation
        condition = QdrantVectorStore._build_namespace_filter(["a", "b"]).must[0]
        assert condition.key == "namespace"
        assert isinstance(condition.match, MatchAny)
        assert condition.match.any == ["a", "b"]

        condition = QdrantVectorStore._build_namespace_filter("a").must[0]
        assert isinstance(condition.match, MatchValue)
        assert condition.match.value == "a"


# --------------------------------------------------------------------------- #
# In-memory store: mismatched batches must not leave a partial registry
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_memory_add_tools_rejects_a_length_mismatch_without_mutating() -> None:
    """``add_skills`` already validated; ``add_tools`` zipped and silently
    stored the matching prefix, so a failed embed left a partial registry."""
    store = InMemoryVectorStore()
    with pytest.raises(ValueError, match="length mismatch"):
        await store.add_tools([_tool("a"), _tool("b"), _tool("c")], [[1.0, 0.0]])
    assert await store.count() == 0


# --------------------------------------------------------------------------- #
# LanceDB: upsert / delete contracts, and the tag filter's fetch window
# --------------------------------------------------------------------------- #


@pytest.fixture
def lancedb_store(tmp_path: Any) -> Any:
    pytest.importorskip("lancedb")
    pytest.importorskip("pyarrow")
    from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

    return LanceDBVectorStore(db_path=str(tmp_path / "db"), dimension=4)


@pytest.mark.asyncio
async def test_lancedb_add_without_upsert_skips_existing(lancedb_store: Any) -> None:
    """The in-memory store skips ids already present; LanceDB inserted a
    duplicate row, which the router then offered twice."""
    await lancedb_store.initialize()
    tool = _tool("dup_tool")
    assert await lancedb_store.add_tools([tool], [[1.0, 0.0, 0.0, 0.0]], upsert=False) == 1
    assert await lancedb_store.add_tools([tool], [[1.0, 0.0, 0.0, 0.0]], upsert=False) == 0
    assert await lancedb_store.count() == 1

    # A mixed batch inserts only the new rows
    inserted = await lancedb_store.add_tools(
        [tool, _tool("fresh_tool")],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        upsert=False,
    )
    assert inserted == 1
    assert await lancedb_store.count() == 2


@pytest.mark.asyncio
async def test_lancedb_add_without_upsert_dedupes_within_one_batch(lancedb_store: Any) -> None:
    """Checking only the table let an id repeated *inside* the batch through."""
    await lancedb_store.initialize()
    tool = _tool("dup_tool")
    inserted = await lancedb_store.add_tools(
        [tool, tool], [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], upsert=False
    )
    assert inserted == 1
    assert await lancedb_store.count() == 1


@pytest.mark.asyncio
async def test_lancedb_delete_reports_a_miss(lancedb_store: Any) -> None:
    """``delete`` returned True for a name that was never there, so
    ``AgentGantry.delete_tool`` reported success for an unknown tool."""
    await lancedb_store.initialize()
    await lancedb_store.add_tools([_tool("real_tool")], [[1.0, 0.0, 0.0, 0.0]])
    assert await lancedb_store.delete("missing_tool") is False
    assert await lancedb_store.delete("real_tool") is True
    assert await lancedb_store.delete("real_tool") is False


@pytest.mark.asyncio
async def test_lancedb_tag_filter_sees_past_the_overfetch_window(lancedb_store: Any) -> None:
    """Tags live inside ``tool_json``, so they are post-filtered. A fixed
    ``limit * 2`` window dropped tagged tools ranked below it entirely."""
    await lancedb_store.initialize()
    # 30 untagged tools rank ahead of the tagged one for this query vector
    untagged = [_tool(f"plain_{i}") for i in range(30)]
    await lancedb_store.add_tools(untagged, [[1.0, 0.0, 0.0, 0.0]] * len(untagged))
    await lancedb_store.add_tools([_tool("tagged_tool", tags=["special"])], [[0.0, 1.0, 0.0, 0.0]])

    results = await lancedb_store.search(
        query_vector=[1.0, 0.0, 0.0, 0.0], limit=3, filters={"tags": ["special"]}
    )
    assert [tool.name for tool, _score in results] == ["tagged_tool"]

    # An untagged search is unaffected by the widening, and a namespace filter
    # still composes with the tag filter.
    plain = await lancedb_store.search(query_vector=[1.0, 0.0, 0.0, 0.0], limit=3)
    assert len(plain) == 3
    await lancedb_store.add_tools(
        [_tool("scoped_tool", namespace="other", tags=["special"])], [[0.0, 1.0, 0.0, 0.0]]
    )
    scoped = await lancedb_store.search(
        query_vector=[1.0, 0.0, 0.0, 0.0],
        limit=5,
        filters={"tags": ["special"], "namespace": ["other"]},
    )
    assert [tool.name for tool, _score in scoped] == ["scoped_tool"]


# --------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------- #


class _FakeModel:
    """Stands in for a sentence-transformers model: unit vectors, as
    ``encode(normalize_embeddings=True)`` returns."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts: Any, **kwargs: Any) -> Any:
        import numpy as np

        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        rows = []
        for index, _text in enumerate(batch):
            vector = np.zeros(self.dimension, dtype=np.float64)
            # Spread the mass so a truncated prefix is genuinely non-unit
            vector[index % self.dimension] = 0.8
            vector[(index + 1) % self.dimension] = 0.6
            rows.append(vector / np.linalg.norm(vector))
        array = np.asarray(rows)
        return array[0] if single else array


def _stub_model(embedder: Any, dimension: int = 8) -> None:
    embedder._model = _FakeModel(dimension)
    embedder._native_dimension = dimension


class TestSentenceTransformersEmbedder:
    def test_embedder_id_does_not_change_when_the_model_loads(self) -> None:
        """The sync manager reads the id *before* the first embed call and
        records it after, so an id of ``dimauto`` that became ``dim8`` forced a
        full re-embed on every process start against a persistent store."""
        from agent_gantry.adapters.embedders.sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        embedder = SentenceTransformersEmbedder(model="stub-model")
        before = embedder.get_embedder_id()
        _stub_model(embedder)
        assert embedder.get_embedder_id() == before

        configured = SentenceTransformersEmbedder(model="stub-model", dimension=4)
        assert configured.get_embedder_id() != before

    @pytest.mark.asyncio
    async def test_truncated_embeddings_are_renormalised(self) -> None:
        """A sliced prefix of a unit vector is not unit length, and LanceDB's
        ``1 - d/2`` cosine conversion is only exact for unit vectors."""
        import numpy as np

        from agent_gantry.adapters.embedders.sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        embedder = SentenceTransformersEmbedder(model="stub-model", dimension=4)
        _stub_model(embedder)
        single = await embedder.embed_text("hello")
        assert len(single) == 4
        assert np.linalg.norm(single) == pytest.approx(1.0)

        batch = await embedder.embed_batch(["a", "b", "c"])
        assert all(len(vector) == 4 for vector in batch)
        assert all(np.linalg.norm(vector) == pytest.approx(1.0) for vector in batch)

    def test_nomic_matryoshka_truncation_is_renormalised(self) -> None:
        import numpy as np

        from agent_gantry.adapters.embedders.nomic import NomicEmbedder

        # 64 is one of Nomic's recommended Matryoshka dimensions
        embedder = NomicEmbedder(dimension=64)
        full = np.zeros(embedder.FULL_DIMENSION)
        full[:96] = np.linspace(1.0, 0.1, 96)
        truncated = embedder._apply_matryoshka_truncation([(full / np.linalg.norm(full)).tolist()])
        assert len(truncated[0]) == 64
        assert np.linalg.norm(truncated[0]) == pytest.approx(1.0)


class _CountingEmbedder:
    """Minimal embedder for the cache tests."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def dimension(self) -> int:
        return 2

    @property
    def model_name(self) -> str:
        return "counting"

    def get_embedder_id(self) -> str:
        return "counting:2"

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        self.calls += len(texts)
        return [[float(len(t)), 1.0] for t in texts]

    async def health_check(self) -> bool:
        return True


def test_cached_embedder_survives_a_second_event_loop(tmp_path: Any) -> None:
    """The cache was guarded by an ``asyncio.Lock`` created in ``__init__``,
    which binds to the first loop that contends on it — so a process-wide
    cache (the point of it) raised "bound to a different event loop" from the
    sync bridge, which runs ``asyncio.run`` per call."""
    cached = CachedEmbedder(_CountingEmbedder(), cache_path=str(tmp_path / "cache.sqlite"))

    async def contend() -> list[list[float]]:
        results = await asyncio.gather(
            cached.embed_batch(["alpha", "beta"]),
            cached.embed_batch(["gamma", "delta"]),
            cached.embed_batch(["alpha", "epsilon"]),
        )
        return results[0]

    first = asyncio.run(contend())
    second = asyncio.run(contend())  # a *different* loop
    assert first == second


def test_cached_embedder_is_usable_from_several_threads(tmp_path: Any) -> None:
    """The SQLite work runs in ``to_thread`` workers, so the guard has to be a
    threading lock rather than an asyncio one."""
    cached = CachedEmbedder(_CountingEmbedder(), cache_path=str(tmp_path / "cache.sqlite"))
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            asyncio.run(cached.embed_batch([f"text-{index}", "shared"]))
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


@pytest.mark.asyncio
async def test_cached_embedder_hits_its_own_first_batch(tmp_path: Any) -> None:
    """With a load-dependent embedder id the second lookup used a different
    cache key from the first write, so nothing ever hit."""
    base = _CountingEmbedder()
    cached = CachedEmbedder(base, cache_path=str(tmp_path / "cache.sqlite"))
    await cached.embed_batch(["one", "two"])
    assert base.calls == 2
    await cached.embed_batch(["one", "two"])
    assert base.calls == 2, "the second call must be served from the cache"
    assert cached.hits == 2
