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
import types
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

    def test_an_empty_namespace_list_selects_nothing_everywhere(self) -> None:
        """``search`` returns nothing for an empty namespace list, but
        ``list_all`` and ``count`` used ``if namespace:``, which cannot tell
        ``None`` -- no filter -- from ``[]`` -- a filter matching nothing -- so
        they returned the whole collection instead."""
        from agent_gantry.adapters.vector_stores.remote import _selects_nothing

        assert _selects_nothing([]) is True
        assert _selects_nothing(()) is True
        assert _selects_nothing(set()) is True
        # a populated filter, a scalar, and "no filter at all" all still query
        assert _selects_nothing(["a"]) is False
        assert _selects_nothing("a") is False
        assert _selects_nothing(None) is False

    @pytest.mark.asyncio
    async def test_empty_namespace_list_short_circuits_list_all_and_count(self) -> None:
        """Each adapter returns the empty answer without reaching its client,
        so the guard holds whether or not a backend is running."""
        from agent_gantry.adapters.vector_stores.remote import (
            ChromaVectorStore,
            PGVectorStore,
            QdrantVectorStore,
        )

        for cls in (QdrantVectorStore, ChromaVectorStore, PGVectorStore):
            store = cls.__new__(cls)

            async def _no_client() -> None:  # the client is never built
                return None

            store.initialize = _no_client  # type: ignore[method-assign]
            assert await cls.list_all(store, namespace=[]) == [], cls.__name__
            assert await cls.count(store, namespace=[]) == 0, cls.__name__

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
        truncated = embedder._truncate([(full / np.linalg.norm(full)).tolist()])
        assert len(truncated[0]) == 64
        assert np.linalg.norm(truncated[0]) == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_nomic_prefixes_documents_and_queries(self) -> None:
        """Documents get the configured task prefix; queries always ``search_query``."""
        import numpy as np

        from agent_gantry.adapters.embedders.nomic import NomicEmbedder

        class RecordingModel(_FakeModel):
            def __init__(self) -> None:
                super().__init__(dimension=768)
                self.calls: list[list[str]] = []

            def encode(self, texts: Any, **kwargs: Any) -> Any:
                self.calls.append(list(texts))
                return super().encode(texts, **kwargs)

        embedder = NomicEmbedder(dimension=64, task_type="clustering")
        model = RecordingModel()
        embedder._model = model
        embedder._native_dimension = 768

        document = await embedder.embed_text("alpha")
        query = await embedder.embed_query("beta")
        await embedder.embed_batch(["c", "d"])
        assert model.calls == [
            ["clustering: alpha"],
            ["search_query: beta"],
            ["clustering: c", "clustering: d"],
        ]
        assert len(document) == len(query) == 64
        assert np.linalg.norm(document) == pytest.approx(1.0)
        assert embedder.get_embedder_id() == "nomic-ai/nomic-embed-text-v1.5:64:clustering"


class TestOpenAIEmbedders:
    def test_azure_builds_from_its_own_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The base class checked OPENAI_API_KEY first, so the documented
        AZURE_OPENAI_API_KEY alone raised the OpenAI error."""
        pytest.importorskip("openai")
        from agent_gantry.adapters.embedders.openai import AzureOpenAIEmbedder
        from agent_gantry.schema.config import EmbedderConfig

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-test-key")
        embedder = AzureOpenAIEmbedder(
            EmbedderConfig(
                type="azure",
                model="text-embedding-3-small",
                api_base="https://example.openai.azure.com",
            )
        )
        assert (
            embedder.get_embedder_id()
            == "azure:text-embedding-3-small:1536@https://example.openai.azure.com"
        )

    def test_openai_still_requires_its_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("openai")
        from agent_gantry.adapters.embedders.openai import OpenAIEmbedder
        from agent_gantry.schema.config import EmbedderConfig

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIEmbedder(EmbedderConfig(model="text-embedding-3-small"))


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


def test_cached_embedder_counters_are_exact_across_threads(tmp_path: Any) -> None:
    """#427: the hit/miss tallies were ``+=``'d outside the lock guarding SQLite.

    One embedder shared across event loops in different threads — the
    configuration the class is built for — could lose increments. Statistics
    only, but the reported hit rate is the one thing these exist for.
    """
    cached = CachedEmbedder(_CountingEmbedder(), cache_path=str(tmp_path / "cache.sqlite"))
    # Warm both keys (document and query scopes differ) so every call the
    # threads make is a hit and the tallies have one right answer.
    asyncio.run(cached.embed_batch(["shared"]))
    asyncio.run(cached.embed_query("shared"))
    rounds, threads = 40, 8
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(rounds):
                asyncio.run(cached.embed_batch(["shared"]))
                asyncio.run(cached.embed_query("shared"))
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    import threading

    pool = [threading.Thread(target=worker) for _ in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()

    assert not errors
    assert cached.misses == 2, "only the two warm-up calls missed"
    assert cached.hits == rounds * threads * 2


# --------------------------------------------------------------------------- #
# LanceDB: zero-row tag filters, error propagation, schema migration
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_lancedb_tag_filter_with_no_rows_returns_nothing(lancedb_store: Any) -> None:
    """A zero row count sized the fetch window at 0, and LanceDB rejects
    ``limit(0)`` on a vector query."""
    await lancedb_store.initialize()
    query = [1.0, 0.0, 0.0, 0.0]
    assert await lancedb_store.search(query, limit=3, filters={"tags": ["x"]}) == []
    await lancedb_store.add_tools([_tool("a", tags=["x"])], [query])
    scoped = await lancedb_store.search(
        query, limit=3, filters={"tags": ["x"], "namespace": ["other"]}
    )
    assert scoped == []
    assert len(await lancedb_store.search(query, limit=3, filters={"tags": ["x"]})) == 1


@pytest.mark.asyncio
async def test_lancedb_table_errors_propagate(lancedb_store: Any) -> None:
    """``list_all``/``count``/``delete`` swallowed table errors into ``[]``/``0``/
    ``False``, which is indistinguishable from an empty store."""
    await lancedb_store.initialize()

    class BrokenTable:
        def search(self) -> Any:
            raise RuntimeError("table exploded")

        def count_rows(self, *args: Any) -> int:
            raise RuntimeError("table exploded")

    lancedb_store._tools_table = BrokenTable()
    for call in (lancedb_store.list_all(), lancedb_store.count(), lancedb_store.delete("x")):
        with pytest.raises(RuntimeError, match="table exploded"):
            await call


@pytest.mark.asyncio
async def test_lancedb_schema_migration_keeps_rows(tmp_path: Any) -> None:
    """A pre-fingerprint table is migrated without losing its rows."""
    lancedb = pytest.importorskip("lancedb")
    pa = pytest.importorskip("pyarrow")
    from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore
    from agent_gantry.utils.fingerprint import compute_tool_fingerprint

    tool = _tool("legacy_tool")
    legacy_schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("name", pa.string()),
            pa.field("namespace", pa.string()),
            pa.field("description", pa.string()),
            pa.field("tool_json", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 4)),
        ]
    )
    db_path = str(tmp_path / "db")
    lancedb.connect(db_path).create_table(
        "tools",
        data=[
            {
                "id": "default.legacy_tool",
                "name": "legacy_tool",
                "namespace": "default",
                "description": tool.description,
                "tool_json": tool.model_dump_json(),
                "vector": [1.0, 0.0, 0.0, 0.0],
            }
        ],
        schema=legacy_schema,
    )

    store = LanceDBVectorStore(db_path=db_path, dimension=4)
    await store.initialize()
    assert await store.count() == 1
    assert await store.get_stored_fingerprints() == {
        "default.legacy_tool": compute_tool_fingerprint(tool)
    }
    listed = store._db.list_tables()
    assert "tools__migrating" not in set(getattr(listed, "tables", listed))


# --------------------------------------------------------------------------- #
# Remote adapters: tag filter, upsert=False and delete contracts (fake clients)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_search_with_tags_widens_until_enough_match() -> None:
    from agent_gantry.adapters.vector_stores.remote import _required_tags, _search_with_tags

    rows: list[Any] = [(_tool(f"plain_{i}"), 1.0 - i / 100) for i in range(10)]
    rows.append((_tool("tagged", tags=["special"]), 0.5))
    sizes: list[int] = []

    async def fetch(size: int) -> list[Any]:
        sizes.append(size)
        return rows[:size]

    assert _required_tags({"tags": ["special"]}) == {"special"}
    assert _required_tags(None) == set() and _required_tags({"namespace": ["a"]}) == set()

    hits = await _search_with_tags(fetch, 2, {"special"})
    assert [tool.name for tool, _ in hits] == ["tagged"]
    assert sizes == [8, 16]  # limit * 4, then doubled until the backend ran out

    assert await _search_with_tags(fetch, 3, set()) == rows[:3]


class _FakeChromaCollection:
    """Enough of ``chromadb.Collection`` for add/get/delete/query/count."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def upsert(self, ids: list[str], embeddings: Any, documents: Any, metadatas: Any) -> None:
        for tool_id, embedding, metadata in zip(ids, embeddings, metadatas):
            self.rows[tool_id] = {"embedding": embedding, "metadata": metadata}

    def get(self, ids: list[str] | None = None, **_: Any) -> dict[str, Any]:
        found = [i for i in (ids or self.rows) if i in self.rows]
        return {"ids": found, "metadatas": [self.rows[i]["metadata"] for i in found]}

    def delete(self, ids: list[str]) -> None:
        for tool_id in ids:
            self.rows.pop(tool_id, None)

    def count(self) -> int:
        return len(self.rows)

    def query(
        self, query_embeddings: Any, n_results: int, where: Any, include: list[str]
    ) -> dict[str, Any]:
        rows = list(self.rows.values())[:n_results]
        out: dict[str, Any] = {
            "metadatas": [[r["metadata"] for r in rows]],
            "distances": [[0.1 * i for i in range(len(rows))]],
        }
        if "embeddings" in include:
            out["embeddings"] = [[r["embedding"] for r in rows]]
        return out


async def _ready() -> None:
    """Stand-in for ``initialize`` on a store built without its client."""


def _chroma_store() -> Any:
    from agent_gantry.adapters.vector_stores.remote import ChromaVectorStore

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store._collection = _FakeChromaCollection()
    store.initialize = _ready  # type: ignore[method-assign]
    return store


@pytest.mark.asyncio
async def test_chroma_upsert_false_skips_stored_ids_and_delete_reports_a_miss() -> None:
    store = _chroma_store()
    tool, other = _tool("a"), _tool("b")
    assert await store.add_tools([tool], [[1.0, 0.0]]) == 1
    # The stored id and an in-batch repeat are skipped; only the new row counts
    assert await store.add_tools([tool, other, other], [[1.0, 0.0]] * 3, upsert=False) == 1
    assert await store.count() == 2
    assert await store.delete("missing") is False
    assert await store.delete("a") is True
    assert await store.delete("a") is False


@pytest.mark.asyncio
async def test_chroma_search_honours_the_tag_filter() -> None:
    store = _chroma_store()
    tools = [_tool(f"plain_{i}") for i in range(6)] + [_tool("tagged", tags=["special"])]
    await store.add_tools(tools, [[1.0, 0.0]] * len(tools))
    hits = await store.search([1.0, 0.0], limit=1, filters={"tags": ["special"]})
    assert [tool.name for tool, _ in hits] == ["tagged"]
    with_vectors = await store.search([1.0, 0.0], limit=2, include_embeddings=True)
    assert [(t.name, e) for t, _, e in with_vectors] == [
        ("plain_0", [1.0, 0.0]),
        ("plain_1", [1.0, 0.0]),
    ]


class _FakeQdrantClient:
    def __init__(self, points: list[Any]) -> None:
        self.points = points
        self.closed = False

    async def query_points(self, *, limit: int, **_: Any) -> Any:
        return types.SimpleNamespace(points=self.points[:limit])

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_qdrant_search_honours_the_tag_filter_and_close_releases_the_client() -> None:
    from agent_gantry.adapters.vector_stores.remote import QdrantVectorStore

    def point(tool: ToolDefinition, score: float) -> Any:
        return types.SimpleNamespace(
            payload={"tool_json": tool.model_dump_json()}, score=score, vector=None
        )

    points = [point(_tool(f"plain_{i}"), 0.9) for i in range(5)]
    points.append(point(_tool("tagged", tags=["special"]), 0.1))
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._collection_name = "c"
    store._quantization = None
    store._client = _FakeQdrantClient(points)
    store.initialize = _ready  # type: ignore[method-assign]

    hits = await store.search([1.0], limit=1, filters={"tags": ["special"]})
    assert [(tool.name, score) for tool, score in hits] == [("tagged", 0.1)]
    await store.close()
    assert store._client.closed


class _AsyncCM:
    def __init__(self, value: Any = None) -> None:
        self.value = value

    async def __aenter__(self) -> Any:
        return self.value

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _FakePGConn:
    """``fetchrow`` applies ON CONFLICT DO NOTHING; ``fetch`` serves an ordered table."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.ids = {row["id"] for row in rows}

    def transaction(self) -> _AsyncCM:
        return _AsyncCM()

    async def fetchrow(self, sql: str, *params: Any) -> dict[str, Any] | None:
        assert "ON CONFLICT (id) DO NOTHING RETURNING id" in sql
        if params[0] in self.ids:
            return None
        self.ids.add(params[0])
        return {"id": params[0]}

    async def fetch(self, sql: str, embedding: str, size: int, *rest: Any) -> list[dict[str, Any]]:
        return self.rows[:size]


def _pg_store(conn: _FakePGConn) -> Any:
    from agent_gantry.adapters.vector_stores.remote import PGVectorStore

    store = PGVectorStore.__new__(PGVectorStore)
    store._table_name = "tools"
    store._pool = types.SimpleNamespace(acquire=lambda: _AsyncCM(conn))
    store.initialize = _ready  # type: ignore[method-assign]
    return store


@pytest.mark.asyncio
async def test_pgvector_upsert_false_counts_only_inserted_rows() -> None:
    conn = _FakePGConn([{"id": "default.a"}])
    store = _pg_store(conn)
    assert (
        await store.add_tools([_tool("a"), _tool("b"), _tool("b")], [[1.0]] * 3, upsert=False) == 1
    )
    assert conn.ids == {"default.a", "default.b"}


@pytest.mark.asyncio
async def test_pgvector_search_honours_the_tag_filter() -> None:
    def row(tool: ToolDefinition, similarity: float) -> dict[str, Any]:
        return {
            "id": tool.qualified_name,
            "tool_json": tool.model_dump_json(),
            "similarity": similarity,
        }

    rows = [row(_tool(f"plain_{i}"), 0.9) for i in range(5)]
    rows.append(row(_tool("tagged", tags=["special"]), 0.2))
    store = _pg_store(_FakePGConn(rows))
    hits = await store.search([1.0], limit=1, filters={"tags": ["special"]})
    assert [(tool.name, score) for tool, score in hits] == [("tagged", 0.2)]
    above = await store.search([1.0], limit=2, score_threshold=0.5)
    assert [tool.name for tool, _ in above] == ["plain_0", "plain_1"]


class TestLazyModelMixin:
    def test_load_model_leaving_model_unset_is_an_error(self) -> None:
        from agent_gantry.adapters.embedders.base import LazyModelMixin

        class Forgetful(LazyModelMixin):
            def __init__(self) -> None:
                self._model = None
                self._load_lock = threading.Lock()

            def _load_model(self) -> None:
                pass  # never assigns self._model

        with pytest.raises(RuntimeError, match="left _model unset"):
            Forgetful()._ensure_initialized()

    @pytest.mark.asyncio
    async def test_concurrent_first_use_loads_once(self) -> None:
        import time

        from agent_gantry.adapters.embedders.base import LazyModelMixin

        class Slow(LazyModelMixin):
            loads = 0

            def __init__(self) -> None:
                self._model = None
                self._load_lock = threading.Lock()

            def _load_model(self) -> None:
                time.sleep(0.05)  # long enough for the second caller to contend
                Slow.loads += 1
                self._model = object()

        lazy = Slow()
        await asyncio.gather(lazy._aensure_initialized(), lazy._aensure_initialized())
        assert Slow.loads == 1 and lazy._model is not None


class TestEmbedderHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_encodes_rather_than_only_loading(self) -> None:
        from agent_gantry.adapters.embedders.sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        healthy = SentenceTransformersEmbedder(model="stub-model")
        _stub_model(healthy)
        assert await healthy.health_check()

        def cannot_encode(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("model loaded but cannot encode")

        broken = SentenceTransformersEmbedder(model="stub-model")
        broken._model = types.SimpleNamespace(encode=cannot_encode)
        assert not await broken.health_check()
