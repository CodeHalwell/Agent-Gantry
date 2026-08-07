"""
Tests for semantic skill selection: the in-memory store's skill support and
the AgentGantry facade API (add/retrieve/prompt-injection/delete/list/count).
"""

from __future__ import annotations

import pytest

from agent_gantry import AgentGantry, Skill, SkillCategory
from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore


def _make_skills() -> list[Skill]:
    return [
        Skill(
            name="api_pagination",
            description="How to implement cursor-based pagination for API endpoints",
            content="Use cursor-based pagination: return an opaque cursor per page...",
            category=SkillCategory.HOW_TO,
            tags=["api", "pagination"],
            related_tools=["fetch_page"],
        ),
        Skill(
            name="retry_backoff",
            description="Pattern for retrying flaky network calls with exponential backoff",
            content="Retry with exponential backoff and jitter: 1s, 2s, 4s...",
            category=SkillCategory.PATTERN,
            tags=["network", "retry"],
        ),
        Skill(
            name="db_migration",
            description="Procedure for running database schema migrations safely",
            content="Always back up before migrating. Apply migrations in a transaction...",
            category=SkillCategory.PROCEDURE,
            namespace="ops",
            tags=["database"],
        ),
    ]


# ---------------------------------------------------------------------------
# InMemoryVectorStore skill support
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_store_skill_crud():
    store = InMemoryVectorStore()
    skills = _make_skills()
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    assert await store.add_skills(skills, embeddings) == 3
    assert await store.count_skills() == 3
    assert await store.count_skills(namespace="ops") == 1

    fetched = await store.get_skill_by_name("api_pagination")
    assert fetched is not None and fetched.category == SkillCategory.HOW_TO

    listed = await store.list_all_skills(namespace="default")
    assert {s.name for s in listed} == {"api_pagination", "retry_backoff"}

    assert await store.delete_skill("db_migration", namespace="ops") is True
    assert await store.count_skills() == 2
    assert await store.delete_skill("db_migration", namespace="ops") is False


@pytest.mark.asyncio
async def test_memory_store_skill_search_and_filters():
    store = InMemoryVectorStore()
    skills = _make_skills()
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    await store.add_skills(skills, embeddings)

    results = await store.search_skills(query_vector=[1.0, 0.0, 0.0], limit=2)
    assert results[0][0].name == "api_pagination"
    assert results[0][1] == pytest.approx(1.0)

    # Category filter
    results = await store.search_skills(
        query_vector=[1.0, 0.0, 0.0], limit=5, filters={"category": "pattern"}
    )
    assert {s.name for s, _ in results} == {"retry_backoff"}

    # Namespace filter
    results = await store.search_skills(
        query_vector=[1.0, 0.0, 0.0], limit=5, filters={"namespace": "ops"}
    )
    assert {s.name for s, _ in results} == {"db_migration"}

    # Score threshold prunes orthogonal matches
    results = await store.search_skills(
        query_vector=[1.0, 0.0, 0.0], limit=5, score_threshold=0.5
    )
    assert {s.name for s, _ in results} == {"api_pagination"}

    # Mismatched query dimension degrades gracefully
    assert await store.search_skills(query_vector=[1.0, 0.0], limit=5) == []


# ---------------------------------------------------------------------------
# AgentGantry facade
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gantry_skill_roundtrip():
    gantry = AgentGantry()
    skills = _make_skills()
    assert await gantry.add_skills(skills) == 3
    assert await gantry.count_skills() == 3

    # Identical text embeds to the identical vector, so querying with a
    # skill's own embedding text must rank that skill first regardless of
    # which embedder backs the store.
    results = await gantry.retrieve_skills(skills[0].to_embedding_text(), limit=2)
    assert results and results[0].skill.name == "api_pagination"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)

    prompt = await gantry.retrieve_skills_as_prompt(skills[0].to_embedding_text(), limit=1)
    assert "Api Pagination" in prompt
    assert "cursor-based pagination" in prompt
    assert "Related tools: fetch_page" in prompt

    # Namespace + category filters pass through
    ops_only = await gantry.retrieve_skills(
        skills[2].to_embedding_text(), limit=5, namespace="ops"
    )
    assert {r.skill.name for r in ops_only} == {"db_migration"}

    assert await gantry.delete_skill("retry_backoff") is True
    assert await gantry.count_skills() == 2
    assert {s.name for s in await gantry.list_skills()} == {"api_pagination", "db_migration"}


@pytest.mark.asyncio
async def test_gantry_skill_prompt_empty_when_no_match():
    gantry = AgentGantry()
    assert await gantry.retrieve_skills_as_prompt("anything") == ""


@pytest.mark.asyncio
async def test_gantry_skills_unsupported_store_raises():
    class NoSkillsStore:
        async def initialize(self) -> None: ...

    gantry = AgentGantry(vector_store=NoSkillsStore())  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="does not support skills"):
        await gantry.add_skills(_make_skills())


@pytest.mark.asyncio
async def test_memory_store_skill_upsert_false_skips_existing():
    store = InMemoryVectorStore()
    skill = _make_skills()[0]
    await store.add_skills([skill], [[1.0, 0.0]])

    updated = skill.model_copy(update={"content": "changed content"})
    assert await store.add_skills([updated], [[0.0, 1.0]], upsert=False) == 0
    stored = await store.get_skill_by_name("api_pagination")
    assert stored is not None and stored.content != "changed content"


@pytest.mark.asyncio
async def test_memory_store_add_skills_length_mismatch_raises():
    store = InMemoryVectorStore()
    with pytest.raises(ValueError):
        await store.add_skills(_make_skills()[:2], [[1.0, 0.0]])


@pytest.mark.asyncio
async def test_gantry_add_skill_singular():
    gantry = AgentGantry()
    await gantry.add_skill(_make_skills()[0])
    assert await gantry.count_skills() == 1


@pytest.mark.asyncio
async def test_skills_reembedded_after_embedder_change():
    """Reopening a store with a different embedder must re-embed persisted
    skills — otherwise queries embed with the new model but search the old
    model's vectors, silently returning wrong guidance."""
    from agent_gantry.adapters.embedders.simple import SimpleEmbedder

    class ShiftedEmbedder(SimpleEmbedder):
        """Same dimension, different vector space and identity."""

        @property
        def model_name(self) -> str:
            return "shifted-model"

        def get_embedder_id(self) -> str:
            return f"shifted:{self.dimension}"

        async def embed_text(self, text: str) -> list[float]:
            base = await super().embed_text(text)
            return list(reversed(base))

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [await self.embed_text(t) for t in texts]

    store = InMemoryVectorStore()
    skills = _make_skills()

    gantry_a = AgentGantry(vector_store=store)
    await gantry_a.add_skills(skills)

    # Same store, different embedder: retrieval must self-heal by
    # re-embedding, so an exact-text query still ranks its skill first with
    # a perfect score
    gantry_b = AgentGantry(vector_store=store, embedder=ShiftedEmbedder())
    results = await gantry_b.retrieve_skills(skills[0].to_embedding_text(), limit=1)
    assert results and results[0].skill.name == "api_pagination"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def _shifted_embedder_cls():
    from agent_gantry.adapters.embedders.simple import SimpleEmbedder

    class ShiftedEmbedder(SimpleEmbedder):
        @property
        def model_name(self) -> str:
            return "shifted-model"

        def get_embedder_id(self) -> str:
            return f"shifted:{self.dimension}"

        async def embed_text(self, text: str) -> list[float]:
            return list(reversed(await super().embed_text(text)))

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [await self.embed_text(t) for t in texts]

    return ShiftedEmbedder


@pytest.mark.asyncio
async def test_concurrent_retrievals_wait_for_skill_migration():
    """Concurrent first calls must not skip past an in-flight re-embed and
    search stale vectors."""
    import asyncio

    store = InMemoryVectorStore()
    skills = _make_skills()
    gantry_a = AgentGantry(vector_store=store)
    await gantry_a.add_skills(skills)

    # Slow the migration's skill listing so a second caller overlaps it
    original_list = store.list_all_skills

    async def slow_list(*args, **kwargs):
        await asyncio.sleep(0.05)
        return await original_list(*args, **kwargs)

    store.list_all_skills = slow_list  # type: ignore[method-assign]

    gantry_b = AgentGantry(vector_store=store, embedder=_shifted_embedder_cls()())
    query = skills[0].to_embedding_text()
    first, second = await asyncio.gather(
        gantry_b.retrieve_skills(query, limit=1),
        gantry_b.retrieve_skills(query, limit=1),
    )
    for results in (first, second):
        assert results and results[0].skill.name == "api_pagination"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)


@pytest.mark.asyncio
async def test_skill_migration_retries_after_transient_failure():
    """A transient failure mid-migration must not permanently mark the
    migration complete — the next call retries it."""
    store = InMemoryVectorStore()
    skills = _make_skills()
    gantry_a = AgentGantry(vector_store=store)
    await gantry_a.add_skills(skills)

    original_get = store.get_metadata
    calls = {"n": 0}

    async def flaky_get(key: str):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient metadata outage")
        return await original_get(key)

    store.get_metadata = flaky_get  # type: ignore[method-assign]

    gantry_b = AgentGantry(vector_store=store, embedder=_shifted_embedder_cls()())
    query = skills[0].to_embedding_text()
    with pytest.raises(RuntimeError, match="transient metadata outage"):
        await gantry_b.retrieve_skills(query, limit=1)

    # Retry migrates and succeeds
    results = await gantry_b.retrieve_skills(query, limit=1)
    assert results and results[0].skill.name == "api_pagination"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
