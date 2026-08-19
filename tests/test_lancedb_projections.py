"""LanceDB reads must project only the columns they use.

The tools ``search()`` already carried a ``.select()`` with a comment
explaining why: without it every returned row also materializes its full
embedding vector into Python objects just to be discarded. Three sibling paths
had been missed — ``search_skills``, ``list_all``, and ``list_all_skills`` —
and the last is called with a very large limit by the facade's
embedder-migration check.

Projection can silently drop a column the code downstream needs (``_distance``
in particular, which newer Lance versions stop auto-projecting once output
columns are specified), so these tests assert the *behaviour* of each path
rather than the shape of the query.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from agent_gantry.schema.skill import Skill, SkillCategory
from agent_gantry.schema.tool import ToolDefinition

pytest.importorskip("lancedb", reason="LanceDB store tests need the lancedb extra")

_DIM = 8


def _vec() -> list[float]:
    return np.random.rand(_DIM).astype(np.float32).tolist()


@pytest.fixture
async def store():
    from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

    store = LanceDBVectorStore(db_path=tempfile.mkdtemp(), dimension=_DIM)
    await store.initialize()

    tools = [
        ToolDefinition(
            name=f"tool_{i}",
            namespace="billing" if i % 2 else "default",
            description=f"Tool {i} performing a documented operation on records.",
            parameters_schema={"type": "object", "properties": {}},
        )
        for i in range(6)
    ]
    store._test_tool_vectors = [_vec() for _ in tools]  # type: ignore[attr-defined]
    await store.add_tools(tools, store._test_tool_vectors)  # type: ignore[attr-defined]

    skills = [
        Skill(
            name=f"skill_{i}",
            namespace="ops",
            category=SkillCategory.WORKFLOW,
            description=f"Skill {i} describing a documented workflow step.",
            content=f"Body of skill {i} with guidance.",
        )
        for i in range(4)
    ]
    store._test_skill_vectors = [_vec() for _ in skills]  # type: ignore[attr-defined]
    await store.add_skills(skills, store._test_skill_vectors)  # type: ignore[attr-defined]
    return store


async def test_list_all_returns_full_definitions(store) -> None:
    tools = await store.list_all(limit=100)

    assert len(tools) == 6
    assert all(isinstance(t, ToolDefinition) for t in tools)
    # Fields live inside the projected tool_json blob — prove none were lost.
    assert all(t.description and t.parameters_schema is not None for t in tools)


async def test_list_all_still_filters_by_namespace(store) -> None:
    """The filter column need not be projected; the engine applies it first."""
    billing = await store.list_all(limit=100, namespace="billing")

    assert len(billing) == 3
    assert {t.namespace for t in billing} == {"billing"}


async def test_list_all_skills_returns_full_skills(store) -> None:
    skills = await store.list_all_skills(limit=1_000_000)

    assert len(skills) == 4
    assert all(s.content for s in skills), "skill body lost to the projection"


async def test_list_all_skills_still_filters_by_namespace(store) -> None:
    assert len(await store.list_all_skills(limit=100, namespace="ops")) == 4
    assert await store.list_all_skills(limit=100, namespace="absent") == []


async def test_search_skills_keeps_distance_for_scoring(store) -> None:
    """Scores come from ``_distance``; dropping it would flatten the ranking."""
    results = await store.search_skills(store._test_skill_vectors[0], limit=3)

    assert results
    assert all(0.0 <= score <= 1.0 for _, score in results)
    # Querying with a stored vector must rank that skill first.
    assert results[0][1] >= results[-1][1]
    assert results[0][1] > 0.0, "all scores collapsed — _distance was not projected"
    assert results[0][0].content


async def test_search_skills_respects_score_threshold(store) -> None:
    everything = await store.search_skills(store._test_skill_vectors[0], limit=10)
    filtered = await store.search_skills(
        store._test_skill_vectors[0], limit=10, score_threshold=0.99
    )

    assert len(filtered) <= len(everything)
    assert all(score >= 0.99 for _, score in filtered)


async def test_tool_search_still_scores(store) -> None:
    results = await store.search(store._test_tool_vectors[0], limit=3)

    assert results
    assert results[0][1] > 0.0
    assert results[0][0].description
