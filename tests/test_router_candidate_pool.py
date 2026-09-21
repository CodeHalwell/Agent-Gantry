"""The vector-search pool must be wide enough for re-scoring to matter.

``SemanticRouter.route`` fetches candidates by raw cosine, then re-scores them
with intent, conversation, health and cost signals. Intent alone is worth 0.15
against a semantic weight of 0.6, so a tool a quarter of a cosine point behind
the leader can still win — but only if it is in the pool. With the pool sized
``limit * 4``, a ``limit=3`` query saw twelve candidates, and the project's own
300-tool demo ("calculate the mean and standard deviation, and generate a
secure password") lost ``calculate_mean`` at cosine rank 13 to three
random-number tools that happened to embed closer.
"""

from __future__ import annotations

from typing import Any

from agent_gantry import AgentGantry
from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore
from agent_gantry.schema.query import ConversationContext, ToolQuery
from agent_gantry.schema.tool import ToolDefinition


class _RankedStore(InMemoryVectorStore):
    """An in-memory store whose search serves a fixed ranking.

    ``limit`` truncates it exactly as a real store's top-k would, which is the
    behaviour under test: whether a strong re-scoring candidate just outside
    ``limit * 4`` is ever seen.
    """

    def __init__(self, ranked: list[tuple[str, float]]) -> None:
        super().__init__()
        self._ranked = ranked
        self.requested: list[int] = []

    async def search(  # type: ignore[override]
        self,
        query_vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        include_embeddings: bool = False,
    ) -> Any:
        self.requested.append(limit)
        by_key = {f"{t.namespace}.{t.name}": t for t in self._tools.values()}
        rows = [
            (by_key[f"default.{name}"], score)
            for name, score in self._ranked
            if score >= (score_threshold or 0.0)
        ]
        return rows[:limit]


def _tool(name: str, description: str, tags: list[str]) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        tags=tags,
        parameters_schema={"type": "object", "properties": {}},
    )


async def _gantry_with_target_at_rank(rank: int) -> tuple[AgentGantry, _RankedStore]:
    """Thirteen+ distractors that embed closer, and the intent-matching tool at ``rank``."""
    distractors = [
        _tool(
            f"random_{i}",
            f"Return a random value drawn from distribution number {i}.",
            ["random"],
        )
        for i in range(rank)
    ]
    target = _tool(
        "calculate_mean",
        "Calculate the arithmetic mean of a list of numbers.",
        ["math", "statistics"],
    )
    ranked = [(d.name, 0.60 - 0.005 * i) for i, d in enumerate(distractors)]
    ranked.append((target.name, 0.60 - 0.005 * rank))
    store = _RankedStore(ranked)
    gantry = AgentGantry(vector_store=store)
    for tool in [*distractors, target]:
        await gantry.add_tool(tool, lambda **kw: kw)
    return gantry, store


async def test_an_intent_match_just_outside_limit_times_four_still_wins() -> None:
    """The demo's shape: the right tool at cosine rank 13, query limit 3."""
    gantry, store = await _gantry_with_target_at_rank(13)
    try:
        result = await gantry.retrieve(
            ToolQuery(
                context=ConversationContext(query="calculate the mean of these numbers"),
                limit=3,
                score_threshold=0.0,
            )
        )
        names = [scored.tool.name for scored in result.tools]
        assert names[0] == "calculate_mean", names
    finally:
        await gantry.close()


async def test_the_pool_still_scales_with_a_large_limit() -> None:
    """A floor must not become a ceiling: ``limit=20`` keeps asking for 80."""
    gantry, store = await _gantry_with_target_at_rank(5)
    try:
        await gantry.retrieve(
            ToolQuery(
                context=ConversationContext(query="calculate the mean"),
                limit=20,
                score_threshold=0.0,
            )
        )
        assert store.requested[-1] >= 80
    finally:
        await gantry.close()
