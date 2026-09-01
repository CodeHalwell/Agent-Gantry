"""The cross-encoder reranker must hand back scores ``ScoredTool`` can hold.

``ScoredTool.semantic_score`` is bounded to ``[0, 1]``. sentence-transformers
applies a sigmoid for single-label rerankers, but a model configured with an
identity activation (or one with several labels) returns raw logits, and one
out-of-range value failed validation and took the whole retrieval down.
"""

from __future__ import annotations

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.rerankers.cross_encoder import CrossEncoderReranker, _bounded_scores
from agent_gantry.schema.query import ConversationContext, ScoredTool, ToolQuery
from agent_gantry.schema.tool import ToolDefinition


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Tool {name} does something useful for the caller.",
        parameters_schema={"type": "object", "properties": {}},
    )


class _LogitModel:
    """Stands in for a loaded ``CrossEncoder`` whose activation is the identity."""

    def predict(self, pairs: list[list[str]]) -> list[float]:
        return [4.2, -1.3, 0.1][: len(pairs)]


def test_probabilities_pass_through_untouched() -> None:
    assert _bounded_scores([0.0, 0.25, 1.0]) == [0.0, 0.25, 1.0]


def test_logits_are_squashed_monotonically() -> None:
    squashed = _bounded_scores([5.0, -2.0, 0.5])
    assert all(0.0 <= s <= 1.0 for s in squashed)
    assert squashed[0] > squashed[2] > squashed[1]
    # Extreme logits must not overflow ``exp``.
    assert _bounded_scores([-1e6, 1e6]) == [0.0, 1.0]


@pytest.mark.asyncio
async def test_rerank_scores_fit_scored_tool() -> None:
    reranker = CrossEncoderReranker()
    reranker._model = _LogitModel()  # skip the model download

    ranked = await reranker.rerank(
        "q", [(_tool("a"), 0.1), (_tool("b"), 0.2), (_tool("c"), 0.3)], top_k=3
    )

    assert [t.name for t, _ in ranked] == ["a", "c", "b"]
    for tool, score in ranked:
        ScoredTool(tool=tool, semantic_score=score)


@pytest.mark.asyncio
async def test_retrieve_with_logit_reranker_does_not_fail_validation() -> None:
    reranker = CrossEncoderReranker()
    reranker._model = _LogitModel()
    gantry = AgentGantry(reranker=reranker)
    for name in ("a", "b", "c"):
        await gantry.add_tool(_tool(name))

    result = await gantry.retrieve(
        ToolQuery(
            context=ConversationContext(query="something useful"),
            limit=3,
            score_threshold=0.0,
            enable_reranking=True,
        )
    )

    assert [t.tool.name for t in result.tools] == ["a", "c", "b"]
