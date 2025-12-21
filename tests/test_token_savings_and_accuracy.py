"""
Tests for provider-backed token savings calculations and retrieval accuracy.
"""

from __future__ import annotations

import pytest

from agent_gantry import AgentGantry
from agent_gantry.metrics import ProviderUsage, calculate_token_savings
from agent_gantry.schema.query import ConversationContext, ToolQuery


def test_calculate_token_savings_uses_provider_usage() -> None:
    """Token savings should be derived from provider-reported usage, not estimates."""
    baseline = ProviderUsage.from_usage(
        {
            "prompt_tokens": 366,  # provider usage from an all-tools prompt
            "completion_tokens": 42,
            "total_tokens": 408,
        }
    )
    optimized = ProviderUsage.from_usage(
        {
            "prompt_tokens": 78,  # provider usage after top-k filtering
            "completion_tokens": 39,
            "total_tokens": 117,
        }
    )

    savings = calculate_token_savings(baseline, optimized)

    assert savings.saved_prompt_tokens == 288
    assert savings.saved_total_tokens == 291
    assert pytest.approx(savings.prompt_savings_pct, rel=1e-4) == 78.6885
    assert pytest.approx(savings.total_savings_pct, rel=1e-4) == 71.3235


@pytest.mark.asyncio
async def test_retrieval_topk_accuracy(sample_tools) -> None:
    """Top-k retrieval should keep the relevant tool present with high accuracy."""
    gantry = AgentGantry()
    for tool in sample_tools:
        await gantry.add_tool(tool)

    queries: dict[str, str] = {
        "send a follow-up email to the customer": "send_email",
        "process a refund for order 123": "process_refund",
        "create a new admin account": "create_user",
        "run an analytics report": "generate_report",
    }

    hits = 0
    for prompt, expected_tool in queries.items():
        result = await gantry.retrieve(
            ToolQuery(context=ConversationContext(query=prompt), limit=2)
        )
        retrieved_names = [scored.tool.name for scored in result.tools]
        if expected_tool in retrieved_names:
            hits += 1

    accuracy = hits / len(queries)
    assert accuracy >= 0.75
