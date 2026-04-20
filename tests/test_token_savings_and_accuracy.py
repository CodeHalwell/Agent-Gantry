"""
Tests for provider-backed token savings calculations and retrieval accuracy.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_gantry import AgentGantry
from agent_gantry.integrations import fetch_framework_tools
from agent_gantry.metrics import ProviderUsage, calculate_token_savings
from agent_gantry.schema.query import ConversationContext, ToolQuery


def test_calculate_token_savings_uses_provider_usage() -> None:
    """Token savings should be derived from provider-reported usage, not estimates."""
    baseline = ProviderUsage.from_usage(
        {
            # Illustrative values mirroring the token_savings_demo.py example (all-tools prompt)
            "prompt_tokens": 366,
            "completion_tokens": 42,
            "total_tokens": 408,
        }
    )
    optimized = ProviderUsage.from_usage(
        {
            # Illustrative values mirroring the token_savings_demo.py example (after top-k filtering)
            "prompt_tokens": 78,
            "completion_tokens": 39,
            "total_tokens": 117,
        }
    )

    savings = calculate_token_savings(baseline, optimized)

    assert savings.saved_prompt_tokens == 288
    assert savings.saved_total_tokens == 291
    assert savings.prompt_savings_pct == pytest.approx(78.6885, rel=1e-4)
    assert savings.total_savings_pct == pytest.approx(71.3235, rel=1e-4)


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
            ToolQuery(context=ConversationContext(query=prompt), limit=2, score_threshold=0.0)
        )
        retrieved_names = [scored.tool.name for scored in result.tools]
        if expected_tool in retrieved_names:
            hits += 1

    accuracy = hits / len(queries)
    # Allow a small buffer for non-deterministic simple embeddings; higher accuracy is expected
    assert accuracy >= 0.75


def test_token_savings_edge_cases() -> None:
    """Edge cases: zero baseline, optimized greater than baseline, both zero, and mixed types."""
    # Baseline zero tokens
    savings_zero = calculate_token_savings(
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    assert savings_zero.saved_prompt_tokens == 0
    assert savings_zero.saved_total_tokens == 0
    assert savings_zero.prompt_savings_pct == 0.0
    assert savings_zero.total_savings_pct == 0.0

    # Optimized exceeds baseline (clamped to zero)
    savings_negative = calculate_token_savings(
        {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
        {"prompt_tokens": 20, "completion_tokens": 0, "total_tokens": 20},
    )
    assert savings_negative.saved_prompt_tokens == 0
    assert savings_negative.saved_total_tokens == 0

    # Raw dicts vs ProviderUsage objects mixed
    baseline = ProviderUsage.from_usage(
        {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    )
    savings_mixed = calculate_token_savings(
        baseline,
        {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    )
    assert savings_mixed.saved_prompt_tokens == 3
    assert savings_mixed.saved_total_tokens == 5


@pytest.mark.asyncio
async def test_framework_adapter_returns_top_k(sample_tools) -> None:
    """Framework adapters should emit limited OpenAI-style tool schemas for all supported frameworks."""
    gantry = AgentGantry()
    for tool in sample_tools:
        await gantry.add_tool(tool)

    for framework in [
        "langgraph",
        "semantic-kernel",
        "crew_ai",
        "google_adk",
        "strands",
    ]:
        tools = await fetch_framework_tools(
            gantry,
            "send a quick email",
            framework=framework,
            limit=1,
            score_threshold=0.0,
        )

        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "send_email"


def test_framework_adapter_unsupported_framework_raises(gantry) -> None:
    with pytest.raises(ValueError):
        asyncio.run(
            fetch_framework_tools(
                gantry,
                "test",
                framework="unsupported",  # type: ignore[arg-type]
            )
        )


def test_provider_usage_coerce_float_error() -> None:
    """Test that _coerce_token_value raises ValueError on non-integer floats."""
    with pytest.raises(ValueError, match="must be an integer token count, got non-integer float"):
        ProviderUsage._coerce_token_value(3.14, "prompt_tokens")


def test_provider_usage_anthropic_naming() -> None:
    """Test that Anthropic-style usage blocks are properly parsed."""
    usage = ProviderUsage.from_usage(
        {
            "input_tokens": 150,
            "output_tokens": 50,
        }
    )
    assert usage.prompt_tokens == 150
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 200


def test_provider_usage_google_naming() -> None:
    """Test that Google-style usage blocks are properly parsed."""
    usage = ProviderUsage.from_usage(
        {
            "prompt_token_count": 100,
            "candidates_token_count": 25,
            "total_token_count": 125,
        }
    )
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 25
    assert usage.total_tokens == 125


def test_provider_usage_derived_total() -> None:
    """Test that total_tokens is properly derived if missing."""
    usage = ProviderUsage.from_usage(
        {
            "prompt_tokens": 80,
            "completion_tokens": 20,
        }
    )
    assert usage.prompt_tokens == 80
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 100


def test_provider_usage_explicit_zero_total() -> None:
    """Test that explicitly provided total_tokens of 0 is preserved."""
    usage = ProviderUsage.from_usage(
        {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 0,
        }
    )
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 0
