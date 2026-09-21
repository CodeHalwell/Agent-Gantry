"""
Normalising token usage from different providers.

``calculate_token_savings`` accepts the raw ``usage`` dicts that Anthropic
(``input_tokens``), Google GenAI (``prompt_token_count``) and OpenAI
(``prompt_tokens``) return, maps them onto one ``ProviderUsage`` shape and
reports the saving. The numbers here are hard-coded to show the mapping; no
provider is called and no API key is needed.

Run with::

    python examples/observability/multi_provider_metrics_demo.py
"""

import asyncio

from agent_gantry import enable_console_logging
from agent_gantry.metrics.token_usage import calculate_token_savings
from agent_gantry.observability.console import ConsoleTelemetryAdapter

# Opt in to console output so record_token_usage() lines are visible.
enable_console_logging()


async def test_multi_provider_metrics():
    telemetry = ConsoleTelemetryAdapter()

    print("=== Multi-Provider Token Savings Test ===\n")

    # 1. Anthropic Format (input_tokens, output_tokens)
    print("--- Testing Anthropic Format ---")
    anthropic_baseline = {"input_tokens": 1200, "output_tokens": 50}
    anthropic_optimized = {"input_tokens": 150, "output_tokens": 60}

    savings_anthropic = calculate_token_savings(anthropic_baseline, anthropic_optimized)
    print(f"Anthropic Savings: {savings_anthropic.prompt_savings_pct:.1f}%")

    await telemetry.record_token_usage(
        usage=savings_anthropic.optimized, model_name="claude-sonnet-4-6", savings=savings_anthropic
    )

    # 2. Google GenAI Format (prompt_token_count, candidates_token_count)
    print("\n--- Testing Google GenAI Format ---")
    google_baseline = {
        "prompt_token_count": 2000,
        "candidates_token_count": 100,
        "total_token_count": 2100,
    }
    google_optimized = {
        "prompt_token_count": 200,
        "candidates_token_count": 110,
        "total_token_count": 310,
    }

    savings_google = calculate_token_savings(google_baseline, google_optimized)
    print(f"Google Savings: {savings_google.prompt_savings_pct:.1f}%")

    await telemetry.record_token_usage(
        usage=savings_google.optimized, model_name="gemini-2.5-flash", savings=savings_google
    )

    # 3. OpenAI Format (prompt_tokens, completion_tokens)
    print("\n--- Testing OpenAI Format ---")
    openai_baseline = {"prompt_tokens": 800, "completion_tokens": 40}
    openai_optimized = {"prompt_tokens": 80, "completion_tokens": 45}

    savings_openai = calculate_token_savings(openai_baseline, openai_optimized)
    print(f"OpenAI Savings: {savings_openai.prompt_savings_pct:.1f}%")

    await telemetry.record_token_usage(
        usage=savings_openai.optimized, model_name="gpt-5.5", savings=savings_openai
    )

    print("\n✅ Multi-provider metrics test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_multi_provider_metrics())
