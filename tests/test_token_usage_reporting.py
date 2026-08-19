"""Provider-reported token usage must actually reach telemetry.

The library's headline claim is a smaller prompt, but nothing measured it:
``TokenUsageEvent`` was defined and never constructed, and
``record_token_usage`` existed on the telemetry protocol and both adapters yet
was never called by library code — only by examples and tests. Every provider
these layers target returns a ``usage`` block, so the actual cost of each call
is now recorded.

Savings are deliberately *not* inferred here: that needs a real baseline (the
same prompt with every tool injected), and ``agent_gantry.metrics.token_usage``
refuses approximate estimators so reported numbers stay auditable.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from agent_gantry import AgentGantry
from agent_gantry.metrics.token_usage import ProviderUsage, calculate_token_savings


class _RecordingTelemetry:
    """Captures ``record_token_usage``; no-ops the rest of the protocol.

    The other members are spelled out rather than caught by ``__getattr__`` so
    ``span`` keeps its async-context-manager shape instead of returning a
    coroutine nobody awaits.
    """

    def __init__(self) -> None:
        self.usages: list[tuple[ProviderUsage, str]] = []

    async def record_token_usage(
        self,
        usage: ProviderUsage,
        model_name: str,
        savings: Any = None,
        trace_id: str | None = None,
    ) -> None:
        self.usages.append((usage, model_name))

    @asynccontextmanager
    async def span(self, *args: Any, **kwargs: Any) -> AsyncIterator[None]:
        yield

    async def record_retrieval(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def record_execution(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def record_health_change(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def health_check(self) -> bool:
        return True


class _OpenAIUsage:
    prompt_tokens = 420
    completion_tokens = 80
    total_tokens = 500


class _OpenAIResponse:
    model = "gpt-4.1"
    usage = _OpenAIUsage()


class TestCacheTokenAccounting:
    """Anthropic reports cached prompt tokens outside ``input_tokens``."""

    def test_cache_tokens_count_towards_the_prompt(self) -> None:
        usage = ProviderUsage.from_usage(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 900,
                "cache_read_input_tokens": 2000,
            }
        )

        assert usage.prompt_tokens == 3000
        assert usage.cached_prompt_tokens == 2900
        assert usage.total_tokens == 3050

    def test_openai_usage_is_unaffected(self) -> None:
        usage = ProviderUsage.from_usage({"prompt_tokens": 500, "completion_tokens": 20})

        assert usage.prompt_tokens == 500
        assert usage.cached_prompt_tokens == 0

    def test_ignoring_cache_tokens_inflated_savings(self) -> None:
        """Regression: a cached run must not look nearly free next to a baseline."""
        savings = calculate_token_savings(
            {"input_tokens": 5000},
            {"input_tokens": 100, "cache_read_input_tokens": 2000},
        )

        # Counting the 2000 cached tokens gives 58%; ignoring them gave 98%.
        assert 55 < savings.prompt_savings_pct < 60
        assert savings.optimized.prompt_tokens == 2100


class TestDecoratorRecordsUsage:
    @pytest.fixture
    def gantry(self) -> AgentGantry:
        return AgentGantry(telemetry=_RecordingTelemetry())

    async def test_usage_is_reported_after_the_call(self, gantry: AgentGantry) -> None:
        from agent_gantry.integrations.semantic_tools import with_semantic_tools

        @with_semantic_tools(gantry)
        async def generate(prompt: str, *, tools: list[Any] | None = None) -> Any:
            return _OpenAIResponse()

        await generate("what is the weather")

        recorded = gantry.telemetry.usages
        assert len(recorded) == 1
        usage, model = recorded[0]
        assert usage.prompt_tokens == 420
        assert usage.completion_tokens == 80
        assert model == "gpt-4.1"

    async def test_a_response_without_usage_is_not_an_error(
        self, gantry: AgentGantry
    ) -> None:
        from agent_gantry.integrations.semantic_tools import with_semantic_tools

        @with_semantic_tools(gantry)
        async def generate(prompt: str, *, tools: list[Any] | None = None) -> str:
            return "a plain string response"

        assert await generate("hello") == "a plain string response"
        assert gantry.telemetry.usages == []

    async def test_accounting_failure_never_breaks_the_call(
        self, gantry: AgentGantry
    ) -> None:
        """Telemetry is observability; it must not take a user's request down."""
        from agent_gantry.integrations.semantic_tools import with_semantic_tools

        async def exploding(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("telemetry backend down")

        gantry.telemetry.record_token_usage = exploding  # type: ignore[method-assign]

        @with_semantic_tools(gantry)
        async def generate(prompt: str, *, tools: list[Any] | None = None) -> Any:
            return _OpenAIResponse()

        assert isinstance(await generate("hello"), _OpenAIResponse)

    async def test_dict_shaped_usage_is_understood(self, gantry: AgentGantry) -> None:
        from agent_gantry.integrations.semantic_tools import with_semantic_tools

        @with_semantic_tools(gantry)
        async def generate(prompt: str, *, tools: list[Any] | None = None) -> dict[str, Any]:
            return {"model": "x", "usage": {"prompt_tokens": 10, "completion_tokens": 2}}

        await generate("hello")

        assert gantry.telemetry.usages[0][0].prompt_tokens == 10


class TestGantryExposesTelemetry:
    def test_telemetry_is_reachable_without_touching_privates(self) -> None:
        adapter = _RecordingTelemetry()
        assert AgentGantry(telemetry=adapter).telemetry is adapter
