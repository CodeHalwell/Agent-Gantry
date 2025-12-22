"""
Tests for observability adapter token usage tracking.
"""

from __future__ import annotations

import pytest

from agent_gantry.metrics.token_usage import ProviderUsage, TokenSavings
from agent_gantry.observability.console import ConsoleTelemetryAdapter
from agent_gantry.observability.opentelemetry_adapter import OpenTelemetryAdapter


class TestOpenTelemetryAdapterTokenUsage:
    """Tests for OpenTelemetryAdapter.record_token_usage method."""

    @pytest.mark.asyncio
    async def test_record_token_usage_basic(self) -> None:
        """Test basic token accumulation without savings."""
        adapter = OpenTelemetryAdapter(service_name="test-service")
        usage = ProviderUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        await adapter.record_token_usage(usage, model_name="gpt-4", savings=None)

        assert adapter.metrics["tokens_prompt_total"] == 100
        assert adapter.metrics["tokens_completion_total"] == 50
        assert adapter.metrics["tokens_total"] == 150
        assert "tokens_saved_prompt_total" not in adapter.metrics
        assert "avg_prompt_savings_pct" not in adapter.metrics

    @pytest.mark.asyncio
    async def test_record_token_usage_accumulation(self) -> None:
        """Test that token usage accumulates across multiple calls."""
        adapter = OpenTelemetryAdapter(service_name="test-service")
        usage1 = ProviderUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        usage2 = ProviderUsage(prompt_tokens=200, completion_tokens=75, total_tokens=275)

        await adapter.record_token_usage(usage1, model_name="gpt-4", savings=None)
        await adapter.record_token_usage(usage2, model_name="gpt-4", savings=None)

        assert adapter.metrics["tokens_prompt_total"] == 300
        assert adapter.metrics["tokens_completion_total"] == 125
        assert adapter.metrics["tokens_total"] == 425

    @pytest.mark.asyncio
    async def test_record_token_usage_with_savings_first_call(self) -> None:
        """Test token savings tracking on first call (initializes running average)."""
        adapter = OpenTelemetryAdapter(service_name="test-service")
        usage = ProviderUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        baseline = ProviderUsage(prompt_tokens=200, completion_tokens=50, total_tokens=250)
        savings = TokenSavings(
            baseline=baseline,
            optimized=usage,
            saved_prompt_tokens=100,
            saved_total_tokens=100,
            prompt_savings_pct=50.0,
            total_savings_pct=40.0,
        )

        await adapter.record_token_usage(usage, model_name="gpt-4", savings=savings)

        assert adapter.metrics["tokens_saved_prompt_total"] == 100
        assert adapter.metrics["avg_prompt_savings_pct"] == 50.0
        assert adapter.metrics["savings_count"] == 1

    @pytest.mark.asyncio
    async def test_record_token_usage_running_average(self) -> None:
        """Test running average calculation for savings percentage."""
        adapter = OpenTelemetryAdapter(service_name="test-service")

        # First call with 50% savings
        usage1 = ProviderUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        baseline1 = ProviderUsage(prompt_tokens=200, completion_tokens=50, total_tokens=250)
        savings1 = TokenSavings(
            baseline=baseline1,
            optimized=usage1,
            saved_prompt_tokens=100,
            saved_total_tokens=100,
            prompt_savings_pct=50.0,
            total_savings_pct=40.0,
        )

        await adapter.record_token_usage(usage1, model_name="gpt-4", savings=savings1)
        assert adapter.metrics["avg_prompt_savings_pct"] == 50.0
        assert adapter.metrics["savings_count"] == 1

        # Second call with 30% savings
        usage2 = ProviderUsage(prompt_tokens=140, completion_tokens=50, total_tokens=190)
        baseline2 = ProviderUsage(prompt_tokens=200, completion_tokens=50, total_tokens=250)
        savings2 = TokenSavings(
            baseline=baseline2,
            optimized=usage2,
            saved_prompt_tokens=60,
            saved_total_tokens=60,
            prompt_savings_pct=30.0,
            total_savings_pct=24.0,
        )

        await adapter.record_token_usage(usage2, model_name="gpt-4", savings=savings2)

        # Running average should be (50.0 + 30.0) / 2 = 40.0
        assert adapter.metrics["avg_prompt_savings_pct"] == 40.0
        assert adapter.metrics["savings_count"] == 2
        assert adapter.metrics["tokens_saved_prompt_total"] == 160

    @pytest.mark.asyncio
    async def test_record_token_usage_zero_count_protection(self) -> None:
        """Test division-by-zero protection when count is 0."""
        adapter = OpenTelemetryAdapter(service_name="test-service")

        # Manually set count to 0 to test the protection
        adapter.metrics["savings_count"] = 0

        usage = ProviderUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        baseline = ProviderUsage(prompt_tokens=200, completion_tokens=50, total_tokens=250)
        savings = TokenSavings(
            baseline=baseline,
            optimized=usage,
            saved_prompt_tokens=100,
            saved_total_tokens=100,
            prompt_savings_pct=50.0,
            total_savings_pct=40.0,
        )

        await adapter.record_token_usage(usage, model_name="gpt-4", savings=savings)

        # Should handle count == 0 case and initialize correctly
        assert adapter.metrics["avg_prompt_savings_pct"] == 50.0
        assert adapter.metrics["savings_count"] == 1

    @pytest.mark.asyncio
    async def test_record_token_usage_mixed_calls(self) -> None:
        """Test mixing calls with and without savings."""
        adapter = OpenTelemetryAdapter(service_name="test-service")

        # Call with savings
        usage1 = ProviderUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        baseline1 = ProviderUsage(prompt_tokens=200, completion_tokens=50, total_tokens=250)
        savings1 = TokenSavings(
            baseline=baseline1,
            optimized=usage1,
            saved_prompt_tokens=100,
            saved_total_tokens=100,
            prompt_savings_pct=50.0,
            total_savings_pct=40.0,
        )
        await adapter.record_token_usage(usage1, model_name="gpt-4", savings=savings1)

        # Call without savings
        usage2 = ProviderUsage(prompt_tokens=80, completion_tokens=40, total_tokens=120)
        await adapter.record_token_usage(usage2, model_name="gpt-4", savings=None)

        # Savings metrics should remain from first call
        assert adapter.metrics["tokens_saved_prompt_total"] == 100
        assert adapter.metrics["avg_prompt_savings_pct"] == 50.0
        assert adapter.metrics["savings_count"] == 1

        # But total tokens should include both calls
        assert adapter.metrics["tokens_prompt_total"] == 180
        assert adapter.metrics["tokens_total"] == 270


class TestConsoleTelemetryAdapterTokenUsage:
    """Tests for ConsoleTelemetryAdapter.record_token_usage method."""

    @pytest.mark.asyncio
    async def test_record_token_usage_basic(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test basic token usage logging without savings."""
        import logging

        adapter = ConsoleTelemetryAdapter(log_level=logging.INFO)
        usage = ProviderUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        await adapter.record_token_usage(
            usage, model_name="gpt-4", savings=None, trace_id="test-trace-123"
        )

        # Check that log was emitted
        assert len(caplog.records) > 0
        log_record = caplog.records[-1]
        assert log_record.message == "Token usage"
        assert log_record.event_type == "token_usage"
        assert log_record.model_name == "gpt-4"
        assert log_record.prompt_tokens == 100
        assert log_record.completion_tokens == 50
        assert log_record.total_tokens == 150
        assert log_record.trace_id == "test-trace-123"

    @pytest.mark.asyncio
    async def test_record_token_usage_with_savings(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test token usage logging with savings data."""
        import logging

        adapter = ConsoleTelemetryAdapter(log_level=logging.INFO)
        usage = ProviderUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        baseline = ProviderUsage(prompt_tokens=200, completion_tokens=50, total_tokens=250)
        savings = TokenSavings(
            baseline=baseline,
            optimized=usage,
            saved_prompt_tokens=100,
            saved_total_tokens=100,
            prompt_savings_pct=50.0,
            total_savings_pct=40.0,
        )

        await adapter.record_token_usage(
            usage, model_name="gpt-4", savings=savings, trace_id="test-trace-456"
        )

        # Check that log includes savings data
        assert len(caplog.records) > 0
        log_record = caplog.records[-1]
        assert log_record.message == "Token usage"
        assert log_record.saved_prompt_tokens == 100
        assert log_record.prompt_savings_pct == "50.0%"

    @pytest.mark.asyncio
    async def test_record_token_usage_without_trace_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test token usage logging without trace_id."""
        import logging

        adapter = ConsoleTelemetryAdapter(log_level=logging.INFO)
        usage = ProviderUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        await adapter.record_token_usage(usage, model_name="gpt-4", savings=None, trace_id=None)

        # Check that log was emitted without error
        assert len(caplog.records) > 0
        log_record = caplog.records[-1]
        assert log_record.trace_id is None

    @pytest.mark.asyncio
    async def test_record_token_usage_formatting(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that savings percentage is properly formatted."""
        import logging

        adapter = ConsoleTelemetryAdapter(log_level=logging.INFO)
        usage = ProviderUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        baseline = ProviderUsage(prompt_tokens=300, completion_tokens=50, total_tokens=350)
        savings = TokenSavings(
            baseline=baseline,
            optimized=usage,
            saved_prompt_tokens=200,
            saved_total_tokens=200,
            prompt_savings_pct=66.66666,
            total_savings_pct=57.14285,
        )

        await adapter.record_token_usage(usage, model_name="gpt-4", savings=savings)

        # Check that percentage is formatted to 1 decimal place
        log_record = caplog.records[-1]
        assert log_record.prompt_savings_pct == "66.7%"
