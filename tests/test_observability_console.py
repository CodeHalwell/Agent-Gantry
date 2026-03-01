"""
Tests for ConsoleTelemetryAdapter methods.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from agent_gantry.observability.console import ConsoleTelemetryAdapter
from agent_gantry.schema.execution import ExecutionStatus, ToolCall, ToolResult
from agent_gantry.schema.query import ConversationContext, RetrievalResult, ScoredTool, ToolQuery
from agent_gantry.schema.tool import ToolDefinition, ToolHealth


class TestConsoleTelemetryAdapterMethods:
    """Tests for ConsoleTelemetryAdapter methods."""

    @pytest.mark.asyncio
    async def test_record_retrieval(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test record_retrieval logs expected information."""
        adapter = ConsoleTelemetryAdapter(log_level=logging.INFO)

        context = ConversationContext(query="What is the weather?")
        query = ToolQuery(context=context, limit=3)

        # Create a mock tool for RetrievalResult
        tool = ToolDefinition(
            name="get_weather",
            description="Get the current weather",
            parameters_schema={"type": "object", "properties": {}},
        )
        scored_tool = ScoredTool(tool=tool, semantic_score=0.9)

        result = RetrievalResult(
            tools=[scored_tool],
            query_embedding_time_ms=10.0,
            vector_search_time_ms=5.0,
            total_time_ms=15.0,
            candidate_count=5,
            filtered_count=2,
            trace_id="test-trace-retrieval",
        )

        await adapter.record_retrieval(query, result)

        assert len(caplog.records) > 0
        log_record = caplog.records[-1]
        assert log_record.message == "Tool retrieval"
        assert log_record.event_type == "retrieval"
        assert log_record.query == "What is the weather?"
        assert log_record.limit == 3
        assert log_record.tools_found == 1
        assert log_record.total_time_ms == 15.0
        assert log_record.trace_id == "test-trace-retrieval"

    @pytest.mark.asyncio
    async def test_record_execution_success(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test record_execution for a successful execution."""
        adapter = ConsoleTelemetryAdapter(log_level=logging.INFO)

        call = ToolCall(tool_name="get_weather", arguments={"location": "Tokyo"})

        now = datetime.now(timezone.utc)
        result = ToolResult(
            tool_name="get_weather",
            status=ExecutionStatus.SUCCESS,
            queued_at=now,
            started_at=now,
            completed_at=now,
            attempt_number=1,
            trace_id="test-trace-exec",
            span_id="test-span-exec",
            result={"temperature": "25C"},
        )

        await adapter.record_execution(call, result)

        assert len(caplog.records) > 0
        log_record = caplog.records[-1]
        assert log_record.levelno == logging.INFO
        assert log_record.message == "Tool execution"
        assert log_record.event_type == "execution"
        assert log_record.tool_name == "get_weather"
        assert log_record.status == ExecutionStatus.SUCCESS.value
        assert hasattr(log_record, "latency_ms")
        assert log_record.attempt_number == 1
        assert log_record.trace_id == "test-trace-exec"

    @pytest.mark.asyncio
    async def test_record_execution_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test record_execution for an execution with an error."""
        adapter = ConsoleTelemetryAdapter(log_level=logging.INFO)

        call = ToolCall(tool_name="get_weather", arguments={"location": "Tokyo"})

        now = datetime.now(timezone.utc)
        result = ToolResult(
            tool_name="get_weather",
            status=ExecutionStatus.FAILURE,
            queued_at=now,
            started_at=now,
            completed_at=now,
            attempt_number=2,
            trace_id="test-trace-error",
            span_id="test-span-error",
            error="Connection timed out",
            error_type="TimeoutError",
        )

        await adapter.record_execution(call, result)

        assert len(caplog.records) > 0
        log_record = caplog.records[-1]
        assert log_record.levelno == logging.ERROR
        assert log_record.message == "Tool execution"
        assert log_record.event_type == "execution"
        assert log_record.tool_name == "get_weather"
        assert log_record.status == ExecutionStatus.FAILURE.value
        assert log_record.attempt_number == 2
        assert log_record.trace_id == "test-trace-error"
        assert log_record.error == "Connection timed out"
        assert log_record.error_type == "TimeoutError"

    @pytest.mark.asyncio
    async def test_record_health_change(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test record_health_change logs normal health changes."""
        adapter = ConsoleTelemetryAdapter(log_level=logging.INFO)

        old_health = ToolHealth(success_rate=1.0, consecutive_failures=0)
        new_health = ToolHealth(success_rate=0.8, consecutive_failures=1)

        await adapter.record_health_change("get_weather", old_health, new_health)

        assert len(caplog.records) > 0
        log_record = caplog.records[-1]
        assert log_record.levelno == logging.INFO
        assert log_record.message == "Tool health changed"
        assert log_record.event_type == "health_change"
        assert log_record.tool_name == "get_weather"
        assert log_record.old_success_rate == 1.0
        assert log_record.new_success_rate == 0.8
        assert log_record.consecutive_failures == 1
        assert log_record.circuit_breaker_open is False

    @pytest.mark.asyncio
    async def test_record_health_change_circuit_breaker(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test record_health_change logs with WARNING when circuit breaker opens."""
        adapter = ConsoleTelemetryAdapter(log_level=logging.INFO)

        old_health = ToolHealth(success_rate=0.5, consecutive_failures=2)
        new_health = ToolHealth(success_rate=0.3, consecutive_failures=3, circuit_breaker_open=True)

        await adapter.record_health_change("get_weather", old_health, new_health)

        assert len(caplog.records) > 0
        log_record = caplog.records[-1]
        assert log_record.levelno == logging.WARNING
        assert log_record.message == "Tool health changed"
        assert log_record.tool_name == "get_weather"
        assert log_record.circuit_breaker_open is True

    @pytest.mark.asyncio
    async def test_span(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test span context manager logs start and complete events."""
        adapter = ConsoleTelemetryAdapter(log_level=logging.INFO)

        async with adapter.span("test_span", attributes={"key": "value"}):
            pass

        assert len(caplog.records) >= 2

        # Check start log
        start_log = caplog.records[-2]
        assert start_log.message == "Span started"
        assert start_log.span_name == "test_span"
        assert start_log.attributes == {"key": "value"}
        assert hasattr(start_log, "timestamp")

        # Check complete log
        complete_log = caplog.records[-1]
        assert complete_log.message == "Span completed"
        assert complete_log.span_name == "test_span"
        assert hasattr(complete_log, "duration_ms")
        assert hasattr(complete_log, "timestamp")

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        """Test health_check method."""
        adapter = ConsoleTelemetryAdapter()
        assert await adapter.health_check() is True
