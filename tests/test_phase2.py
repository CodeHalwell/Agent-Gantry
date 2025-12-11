"""
Tests for Phase 2: Execution Engine, Security, and Health.

Tests circuit breaker, retries, timeouts, permissions, validation, and async operations.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_gantry import AgentGantry
from agent_gantry.core.security import (
    ConfirmationRequired,
    PermissionChecker,
    SecurityPolicy,
)
from agent_gantry.schema.execution import ExecutionStatus, ToolCall
from agent_gantry.schema.tool import ToolCapability


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, gantry: AgentGantry) -> None:
        """Test that circuit breaker opens after consecutive failures."""

        @gantry.register
        def failing_tool(x: int) -> int:
            """A tool that always fails for testing circuit breaker."""
            raise ValueError("Intentional failure")

        await gantry.sync()

        # Execute multiple times to trigger circuit breaker (threshold is 5 by default)
        for i in range(6):
            result = await gantry.execute(
                ToolCall(tool_name="failing_tool", arguments={"x": i}, retry_count=0)
            )
            if i < 5:
                assert result.status == ExecutionStatus.FAILURE
            else:
                # Circuit should open on 6th attempt
                assert result.status == ExecutionStatus.CIRCUIT_OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovers(self, gantry: AgentGantry) -> None:
        """Test that circuit breaker allows recovery attempts after timeout."""
        from agent_gantry.core.executor import ExecutionEngine

        # Use a custom executor with very short timeout for testing
        custom_executor = ExecutionEngine(
            registry=gantry._registry,
            circuit_breaker_threshold=3,
            circuit_breaker_timeout_s=1,  # 1 second timeout
            security_policy=gantry._security_policy,
            telemetry=gantry._telemetry,
        )
        gantry._executor = custom_executor

        @gantry.register
        def recovering_tool(x: int) -> int:
            """A tool that always fails for testing recovery."""
            raise ValueError("Failing")

        await gantry.sync()

        # Trigger circuit breaker (3 failures)
        for i in range(3):
            await gantry.execute(
                ToolCall(tool_name="recovering_tool", arguments={"x": i}, retry_count=0)
            )

        # Circuit should be open now
        result = await gantry.execute(
            ToolCall(tool_name="recovering_tool", arguments={"x": 10}, retry_count=0)
        )
        assert result.status == ExecutionStatus.CIRCUIT_OPEN

        # Wait for circuit breaker timeout
        await asyncio.sleep(1.1)

        # Should attempt execution again (will fail but circuit is attempting recovery)
        result = await gantry.execute(
            ToolCall(tool_name="recovering_tool", arguments={"x": 10}, retry_count=0)
        )
        # After timeout, it should try again and fail (not be circuit_open)
        assert result.status == ExecutionStatus.FAILURE


class TestRetryAndTimeout:
    """Tests for retry and timeout behavior."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, gantry: AgentGantry) -> None:
        """Test that failed executions are retried."""

        attempt_count = 0

        @gantry.register
        def flaky_tool(x: int) -> int:
            """A tool that fails twice then succeeds."""
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary failure")
            return x * 2

        await gantry.sync()

        result = await gantry.execute(
            ToolCall(tool_name="flaky_tool", arguments={"x": 5}, retry_count=3)
        )
        assert result.status == ExecutionStatus.SUCCESS
        assert result.result == 10
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_timeout_handling(self, gantry: AgentGantry) -> None:
        """Test that long-running tools timeout correctly."""

        @gantry.register
        async def slow_tool(x: int) -> int:
            """A tool that takes too long."""
            await asyncio.sleep(2)
            return x * 2

        await gantry.sync()

        result = await gantry.execute(
            ToolCall(
                tool_name="slow_tool",
                arguments={"x": 5},
                timeout_ms=500,  # 500ms timeout
                retry_count=0,
            )
        )
        assert result.status == ExecutionStatus.TIMEOUT
        assert result.error_type == "TimeoutError"


class TestSecurityPolicy:
    """Tests for security policy enforcement."""

    @pytest.mark.asyncio
    async def test_destructive_tool_requires_confirmation(self, gantry: AgentGantry) -> None:
        """Test that destructive tools require confirmation."""

        @gantry.register(requires_confirmation=True)
        def delete_user(user_id: str) -> str:
            """Delete a user from the system."""
            return f"Deleted user {user_id}"

        await gantry.sync()

        result = await gantry.execute(
            ToolCall(tool_name="delete_user", arguments={"user_id": "123"})
        )
        assert result.status == ExecutionStatus.PENDING_CONFIRMATION

    @pytest.mark.asyncio
    async def test_security_policy_pattern_matching(self) -> None:
        """Test that security policy matches tool name patterns."""
        policy = SecurityPolicy(require_confirmation=["delete_*", "payment_*"])

        with pytest.raises(ConfirmationRequired):
            policy.check_permission("delete_user", {})

        with pytest.raises(ConfirmationRequired):
            policy.check_permission("payment_process", {})

        # Should not raise for non-matching patterns
        policy.check_permission("get_user", {})

    @pytest.mark.asyncio
    async def test_confirmation_override(self, gantry: AgentGantry) -> None:
        """Test that confirmation can be explicitly set in tool call."""

        @gantry.register
        def safe_tool(x: int) -> int:
            """A safe tool that doesn't normally require confirmation."""
            return x * 2

        await gantry.sync()

        # Force confirmation even though tool doesn't require it
        result = await gantry.execute(
            ToolCall(
                tool_name="safe_tool", arguments={"x": 5}, require_confirmation=True
            )
        )
        assert result.status == ExecutionStatus.PENDING_CONFIRMATION


class TestPermissionChecker:
    """Tests for capability-based permission checking."""

    def test_permission_checker_allows_with_capabilities(self) -> None:
        """Test that users with required capabilities can use tools."""
        from agent_gantry.schema.tool import ToolDefinition

        tool = ToolDefinition(
            name="test_tool",
            description="A tool requiring read access to test permissions.",
            parameters_schema={"type": "object", "properties": {}, "required": []},
            capabilities=[ToolCapability.READ_DATA],
        )

        checker = PermissionChecker([ToolCapability.READ_DATA, ToolCapability.WRITE_DATA])
        can_use, error = checker.can_use(tool)
        assert can_use
        assert error is None

    def test_permission_checker_denies_without_capabilities(self) -> None:
        """Test that users without required capabilities cannot use tools."""
        from agent_gantry.schema.tool import ToolDefinition

        tool = ToolDefinition(
            name="test_tool",
            description="A tool requiring write access to test permission denial.",
            parameters_schema={"type": "object", "properties": {}, "required": []},
            capabilities=[ToolCapability.WRITE_DATA, ToolCapability.DELETE_DATA],
        )

        checker = PermissionChecker([ToolCapability.READ_DATA])
        can_use, error = checker.can_use(tool)
        assert not can_use
        assert "Missing capabilities" in error
        assert "write_data" in error.lower()

    def test_permission_checker_filters_tools(self) -> None:
        """Test that permission checker filters tool lists correctly."""
        from agent_gantry.schema.tool import ToolDefinition

        tools = [
            ToolDefinition(
                name="read_tool",
                description="A tool for reading data in permission filter tests.",
                parameters_schema={"type": "object", "properties": {}, "required": []},
                capabilities=[ToolCapability.READ_DATA],
            ),
            ToolDefinition(
                name="write_tool",
                description="A tool for writing data in permission filter tests.",
                parameters_schema={"type": "object", "properties": {}, "required": []},
                capabilities=[ToolCapability.WRITE_DATA],
            ),
            ToolDefinition(
                name="delete_tool",
                description="A tool for deleting data in permission filter tests.",
                parameters_schema={"type": "object", "properties": {}, "required": []},
                capabilities=[ToolCapability.DELETE_DATA],
            ),
        ]

        checker = PermissionChecker([ToolCapability.READ_DATA, ToolCapability.WRITE_DATA])
        allowed = checker.filter_tools(tools)
        assert len(allowed) == 2
        assert all(t.name in ["read_tool", "write_tool"] for t in allowed)


class TestArgumentValidation:
    """Tests for argument validation."""

    @pytest.mark.asyncio
    async def test_missing_required_argument(self, gantry: AgentGantry) -> None:
        """Test that missing required arguments are caught."""

        @gantry.register
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together for validation testing."""
            return a + b

        await gantry.sync()

        result = await gantry.execute(
            ToolCall(tool_name="add_numbers", arguments={"a": 5})  # Missing 'b'
        )
        assert result.status == ExecutionStatus.FAILURE
        assert "missing required parameter" in result.error.lower()

    @pytest.mark.asyncio
    async def test_wrong_argument_type(self, gantry: AgentGantry) -> None:
        """Test that wrong argument types are caught."""

        @gantry.register
        def multiply(x: int, y: int) -> int:
            """Multiply two integers for type validation testing."""
            return x * y

        await gantry.sync()

        result = await gantry.execute(
            ToolCall(tool_name="multiply", arguments={"x": "not_a_number", "y": 5})
        )
        assert result.status == ExecutionStatus.FAILURE
        assert "must be an integer" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unknown_argument(self, gantry: AgentGantry) -> None:
        """Test that unknown arguments are caught."""

        @gantry.register
        def simple_tool(x: int) -> int:
            """A simple tool for testing unknown argument detection."""
            return x * 2

        await gantry.sync()

        result = await gantry.execute(
            ToolCall(
                tool_name="simple_tool", arguments={"x": 5, "unknown_param": "value"}
            )
        )
        assert result.status == ExecutionStatus.FAILURE
        assert "unknown parameter" in result.error.lower()


class TestAsyncOperations:
    """Tests for async tool execution."""

    @pytest.mark.asyncio
    async def test_async_tool_execution(self, gantry: AgentGantry) -> None:
        """Test that async tools execute correctly."""

        @gantry.register
        async def async_tool(x: int) -> int:
            """An async tool for testing async execution."""
            await asyncio.sleep(0.1)
            return x * 2

        await gantry.sync()

        result = await gantry.execute(
            ToolCall(tool_name="async_tool", arguments={"x": 5})
        )
        assert result.status == ExecutionStatus.SUCCESS
        assert result.result == 10

    @pytest.mark.asyncio
    async def test_sync_tool_execution(self, gantry: AgentGantry) -> None:
        """Test that sync tools execute correctly in async context."""

        @gantry.register
        def sync_tool(x: int) -> int:
            """A sync tool for testing sync execution in async context."""
            return x * 3

        await gantry.sync()

        result = await gantry.execute(
            ToolCall(tool_name="sync_tool", arguments={"x": 5})
        )
        assert result.status == ExecutionStatus.SUCCESS
        assert result.result == 15

    @pytest.mark.asyncio
    async def test_batch_execution(self, gantry: AgentGantry) -> None:
        """Test batch execution of multiple tools."""
        from agent_gantry.schema.execution import BatchToolCall

        @gantry.register
        def tool_one(x: int) -> int:
            """First tool for batch execution testing."""
            return x + 1

        @gantry.register
        def tool_two(x: int) -> int:
            """Second tool for batch execution testing."""
            return x * 2

        await gantry.sync()

        batch = BatchToolCall(
            calls=[
                ToolCall(tool_name="tool_one", arguments={"x": 5}),
                ToolCall(tool_name="tool_two", arguments={"x": 10}),
            ]
        )

        result = await gantry.execute_batch(batch)
        assert result.successful_count == 2
        assert result.failed_count == 0
        assert len(result.results) == 2


class TestHealthTracking:
    """Tests for health metrics tracking."""

    @pytest.mark.asyncio
    async def test_health_metrics_update_on_success(self, gantry: AgentGantry) -> None:
        """Test that health metrics update correctly on success."""

        @gantry.register
        def success_tool(x: int) -> int:
            """A tool that succeeds for health metrics testing."""
            return x * 2

        await gantry.sync()
        tool = await gantry.get_tool("success_tool")
        assert tool is not None

        # Initial health state
        assert tool.health.total_calls == 0
        assert tool.health.success_rate == 1.0

        # Execute successfully
        result = await gantry.execute(
            ToolCall(tool_name="success_tool", arguments={"x": 5})
        )
        assert result.status == ExecutionStatus.SUCCESS

        # Check health updated
        tool_after = await gantry.get_tool("success_tool")
        assert tool_after is not None
        assert tool_after.health.total_calls == 1
        assert tool_after.health.success_rate == 1.0
        assert tool_after.health.consecutive_failures == 0
        assert not tool_after.health.circuit_breaker_open

    @pytest.mark.asyncio
    async def test_health_metrics_update_on_failure(self, gantry: AgentGantry) -> None:
        """Test that health metrics update correctly on failure."""

        @gantry.register
        def failing_tool(x: int) -> int:
            """A tool that fails for health metrics testing."""
            raise ValueError("Intentional failure")

        await gantry.sync()

        # Execute and fail
        result = await gantry.execute(
            ToolCall(tool_name="failing_tool", arguments={"x": 5}, retry_count=0)
        )
        assert result.status == ExecutionStatus.FAILURE

        # Check health updated
        tool = await gantry.get_tool("failing_tool")
        assert tool is not None
        assert tool.health.total_calls == 1
        assert tool.health.success_rate < 1.0
        assert tool.health.consecutive_failures == 1
