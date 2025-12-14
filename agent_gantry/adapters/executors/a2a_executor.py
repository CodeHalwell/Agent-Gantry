"""
A2A Executor Adapter.

Executes tools that originate from A2A agents by sending tasks to the remote agent.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from agent_gantry.schema.execution import ExecutionStatus, ToolResult
from agent_gantry.schema.tool import ToolSource

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable

    from agent_gantry.schema.execution import ToolCall
    from agent_gantry.schema.tool import ToolDefinition

logger = logging.getLogger(__name__)


class A2AExecutor:
    """
    Executor for A2A agent tools.

    Sends tasks to remote A2A agents and returns results.
    """

    def __init__(self) -> None:
        """Initialize A2A executor."""
        self._clients: dict[str, Any] = {}

    def _get_client(self, tool: ToolDefinition) -> Any:
        """
        Get or create an A2A client for the tool.

        Args:
            tool: Tool definition with A2A metadata

        Returns:
            A2A client instance

        Raises:
            ValueError: If tool is not from an A2A agent
        """
        if tool.source != ToolSource.A2A_AGENT:
            raise ValueError(f"Tool {tool.name} is not from an A2A agent")

        agent_name = tool.metadata.get("a2a_agent")
        if not agent_name:
            raise ValueError(f"Tool {tool.name} missing a2a_agent metadata")

        # Lazy import to avoid circular dependency
        from agent_gantry.providers.a2a_client import A2AClient
        from agent_gantry.schema.config import A2AAgentConfig

        # Return cached client or create new one
        if agent_name not in self._clients:
            config = A2AAgentConfig(
                name=agent_name,
                url=tool.metadata["a2a_url"],
                namespace=tool.namespace,
            )
            self._clients[agent_name] = A2AClient(config)

        return self._clients[agent_name]

    async def execute(
        self,
        tool: ToolDefinition,
        call: ToolCall,
        handler: Callable[..., Awaitable[Any]] | None = None,
    ) -> ToolResult:
        """
        Execute a tool by sending a task to the A2A agent.

        Args:
            tool: Tool definition
            call: Tool call parameters
            handler: Optional direct handler (not used for A2A)

        Returns:
            Tool result
        """
        trace_id = call.trace_id or "unknown"
        span_id = f"a2a-{trace_id}"
        queued_at = datetime.now(timezone.utc)
        started_at = datetime.now(timezone.utc)

        try:
            # Get A2A client
            client = self._get_client(tool)

            # Extract skill_id from metadata
            skill_id = tool.metadata.get("skill_id")
            if not skill_id:
                raise ValueError(f"Tool {tool.name} missing skill_id metadata")

            # Extract query from arguments
            if "query" not in call.arguments:
                raise ValueError("Missing required argument: query")
            query = call.arguments["query"]

            # Send task to A2A agent
            result = await client.send_task(
                skill_id=skill_id,
                query=query,
                metadata={"trace_id": trace_id},
            )

            completed_at = datetime.now(timezone.utc)

            return ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.SUCCESS,
                result=result,
                queued_at=queued_at,
                started_at=started_at,
                completed_at=completed_at,
                trace_id=trace_id,
                span_id=span_id,
            )

        except Exception as e:
            completed_at = datetime.now(timezone.utc)
            logger.error(f"A2A execution failed for {tool.name}: {e}")

            return ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.FAILURE,
                error=str(e),
                error_type=type(e).__name__,
                queued_at=queued_at,
                started_at=started_at,
                completed_at=completed_at,
                trace_id=trace_id,
                span_id=span_id,
            )

    async def validate_arguments(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """
        Validate arguments for an A2A tool.

        Args:
            tool: Tool definition
            arguments: Arguments to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic validation: check for required query parameter
        if "query" not in arguments:
            return False, "Missing required argument: query"

        if not isinstance(arguments["query"], str):
            return False, "Argument 'query' must be a string"

        return True, None

    def supports_source(self, source: ToolSource) -> bool:
        """
        Check if this executor supports the given source.

        Args:
            source: Tool source type

        Returns:
            True if A2A_AGENT source
        """
        return source == ToolSource.A2A_AGENT
