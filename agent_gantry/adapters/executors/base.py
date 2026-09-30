"""
Base executor adapter protocol.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agent_gantry.schema.execution import ToolCall, ToolResult
    from agent_gantry.schema.tool import ToolDefinition


class ExecutorAdapter(Protocol):
    """
    Execution backend for tools.

    Implementations: DirectExecutor, SandboxExecutor, DockerExecutor,
                     MCPExecutor, A2AExecutor, HTTPExecutor.
    """

    @abstractmethod
    async def execute(
        self,
        tool: ToolDefinition,
        call: ToolCall,
        handler: Callable[..., Awaitable[Any]] | None = None,
    ) -> ToolResult:
        """
        Execute a tool call.

        Args:
            tool: The tool definition
            call: The tool call to execute
            handler: Optional handler function

        Returns:
            Result of the execution
        """
        ...
