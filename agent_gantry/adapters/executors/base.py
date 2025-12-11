"""
Base executor adapter protocol.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agent_gantry.schema.execution import ToolCall, ToolResult
    from agent_gantry.schema.tool import ToolDefinition, ToolSource


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

    @abstractmethod
    async def validate_arguments(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """
        Validate arguments for a tool call.

        Args:
            tool: The tool definition
            arguments: Arguments to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        ...

    @abstractmethod
    def supports_source(self, source: ToolSource) -> bool:
        """
        Check if this executor supports a tool source.

        Args:
            source: The tool source type

        Returns:
            True if supported
        """
        ...
