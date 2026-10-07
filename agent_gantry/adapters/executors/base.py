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

    The implementations in this package are ``A2AExecutor`` and the MCP client in
    ``mcp_client.py``. In-process Python handlers are run by ``ExecutionEngine``
    itself; there is no sandboxed, containerised or HTTP executor.
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
