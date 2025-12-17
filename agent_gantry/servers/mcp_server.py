"""
MCP Server implementation for Agent-Gantry.

Exposes AgentGantry as an MCP server with dynamic tool discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server import Server  # type: ignore[import-not-found]
from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]
from mcp.types import Tool  # type: ignore[import-not-found]

from agent_gantry.schema.query import ConversationContext, ToolQuery

if TYPE_CHECKING:
    from agent_gantry import AgentGantry


class MCPServer:
    """
    MCP Server for Agent-Gantry.

    Supports three modes:
    - dynamic: Only expose meta-tools (find_relevant_tools, execute_tool)
    - static: Expose all tools directly
    - hybrid: Expose common tools + meta-tools for the rest
    """

    def __init__(
        self,
        gantry: AgentGantry,
        mode: str = "dynamic",
        name: str = "agent-gantry",
    ) -> None:
        """
        Initialize MCP server.

        Args:
            gantry: AgentGantry instance to serve
            mode: Server mode (dynamic, static, or hybrid)
            name: Server name for identification
        """
        self.gantry = gantry
        self.mode = mode
        self.name = name
        self.server = Server(name)
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup MCP server handlers."""

        @self.server.list_tools()  # type: ignore[untyped-decorator]
        async def list_tools() -> list[Tool]:
            """List available tools based on mode."""
            if self.mode == "dynamic":
                # Dynamic mode: only expose meta-tools
                return [
                    Tool(
                        name="find_relevant_tools",
                        description=(
                            "Search for tools relevant to your current task. "
                            "Use this before calling other tools to discover what's available."
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "What you're trying to accomplish",
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Max tools to return",
                                    "default": 5,
                                },
                            },
                            "required": ["query"],
                        },
                    ),
                    Tool(
                        name="execute_tool",
                        description=(
                            "Execute a tool by name. Use find_relevant_tools first "
                            "to discover available tools and their schemas."
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "tool_name": {
                                    "type": "string",
                                    "description": "Name of the tool to execute",
                                },
                                "arguments": {
                                    "type": "object",
                                    "description": "Arguments for the tool",
                                },
                            },
                            "required": ["tool_name", "arguments"],
                        },
                    ),
                ]
            elif self.mode == "static":
                # Static mode: expose all tools directly
                tools = await self.gantry.list_tools()
                return [self._convert_tool(tool) for tool in tools]
            else:  # hybrid mode
                # TODO: Implement hybrid mode logic
                # For now, fall back to dynamic mode tools
                return [
                    Tool(
                        name="find_relevant_tools",
                        description=(
                            "Search for tools relevant to your current task. "
                            "Use this before calling other tools to discover what's available."
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "What you're trying to accomplish",
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Max tools to return",
                                    "default": 5,
                                },
                            },
                            "required": ["query"],
                        },
                    ),
                    Tool(
                        name="execute_tool",
                        description=(
                            "Execute a tool by name. Use find_relevant_tools first "
                            "to discover available tools and their schemas."
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "tool_name": {
                                    "type": "string",
                                    "description": "Name of the tool to execute",
                                },
                                "arguments": {
                                    "type": "object",
                                    "description": "Arguments for the tool",
                                },
                            },
                            "required": ["tool_name", "arguments"],
                        },
                    ),
                ]

        @self.server.call_tool()  # type: ignore[untyped-decorator]
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
            """Handle tool calls."""
            if name == "find_relevant_tools":
                return await self._handle_find_relevant_tools(arguments)
            elif name == "execute_tool":
                return await self._handle_execute_tool(arguments)
            else:
                # Direct tool execution (static mode)
                return await self._handle_execute_tool(
                    {"tool_name": name, "arguments": arguments}
                )

    def _convert_tool(self, tool_def: Any) -> Tool:
        """
        Convert ToolDefinition to MCP Tool.

        Args:
            tool_def: ToolDefinition object

        Returns:
            MCP Tool object
        """
        return Tool(
            name=tool_def.name,
            description=tool_def.description,
            inputSchema=tool_def.parameters_schema,
        )

    async def _handle_find_relevant_tools(
        self, arguments: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Handle find_relevant_tools meta-tool.

        Args:
            arguments: Query and limit parameters

        Returns:
            List of tool descriptions
        """
        query = arguments.get("query", "")
        limit = arguments.get("limit", 5)

        # Use AgentGantry's semantic routing
        context = ConversationContext(query=query)
        tool_query = ToolQuery(context=context, limit=limit)
        result = await self.gantry.retrieve(tool_query)

        # Format results for MCP
        tools_info = []
        for scored_tool in result.tools:
            tool = scored_tool.tool
            tools_info.append(
                {
                    "type": "text",
                    "text": (
                        f"Tool: {tool.name}\n"
                        f"Description: {tool.description}\n"
                        f"Parameters: {tool.parameters_schema}\n"
                        f"Relevance Score: {scored_tool.semantic_score:.2f}\n"
                    ),
                }
            )

        return tools_info

    async def _handle_execute_tool(
        self, arguments: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Handle execute_tool meta-tool or direct tool execution.

        Args:
            arguments: Tool name and arguments

        Returns:
            Tool execution result
        """
        from agent_gantry.schema.execution import ToolCall

        tool_name = arguments.get("tool_name", "")
        tool_arguments = arguments.get("arguments", {})

        # Execute the tool
        call = ToolCall(tool_name=tool_name, arguments=tool_arguments)
        result = await self.gantry.execute(call)

        # Format result for MCP
        if result.status.value == "success":
            return [{"type": "text", "text": str(result.result)}]
        else:
            return [
                {
                    "type": "text",
                    "text": f"Error: {result.error or 'Unknown error'}",
                }
            ]

    async def run_stdio(self) -> None:
        """Run the server with stdio transport."""
        async with stdio_server() as (read, write):
            await self.server.run(
                read,
                write,
                self.server.create_initialization_options(),
            )

    async def run_sse(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """
        Run the server with SSE transport.

        Args:
            host: Host to bind to
            port: Port to listen on
        """
        # SSE transport implementation
        # This would typically use a web framework like FastAPI
        raise NotImplementedError("SSE transport not yet implemented")


def create_mcp_server(
    gantry: AgentGantry, mode: str = "dynamic", name: str = "agent-gantry"
) -> MCPServer:
    """
    Create an MCP server for AgentGantry.

    Args:
        gantry: AgentGantry instance to serve
        mode: Server mode (dynamic, static, or hybrid)
        name: Server name for identification

    Returns:
        MCPServer instance
    """
    return MCPServer(gantry, mode, name)
