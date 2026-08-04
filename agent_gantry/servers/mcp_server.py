"""
MCP Server implementation for Agent-Gantry.

Exposes AgentGantry as an MCP server with dynamic tool discovery.

Supports both mcp 1.x and mcp 2.x. The wire types (``Tool``, ``TextContent``)
and the stdio transport (``stdio_server`` + ``Server.run``) are identical
across both major versions; the one breaking seam is handler registration —
mcp 2.0 removed the ``@server.list_tools()`` / ``@server.call_tool()``
decorators in favour of constructor callbacks (``on_list_tools`` /
``on_call_tool``) whose handlers take ``(ctx, params)`` and return full
result models (``ListToolsResult`` / ``CallToolResult``), with no automatic
wrapping of returned content lists or raised exceptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mcp
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from agent_gantry.schema.query import ConversationContext, ToolQuery

if TYPE_CHECKING:
    from agent_gantry import AgentGantry

# mcp 2.x introduced the high-level `Client`; 1.x has no such attribute.
_MCP_V2 = hasattr(mcp, "Client")


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
        if _MCP_V2:
            self.server = Server(
                name,
                on_list_tools=self._on_list_tools_v2,
                on_call_tool=self._on_call_tool_v2,
            )
        else:
            self.server = Server(name)
            self._setup_v1_handlers()

    async def _list_tools(self) -> list[Tool]:
        """List available tools based on mode (shared across mcp versions)."""
        if self.mode == "static":
            # Static mode: expose all tools directly
            tools = await self.gantry.list_tools()
            return [self._convert_tool(tool) for tool in tools]
        # Dynamic and hybrid modes: expose meta-tools for context savings
        # (hybrid could add common tools later)
        return self._get_meta_tools()

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> list[Any]:
        """Dispatch a tool call (shared across mcp versions)."""
        if name == "find_relevant_tools":
            return await self._handle_find_relevant_tools(arguments)
        elif name == "execute_tool":
            return await self._handle_execute_tool(arguments)
        else:
            # Direct tool execution (static mode)
            return await self._handle_execute_tool({"tool_name": name, "arguments": arguments})

    def _setup_v1_handlers(self) -> None:
        """Register handlers via the mcp 1.x decorators."""

        @self.server.list_tools()  # type: ignore[misc, no-untyped-call]
        async def list_tools() -> list[Tool]:
            return await self._list_tools()

        @self.server.call_tool()  # type: ignore[misc, no-untyped-call]
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
            return await self._call_tool(name, arguments)

    async def _on_list_tools_v2(self, ctx: Any, params: Any) -> Any:
        """mcp 2.x list-tools handler: wrap into a full ListToolsResult."""
        import mcp.types as types

        return types.ListToolsResult(tools=await self._list_tools())

    async def _on_call_tool_v2(self, ctx: Any, params: Any) -> Any:
        """mcp 2.x call-tool handler.

        Unlike 1.x, the SDK no longer wraps returned content lists or raised
        exceptions — return a full CallToolResult and mark failures with
        ``is_error`` ourselves.
        """
        import mcp.types as types

        try:
            content = await self._call_tool(params.name, params.arguments or {})
            return types.CallToolResult(content=content)
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error: {e}")],
                is_error=True,
            )

    def _get_meta_tools(self) -> list[Tool]:
        """Return the meta-tools for dynamic tool discovery."""
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

    async def _handle_find_relevant_tools(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
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

    async def _handle_execute_tool(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
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
