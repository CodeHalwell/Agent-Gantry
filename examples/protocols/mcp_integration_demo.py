"""
MCP Integration Demo for Agent-Gantry.

Walks through, without opening any connection:
1. What an MCP client sees when AgentGantry is served in dynamic mode
2. How an external MCP server is configured for ``add_mcp_server``
3. The meta-tool flow (``find_relevant_tools`` then ``execute_tool``), run
   directly against the gantry the server would wrap
4. How much schema text dynamic mode keeps out of the prompt, measured

No API key and no MCP server needed. Nothing here starts a server or spawns
one — the calls that would (``serve_mcp``, ``add_mcp_server``) are printed,
not run. To actually serve, use ``agent-gantry serve-mcp`` (see
``claude_desktop_config.json``) or uncomment the line shown.

Run with::

    python examples/protocols/mcp_integration_demo.py
"""

import asyncio
import json

from agent_gantry import AgentGantry
from agent_gantry.schema.config import MCPServerConfig
from agent_gantry.schema.query import ConversationContext, ToolQuery


async def demo_mcp_server():
    """Demo: Serve AgentGantry as an MCP server in dynamic mode."""
    print("\n=== MCP Server Demo (Dynamic Mode) ===\n")

    # Create a gantry instance with some tools
    gantry = AgentGantry()
    try:

        @gantry.register(examples=["add 3 and 4", "what is 10 plus 5"])
        def calculate_sum(a: int, b: int) -> int:
            """Calculate the sum of two numbers."""
            return a + b

        @gantry.register(examples=["multiply 6 by 7", "what is 12 times 3"])
        def calculate_product(a: int, b: int) -> int:
            """Calculate the product of two numbers."""
            return a * b

        @gantry.register(examples=["what's the weather in London", "is it raining in Leeds"])
        def get_weather(city: str) -> str:
            """Get the current weather for a city."""
            return f"Weather in {city}: Sunny, 72°F"

        await gantry.sync()

        print(f"Registered {gantry.tool_count} tools")
        print("\nIn dynamic mode, MCP clients see only 2 meta-tools:")
        print("  1. find_relevant_tools - Discover tools by query")
        print("  2. execute_tool - Execute discovered tools")
        print("\nThis minimizes context window usage!")
        print("\nTo start the MCP server, uncomment the following line:")
        print("# await gantry.serve_mcp(transport='stdio', mode='dynamic')")
    finally:
        await gantry.close()


async def demo_mcp_client():
    """Demo: Connect to external MCP servers as a client."""
    print("\n=== MCP Client Demo ===\n")

    gantry = AgentGantry()
    try:
        # Example configuration for connecting to an external MCP server
        # Note: This requires an actual MCP server to be running
        config = MCPServerConfig(
            name="example-server",
            command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
            args=["/tmp"],  # the directories the server may access, as positionals
            namespace="filesystem",
        )

        print(f"Configuration for connecting to: {config.name}")
        print(f"Command: {' '.join(config.command + config.args)}")
        print(f"Namespace: {config.namespace}")
        print("\nTo actually connect, uncomment the following lines:")
        print("# count = await gantry.add_mcp_server(config)")
        print("# print(f'Discovered {count} tools from MCP server')")

        # Mock demonstration of what would happen
        print("\n--- What happens when connected: ---")
        print("1. MCPClient connects via stdio subprocess")
        print("2. Performs MCP handshake (initialize/initialized)")
        print("3. Calls tools/list to discover available tools")
        print("4. Converts each MCP tool to ToolDefinition")
        print("5. Registers tools in AgentGantry's vector store")
        print("6. Tools become available for semantic routing!")
    finally:
        await gantry.close()


async def demo_meta_tool_flow():
    """Demo: Meta-tool discovery and execution flow."""
    print("\n=== Meta-Tool Discovery Flow Demo ===\n")

    gantry = AgentGantry()
    try:
        # Register various tools
        @gantry.register(
            tags=["math", "calculation"], examples=["add 3 and 4", "what is 10 plus 5"]
        )
        def add_numbers(x: int, y: int) -> int:
            """Add two numbers together."""
            return x + y

        @gantry.register(
            tags=["math", "calculation"],
            examples=["I need to multiply two numbers", "what is 12 times 3"],
        )
        def multiply_numbers(x: int, y: int) -> int:
            """Multiply two numbers together."""
            return x * y

        @gantry.register(
            tags=["string", "text"], examples=["reverse this word", "write 'hello' backwards"]
        )
        def reverse_string(text: str) -> str:
            """Reverse a string."""
            return text[::-1]

        @gantry.register(
            tags=["data", "conversion"],
            examples=["make this all caps", "convert my text to uppercase"],
        )
        def convert_to_uppercase(text: str) -> str:
            """Convert text to uppercase."""
            return text.upper()

        await gantry.sync()

        print("Scenario: Claude Desktop connects to AgentGantry MCP server")
        print("\nStep 1: Claude calls find_relevant_tools")
        print("Query: 'I need to multiply two numbers'")

        # Simulate what happens in the MCP server
        context = ConversationContext(query="I need to multiply two numbers")
        query = ToolQuery(context=context, limit=3, score_threshold=0.0)
        result = await gantry.retrieve(query)

        print(f"\nReturned {len(result.tools)} relevant tools:")
        for scored_tool in result.tools:
            print(f"  - {scored_tool.tool.name}: {scored_tool.tool.description}")
            print(f"    Relevance: {scored_tool.semantic_score:.2f}")

        print("\nStep 2: Claude calls execute_tool")
        print("Tool: multiply_numbers")
        print("Arguments: {x: 5, y: 7}")

        from agent_gantry.schema.execution import ToolCall

        call = ToolCall(tool_name="multiply_numbers", arguments={"x": 5, "y": 7})
        result = await gantry.execute(call)

        print(f"\nResult: {result.result}")
        print(f"Status: {result.status.value}")
        print(f"Latency: {result.latency_ms:.2f}ms")
    finally:
        await gantry.close()


async def demo_context_window_savings():
    """Demo: Context window savings with dynamic mode."""
    print("\n=== Context Window Savings Demo ===\n")

    gantry = AgentGantry()
    try:
        # Register many tools. A factory gives each closure its own `i` and a
        # real docstring (an f-string in docstring position is not a docstring).
        def make_tool(i: int):
            def tool_fn(x: int) -> int:
                return x + i

            tool_fn.__doc__ = f"Tool number {i} for various operations and demonstrations."
            return tool_fn

        print("Registering 50 tools...")
        for i in range(50):
            gantry.register(
                make_tool(i),
                name=f"tool_{i}",
                tags=[f"category_{i % 5}"],
                examples=[f"use tool {i}", f"run tool number {i}"],
            )

        await gantry.sync()

        print(f"\nTotal tools: {gantry.tool_count}")

        # Measure the schema text a client would receive rather than guessing.
        # Roughly four characters per token.
        all_schemas = [t.to_dialect("openai") for t in await gantry.list_tools()]
        all_chars = len(json.dumps(all_schemas))

        context = ConversationContext(query="use tool 25")
        query = ToolQuery(context=context, limit=3, score_threshold=0.0)
        result = await gantry.retrieve(query)
        slice_schemas = [s.tool.to_dialect("openai") for s in result.tools]
        slice_chars = len(json.dumps(slice_schemas))

        print("\n--- Static Mode (Traditional) ---")
        print(f"All {len(all_schemas)} tool schemas listed to the client on every request:")
        print(f"  {all_chars:,} chars (~{all_chars // 4:,} tokens)")

        print("\n--- Dynamic Mode (Agent-Gantry MCP) ---")
        print("Only the two meta-tools are listed; the client asks for what it needs:")
        print("  Query: 'use tool 25'")
        print(f"  Relevant tools returned: {[s.tool.name for s in result.tools]}")
        print(f"  {slice_chars:,} chars (~{slice_chars // 4:,} tokens)")
        print(
            f"\nTool schema text in the prompt cut by {100 * (1 - slice_chars / all_chars):.0f}%"
            " (before adding the two small meta-tool schemas)."
        )
    finally:
        await gantry.close()


async def main():
    """Run all demos."""
    print("=" * 60)
    print("Agent-Gantry MCP Integration Demo")
    print("=" * 60)

    await demo_mcp_server()
    await demo_mcp_client()
    await demo_meta_tool_flow()
    await demo_context_window_savings()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("\nKey Benefits of MCP Integration:")
    print("  ✓ Universal protocol compatibility (Claude, custom clients)")
    print("  ✓ Large reduction in tool-schema context (measured above)")
    print("  ✓ Dynamic tool discovery at runtime")
    print("  ✓ Seamless integration with existing tools")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
