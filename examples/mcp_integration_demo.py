"""
MCP Integration Demo for Agent-Gantry.

Demonstrates how to:
1. Serve AgentGantry as an MCP server (dynamic mode)
2. Connect to external MCP servers as a client
3. Use the meta-tool discovery flow
"""

import asyncio
from agent_gantry import AgentGantry
from agent_gantry.schema.config import MCPServerConfig
from agent_gantry.schema.query import ConversationContext, ToolQuery


async def demo_mcp_server():
    """Demo: Serve AgentGantry as an MCP server in dynamic mode."""
    print("\n=== MCP Server Demo (Dynamic Mode) ===\n")
    
    # Create a gantry instance with some tools
    gantry = AgentGantry()
    
    @gantry.register
    def calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers."""
        return a + b
    
    @gantry.register
    def calculate_product(a: int, b: int) -> int:
        """Calculate the product of two numbers."""
        return a * b
    
    @gantry.register
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


async def demo_mcp_client():
    """Demo: Connect to external MCP servers as a client."""
    print("\n=== MCP Client Demo ===\n")
    
    gantry = AgentGantry()
    
    # Example configuration for connecting to an external MCP server
    # Note: This requires an actual MCP server to be running
    config = MCPServerConfig(
        name="example-server",
        command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
        args=["--path", "/tmp"],
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


async def demo_meta_tool_flow():
    """Demo: Meta-tool discovery and execution flow."""
    print("\n=== Meta-Tool Discovery Flow Demo ===\n")
    
    gantry = AgentGantry()
    
    # Register various tools
    @gantry.register(tags=["math", "calculation"])
    def add_numbers(x: int, y: int) -> int:
        """Add two numbers together."""
        return x + y
    
    @gantry.register(tags=["math", "calculation"])
    def multiply_numbers(x: int, y: int) -> int:
        """Multiply two numbers together."""
        return x * y
    
    @gantry.register(tags=["string", "text"])
    def reverse_string(text: str) -> str:
        """Reverse a string."""
        return text[::-1]
    
    @gantry.register(tags=["data", "conversion"])
    def convert_to_uppercase(text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()
    
    await gantry.sync()
    
    print("Scenario: Claude Desktop connects to AgentGantry MCP server")
    print("\nStep 1: Claude calls find_relevant_tools")
    print("Query: 'I need to multiply two numbers'")
    
    # Simulate what happens in the MCP server
    context = ConversationContext(query="I need to multiply two numbers")
    query = ToolQuery(context=context, limit=3)
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


async def demo_context_window_savings():
    """Demo: Context window savings with dynamic mode."""
    print("\n=== Context Window Savings Demo ===\n")
    
    gantry = AgentGantry()
    
    # Register many tools
    print("Registering 50 tools...")
    for i in range(50):
        @gantry.register(name=f"tool_{i}", tags=[f"category_{i % 5}"])
        def tool_fn(x: int) -> int:
            f"""Tool number {i} for various operations and demonstrations."""
            return x + i
    
    await gantry.sync()
    
    print(f"\nTotal tools: {gantry.tool_count}")
    
    print("\n--- Static Mode (Traditional) ---")
    print("All 50+ tools sent to LLM in every request")
    print("Context tokens: ~5000-10000 (depending on schema size)")
    print("Cost per request: HIGH")
    
    print("\n--- Dynamic Mode (Agent-Gantry MCP) ---")
    print("Only 2 meta-tools sent to LLM initially")
    print("Context tokens: ~200-300")
    print("Cost savings: ~95%!")
    
    print("\nLLM discovers relevant tools dynamically:")
    context = ConversationContext(query="use tool 25")
    query = ToolQuery(context=context, limit=3)
    result = await gantry.retrieve(query)
    
    print(f"  Query: 'use tool 25'")
    print(f"  Relevant tools returned: {len(result.tools)}")
    print(f"  Context tokens for these tools: ~150-200")
    print(f"  Total context: ~350-500 tokens (vs 5000-10000)")


async def main():
    """Run all demos."""
    print("=" * 60)
    print("Agent-Gantry Phase 5: MCP Integration Demo")
    print("=" * 60)
    
    await demo_mcp_server()
    await demo_mcp_client()
    await demo_meta_tool_flow()
    await demo_context_window_savings()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("\nKey Benefits of MCP Integration:")
    print("  ✓ Universal protocol compatibility (Claude, custom clients)")
    print("  ✓ 90%+ reduction in context window usage")
    print("  ✓ Dynamic tool discovery at runtime")
    print("  ✓ Seamless integration with existing tools")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
