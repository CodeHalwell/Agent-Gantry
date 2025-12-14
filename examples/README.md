# Agent-Gantry Examples

This directory contains examples demonstrating various features of Agent-Gantry.

## MCP Integration Demo

Run the comprehensive MCP integration demo:

```bash
python examples/mcp_integration_demo.py
```

This demonstrates:
- Serving Agent-Gantry as an MCP server in dynamic mode
- Connecting to external MCP servers as a client
- Meta-tool discovery and execution flow
- Context window savings (90%+ reduction)

## Claude Desktop Integration

To integrate Agent-Gantry with Claude Desktop:

1. Create your Agent-Gantry server script:

```python
# my_agent_gantry_server.py
import asyncio
from agent_gantry import AgentGantry

async def main():
    gantry = AgentGantry()
    
    # Register your tools
    @gantry.register
    def my_tool(x: int) -> int:
        """My custom tool."""
        return x * 2
    
    await gantry.sync()
    
    # Start MCP server in dynamic mode
    await gantry.serve_mcp(transport="stdio", mode="dynamic")

if __name__ == "__main__":
    asyncio.run(main())
```

2. Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "agent-gantry": {
      "command": "python",
      "args": ["/path/to/my_agent_gantry_server.py"]
    }
  }
}
```

3. Restart Claude Desktop

4. Claude will now be able to discover and use your tools dynamically through the `find_relevant_tools` and `execute_tool` meta-tools!

## Key Benefits

- **Context Window Minimization**: Only 2 meta-tools exposed instead of all tools
- **Dynamic Discovery**: Tools found on-demand through semantic search
- **Universal Protocol**: Works with any MCP client (Claude, custom implementations)
- **Seamless Integration**: Existing Agent-Gantry tools automatically available

## More Examples Coming Soon

- LangChain integration
- AutoGen integration
- Production deployment examples
- Multi-server MCP configurations
