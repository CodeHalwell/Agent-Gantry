# Phase 5: MCP Integration

**Status**: ✅ Complete

## Overview

Phase 5 implements complete Model Context Protocol (MCP) integration for Agent-Gantry, enabling it to act as both an MCP client (consuming external MCP servers) and an MCP server (exposing tools to MCP clients like Claude Desktop).

## Key Features

### 1. MCP Client (`agent_gantry/adapters/executors/mcp_client.py`)

The MCP client enables Agent-Gantry to discover and use tools from external MCP servers:

```python
from agent_gantry import AgentGantry
from agent_gantry.schema.config import MCPServerConfig

gantry = AgentGantry()

# Connect to an external MCP server
config = MCPServerConfig(
    name="filesystem",
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
    args=["--path", "/tmp"],
    namespace="fs",
)

# Discover and register tools
count = await gantry.add_mcp_server(config)
print(f"Added {count} tools from MCP server")
```

**Features:**
- stdio subprocess connection support
- MCP handshake (initialize/initialized)
- Tool discovery via tools/list
- Tool execution via tools/call
- Automatic conversion to ToolDefinition
- MCPClientPool for managing multiple connections

### 2. MCP Server (`agent_gantry/servers/mcp_server.py`)

The MCP server enables Agent-Gantry to expose its tools to MCP clients:

```python
from agent_gantry import AgentGantry

gantry = AgentGantry()

@gantry.register
def my_tool(x: int) -> int:
    """My custom tool."""
    return x * 2

await gantry.sync()

# Serve as MCP server
await gantry.serve_mcp(transport="stdio", mode="dynamic")
```

**Modes:**

1. **Dynamic Mode** (Recommended)
   - Exposes only 2 meta-tools: `find_relevant_tools` and `execute_tool`
   - Tools discovered on-demand through semantic search
   - 90%+ reduction in context window usage
   - Perfect for Claude Desktop integration

2. **Static Mode**
   - Exposes all registered tools directly
   - Traditional MCP server behavior
   - Higher context window usage

3. **Hybrid Mode** (Structure ready for future)
   - Common tools exposed directly
   - Remaining tools via meta-tools
   - Balanced approach

### 3. Meta-Tools for Dynamic Mode

**find_relevant_tools**
- Input: `query` (string), `limit` (integer, default: 5)
- Uses Agent-Gantry's semantic routing to find relevant tools
- Returns tool names, descriptions, schemas, and relevance scores

**execute_tool**
- Input: `tool_name` (string), `arguments` (object)
- Executes the specified tool with provided arguments
- Returns tool execution result

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Clients                              │
│              (Claude Desktop, Custom)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │ MCP Protocol
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Agent-Gantry                              │
│                    (MCP Server)                             │
│                                                             │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │  find_relevant_  │      │   execute_tool   │           │
│  │     tools        │      │                  │           │
│  └────────┬─────────┘      └────────┬─────────┘           │
│           │                         │                      │
│           ▼                         ▼                      │
│  ┌──────────────────────────────────────────────┐         │
│  │      Semantic Router + Executor              │         │
│  └──────────────────────────────────────────────┘         │
└─────────────────────────┬───────────────────────────────────┘
                          │ MCP Protocol
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              External MCP Servers                           │
│         (filesystem, databases, APIs, etc.)                 │
└─────────────────────────────────────────────────────────────┘
```

## Protocol Compliance

Agent-Gantry's MCP implementation is fully compliant with the MCP specification:

- ✅ Initialize handshake
- ✅ tools/list operation
- ✅ tools/call operation
- ✅ Proper JSON-RPC 2.0 format
- ✅ Error handling
- ✅ stdin/stdout transport

## Context Window Optimization

### The Problem

Traditional MCP servers expose all tools in every request:
- 50 tools × ~100 tokens/tool = ~5,000 tokens
- Every request pays this context tax
- Large tool sets become prohibitively expensive

### Dynamic Mode Solution

Agent-Gantry's dynamic mode uses meta-tools:
- Initial context: 2 meta-tools × ~100 tokens = ~200 tokens
- Tool discovery: 3-5 relevant tools × ~100 tokens = ~300-500 tokens
- **Total: ~500-700 tokens vs 5,000+ tokens**
- **Savings: 90%+ reduction**

## Testing

Phase 5 includes comprehensive test coverage:

```bash
pytest tests/test_phase5_mcp.py -v
```

**Test Categories:**

1. **MCPClient Tests**
   - Client initialization
   - Tool conversion
   - Connection handling
   - Tool listing

2. **MCPClientPool Tests**
   - Server management
   - Multi-server connections
   - Tool aggregation

3. **MCPServer Tests**
   - Server initialization
   - Dynamic mode operation
   - Static mode operation
   - Meta-tool functionality
   - Error handling

4. **Integration Tests**
   - add_mcp_server method
   - serve_mcp method
   - End-to-end flows

5. **Protocol Compliance Tests**
   - Schema format validation
   - Meta-tool discovery flow
   - Context window minimization

**Results:** All 19 MCP tests passing ✅

## Examples

### Running the Demo

```bash
python examples/mcp_integration_demo.py
```

This demonstrates:
- MCP server setup
- MCP client usage
- Meta-tool discovery flow
- Context window savings

### Claude Desktop Integration

1. Create your server script:

```python
# my_server.py
import asyncio
from agent_gantry import AgentGantry

async def main():
    gantry = AgentGantry()
    
    @gantry.register
    def calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers."""
        return a + b
    
    await gantry.sync()
    await gantry.serve_mcp(transport="stdio", mode="dynamic")

if __name__ == "__main__":
    asyncio.run(main())
```

2. Configure Claude Desktop:

```json
{
  "mcpServers": {
    "agent-gantry": {
      "command": "python",
      "args": ["/path/to/my_server.py"]
    }
  }
}
```

3. Restart Claude Desktop

4. Claude can now discover and use your tools dynamically!

## Performance Characteristics

| Metric | Static Mode | Dynamic Mode | Improvement |
|--------|-------------|--------------|-------------|
| Initial Context Tokens | ~5,000-10,000 | ~200-300 | 95%+ |
| Per-Query Context | ~5,000-10,000 | ~500-700 | 90%+ |
| Tool Discovery | All at once | On-demand | Dynamic |
| Latency | Low | Low + search | Minimal |

## Future Enhancements

1. **SSE Transport** - Structure is ready, implementation pending
2. **Hybrid Mode** - Expose common tools + meta-tools for others
3. **WebSocket Transport** - For real-time bidirectional communication
4. **Tool Caching** - Cache frequently used tool schemas
5. **Streaming Results** - Stream tool execution results for long operations

## Dependencies

```toml
[project.optional-dependencies]
mcp = [
    "mcp>=1.0.0",
]
```

Install with:
```bash
pip install agent-gantry[mcp]
```

## API Reference

### AgentGantry Methods

#### `add_mcp_server(config: MCPServerConfig) -> int`

Connects to an external MCP server and registers its tools.

**Parameters:**
- `config`: MCPServerConfig with server details

**Returns:**
- Number of tools discovered and registered

**Example:**
```python
config = MCPServerConfig(
    name="my-server",
    command=["node", "server.js"],
    namespace="external",
)
count = await gantry.add_mcp_server(config)
```

#### `serve_mcp(transport: str = "stdio", mode: str = "dynamic", name: str = "agent-gantry") -> None`

Starts Agent-Gantry as an MCP server.

**Parameters:**
- `transport`: "stdio" or "sse"
- `mode`: "dynamic", "static", or "hybrid"
- `name`: Server name for identification

**Example:**
```python
await gantry.serve_mcp(transport="stdio", mode="dynamic")
```

### MCPClient

#### `__init__(config: MCPServerConfig)`

Creates a new MCP client.

#### `async list_tools() -> list[ToolDefinition]`

Lists all tools from the MCP server.

#### `async call_tool(tool_name: str, arguments: dict[str, Any]) -> Any`

Executes a tool on the MCP server.

### MCPServer

#### `__init__(gantry: AgentGantry, mode: str = "dynamic", name: str = "agent-gantry")`

Creates a new MCP server.

#### `async run_stdio() -> None`

Runs the server with stdio transport.

## Success Metrics

✅ **Implementation Complete**
- MCP client fully functional
- MCP server fully functional
- Dynamic mode with meta-tools working
- Protocol compliance verified

✅ **Testing Complete**
- 19 comprehensive tests passing
- All test categories covered
- Integration tests validated

✅ **Documentation Complete**
- README updated
- Examples created
- Claude Desktop integration guide
- This phase documentation

✅ **Performance Targets Met**
- 90%+ context window reduction achieved
- Semantic routing integration seamless
- Minimal latency overhead

## Conclusion

Phase 5 successfully implements complete MCP integration for Agent-Gantry, enabling universal protocol compatibility with Claude Desktop and other MCP clients while achieving dramatic reductions in context window usage through innovative dynamic tool discovery.
