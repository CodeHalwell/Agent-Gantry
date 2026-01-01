# Dynamic MCP Server Selection

**Status**: 🚧 In Progress (Placeholder Implementation)

## Overview

Dynamic MCP Server Selection enables Agent-Gantry to intelligently route queries to the most relevant MCP servers using semantic search, similar to how tools are selected. This feature allows you to register multiple MCP servers with rich metadata and have Agent-Gantry automatically determine which servers to connect to based on the user's query.

## Key Benefits

### 🎯 Intelligent Server Selection
- **Semantic Routing**: Uses vector embeddings to find servers relevant to the query
- **Context-Aware**: Considers tags, examples, capabilities, and descriptions
- **Flexible Filtering**: Filter by capabilities, namespaces, or health status

### ⚡ Performance & Efficiency
- **Lazy Loading**: Connects to servers only when needed
- **No Upfront Cost**: Register servers without immediate connections
- **Reduced Initialization Time**: Skip connecting to unused servers

### 🔒 Security & Resource Management
- **Minimal Attack Surface**: Only connects to necessary servers
- **Health Tracking**: Monitors server availability and connection success
- **Capability-Based Access**: Ensures servers have required capabilities

### 🧩 Plug & Play Philosophy
- **Simple Registration**: Register servers with intuitive metadata
- **Backward Compatible**: Existing `add_mcp_server()` continues to work
- **Consistent API**: Follows same patterns as tool registration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
│              "I need to read a file"                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              retrieve_mcp_servers()                          │
│           (Semantic Search on Metadata)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Ranked MCP Servers                                 │
│    1. filesystem (0.92 relevance)                           │
│    2. storage (0.78 relevance)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│        discover_tools_from_server()                         │
│          (Lazy Connect & Import)                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Available Tools                                 │
│       - read_file                                           │
│       - write_file                                          │
│       - list_directory                                      │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Register MCP Servers

Register servers with rich metadata instead of immediately connecting:

```python
from agent_gantry import AgentGantry

gantry = AgentGantry()

# Register filesystem server
gantry.register_mcp_server(
    name="filesystem",
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
    description="Provides tools for reading and writing files on the local filesystem",
    args=["--path", "/home/user/documents"],
    tags=["filesystem", "files", "io", "local"],
    examples=[
        "read a file",
        "write to a file",
        "list directory contents",
    ],
    capabilities=["read_files", "write_files", "list_directory"],
)

# Register database server
gantry.register_mcp_server(
    name="postgresql",
    command=["python", "-m", "mcp_postgresql"],
    description="Access PostgreSQL databases for querying and data manipulation",
    args=["--connection-string", "postgresql://localhost/mydb"],
    tags=["database", "sql", "data"],
    examples=["query database", "insert record", "update data"],
    capabilities=["read_data", "write_data", "execute_sql"],
)
```

### 2. Sync Server Metadata

Sync server metadata to enable semantic search:

```python
# Embed server metadata for semantic routing
await gantry.sync_mcp_servers()
```

### 3. Discover Relevant Servers

Find servers relevant to your query:

```python
# Semantic search for relevant servers
servers = await gantry.retrieve_mcp_servers(
    query="I need to read a configuration file",
    limit=2,
    score_threshold=0.5,
)

for server in servers:
    print(f"Server: {server.name}")
    print(f"Description: {server.description}")
    print(f"Capabilities: {server.capabilities}")
```

### 4. Load Tools On-Demand

Connect to selected servers and load their tools:

```python
# Discover tools from the selected server
count = await gantry.discover_tools_from_server("filesystem")
print(f"Discovered {count} tools from filesystem server")

# Now use the tools
tools = await gantry.retrieve_tools("read my config.yaml")
result = await gantry.execute(...)
```

## Complete Example

```python
import asyncio
from agent_gantry import AgentGantry

async def main():
    # Initialize
    gantry = AgentGantry()
    
    # Register servers (no immediate connection)
    gantry.register_mcp_server(
        name="filesystem",
        command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
        description="Local filesystem operations",
        args=["--path", "/tmp"],
        tags=["files", "io"],
        examples=["read file", "write file"],
        capabilities=["read_files", "write_files"],
    )
    
    gantry.register_mcp_server(
        name="api_client",
        command=["node", "mcp-rest-api.js"],
        description="REST API client for external services",
        tags=["api", "http", "rest"],
        examples=["call API", "fetch data"],
        capabilities=["network_access", "external_api"],
    )
    
    # Sync for semantic search
    await gantry.sync_mcp_servers()
    
    # User query determines which servers to use
    user_query = "I need to fetch data from an API"
    
    # Find relevant servers
    servers = await gantry.retrieve_mcp_servers(user_query, limit=1)
    
    if servers:
        # Connect and load tools only from relevant server
        server = servers[0]
        print(f"Selected: {server.name}")
        count = await gantry.discover_tools_from_server(server.name)
        print(f"Loaded {count} tools")
    
    # Continue with tool retrieval and execution
    tools = await gantry.retrieve_tools(user_query)
    # ... use tools

asyncio.run(main())
```

## API Reference

### `register_mcp_server()`

Register an MCP server with metadata for semantic routing.

```python
gantry.register_mcp_server(
    name: str,                      # Unique server name
    command: list[str],             # Command to start server
    description: str,               # What the server provides (min 10 chars)
    namespace: str = "default",     # Namespace for organization
    args: list[str] | None = None,  # Command-line arguments
    env: dict[str, str] | None = None,  # Environment variables
    tags: list[str] | None = None,  # Categorization tags
    examples: list[str] | None = None,  # Example queries
    capabilities: list[str] | None = None,  # Server capabilities
) -> None
```

**Parameters:**
- `name`: Unique identifier for the server
- `command`: Command to execute the MCP server process
- `description`: Detailed description of server functionality (used for semantic search)
- `namespace`: Logical grouping namespace (default: "default")
- `args`: Additional command-line arguments to pass
- `env`: Environment variables for the server process
- `tags`: List of tags for categorization and filtering
- `examples`: Example queries that this server handles well
- `capabilities`: List of capabilities (e.g., "read_files", "write_data")

**Example:**
```python
gantry.register_mcp_server(
    name="weather_api",
    command=["python", "-m", "mcp_weather"],
    description="Access weather data from multiple providers including forecasts and alerts",
    namespace="apis",
    args=["--api-key", "xyz"],
    env={"WEATHER_PROVIDER": "openweathermap"},
    tags=["weather", "api", "forecast"],
    examples=["get weather", "forecast tomorrow", "weather alerts"],
    capabilities=["read_weather", "forecasts"],
)
```

### `retrieve_mcp_servers()`

Find relevant MCP servers using semantic search.

```python
await gantry.retrieve_mcp_servers(
    query: str,                          # Search query
    limit: int = 3,                      # Max servers to return
    score_threshold: float | None = None,  # Min similarity score
    namespaces: list[str] | None = None,  # Filter by namespaces
) -> list[MCPServerDefinition]
```

**Parameters:**
- `query`: Natural language query describing needed functionality
- `limit`: Maximum number of servers to return (default: 3)
- `score_threshold`: Minimum similarity score (0-1) to include a server
- `namespaces`: Filter results to specific namespaces

**Returns:** List of `MCPServerDefinition` objects, ordered by relevance

**Example:**
```python
servers = await gantry.retrieve_mcp_servers(
    query="send email with attachments",
    limit=2,
    score_threshold=0.6,
    namespaces=["communication"],
)
```

### `discover_tools_from_server()`

Connect to a server and load its tools on-demand.

```python
await gantry.discover_tools_from_server(
    server_name: str,             # Server name
    namespace: str = "default",   # Server namespace
) -> int
```

**Parameters:**
- `server_name`: Name of the registered MCP server
- `namespace`: Namespace of the server (default: "default")

**Returns:** Number of tools discovered and registered

**Raises:**
- `ValueError`: If the server is not registered

**Example:**
```python
count = await gantry.discover_tools_from_server("filesystem")
print(f"Loaded {count} tools from filesystem server")
```

### `sync_mcp_servers()`

Sync server metadata to vector store for semantic search.

```python
await gantry.sync_mcp_servers(
    batch_size: int = 100,   # Batch size for embedding
    force: bool = False,     # Force re-embedding
) -> int
```

**Parameters:**
- `batch_size`: Number of servers to process per batch (default: 100)
- `force`: Re-embed all servers regardless of changes (default: False)

**Returns:** Number of servers synced

**Example:**
```python
synced = await gantry.sync_mcp_servers()
print(f"Synced {synced} servers")
```

## Metadata Schema

### MCPServerDefinition

Server metadata model for semantic routing:

```python
class MCPServerDefinition(BaseModel):
    # Identity
    name: str                           # Server name
    namespace: str = "default"          # Namespace
    
    # Discovery (used for semantic search)
    description: str                    # Detailed description (min 10 chars)
    extended_description: str | None    # Additional details
    tags: list[str] = []               # Categorization tags
    examples: list[str] = []           # Example queries
    
    # Connection
    command: list[str]                 # Start command
    args: list[str] = []              # Command arguments
    env: dict[str, str] = {}          # Environment variables
    
    # Capabilities & Cost
    capabilities: list[str] = []       # Server capabilities
    cost: MCPServerCost                # Cost/latency model
    
    # Health (runtime)
    health: MCPServerHealth            # Health metrics
    
    # Metadata
    metadata: dict[str, Any] = {}     # Additional metadata
    created_at: datetime               # Registration time
    deprecated: bool = False           # Deprecation flag
```

**Key Fields for Semantic Search:**
- `description`: Primary text for semantic matching
- `tags`: Keywords for categorization and filtering
- `examples`: Sample queries the server handles well
- `capabilities`: Functional capabilities for filtering

## Comparison: Traditional vs Dynamic

### Traditional Approach (`add_mcp_server`)

```python
# Connect to ALL servers at startup
count1 = await gantry.add_mcp_server(config1)
count2 = await gantry.add_mcp_server(config2)
count3 = await gantry.add_mcp_server(config3)
# All 3 servers connected, all tools loaded upfront
```

**Characteristics:**
- ❌ Connects to all servers immediately
- ❌ Loads all tools from all servers
- ❌ High initialization cost
- ❌ No semantic server selection
- ✅ All tools always available

### Dynamic Approach (`register_mcp_server`)

```python
# Register servers (no connection)
gantry.register_mcp_server(name="server1", ...)
gantry.register_mcp_server(name="server2", ...)
gantry.register_mcp_server(name="server3", ...)
await gantry.sync_mcp_servers()

# Find relevant servers for query
servers = await gantry.retrieve_mcp_servers(query)

# Connect only to selected server
await gantry.discover_tools_from_server(servers[0].name)
```

**Characteristics:**
- ✅ No upfront connections
- ✅ Loads tools only when needed
- ✅ Fast initialization
- ✅ Semantic server selection
- ✅ Minimal resource usage

## Use Cases

### 1. Multi-Domain Agent Systems

Register servers for different domains and let semantic routing select the right one:

```python
# Register domain-specific servers
gantry.register_mcp_server(name="filesystem", ...)
gantry.register_mcp_server(name="database", ...)
gantry.register_mcp_server(name="email", ...)
gantry.register_mcp_server(name="calendar", ...)

# Query determines which domain to use
servers = await gantry.retrieve_mcp_servers("schedule a meeting")
# → Returns calendar server
```

### 2. Resource-Constrained Environments

Minimize connections and memory usage:

```python
# Register 20+ servers
for server_config in all_servers:
    gantry.register_mcp_server(**server_config)

# Only connect to 2-3 most relevant servers
servers = await gantry.retrieve_mcp_servers(user_query, limit=2)
for server in servers:
    await gantry.discover_tools_from_server(server.name)
```

### 3. Capability-Based Selection

Filter servers by required capabilities:

```python
# Find servers with specific capabilities
from agent_gantry.core.mcp_router import MCPRouter

servers = await gantry.retrieve_mcp_servers("data operations")
filtered = await gantry._mcp_router.filter_by_capabilities(
    servers,
    required_capabilities=["write_data", "execute_sql"]
)
```

### 4. Health-Aware Routing

Avoid unavailable servers automatically:

```python
# Retrieval automatically excludes unhealthy servers
servers = await gantry.retrieve_mcp_servers(query)
# Only returns servers with health.available == True
```

## Advanced Features

### Health Tracking

Server health is automatically tracked:

```python
# Access server health
server = gantry._mcp_registry.get_server("filesystem")
if server:
    print(f"Success rate: {server.health.success_rate}")
    print(f"Available: {server.health.available}")
    print(f"Consecutive failures: {server.health.consecutive_failures}")
```

### Manual Health Updates

```python
# Update server health manually
gantry._mcp_registry.update_health(
    "filesystem",
    namespace="default",
    available=False,
    consecutive_failures=3,
)
```

### Lazy Client Management

Clients are created only when needed:

```python
# Get or create client for a server
client = gantry._mcp_registry.get_client("filesystem")
# First call creates client, subsequent calls return same instance
```

### Server Listing

```python
# List all registered servers
all_servers = gantry._mcp_registry.list_servers()

# Filter by namespace
ns_servers = gantry._mcp_registry.list_servers(namespace="apis")

# Check server count
count = gantry._mcp_registry.server_count
```

## Future Enhancements

### Phase 5: Vector Store Integration

**Current Status**: Placeholder implementation  
**Goal**: Full semantic search with vector embeddings

**What's Needed:**
1. Extend vector store to support multiple entity types (tools + servers)
2. Implement metadata-based filtering in vector store
3. Complete MCPRouter.route() with actual search results

**When Complete:**
```python
# Will work with actual semantic search
servers = await gantry.retrieve_mcp_servers("read files")
# Returns servers ranked by semantic similarity
```

### Planned Features

- **Auto-discovery**: Automatically detect available MCP servers
- **Connection Pooling**: Reuse connections across queries
- **Hybrid Mode**: Mix static and dynamic server loading
- **Server Dependencies**: Express dependencies between servers
- **Cost Optimization**: Prefer low-latency servers

## Troubleshooting

### Server Not Found

```python
# Check if server is registered
server = gantry._mcp_registry.get_server("myserver")
if not server:
    print("Server not registered!")
```

### Connection Failures

```python
try:
    count = await gantry.discover_tools_from_server("myserver")
except Exception as e:
    print(f"Failed to connect: {e}")
    # Check server health
    server = gantry._mcp_registry.get_server("myserver")
    if server and not server.health.available:
        print("Server marked as unavailable")
```

### No Servers Returned

```python
servers = await gantry.retrieve_mcp_servers("query", score_threshold=0.8)
if not servers:
    # Try lower threshold
    servers = await gantry.retrieve_mcp_servers("query", score_threshold=0.5)
```

## Best Practices

1. **Rich Descriptions**: Provide detailed, keyword-rich descriptions
2. **Relevant Examples**: Include example queries that match your use cases
3. **Meaningful Tags**: Use consistent, searchable tags
4. **Capability Modeling**: Define capabilities that match your security model
5. **Health Monitoring**: Check server health before critical operations
6. **Sync Regularly**: Call `sync_mcp_servers()` after batch registrations
7. **Namespace Organization**: Group related servers in namespaces

## Migration Guide

### From `add_mcp_server()` to `register_mcp_server()`

**Before:**
```python
config = MCPServerConfig(
    name="filesystem",
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
    args=["--path", "/tmp"],
    namespace="default",
)
count = await gantry.add_mcp_server(config)
```

**After:**
```python
gantry.register_mcp_server(
    name="filesystem",
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
    description="Local filesystem operations for reading and writing files",
    args=["--path", "/tmp"],
    namespace="default",
    tags=["filesystem", "files"],
    examples=["read file", "write file"],
    capabilities=["read_files", "write_files"],
)
await gantry.sync_mcp_servers()

# Later, discover tools on-demand
servers = await gantry.retrieve_mcp_servers("need to read a file")
if servers:
    await gantry.discover_tools_from_server(servers[0].name)
```

**Benefits:**
- No upfront connection overhead
- Semantic server selection
- Better resource management

**Note:** Both methods work and can be used together!

## See Also

- [MCP Integration Documentation](phase5_mcp.md)
- [Example: dynamic_mcp_selection_demo.py](../examples/protocols/dynamic_mcp_selection_demo.py)
- [Tests: test_dynamic_mcp_selection.py](../tests/test_dynamic_mcp_selection.py)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
