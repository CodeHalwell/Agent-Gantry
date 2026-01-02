---
layout: default
title: API Reference
parent: Reference
nav_order: 1
description: "Complete API documentation for Agent-Gantry"
---

# API Reference

Complete API documentation for Agent-Gantry v0.1.2.

---

## Core Classes

### AgentGantry

The main facade class for Agent-Gantry. This is your primary entry point for tool orchestration.

#### Constructor

```python
from agent_gantry import AgentGantry

gantry = AgentGantry(
    config: Optional[AgentGantryConfig] = None,
    vector_store: Optional[VectorStoreAdapter] = None,
    embedder: Optional[EmbedderAdapter] = None,
    reranker: Optional[RerankerAdapter] = None,
    executor: Optional[ExecutorAdapter] = None,
    modules: Optional[list[str]] = None
)
```

**Parameters:**

- `config` (Optional[AgentGantryConfig]): Configuration object. If None, uses defaults.
- `vector_store` (Optional[VectorStoreAdapter]): Custom vector store adapter. Defaults to InMemoryVectorStore.
- `embedder` (Optional[EmbedderAdapter]): Custom embedder adapter. Defaults to SimpleEmbedder or NomicEmbedder if available.
- `reranker` (Optional[RerankerAdapter]): Custom reranker adapter. Optional.
- `executor` (Optional[ExecutorAdapter]): Custom executor adapter. Defaults to DirectExecutor.
- `modules` (Optional[list[str]]): List of module paths to import tools from.

**Example:**

```python
from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.nomic import NomicEmbedder
from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

gantry = AgentGantry(
    embedder=NomicEmbedder(dimension=768),
    vector_store=LanceDBVectorStore(
        db_path="my_tools.lancedb",
        dimension=768
    )
)
```

#### Methods

##### register()

Register a tool function.

```python
@gantry.register(
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[list[str]] = None,
    namespace: str = "default",
    metadata: Optional[dict[str, Any]] = None
) -> Callable
```

**Parameters:**

- `name`: Tool name. Defaults to function name.
- `description`: Tool description. Extracted from docstring if not provided.
- `tags`: Tags for categorization and semantic search.
- `namespace`: Namespace for organizing tools. Default: "default"
- `metadata`: Additional metadata dictionary.

**Example:**

```python
@gantry.register(
    tags=["weather", "forecast"],
    description="Get current weather for a city"
)
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"
```

##### sync()

Sync all registered tools to the vector store for semantic search.

```python
await gantry.sync(
    batch_size: int = 100,
    force: bool = False
) -> int
```

**Parameters:**

- `batch_size`: Number of tools to process in each batch.
- `force`: If True, re-embed all tools even if unchanged.

**Returns:** Number of tools synced.

**Example:**

```python
count = await gantry.sync()
print(f"Synced {count} tools")
```

##### retrieve_tools()

Retrieve semantically relevant tools for a query.

```python
await gantry.retrieve_tools(
    query: str,
    limit: int = 10,
    score_threshold: float = 0.0,
    filters: Optional[dict[str, Any]] = None,
    namespace: Optional[str] = None,
    rerank: bool = False
) -> list[ToolDefinition]
```

**Parameters:**

- `query`: User query or task description.
- `limit`: Maximum number of tools to return.
- `score_threshold`: Minimum similarity score (0.0 to 1.0).
- `filters`: Metadata filters (e.g., `{"tags": ["weather"]}`).
- `namespace`: Filter by namespace.
- `rerank`: If True, apply reranking to results.

**Returns:** List of ToolDefinition objects, ordered by relevance.

**Example:**

```python
tools = await gantry.retrieve_tools(
    query="I need to check the weather",
    limit=3,
    score_threshold=0.1
)

for tool in tools:
    print(f"{tool.name}: {tool.description}")
```

##### execute()

Execute a tool with the given arguments.

```python
await gantry.execute(
    call: ToolCall,
    timeout: Optional[float] = None
) -> ToolResult
```

**Parameters:**

- `call`: ToolCall object with tool_name and arguments.
- `timeout`: Execution timeout in seconds.

**Returns:** ToolResult object with output and metadata.

**Example:**

```python
from agent_gantry.schema.execution import ToolCall

result = await gantry.execute(
    ToolCall(
        tool_name="get_weather",
        arguments={"city": "Paris"}
    ),
    timeout=10.0
)

print(result.output)  # "Weather in Paris: Sunny, 72°F"
```

##### list_tools()

List all registered tools.

```python
gantry.list_tools(
    namespace: Optional[str] = None
) -> list[ToolDefinition]
```

**Parameters:**

- `namespace`: Filter by namespace. If None, returns all tools.

**Returns:** List of all registered ToolDefinition objects.

**Example:**

```python
tools = gantry.list_tools()
print(f"Total tools: {len(tools)}")
```

##### get_tool()

Get a specific tool by name.

```python
gantry.get_tool(
    name: str,
    namespace: str = "default"
) -> Optional[ToolDefinition]
```

**Parameters:**

- `name`: Tool name.
- `namespace`: Tool namespace.

**Returns:** ToolDefinition or None if not found.

**Example:**

```python
tool = gantry.get_tool("get_weather")
if tool:
    print(tool.description)
```

##### delete_tool()

Delete a tool from the registry.

```python
gantry.delete_tool(
    name: str,
    namespace: str = "default"
) -> bool
```

**Parameters:**

- `name`: Tool name.
- `namespace`: Tool namespace.

**Returns:** True if deleted, False if not found.

##### from_modules() (Class Method)

Create an AgentGantry instance and load tools from modules.

```python
gantry = await AgentGantry.from_modules(
    modules: list[str],
    attr: str = "tools",
    **kwargs
) -> AgentGantry
```

**Parameters:**

- `modules`: List of module paths to import.
- `attr`: Attribute name containing the AgentGantry instance.
- `**kwargs`: Additional arguments passed to AgentGantry constructor.

**Example:**

```python
gantry = await AgentGantry.from_modules([
    "myapp.tools.weather",
    "myapp.tools.math"
], attr="tools")
```

---

## MCP Integration

### register_mcp_server()

Register an MCP server with metadata for semantic selection.

```python
gantry.register_mcp_server(
    name: str,
    command: list[str],
    description: str,
    args: Optional[list[str]] = None,
    env: Optional[dict[str, str]] = None,
    namespace: str = "default",
    tags: Optional[list[str]] = None,
    examples: Optional[list[str]] = None,
    capabilities: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None
) -> MCPServerDefinition
```

**Parameters:**

- `name`: Server identifier.
- `command`: Command to start the server (e.g., `["npx", "-y", "@modelcontextprotocol/server-filesystem"]`).
- `description`: Human-readable description.
- `args`: Command-line arguments.
- `env`: Environment variables.
- `namespace`: Namespace for organization.
- `tags`: Tags for semantic search.
- `examples`: Example queries the server can handle.
- `capabilities`: List of capabilities (e.g., `["read_files", "write_files"]`).
- `metadata`: Additional metadata.

**Example:**

```python
gantry.register_mcp_server(
    name="filesystem",
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
    description="Provides tools for reading and writing files",
    args=["--path", "/home/user/docs"],
    tags=["filesystem", "files", "io"],
    examples=["read a file", "write to disk"],
    capabilities=["read_files", "write_files"]
)
```

### sync_mcp_servers()

Sync MCP server metadata to vector store for semantic search.

```python
await gantry.sync_mcp_servers(
    batch_size: int = 100,
    force: bool = False
) -> int
```

**Parameters:**

- `batch_size`: Batch size for processing.
- `force`: Force re-sync even if unchanged.

**Returns:** Number of servers synced.

### retrieve_mcp_servers()

Retrieve relevant MCP servers for a query.

```python
await gantry.retrieve_mcp_servers(
    query: str,
    limit: int = 5,
    score_threshold: float = 0.0,
    filters: Optional[dict[str, Any]] = None
) -> list[MCPServerScore]
```

**Parameters:**

- `query`: User query or task description.
- `limit`: Maximum servers to return.
- `score_threshold`: Minimum similarity score.
- `filters`: Metadata filters.

**Returns:** List of MCPServerScore objects with server and score.

**Example:**

```python
servers = await gantry.retrieve_mcp_servers(
    query="I need to read a configuration file",
    limit=2
)

for scored_server in servers:
    print(f"{scored_server.server.name}: {scored_server.score:.2f}")
```

### discover_tools_from_server()

Connect to an MCP server and discover its tools.

```python
await gantry.discover_tools_from_server(
    server_name: str,
    namespace: str = "default",
    timeout: float = 30.0
) -> int
```

**Parameters:**

- `server_name`: Name of registered MCP server.
- `namespace`: Server namespace.
- `timeout`: Connection timeout in seconds.

**Returns:** Number of tools discovered and registered.

**Example:**

```python
count = await gantry.discover_tools_from_server("filesystem")
print(f"Discovered {count} tools from filesystem server")
```

### add_mcp_server()

Connect to an external MCP server and register its tools (traditional approach).

```python
await gantry.add_mcp_server(
    config: MCPServerConfig
) -> int
```

**Parameters:**

- `config`: MCPServerConfig object with server connection details.

**Returns:** Number of tools added.

---

## Decorators

### @with_semantic_tools

Automatically inject semantically relevant tools into LLM function calls.

```python
from agent_gantry import with_semantic_tools

@with_semantic_tools(
    limit: int = 10,
    score_threshold: float = 0.0,
    dialect: str = "openai",
    prompt_param: str = "prompt",
    tools_param: str = "tools",
    gantry: Optional[AgentGantry] = None
)
async def my_llm_function(...):
    ...
```

**Parameters:**

- `limit`: Maximum tools to inject.
- `score_threshold`: Minimum similarity score.
- `dialect`: Target LLM format (`"openai"`, `"anthropic"`, `"gemini"`).
- `prompt_param`: Name of parameter containing the user prompt.
- `tools_param`: Name of keyword parameter to inject tools into.
- `gantry`: AgentGantry instance (optional if `set_default_gantry()` was called).

**Example:**

```python
from openai import AsyncOpenAI
from agent_gantry import with_semantic_tools, set_default_gantry

client = AsyncOpenAI()
gantry = AgentGantry()
set_default_gantry(gantry)

@with_semantic_tools(limit=3, dialect="openai")
async def chat(prompt: str, *, tools=None):
    return await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        tools=tools  # Automatically injected
    )
```

---

## Data Models

### ToolDefinition

Represents a registered tool.

```python
from agent_gantry.schema.tool import ToolDefinition

tool = ToolDefinition(
    name: str,
    description: str,
    parameters: dict[str, Any],
    namespace: str = "default",
    tags: list[str] = [],
    metadata: dict[str, Any] = {},
    capabilities: list[str] = []
)
```

**Attributes:**

- `name`: Tool identifier.
- `description`: Human-readable description.
- `parameters`: JSON Schema for tool parameters.
- `namespace`: Tool namespace.
- `tags`: Tags for categorization.
- `metadata`: Additional metadata.
- `capabilities`: Required capabilities.

### ToolCall

Represents a request to execute a tool.

```python
from agent_gantry.schema.execution import ToolCall

call = ToolCall(
    tool_name: str,
    arguments: dict[str, Any],
    tool_id: Optional[str] = None
)
```

### ToolResult

Result of a tool execution.

```python
from agent_gantry.schema.execution import ToolResult

# Returned by gantry.execute()
result = ToolResult(
    tool_name: str,
    output: Any,
    success: bool,
    error: Optional[str] = None,
    execution_time_ms: float,
    metadata: dict[str, Any] = {}
)
```

**Attributes:**

- `tool_name`: Name of executed tool.
- `output`: Tool output (any JSON-serializable type).
- `success`: True if execution succeeded.
- `error`: Error message if failed.
- `execution_time_ms`: Execution duration in milliseconds.
- `metadata`: Additional execution metadata.

---

## Configuration

### AgentGantryConfig

Main configuration object.

```python
from agent_gantry.schema.config import AgentGantryConfig

config = AgentGantryConfig(
    embedder: Optional[EmbedderConfig] = None,
    vector_store: Optional[VectorStoreConfig] = None,
    reranker: Optional[RerankerConfig] = None,
    executor: Optional[ExecutorConfig] = None,
    security: Optional[SecurityConfig] = None,
    telemetry: Optional[TelemetryConfig] = None
)
```

See [Configuration Reference]({{ '/reference/configuration' | relative_url }}) for detailed configuration options.

---

## Helper Functions

### set_default_gantry()

Set the default AgentGantry instance for the current context.

```python
from agent_gantry import set_default_gantry

gantry = AgentGantry()
set_default_gantry(gantry)
```

**Why use this?**

- Enables `@with_semantic_tools` decorator without explicit gantry parameter
- Thread-safe and async-safe using context variables
- Simplifies code when using a single gantry instance

### create_default_gantry()

Factory function to create an AgentGantry with sensible defaults.

```python
from agent_gantry import create_default_gantry

gantry = create_default_gantry()
```

**What it does:**

- Automatically selects NomicEmbedder if available, falls back to SimpleEmbedder
- Uses InMemoryVectorStore
- Uses DirectExecutor
- Good for quick prototyping

---

## Type Hints

Agent-Gantry is fully type-hinted and passes strict mypy checks. Import types as needed:

```python
from agent_gantry.schema.tool import ToolDefinition
from agent_gantry.schema.execution import ToolCall, ToolResult
from agent_gantry.schema.config import AgentGantryConfig
from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
from agent_gantry.adapters.embedders.base import EmbedderAdapter
```

---

## Next Steps

- **[Configuration Reference]({{ '/reference/configuration' | relative_url }})** - Detailed configuration options
- **[LLM SDK Compatibility]({{ '/reference/llm_sdk_compatibility' | relative_url }})** - Provider-specific integration guides
- **[Best Practices]({{ '/architecture/best-practices' | relative_url }})** - Production deployment patterns

---

<div style="display: flex; justify-content: space-between; margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border-color);">

<a href="{{ '/guides/local_persistence_and_skills' | relative_url }}" style="display: flex; flex-direction: column; padding: 1rem; max-width: 45%;">
<span style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Previous</span>
<span style="font-weight: 600; color: var(--text-primary);">← Local Persistence & Skills</span>
</a>

<a href="{{ '/reference/configuration' | relative_url }}" style="display: flex; flex-direction: column; padding: 1rem; max-width: 45%; text-align: right;">
<span style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Next</span>
<span style="font-weight: 600; color: var(--text-primary);">Configuration →</span>
</a>

</div>
