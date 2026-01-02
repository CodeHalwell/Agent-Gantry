---
layout: default
title: Troubleshooting
nav_order: 10
description: "Common issues and solutions for Agent-Gantry"
---

# Troubleshooting

Solutions to common issues when using Agent-Gantry.

---

## Installation Issues

### ImportError: No module named 'agent_gantry'

**Problem:** Package not installed or installed in wrong environment.

**Solution:**

```bash
# Ensure you're in the correct virtual environment
which python  # Should point to your venv

# Install the package
pip install agent-gantry

# Verify installation
python -c "import agent_gantry; print(agent_gantry.__version__)"
```

### ModuleNotFoundError: No module named 'nomic'

**Problem:** Optional dependency not installed.

**Solution:**

```bash
# Install with nomic support
pip install agent-gantry[nomic]

# Or install all optional dependencies
pip install agent-gantry[all]
```

### ImportError: cannot import name 'AsyncOpenAI'

**Problem:** Outdated OpenAI SDK version.

**Solution:**

```bash
# Upgrade OpenAI SDK
pip install --upgrade openai

# Requires openai >= 1.0.0
```

## Tool Registration Issues

### Tools Not Being Selected Correctly

**Problem:** Semantic search returns irrelevant tools.

**Diagnostic Steps:**

1. **Check if tools are synced:**

```python
await gantry.sync()
print(f"Synced {len(gantry._registry._tools)} tools")
```

2. **Inspect tool descriptions:**

```python
tools = gantry.list_tools()
for tool in tools:
    print(f"{tool.name}: {tool.description}")
    print(f"  Tags: {tool.tags}")
```

3. **Test vector search directly:**

```python
tools = await gantry.retrieve_tools("your query", limit=5)
for tool in tools:
    print(f"{tool.name} - {tool.description}")
```

**Solutions:**

- **Add more descriptive tool descriptions**

```python
# ❌ Too vague
@gantry.register
def fetch(url: str) -> str:
    """Fetch URL."""
    ...

# ✅ Clear and detailed
@gantry.register(
    tags=["web", "http", "download", "scraping"],
    description="Fetch and download content from any web URL using HTTP/HTTPS"
)
def fetch_url(url: str) -> str:
    """
    Fetch web page content from a URL.

    Supports HTTP and HTTPS protocols. Returns the page content as text.
    """
    ...
```

- **Use relevant tags:**

```python
@gantry.register(
    tags=["weather", "forecast", "temperature", "meteorology"],
    description="Get current weather conditions for any city"
)
def get_weather(city: str) -> str:
    ...
```

- **Ensure embedder quality:**

```python
# Use NomicEmbedder for better semantic understanding
from agent_gantry.adapters.embedders.nomic import NomicEmbedder

gantry = AgentGantry(embedder=NomicEmbedder(dimension=768))
```

### "No default gantry set" Error

**Problem:** Using `@with_semantic_tools` without setting default gantry.

**Error:**

```
RuntimeError: No default AgentGantry instance set. Call set_default_gantry() first.
```

**Solution:**

```python
from agent_gantry import AgentGantry, set_default_gantry

gantry = AgentGantry()
set_default_gantry(gantry)  # ← This is required!

@with_semantic_tools(limit=3)
async def chat(prompt: str, *, tools=None):
    ...
```

**Alternative:** Pass gantry explicitly (deprecated pattern):

```python
@with_semantic_tools(gantry, limit=3)
async def chat(prompt: str, *, tools=None):
    ...
```

### Tool Registration Doesn't Update

**Problem:** Tool changes not reflected after modification.

**Solution:**

Force re-sync:

```python
# Re-sync with force=True to re-embed all tools
await gantry.sync(force=True)
```

Or delete and re-register:

```python
gantry.delete_tool("my_tool")

@gantry.register
def my_tool(arg: str) -> str:
    """Updated implementation."""
    ...

await gantry.sync()
```

## Tool Execution Issues

### ToolExecutionError: Tool not found

**Problem:** Attempting to execute a tool that isn't registered.

**Diagnostic:**

```python
# Check if tool exists
tool = gantry.get_tool("my_tool_name")
if tool is None:
    print("Tool not registered")
else:
    print(f"Found tool: {tool.name}")

# List all tools
all_tools = gantry.list_tools()
print(f"Registered tools: {[t.name for t in all_tools]}")
```

**Solution:**

- Ensure tool is registered before execution
- Check spelling of tool name
- Verify correct namespace (if using namespaces)

### TypeError: execute() missing required argument

**Problem:** Incorrect ToolCall structure.

**Error:**

```
TypeError: ToolCall.__init__() missing 1 required positional argument: 'arguments'
```

**Solution:**

```python
from agent_gantry.schema.execution import ToolCall

# ✅ Correct usage
result = await gantry.execute(ToolCall(
    tool_name="get_weather",
    arguments={"city": "Paris"}  # Dict of arguments
))

# ❌ Wrong - missing arguments
result = await gantry.execute(ToolCall(tool_name="get_weather"))
```

### Tool Execution Timeouts

**Problem:** Tools taking too long to execute.

**Error:**

```
TimeoutError: Tool execution exceeded timeout of 30.0s
```

**Solutions:**

1. **Increase timeout:**

```python
result = await gantry.execute(
    ToolCall(tool_name="slow_tool", arguments={}),
    timeout=60.0  # 60 seconds
)
```

2. **Configure global timeout:**

```python
from agent_gantry.schema.config import AgentGantryConfig, ExecutorConfig

config = AgentGantryConfig(
    executor=ExecutorConfig(timeout=60.0)
)
gantry = AgentGantry(config=config)
```

3. **Optimize tool implementation:**

```python
# Use async operations
import httpx

@gantry.register
async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=10.0)
        return response.json()
```

## Vector Store Issues

### "Dimension mismatch" Error

**Problem:** Vector store dimension doesn't match embedder dimension.

**Error:**

```
ValueError: Expected embedding dimension 768, got 1536
```

**Solution:**

Ensure vector store and embedder dimensions match:

```python
from agent_gantry.adapters.embedders.nomic import NomicEmbedder
from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

# Both use 768 dimensions
embedder = NomicEmbedder(dimension=768)
vector_store = LanceDBVectorStore(
    db_path="tools.lancedb",
    dimension=768  # ← Must match embedder
)

gantry = AgentGantry(
    embedder=embedder,
    vector_store=vector_store
)
```

### LanceDB Permission Denied

**Problem:** Cannot write to LanceDB directory.

**Error:**

```
PermissionError: [Errno 13] Permission denied: 'gantry_tools.lancedb'
```

**Solution:**

```bash
# Ensure directory is writable
chmod -R 755 gantry_tools.lancedb

# Or use a different path
mkdir -p /tmp/gantry_tools
chmod 755 /tmp/gantry_tools
```

```python
gantry = AgentGantry(
    vector_store=LanceDBVectorStore(db_path="/tmp/gantry_tools/db.lancedb")
)
```

### "No embeddings found" Error

**Problem:** Calling `retrieve_tools()` before `sync()`.

**Solution:**

Always sync after registering tools:

```python
# Register tools
@gantry.register
def my_tool() -> str:
    return "result"

# Sync before using retrieve_tools()
await gantry.sync()

# Now you can search
tools = await gantry.retrieve_tools("query")
```

## MCP Integration Issues

### MCP Server Connection Failed

**Problem:** Cannot connect to MCP server.

**Error:**

```
MCPConnectionError: Failed to start MCP server 'filesystem'
```

**Diagnostic:**

1. **Verify command works manually:**

```bash
npx -y @modelcontextprotocol/server-filesystem --path /tmp
```

2. **Check if npx is installed:**

```bash
which npx
npm --version
```

3. **Review server configuration:**

```python
server = gantry._mcp_registry.get_server("filesystem")
if server:
    print(f"Command: {server.command}")
    print(f"Args: {server.args}")
    print(f"Env: {server.env}")
```

**Solutions:**

- **Install required dependencies:**

```bash
# For Node.js based MCP servers
npm install -g npm
```

- **Use absolute paths:**

```python
gantry.register_mcp_server(
    name="filesystem",
    command=["/usr/local/bin/npx", "-y", "@modelcontextprotocol/server-filesystem"],
    args=["--path", "/absolute/path/to/dir"]
)
```

- **Set environment variables:**

```python
gantry.register_mcp_server(
    name="custom",
    command=["python", "-m", "my_mcp_server"],
    env={"CUSTOM_VAR": "value"}
)
```

### MCP Server Timeout

**Problem:** Server takes too long to respond.

**Solution:**

Increase timeout:

```python
count = await gantry.discover_tools_from_server(
    "filesystem",
    timeout=60.0  # 60 seconds
)
```

### MCP Tools Not Appearing

**Problem:** `discover_tools_from_server()` returns 0 tools.

**Diagnostic:**

```python
count = await gantry.discover_tools_from_server("filesystem")
print(f"Discovered {count} tools")

# Check if server is healthy
server = gantry._mcp_registry.get_server("filesystem")
if server:
    print(f"Health: {server.health}")
```

**Solution:**

- Ensure server supports MCP protocol correctly
- Check server logs for errors
- Verify server is running and accessible

## LLM Integration Issues

### "Tools not being passed to LLM"

**Problem:** LLM doesn't receive tools.

**Diagnostic:**

```python
@with_semantic_tools(limit=3, dialect="openai")
async def chat(prompt: str, *, tools=None):
    print(f"Tools received: {tools}")  # Debug print
    ...
```

**Solution:**

Ensure `tools=None` is a **keyword argument**:

```python
# ✅ Correct - keyword argument
async def chat(prompt: str, *, tools=None):
    ...

# ❌ Wrong - positional argument
async def chat(prompt: str, tools=None):
    ...
```

### "Dialect mismatch" Error

**Problem:** Wrong tool format for LLM provider.

**Solution:**

Use correct dialect:

```python
# OpenAI, Groq, Mistral (OpenAI-compatible)
@with_semantic_tools(dialect="openai")
async def chat_openai(prompt: str, *, tools=None):
    ...

# Anthropic Claude
@with_semantic_tools(dialect="anthropic")
async def chat_anthropic(prompt: str, *, tools=None):
    ...

# Google Gemini
@with_semantic_tools(dialect="gemini")
async def chat_gemini(prompt: str, *, tools=None):
    ...
```

### LLM Generates Invalid Tool Calls

**Problem:** LLM hallucinates tool names or arguments.

**Solution:**

Add validation:

```python
import json

for tool_call in response.choices[0].message.tool_calls:
    # Validate tool exists
    tool = gantry.get_tool(tool_call.function.name)
    if not tool:
        print(f"Warning: LLM called non-existent tool '{tool_call.function.name}'")
        continue

    # Validate arguments can be parsed
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in tool arguments: {e}")
        continue

    # Execute
    result = await gantry.execute(ToolCall(
        tool_name=tool.name,
        arguments=args
    ))
```

## Performance Issues

### Slow Tool Retrieval

**Problem:** `retrieve_tools()` is slow.

**Solutions:**

1. **Use local embedder:**

```python
# Instead of OpenAI (network latency)
from agent_gantry.adapters.embedders.nomic import NomicEmbedder

gantry = AgentGantry(embedder=NomicEmbedder(dimension=768))
```

2. **Use persistent vector store:**

```python
# LanceDB is faster than Qdrant for local deployments
from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

gantry = AgentGantry(
    vector_store=LanceDBVectorStore(db_path="tools.lancedb")
)
```

3. **Reduce search limit:**

```python
# Retrieve fewer tools
tools = await gantry.retrieve_tools("query", limit=3)  # Instead of 10
```

### High Memory Usage

**Problem:** Agent-Gantry using too much memory.

**Solutions:**

- Use smaller embedding dimensions
- Clear vector store periodically
- Use disk-based vector store (LanceDB) instead of in-memory

## Common Error Messages

### "Circuit breaker is open"

**Meaning:** Tool has failed repeatedly and is temporarily disabled.

**Solution:**

Wait for circuit breaker to reset (default: 60 seconds) or manually reset:

```python
# Circuit breakers auto-reset after timeout
# Or force enable the tool
gantry._executor._circuit_breakers.pop("tool_name", None)
```

### "Tool execution failed: [Errno 2] No such file or directory"

**Meaning:** Tool tries to access non-existent file.

**Solution:**

- Validate file paths in tool implementation
- Use absolute paths
- Check file permissions

## Getting Help

If you can't find a solution here:

1. **Check the GitHub Issues:** [https://github.com/CodeHalwell/Agent-Gantry/issues](https://github.com/CodeHalwell/Agent-Gantry/issues)
2. **Review the Examples:** [https://github.com/CodeHalwell/Agent-Gantry/tree/main/examples](https://github.com/CodeHalwell/Agent-Gantry/tree/main/examples)
3. **Read the API Reference:** [API Documentation]({{ '/reference/api-reference' | relative_url }})
4. **Open a New Issue:** [Create Issue](https://github.com/CodeHalwell/Agent-Gantry/issues/new)

When reporting issues, include:
- Agent-Gantry version (`agent_gantry.__version__`)
- Python version
- Minimal reproducible example
- Full error traceback
- Operating system

---

**Still stuck?** Open an issue on GitHub with the `question` label.

