# Phase 6: A2A Integration

## Overview

Phase 6 implements the Agent-to-Agent (A2A) protocol, enabling AgentGantry to both consume external A2A agents as tools and expose itself as an A2A agent that other systems can interact with.

## Key Features

### 1. Agent Card

The Agent Card is a JSON document served at `/.well-known/agent.json` that describes an agent's capabilities:

```json
{
  "name": "AgentGantry",
  "description": "Intelligent tool routing and execution service",
  "url": "http://localhost:8080",
  "version": "1.0.0",
  "skills": [
    {
      "id": "tool_discovery",
      "name": "Tool Discovery",
      "description": "Find relevant tools for a given task using semantic search"
    },
    {
      "id": "tool_execution",
      "name": "Tool Execution",
      "description": "Execute registered tools with retries and timeouts"
    }
  ]
}
```

### 2. A2A Client

The A2A client (`A2AClient`) discovers and interacts with external A2A agents:

```python
from agent_gantry.providers.a2a_client import A2AClient
from agent_gantry.schema.config import A2AAgentConfig

config = A2AAgentConfig(
    name="external-agent",
    url="https://external-agent.example.com",
    namespace="external"
)

client = A2AClient(config)

# Discover agent capabilities
agent_card = await client.discover()

# List tools (skills)
tools = await client.list_tools()

# Send task to agent
result = await client.send_task(
    skill_id="translation",
    query="Translate 'hello' to Spanish",
    metadata={"trace_id": "123"}
)
```

### 3. A2A Server

The A2A server exposes AgentGantry as an A2A agent via FastAPI:

```python
from agent_gantry import AgentGantry

gantry = AgentGantry()

@gantry.register
def analyze_data(data: str) -> str:
    """Analyze data and provide insights."""
    return f"Analysis: {data}"

await gantry.sync()

# Start A2A server
gantry.serve_a2a(host="0.0.0.0", port=8080)
```

The server exposes:
- **/.well-known/agent.json** - Agent Card endpoint
- **/tasks/send** - JSON-RPC task execution endpoint

### 4. Skill Mapping

External agent skills are automatically mapped to ToolDefinition objects:

- **Skill ID**: `translate_text`
- **Tool Name**: `a2a_external_agent_translate_text`
- **Namespace**: Configured namespace (default: "default")
- **Source**: `ToolSource.A2A_AGENT`
- **Metadata**: Contains agent name, URL, skill ID, and input/output modes

### 5. A2A Executor

The `A2AExecutor` handles execution of tools that originate from A2A agents:

- Validates that the tool is from an A2A agent
- Extracts skill_id and agent metadata
- Sends JSON-RPC task request to the remote agent
- Returns structured ToolResult

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AgentGantry                            │
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                │
│  │ A2A Client   │         │  A2A Server  │                │
│  │              │         │              │                │
│  │ - discover() │         │ - Agent Card │                │
│  │ - list_tools│         │ - tasks/send │                │
│  │ - send_task()│         │              │                │
│  └──────┬───────┘         └──────┬───────┘                │
│         │                        │                        │
│         ▼                        ▼                        │
│  ┌──────────────────────────────────────┐                │
│  │         Tool Registry                 │                │
│  │  - Python functions                   │                │
│  │  - MCP tools                          │                │
│  │  - A2A agent skills                   │                │
│  └──────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
           │                            │
           ▼                            ▼
    External A2A Agents          External Clients
```

## Skills Provided by AgentGantry

When AgentGantry is exposed as an A2A agent, it provides two skills:

### 1. tool_discovery

**Description**: Find relevant tools for a given task using semantic search

**Input**: Text query describing what the user wants to accomplish

**Output**: JSON containing discovered tools with their names, descriptions, and schemas

**Example**:
```json
{
  "query": "calculate sum of numbers",
  "tools_found": 2,
  "tools": [
    {
      "name": "calculate_sum",
      "description": "Calculate the sum of two numbers",
      "parameters": { ... }
    }
  ]
}
```

### 2. tool_execution

**Description**: Execute a registered tool by name with provided arguments

**Input**: Text query (in production, would be structured JSON with tool name and arguments)

**Output**: Result of tool execution or error message

**Note**: The current implementation returns a message about requiring structured input. In production deployments, this would accept JSON-RPC formatted tool calls.

## Usage Examples

### Consuming External A2A Agents

```python
from agent_gantry import AgentGantry
from agent_gantry.schema.config import A2AAgentConfig

gantry = AgentGantry()

# Configure external agent
external_agent = A2AAgentConfig(
    name="translation-agent",
    url="https://translation-agent.example.com",
    namespace="external"
)

# Discover and register agent's skills
count = await gantry.add_a2a_agent(external_agent)
print(f"Added {count} skills from external agent")

# Use agent skills like regular tools
tools = await gantry.retrieve_tools("translate hello to Spanish")
# Returns: a2a_translation_agent_translate skill
```

### Serving AgentGantry as A2A Agent

```python
from agent_gantry import AgentGantry

gantry = AgentGantry()

# Register your tools
@gantry.register(tags=["math"])
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b

await gantry.sync()

# Start A2A server (blocking call)
gantry.serve_a2a(host="0.0.0.0", port=8080)
```

### Programmatic Agent Card Access

```python
from agent_gantry.servers.a2a_server import generate_agent_card

# Generate agent card
agent_card = generate_agent_card(gantry, "http://localhost:8080")

print(f"Agent: {agent_card.name}")
print(f"Skills: {len(agent_card.skills)}")

for skill in agent_card.skills:
    print(f"  - {skill.name}: {skill.description}")
```

## Configuration

### A2AAgentConfig

Configuration for connecting to an external A2A agent:

```python
from agent_gantry.schema.config import A2AAgentConfig

config = A2AAgentConfig(
    name="agent-name",          # Unique identifier
    url="https://agent.com",    # Base URL
    namespace="namespace"       # Namespace for tools (default: "default")
)
```

### A2AConfig

Configuration for A2A integration in AgentGantryConfig:

```python
from agent_gantry.schema.config import AgentGantryConfig, A2AConfig, A2AAgentConfig

config = AgentGantryConfig(
    a2a=A2AConfig(
        agents=[
            A2AAgentConfig(
                name="external-agent",
                url="https://external.com",
                namespace="external"
            )
        ],
        serve_a2a=True,     # Whether to serve as A2A agent
        a2a_port=8080       # Port for A2A server
    )
)

gantry = AgentGantry.from_config(config)
```

## Installation

A2A integration requires optional dependencies:

```bash
pip install agent-gantry[a2a]
```

This installs:
- `httpx` - HTTP client for A2A communication
- `fastapi` - Web framework for A2A server
- `uvicorn` - ASGI server for FastAPI

## Testing

Run A2A tests:

```bash
pytest tests/test_phase6_a2a.py -v
```

Test coverage includes:
- Agent Card generation
- Agent Skill model
- A2A client discovery and task sending
- Skill to ToolDefinition conversion
- Server endpoint functionality
- End-to-end integration

## Security Considerations

### Authentication

The Agent Card supports authentication configuration:

```python
agent_card = AgentCard(
    name="SecureAgent",
    description="Agent with authentication",
    url="https://secure-agent.example.com",
    skills=[...],
    authentication={
        "type": "bearer",
        "required": True
    }
)
```

### Best Practices

1. **Use HTTPS** for production A2A agents
2. **Validate inputs** from external agents before execution
3. **Rate limit** A2A endpoints to prevent abuse
4. **Monitor** A2A agent interactions via telemetry
5. **Circuit breakers** protect against failing external agents

## Protocol Details

### JSON-RPC Format

A2A uses JSON-RPC 2.0 for task requests:

```json
{
  "jsonrpc": "2.0",
  "method": "tasks/send",
  "params": {
    "skill_id": "tool_discovery",
    "messages": [
      {
        "role": "user",
        "parts": [
          {
            "type": "text",
            "text": "find tools for math calculations"
          }
        ]
      }
    ],
    "metadata": {}
  },
  "id": 1
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "result": {
    "status": "success",
    "result": { ... }
  },
  "id": 1
}
```

## Future Enhancements

- **Streaming responses** for long-running tasks
- **Webhook support** for async task completion
- **Enhanced authentication** (OAuth, mTLS)
- **Agent registry** for discovery of multiple agents
- **Task cancellation** support
- **Batch task execution** across multiple agents

## References

- [Plan.md - Phase 6 Specification](../plan.md)
- [A2A Protocol Documentation](https://google.github.io/A2A/)
- [Example: A2A Integration Demo](../examples/a2a_integration_demo.py)
