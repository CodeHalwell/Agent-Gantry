# agent_gantry/servers

Server entry points that expose Agent-Gantry over wire protocols so other agents can discover and
invoke your tools.

## Modules

- `a2a_server.py`: FastAPI app factory for serving as an A2A agent. Exposes the Agent Card at
  `/.well-known/agent.json` and implements the `tool_discovery` and `tool_execution` skills.
- `mcp_server.py`: MCP server factory supporting stdio/SSE transports in static or dynamic mode.
  Dynamic mode exposes only meta-tools (`find_relevant_tools`, `execute_tool`) to minimize context
  window usage.

## Usage

```python
from agent_gantry import AgentGantry

gantry = AgentGantry()
@gantry.register
def echo(text: str) -> str:
    return text

await gantry.sync()
await gantry.serve_mcp(transport="stdio", mode="dynamic")  # or gantry.serve_a2a(port=8080)
```

Server helpers are thin wrappers around the core facade and adapters. They do not require additional
configuration beyond what is already present in `AgentGantryConfig`.***
