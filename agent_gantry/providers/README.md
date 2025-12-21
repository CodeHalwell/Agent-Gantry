# agent_gantry/providers

Providers pull external skills/tools into the local registry so they can be routed and executed just
like native Python callables. They are typically used by higher-level server helpers (`serve_a2a`,
`add_a2a_agent`, etc.).

## Modules

- `a2a_client.py`: Discovers skills from remote A2A agents, converts them into `ToolDefinition`
  objects, and wires them to the A2A executor for remote invocation. Handles agent cards and
  namespace scoping.

## Example

```python
from agent_gantry import AgentGantry
from agent_gantry.schema.config import A2AAgentConfig

gantry = AgentGantry()
await gantry.add_a2a_agent(
    A2AAgentConfig(name="reports", url="https://reports.example.com", namespace="reports")
)

tools = await gantry.retrieve_tools("generate revenue report", limit=5)
```

The retrieved tools will include skills exposed by the remote agent and will execute through the
configured A2A executor transparently.***
