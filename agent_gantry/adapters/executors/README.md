# agent_gantry/adapters/executors

Execution adapters allow Agent-Gantry to dispatch tool calls outside the current process. They all
implement the `ExecutorAdapter` interface in `base.py`, which mirrors the contract expected by the
core `ExecutionEngine`.

## Modules

- `base.py`: Declares the `ExecutorAdapter` protocol, whose one method is `execute(tool, call, handler)`.
- `a2a_executor.py`: Runs tool calls against remote A2A agents over HTTP, mapping A2A skill metadata
  into `ToolDefinition` objects and converting responses into `ToolResult` instances.
- `mcp_client.py`: Discovers and executes tools hosted on MCP servers (local stdio subprocesses, or remote Streamable HTTP / SSE endpoints via `MCPServerConfig(url=...)`), handling
  MCP meta-tools as well as direct tool invocations.

## When to use an executor

- **Local tools**: No executor needed; `ExecutionEngine` invokes the Python callable itself, in
  this process. There is no sandboxed or containerised executor; `ExecutionConfig` refuses
  `enable_sandbox=True` for that reason.
- **Remote agents (A2A)**: `await gantry.add_a2a_agent(...)` discovers the agent's skills and
  registers them; the engine owns the `A2AExecutor` that calls them.
- **External MCP servers**: Use `add_mcp_server` on `AgentGantry`; the MCP client executor is
  attached automatically to discovered tools.

## Minimal example

```python
from agent_gantry import AgentGantry
from agent_gantry.schema.config import A2AAgentConfig

gantry = AgentGantry()
# Fetches the agent card, registers each skill as a tool under the "calc" namespace,
# and returns how many it found. The engine's A2AExecutor makes the calls.
count = await gantry.add_a2a_agent(A2AAgentConfig(name="calc", url="https://calc.example.com"))
```

`AgentGantry` has no hook for supplying an executor of your own: `ExecutionEngine` picks the A2A
executor for tools whose `source` is an A2A agent, the MCP client for MCP tools, and otherwise calls
the registered Python handler. A new remote transport means extending the engine.
