# agent_gantry/adapters/executors

Execution adapters allow Agent-Gantry to dispatch tool calls outside the current process. They all
implement the `ExecutorAdapter` interface in `base.py`, which mirrors the contract expected by the
core `ExecutionEngine`.

## Modules

- `base.py`: Declares the adapter contract (`execute_tool`, `execute_batch`) and shared helpers for
  translating results and errors.
- `a2a_executor.py`: Runs tool calls against remote A2A agents over HTTP, mapping A2A skill metadata
  into `ToolDefinition` objects and converting responses into `ToolResult` instances.
- `mcp_client.py`: Discovers and executes tools hosted on MCP servers (stdio or sockets), handling
  MCP meta-tools as well as direct tool invocations.

## When to use an executor

- **Local tools**: No executor needed; `ExecutionEngine` invokes the Python callable directly.
- **Remote agents (A2A)**: Register an `A2AExecutor` so retrieved tools can call into another agent.
- **External MCP servers**: Use `add_mcp_server` on `AgentGantry`; the MCP client executor is
  attached automatically to discovered tools.

## Minimal example

```python
from agent_gantry import AgentGantry
from agent_gantry.adapters.executors.a2a_executor import A2AExecutor
from agent_gantry.schema.config import A2AAgentConfig

executor = A2AExecutor(agent=A2AAgentConfig(name="calc", url="https://calc.example.com"))
gantry = AgentGantry()
gantry.register_executor(executor)  # makes remote skills available under executor.namespace
```

If you need a custom remote transport, subclass `ExecutorAdapter` and supply it to `AgentGantry`; the
core router and executor require no additional changes.*** End Patch"|()
