# agent_gantry/adapters/executors

Execution adapters that delegate tool calls or discover tools from remote systems.

- `__init__.py`: Exposes executor adapter types.
- `base.py`: Common interface for executor adapters and shared helpers.
- `a2a_executor.py`: Runs tool calls against remote A2A agents via HTTP.
- `mcp_client.py`: Discovers and executes tools hosted on MCP servers.
