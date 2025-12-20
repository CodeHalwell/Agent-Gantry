# agent_gantry/servers

Server entry points that expose Agent-Gantry over different protocols.

- `__init__.py`: Server package exports.
- `a2a_server.py`: FastAPI app factory for serving Agent-Gantry as an A2A agent.
- `mcp_server.py`: MCP server factory for stdio/SSE transports with dynamic/static modes.
