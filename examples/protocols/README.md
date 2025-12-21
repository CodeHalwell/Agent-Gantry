# examples/protocols

Demonstrations of Agent-Gantry's protocol support for MCP (Model Context Protocol) and A2A
(Agent-to-Agent).

## Files
- `mcp_integration_demo.py`: Starts an MCP server (dynamic mode) and connects to external MCP
  servers. Walks through discovery (`list_tools`), meta-tools, and execution.
- `a2a_integration_demo.py`: Serves Agent-Gantry over HTTP using the A2A protocol and consumes a
  remote agent's skills.
- `claude_desktop_config.json`: Sample configuration for pointing Claude Desktop at the MCP demo.

## Run commands

```bash
python examples/protocols/mcp_integration_demo.py
python examples/protocols/a2a_integration_demo.py
```

The MCP demo is great for quick Claude Desktop validation. The A2A demo exposes the generated Agent
Card at `/.well-known/agent.json` and shows how remote skills are brought into the local registry.
