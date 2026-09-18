# examples/protocols

Demonstrations of Agent-Gantry's protocol support for MCP (Model Context Protocol) and A2A
(Agent-to-Agent).

## Files
- `mcp_integration_demo.py`: Starts an MCP server (dynamic mode) and connects to external MCP
  servers. Walks through discovery (`list_tools`), meta-tools, and execution.
- `a2a_integration_demo.py`: Serves Agent-Gantry over HTTP using the A2A protocol and consumes a
  remote agent's skills.
- `dynamic_mcp_selection_demo.py`: Semantic selection of MCP servers, so a prompt connects to the
  one server it needs rather than all of them.
- `jev_mcp_selection_demo.py`: The same job through a decision model instead of embeddings, covering
  both MCP servers and Agent Skills, and ending with the fragment that a selected skill injects
  into the system prompt.
- `claude_desktop_config.json`: Sample configuration for pointing Claude Desktop at the MCP demo.

## Run commands

```bash
python examples/protocols/mcp_integration_demo.py
python examples/protocols/a2a_integration_demo.py
python examples/protocols/jev_mcp_selection_demo.py   # needs agent-gantry[jev,mcp]
```

The MCP demo is great for quick Claude Desktop validation. The A2A demo exposes the generated Agent
Card at `/.well-known/agent.json` and shows how remote skills are brought into the local registry.
