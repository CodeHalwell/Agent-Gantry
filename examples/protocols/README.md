# examples/protocols

Demonstrations of Agent-Gantry's protocol support for MCP (Model Context Protocol) and A2A
(Agent-to-Agent). None of the Python demos opens a network connection or spawns a server: they
build the same objects the servers and clients use, run the routing locally, and print the one-line
call that would go live.

## Files
- `mcp_integration_demo.py`: What an MCP client sees in dynamic mode (two meta-tools), how an
  external server is configured for `add_mcp_server`, the `find_relevant_tools` -> `execute_tool`
  flow run against the gantry directly, and a measured comparison of schema text with and without
  dynamic mode.
- `a2a_integration_demo.py`: Generates the Agent Card served at `/.well-known/agent.json`,
  configures an external agent, and shows how a (mocked) remote agent's skills map onto
  `ToolDefinition` entries. Prints the `serve_a2a` call rather than binding a port.
- `dynamic_mcp_selection_demo.py`: Registers MCP servers by metadata only, syncs it and uses
  `retrieve_mcp_servers` to pick the one server a prompt needs. Connecting to it
  (`discover_tools_from_server`) is shown but not run. Needs `agent-gantry[mcp]`.
- `jev_mcp_selection_demo.py`: The same job through a decision model instead of embeddings, covering
  both MCP servers and Agent Skills, and ending with the fragment that a selected skill injects
  into the system prompt. Runs without a key (falls back to semantic search).
- `claude_desktop_config.json`: Points Claude Desktop at the `agent-gantry serve-mcp` CLI over
  stdio. Replace `my_project.tools` with the module that holds your `AgentGantry` instance
  (`pkg.module` or `pkg.module:attr`; the attribute defaults to `tools`). Drop the `--module`
  argument to serve the CLI's built-in demo tools instead.

## Run commands

```bash
python examples/protocols/mcp_integration_demo.py
python examples/protocols/a2a_integration_demo.py
python examples/protocols/dynamic_mcp_selection_demo.py   # needs agent-gantry[mcp]
python examples/protocols/jev_mcp_selection_demo.py       # needs agent-gantry[jev,mcp]; runs without a key
```

To serve for real: `agent-gantry serve-mcp --module my_project.tools` (stdio, dynamic mode) for
Claude Desktop or Claude Code, or `gantry.serve_a2a(host, port)` for A2A, which needs
`agent-gantry[a2a]` (FastAPI + uvicorn) and exposes the Agent Card at `/.well-known/agent.json`
and tasks at `/tasks/send`.
