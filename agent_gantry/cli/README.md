# agent_gantry/cli

The command-line interface ships a small set of utilities for inspecting tools and running demo
servers. It is implemented with Click in `main.py` and is available as the `agent-gantry` entry
point after installation.

## Commands

- `agent-gantry list`: Bootstraps demo tools using the in-memory embedder/vector store and prints the
  registered tools.
- `agent-gantry search "<query>" --limit 3`: Runs semantic retrieval over the demo registry and shows
  scored results.
- `agent-gantry serve-mcp`: Starts an MCP server in dynamic mode with the demo tools (useful for
  quick Claude Desktop trials).

You can also invoke it module-style:

```bash
python -m agent_gantry.cli search "refund an order" --limit 2
```

The CLI is intentionally minimal and delegates all heavy lifting to the `AgentGantry` facade in
`core/gantry.py`. If you want to add new commands, follow the patterns in `main.py` so the tooling
stays consistent with the rest of the codebase.***
