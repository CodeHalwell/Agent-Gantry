# dotfetch/cli

The command-line interface ships as the `dotfetch` entry point (`python -m dotfetch.cli`
works too). It is a thin argparse front end over the `DotFetch` facade in `core/fetcher.py`.

## Pointing it at your registry

Every inspection command accepts `--module pkg.tools[:attr]` naming the module that holds your
`DotFetch` instance (attribute `tools` by default, or `--attr NAME`). With one `--module` the
CLI uses that instance directly, so it sees your configured embedder and vector store; with
several, their tools are merged into one fresh fetcher (optionally built from `--config path.yaml`).
Without `--module` a three-tool demo registry is used and a note is printed on stderr.

## Commands

- `dotfetch list [--namespace NS]` — print the registered tools.
- `dotfetch search "<query>" [--limit N] [--namespace NS]` — semantic retrieval with scores.
- `dotfetch lint` — flag description cross-references, near-duplicate tools, over-used tags and tools registered without `examples=[...]`. Add `--source PATH` (repeatable) to also scan Python files for non-zero `score_threshold` overrides with no justifying comment (or a comment claiming they *relax* the filter — it is an absolute cosine cutoff, so they only tighten it) and for `__main__` scripts that build a `DotFetch` and never `close()` it. `--source` on its own runs only the file scan and builds no fetcher.
- `dotfetch sim tool_a tool_b` — cosine similarity between two tools' searchable text.
- `dotfetch sync [--dry-run] [--force] [--prune]` — embed changed tools into the vector store;
  `--prune` also removes stored tools that are no longer registered.
- `dotfetch serve-mcp [--transport stdio|http|sse] [--mode dynamic|static|hybrid]
  [--expose TOOL ...] [--host H] [--port P] [--path PATH]` — expose the registry as an MCP server.
  `stdio` is what Claude Desktop / Claude Code launch; `http` is the Streamable HTTP transport for
  remote clients. `--path` defaults to each transport's own endpoint (`/mcp` for `http`, `/sse` for
  `sse`), and the printed address is the one the server will serve.
- `dotfetch install-skill [--claude | --target DIR] [--overwrite] [--print-path]` — vendor the
  bundled Claude Skill.

```bash
dotfetch search "refund an order" --module my_app.tools --limit 3
dotfetch serve-mcp --module my_app.tools:fetcher --mode hybrid --expose get_weather
```

For Claude Desktop, register the stdio server in `claude_desktop_config.json`:

```json
{"mcpServers": {"my-tools": {"command": "dotfetch",
                             "args": ["serve-mcp", "--module", "my_app.tools"]}}}
```
