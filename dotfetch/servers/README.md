# dotfetch/servers

Server entry points that expose dotfetch over wire protocols so other agents can discover and
invoke your tools.

## Modules

- `a2a_server.py`: FastAPI app factory for serving as an A2A agent. Exposes the Agent Card at
  `/.well-known/agent.json` and implements the `tool_discovery` and `tool_execution` skills.
- `mcp_server.py`: MCP server factory with stdio, Streamable HTTP (`run_http` / `streamable_http_app`) and legacy SSE (`run_sse` / `sse_app`) transports, in dynamic (meta-tools), static (every tool) or hybrid (`expose=[...]` pinned tools + meta-tools) mode.
  Dynamic mode exposes only meta-tools (`find_relevant_tools`, `execute_tool`) to minimize context
  window usage.

## Usage

```python
from dotfetch import DotFetch

fetcher = DotFetch()
@fetcher.register
def echo(text: str) -> str:
    return text

await fetcher.sync()
await fetcher.serve_mcp(transport="stdio", mode="dynamic")  # or "http" / "sse"; or fetcher.serve_a2a(port=8080)
```

Server helpers are thin wrappers around the core facade and adapters. They do not require additional
configuration beyond what is already present in `DotFetchConfig`.***
