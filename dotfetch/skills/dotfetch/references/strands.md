# Strands Agents

> Part of the `dotfetch` skill. Core concepts, debugging and common
> pitfalls are in [`SKILL.md`](../SKILL.md); read those first.

Install with `uv add "dotfetch[strands]"` (pins `strands-agents>=1.38.0`; tested on 1.38.0 and the latest).

```python
from dotfetch.strands import StrandsAdapter, current_tool_context

agent = StrandsAdapter(fetcher).agent(model=my_model, limit=5)   # error_formatter=... on the adapter
result = await agent.invoke_async("refund order 4471", user=current_user)
if result.stop_reason == "interrupt":                            # a dotfetch-gated call
    [interrupt] = result.interrupts
    result = await agent.invoke_async([{"interruptResponse": {"interruptId": interrupt.id, "response": "yes"}}])
```

- **Errors**: error `ToolResult` (`{"status": "error", "content": [{"text": "<type>: <message>"}]}`) everywhere; `error_mode="raise"` raises `ToolExecutionError`.
- **Approval**: gated tools call `tool_context.interrupt("fetcher-approval", reason=...)`; "yes"/`True`/`{"approved": True}` runs once on resume, anything else rejects. Direct calls report "needs approval".
- **Run context**: `current_tool_context()` gives Strands' `ToolContext` (`.invocation_state` = kwargs passed to `agent(...)`, `.agent`, `.tool_use`).
- **Per-turn**: `tool_hook()` swaps the tool registry in `BeforeModelCallEvent`; on a retrieval failure the previous turn's tools stay.
