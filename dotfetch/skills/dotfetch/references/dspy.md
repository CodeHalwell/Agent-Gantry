# DSPy

> Part of the `dotfetch` skill. Core concepts, debugging and common
> pitfalls are in [`SKILL.md`](../SKILL.md); read those first.

Install with `uv add "dotfetch[dspy]"` (pins `dspy>=3.5.0`, the redesigned `dspy.ReAct` with native `tool_calls` and a `submit` tool; tested on 3.5.0 and the latest).

```python
import dspy
from dotfetch.dspy import DSPyAdapter, current_dspy_module

adapter = DSPyAdapter(fetcher, approve=lambda request: request.tool_name != "delete_account")
react = dspy.ReAct("question -> answer", tools=await adapter.select("refund order 4471", limit=5))
pred = react(question="refund order 4471")          # or: await react.acall(...)
builder = adapter.agent_builder("question -> answer")  # per call: react = await builder.build(query)
```

- **Errors**: `DSPyToolError` (a `ToolExecutionError`), which `dspy.ReAct` records as an error tool result (`is_error=True`); the text ends with `"<type>: <message>"` after DSPy's own short traceback, with no chained original error, so `error_formatter` redaction holds. `error_mode="raise"` raises the plain `ToolExecutionError`.
- **Approval**: DSPy has no HITL, so `approve=fn` decides gated calls (`ApprovalRequest`: `tool_name`, `arguments`, `description`, `module`); `True` runs the call once, anything else (or a raising approver) blocks it. Without an approver, gated calls report "needs approval".
- **Run context**: `current_dspy_module()` gives the module running the call (normally the `dspy.ReAct`). DSPy has no run or call ids.
- **Per-run**: `dspy.ReAct` fixes its tools at construction, so `agent_builder()` builds a fresh one per query; pins are offered if retrieval fails.
