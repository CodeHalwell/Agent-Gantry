# Agno

> Part of the `dotfetch` skill. Core concepts, debugging and common
> pitfalls are in [`SKILL.md`](../SKILL.md); read those first.

Install with `uv add "dotfetch[agno]"` (pins `agno>=2.6.10`; tested on 2.6.10 and the latest).

```python
from dotfetch.agno import AgnoAdapter, current_run_context

agent = AgnoAdapter(fetcher).agent(model=my_model, limit=5)   # error_formatter=... on the adapter
run = await agent.arun("refund order 4471", user_id="dan", dependencies={"db": db})
if run.is_paused:                                             # a dotfetch-gated call
    for requirement in run.active_requirements:
        requirement.confirm()
    run = await agent.acontinue_run(run_response=run)
```

- **Errors**: `AgnoToolError` (a `ToolExecutionError`), which Agno records as a failed tool result showing `"<type>: <message>"`; `error_mode="raise"` raises the plain `ToolExecutionError`.
- **Approval**: gated tools are `Function(requires_confirmation=True)`; a confirmed requirement runs the call once on `continue_run`, a rejected one never runs. Direct calls report "needs approval".
- **Run context**: `current_run_context()` gives Agno's `RunContext` (`.dependencies`, `.session_state`, `.user_id`, `.session_id`, `.run_id`).
- **Per-run**: `agent()` re-selects inside Agno (callable `tools` factory) for each run and keeps the selection for `continue_run`; works with `run()` and `arun()`.
