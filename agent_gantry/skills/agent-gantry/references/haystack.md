# Haystack

> Part of the `agent-gantry` skill. Core concepts, debugging and common
> pitfalls are in [`SKILL.md`](../SKILL.md); read those first.

Install with `uv add "agent-gantry[haystack]"` (pins `haystack-ai>=3.0.0`; tested on 3.0.0 and the latest).

```python
from agent_gantry.haystack import HaystackAdapter, current_agent_state
from haystack.hooks.human_in_the_loop import SimpleConsoleUI

agent = HaystackAdapter(gantry).agent(
    chat_generator=generator, limit=5, confirmation_ui=SimpleConsoleUI()  # error_formatter=... on the adapter
)
result = agent.run(messages=[ChatMessage.from_user("refund order 4471")])
```

- **Errors**: `HaystackToolError` (a `ToolExecutionError`), which the Agent turns into an error tool result showing `"<type>: <message>"`; `error_mode="raise"` raises the plain `ToolExecutionError`.
- **Approval**: `confirmation_hook(ui)` (added by `agent(confirmation_ui=...)`) asks a Haystack `ConfirmationUI` about gated Gantry tools only; confirm/modify runs the call once, reject never runs it. Without a UI, gated calls report "needs approval".
- **Run context**: `current_agent_state()` gives the Agent's `State` (`.get(key)`, `.data`); `current_tool_invocation().deps` is `state.data`.
- **Per-step**: `agent()` re-selects before the first step and after every tool step (dynamic `Toolset` + `before_run`/`after_tool` hooks); sticky up to `max_offered`; previous tools plus pins on a retrieval failure.
