# LlamaIndex

> Part of the `agent-gantry` skill. Core concepts, debugging and common
> pitfalls are in [`SKILL.md`](../SKILL.md); read those first.

Install with `uv add "agent-gantry[llamaindex]"` (pins `llama-index-core>=0.13.0`; tested on 0.13.0 and the latest).

```python
from llama_index.core.workflow import HumanResponseEvent, InputRequiredEvent
from agent_gantry.llamaindex import LlamaIndexAdapter, current_workflow_context

agent = LlamaIndexAdapter(gantry).function_agent(llm, limit=5)   # error_formatter=... on the adapter
handler = agent.run(user_msg="refund order 4471")
async for event in handler.stream_events():
    if isinstance(event, InputRequiredEvent):                     # a Gantry-gated call waiting
        handler.ctx.send_event(HumanResponseEvent(response="yes", gantry_approval_id=event.gantry_approval_id))
result = await handler
```

- **Errors**: `LlamaIndexToolError` (text `<type>: <message>`), shown to the model as `ToolOutput(is_error=True)`.
- **Approval**: gated calls `ctx.wait_for_event(HumanResponseEvent, waiter_event=InputRequiredEvent(gantry_tool=..., gantry_arguments=..., gantry_approval_id=...))`; answer with a `HumanResponseEvent` echoing `gantry_approval_id` (so an answer only decides its own call); "yes"/"y"/"approve"/"approved" runs once. Use `function_agent()` or `GantryFunctionAgent(tools=...)` for gated tools, so each call has its own request. Direct `tool.call` reports "needs approval".
- **Run context**: hidden `ctx` parameter; `current_workflow_context()` returns the workflow `Context` (`.store` for state).
- **Per step**: `tool_retriever()` re-selects each step; on a retrieval failure only the pins are offered.
