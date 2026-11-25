# LangChain and LangGraph

> Part of the `dotfetch` skill. Core concepts, debugging and common
> pitfalls are in [`SKILL.md`](../SKILL.md); read those first.

Install with `uv add "dotfetch[langgraph]"` (or `[langchain]`; pins `langchain`/`langchain-core`/`langgraph>=1.0.0`, tested on 1.0.0 and the latest).

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from dotfetch.langgraph import LangGraphAdapter, current_tool_runtime

adapter = LangGraphAdapter(fetcher)                    # error_formatter=... to redact failure text
agent = create_agent(model, middleware=[await adapter.middleware(limit=5)],
                     checkpointer=InMemorySaver(), context_schema=MyContext)
config = {"configurable": {"thread_id": "t1"}}
result = await agent.ainvoke({"messages": [("user", "refund order 4471")]}, config, context=my_ctx)
if "__interrupt__" in result:                         # dotfetch's confirmation gate -> interrupt()
    result = await agent.ainvoke(Command(resume=True), config)   # approved call runs exactly once
```

- **Per-turn selection** (`middleware()`): carries the catalogue as its `tools` and narrows `request.tools` in `wrap_model_call` each model turn; your own `create_agent(tools=...)` stay. Query = newest user message or last *successful* tool result (`query_from_results=False` to exclude output). Per-run state keyed by thread + newest user message; cached, sticky up to `max_offered` (default `3 * limit`); a retrieval failure keeps the previous tools plus pins. `areact_agent(model, ...)` builds the same agent in one call.
- **Errors**: LangChain's `ToolException` with `handle_tool_error=True`, so the model gets an error `ToolMessage` (`status="error"`) and the run continues. `error_mode="raise"` restores `ToolExecutionError`.
- **Approval**: gated calls (`fetcher.requires_confirmation(tool)`) call LangGraph's `interrupt()` with `{"type": "fetcher_approval", "tool", "arguments", "tool_call_id"}`; resume with `Command(resume=True)` / `{"approved": True}` to run, anything else rejects. Needs a checkpointer; without one the model is told the call needs approval and it is not run.
- **Run context**: `current_tool_runtime()` gives `.context` (the graph's `context=`), `.store`, `.config`, `.tool_call_id`; `current_tool_invocation().metadata["thread_id"]`.
- **Tracing**: an OpenTelemetry span, else the current LangSmith run, becomes the `ToolCall`'s `trace_id`/`parent_span_id`.
