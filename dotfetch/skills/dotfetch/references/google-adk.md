# Google ADK

> Part of the `dotfetch` skill. Core concepts, debugging and common
> pitfalls are in [`SKILL.md`](../SKILL.md); read those first.

Install with `uv add "dotfetch[google-adk]"` (pins `google-adk>=2.6.1`; tested on 2.6.1 and the latest).

```python
from google.adk.runners import InMemoryRunner
from google.genai import types
from dotfetch.google_adk import GoogleADKAdapter, current_tool_context

adapter = GoogleADKAdapter(fetcher)                    # error_formatter=... to redact failure text
agent = adapter.agent(model="gemini-2.5-flash", name="support", limit=5)
runner = InMemoryRunner(agent=agent, app_name="app")
# a gated tool emits an "adk_request_confirmation" function call; answer it with
# FunctionResponse(name="adk_request_confirmation", id=call.id, response={"confirmed": True})
```

- **Per-turn selection** (`agent()` / `before_model_callback()`): injects each model request's tools with `LlmRequest.append_tools`; the agent's own tools stay. Query = newest user message + latest successful tool result (`query_from_results=False` to exclude output). Per-run state keyed by the ADK invocation; cached, sticky up to `max_offered` (default `3 * limit`); a retrieval failure keeps the previous tools plus pins.
- **Errors**: the function response is `{"error": "<type>: <message>"}` (ADK's convention), so the model can correct itself; an exception would end the run. `error_mode="raise"` restores `ToolExecutionError`.
- **Approval**: `fetcher.requires_confirmation(tool)` drives ADK's `require_confirmation`; a confirmed call runs exactly once, ADK refuses a rejected one. A hand-built `Agent(before_model_callback=adapter.before_model_callback())` also needs `tools=[adapter.confirmation_toolset()]` so the confirmed call can resume (`adapter.agent()` adds it).
- **Run context**: `current_tool_context()` returns the ADK `ToolContext` (`.state`, `.user_id`, `.invocation_id`, `.function_call_id`, artifacts, memory).
- **Tracing**: ADK's OpenTelemetry span becomes the `ToolCall`'s `trace_id`/`parent_span_id`.
