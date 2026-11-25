# OpenAI Agents SDK

> Part of the `dotfetch` skill. Core concepts, debugging and common
> pitfalls are in [`SKILL.md`](../SKILL.md); read those first.

Install with `uv add "dotfetch[openai-agents]"` (pins `openai-agents>=0.23.1`; tested on 0.23.1 and the latest). It cannot share an environment with `[agent-frameworks]`/`[all]`: `openai-agents>=0.23` needs `openai>=3`, CrewAI pins `openai<3`.

```python
from agents import Agent, Runner
from dotfetch.openai_agents import OpenAIAgentsAdapter
#   also: current_run_context, DotFetchDynamicTools; DotFetch.requires_confirmation(tool)

adapter = OpenAIAgentsAdapter(fetcher)                 # error_formatter=... to redact failure text
agent = Agent(name="assistant", tools=await adapter.dynamic_tools(limit=5))
result = await Runner.run(agent, "refund order 4471", context=my_deps)
if result.interruptions:                              # dotfetch's confirmation gate -> needs_approval
    state = result.to_state()
    for item in result.interruptions:
        state.approve(item)                           # or state.reject(item)
    result = await Runner.run(agent, state)           # the approved call runs exactly once
```

- **Per-turn selection** (`dynamic_tools`): the catalogue as `FunctionTool`s gated by `is_enabled`. Each turn's query is the newest user message, or after a dotfetch tool call that message plus the tool's output (`query_from_results=False` to keep output out of retrieval). Cached per run, sticky up to `max_offered` (default `3 * limit`), state keyed to the run, so concurrent runs are independent and `agent.tools` is never rewritten. Tools registered later need a fresh `dynamic_tools()` call.
- **Errors**: returned to the model as tool output, `Error: <type>: <message>` (capped at 2,000 chars), so it can correct bad arguments. The SDK would otherwise end the run with `UserError`. `error_mode="raise"` restores `ToolExecutionError`.
- **Approval**: `needs_approval` follows `fetcher.requires_confirmation(tool)` (the tool's flag or a `SecurityPolicy` pattern). A custom policy that only decides at call time still blocks the call; the model is told it needs approval.
- **Run context**: `dotfetch.openai_agents.current_run_context()` returns the call's `ToolContext` (`.context` is what you passed to `Runner.run(context=...)`, plus `.tool_call_id`, `.tool_name`).
- **Tracing**: the SDK's current span (when tracing) or an OpenTelemetry span becomes the `ToolCall`'s `trace_id`/`parent_span_id`.
- The older `session` / `run` / `run_hooks` helpers rewrite `agent.tools`; use them for one run at a time per agent. `run_hooks` selects at `on_agent_start` (so the first model request has the tools) and re-selects before each later request.
- **First request, older SDKs**: before `openai-agents` 0.23.1 the SDK lists tools before it records the turn's input, so `dynamic_tools()` raises `ImportError` and `run_hooks()` cannot cover request one (it warns). There, use `adapter.run(agent, input)`, `adapter.session(agent).run(input)` or `adapter.refresh(agent, input)`, which take the input from you.
