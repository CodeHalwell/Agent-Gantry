# Pydantic AI

> Part of the `dotfetch` skill. Core concepts, debugging and common
> pitfalls are in [`SKILL.md`](../SKILL.md); read those first.

Install with `uv add "dotfetch[pydantic-ai]"` (pins `pydantic-ai-slim>=2.0.0`; tested on 2.0.0 and the latest 2.x). dotfetch uses Pydantic AI's own extension points, so tool failures, approvals, `deps` and tracing behave like hand-written Pydantic AI tools.

## Pick a mode

| Mode | Build | Plug into | Use when |
|---|---|---|---|
| Per-turn (default) | `adapter.capability()` or `adapter.toolset()` (= `.live()`) | `Agent(capabilities=[...])` / `Agent(toolsets=[...])` | Most agents: top-k re-selected every model step |
| Search | `adapter.capability(mode="search")` | `Agent(capabilities=[...])` | Large catalogues; prompt-cache-sensitive agents (tool list stays stable) |
| Static | `await adapter.select(query)` | `Agent(tools=[...])` | Fixed task, one selection |

```python
from pydantic_ai import Agent, DeferredToolRequests
from dotfetch.pydantic_ai import PydanticAIAdapter
#   also: current_run_context, DotFetchCapability, DotFetchToolset,
#         DotFetchDeferredToolset, fetcher_search_strategy, ErrorMode

adapter = PydanticAIAdapter(fetcher)          # or fetcher.framework("pydantic-ai")
agent = Agent(
    "openai:gpt-5.5",
    deps_type=MyDeps,
    output_type=[str, DeferredToolRequests],  # only if a tool can need approval
    capabilities=[adapter.capability(limit=5, skills=2)],
)
```

`capability(skills=N)` also injects the N dotfetch skills relevant to each turn as instructions (`skill_namespaces=[...]` to restrict). Search mode is `deferred_toolset()` (catalogue marked `defer_loading`; `required`/`always_include` stay visible) plus `tool_search()` (Pydantic AI's `ToolSearch` with dotfetch's router as `strategy`); compose those yourself if you need to.

## Per-turn behaviour

- Each step's query is the newest user prompt, or after a tool call the run's prompt plus the newest tool result (first 2,000 chars) — so "look up the customer, then email them" surfaces the email tool mid-run.
- Tools already called feed the router's `tools_already_used` signal.
- Tool-result text is sent to the embedder; `query_from_results=False` (on `toolset()` / `capability()`) keeps tool output out of retrieval when it may hold sensitive data.
- One retrieval per distinct query per run (cached), not one per step.
- `sticky=True` (default) keeps tools offered earlier in the run, up to `max_offered` per step (default `3 * limit`; the current selection and pins always stay, older tools drop least recently selected first); `sticky=False` for a strict top-k.
- State lives on the per-run copy from `for_run`, so concurrent runs can share one toolset.
- Retrieval failure mid-run: keeps the run's earlier tools, logs a WARNING, never raises.
- Several dotfetch toolsets on one agent need distinct `id=` values and non-clashing names (`namespaces=` or `.prefixed(...)`).

## Errors and approval

Pydantic AI ends a run on an ordinary tool exception, so with the default `error_mode="model"`:

| dotfetch outcome | Pydantic AI exception | What the model sees |
|---|---|---|
| Bad arguments (`ValidationError`, `ToolNotFound`) | `ModelRetry` | The error; it retries within `max_retries` (default 1) |
| Handler error, timeout, circuit open, permission denied | `ToolFailed` (on early 2.x, returned as the tool's `Error: ...` result) | A terminal failure it can adapt to |
| Confirmation required (`requires_confirmation=True` or a `SecurityPolicy` pattern) | `ApprovalRequired` | Nothing yet — the run pauses with `DeferredToolRequests` |

```python
from pydantic_ai import DeferredToolRequests, DeferredToolResults

result = await agent.run("refund order 4471")
if isinstance(result.output, DeferredToolRequests):
    decisions = {c.tool_call_id: approve(c) for c in result.output.approvals}
    result = await agent.run(
        message_history=result.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals=decisions),
    )
```

The approved replay runs once through `fetcher.execute` with only the confirmation gate cleared. `result.output.metadata[call_id]["dotfetch"]` holds the tool's qualified name and the reason. Pass `error_mode="raise"` (adapter or any builder) to get `ToolExecutionError` instead, like the other adapters. The model sees the failure's text (capped at 2,000 chars); to redact it, pass `PydanticAIAdapter(fetcher, error_formatter=lambda exc: ...)`, which rewrites every model-facing failure and falls back to a generic message if the formatter itself raises.

## Run context and tracing

```python
from dotfetch.pydantic_ai import current_run_context

@fetcher.register
async def get_balance(account: str) -> str:
    """Fetch an account balance for the signed-in user."""
    ctx = current_run_context()        # pydantic_ai.RunContext, or None outside a run
    return await ctx.deps.bank.balance(account)
```

`dotfetch.current_tool_invocation()` is the framework-neutral form (`.deps`, `.run_id`, `.tool_call_id`, `.native_context`). Not available to out-of-process executors (sandbox, MCP, A2A, HTTP). When Pydantic AI is instrumented (Logfire / OpenTelemetry), the tool span's ids become the `ToolCall`'s `trace_id`/`parent_span_id`, so `on_tool_call` events join the agent's trace.

## Debugging

There is no `last_selection` on the Pydantic AI toolset; reproduce a step's selection with `await adapter.select(query, limit=...)` (same router, same pins), or spy with `fetcher.on_tool_call`. In search mode, `search_tools` results appear as tool-return parts in `result.all_messages()`.
