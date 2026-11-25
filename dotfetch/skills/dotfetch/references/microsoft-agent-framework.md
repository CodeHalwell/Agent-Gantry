# Microsoft Agent Framework

> Part of the `dotfetch` skill. Core concepts, debugging and common
> pitfalls are in [`SKILL.md`](../SKILL.md); read those first.

The native, idiomatic integration. The one-class entry point is
`AgentFrameworkAdapter`; `AgentFrameworkAdapter(fetcher).context_provider(top_k=...)`
returns a `DotFetchContextProvider` — an AF `ContextProvider` that runs at
`before_run` (per-run mode) or on every chat-completion round (per-turn mode, via
a paired middleware). The adapter's other methods build the tool bridge and the
approval / observability / tool-choice middleware:
`.tool_bridge()`, `.approval_middleware(policy)`, `.observability_middleware()`,
`.tool_choice_middleware(decider)`. The returned types (`DotFetchContextProvider`,
`DotFetchToolBridge`, the middleware classes) are still importable directly for type
annotations.

## Per-run mode (default — fixed tool set for one `agent.run(...)` call)

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from dotfetch import DotFetch
from dotfetch.agent_framework import AgentFrameworkAdapter

fetcher = DotFetch()
# ... register tools, await fetcher.sync() ...

provider = AgentFrameworkAdapter(fetcher).context_provider(top_k=5)  # -> DotFetchContextProvider

agent = Agent(
    OpenAIChatClient(),
    "You are a helpful assistant.",
    context_providers=[provider],
)

result = await agent.run("Book me a flight to Tokyo")
```

## Per-turn mode (re-runs retrieval every chat round)

Use when the agent reasons in multiple steps and needs different tools at different stages. The chat middleware is **required** — without it the per-turn mode silently degrades to per-run.

```python
from dotfetch.agent_framework import AgentFrameworkAdapter

provider = AgentFrameworkAdapter(fetcher).context_provider(
    top_k=3,
    query_strategy="per_turn",
    # Default for per_turn is latest_activity: the newest user message, plus
    # the newest tool result when one came after it. Pass query_generator=...
    # only for a different rotation (e.g. partial(latest_activity,
    # include_request=False) to query on the tool's output alone in a pipeline
    # whose results name the next step).
)

agent = Agent(
    OpenAIChatClient(),
    "...",
    context_providers=[provider],
    middleware=[provider.as_chat_middleware()],  # REQUIRED for per_turn
)
```

Or use the one-call helper:

```python
agent = Agent(OpenAIChatClient(), "...")
provider.attach_to(agent)                 # appends provider + middleware in one shot
provider.attach_to(agent, trace=True)     # ...and a console trace of every tool call
```

`trace=True` also installs the built-in trace middleware (see
[Observability & tracing](../SKILL.md#observability--tracing)) — the library-owned
replacement for hand-rolled `@function_middleware` logging.

## Pinning tools that must always be visible

```python
provider = AgentFrameworkAdapter(fetcher).context_provider(
    top_k=5,
    required=["validate_input"],     # MissingRequiredToolError at construction if absent
    always_include=["log_event"],    # warning + skip if absent
    static_tools=[some_af_native_tool],  # tool not registered with fetcher
)
```

`required` and `always_include` reference *fetcher-registered* tool names. `static_tools` is for AF-native `@tool` callables that live outside the fetcher registry — they're appended every round and never filtered.

## Static (no per-turn retrieval — bake once)

Use `DotFetchToolBridge` directly when the tool set is fixed at construction:

```python
from dotfetch.integrations.agent_framework_bridge import DotFetchToolBridge

bridge = DotFetchToolBridge(fetcher)
tools = await bridge.get_tools("customer support tasks", limit=5)
agent = Agent(OpenAIChatClient(), "...", tools=tools)
```

The bridge also exposes one-call agent constructors: `bridge.as_agent(...)`, `bridge.build_handoff_workflow(...)`, `bridge.build_sequential_workflow(...)`, `bridge.build_workflow(...)`.

## Approval middleware + observability

```python
from dotfetch.core.security import SecurityPolicy
from dotfetch.integrations.agent_framework_middleware import (
    DotFetchApprovalMiddleware,
    DotFetchObservabilityMiddleware,
    DotFetchToolChoiceMiddleware,
)

policy = SecurityPolicy(require_confirmation=["delete_*", "refund_*"])

# tool_choice modulation: force tool calls for the first N rounds, then auto.
rounds = {"n": 0}
def choose(_ctx):
    rounds["n"] += 1
    return "required" if rounds["n"] <= 5 else "auto"

agent = Agent(
    OpenAIChatClient(),
    "...",
    context_providers=[provider],
    middleware=[
        provider.as_chat_middleware(),
        DotFetchApprovalMiddleware(policy),
        DotFetchObservabilityMiddleware(fetcher),
        DotFetchToolChoiceMiddleware(choose),
    ],
)
```
