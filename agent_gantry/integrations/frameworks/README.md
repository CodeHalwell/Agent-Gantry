# Native framework tool adapters

Agent-Gantry selects a small, relevant slice of tools from a large registry.
These adapters hand that slice to a concrete agent framework as **native tool
objects** — not just JSON schemas — so the framework can introspect and call
them, while every invocation still routes through `gantry.execute` (retries,
timeouts, circuit breakers, security policy all apply).

## Start from your registry

Access the adapter directly, with concrete adapter types available to your editor:

```python
from agent_gantry import AgentGantry, supported_frameworks

gantry = AgentGantry()

@gantry.register
def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b

tools = await gantry.framework("langchain").select("add two numbers", limit=3)
assert await tools[0].ainvoke({"a": 2, "b": 3}) == 5
print(supported_frameworks())
```

Selection syncs pending registrations automatically. Keep an adapter around to
reuse its configuration, for example `adapter = gantry.framework("langchain",
default_limit=3)`. Unknown names raise a `ValueError` with a suggested spelling
and the supported names; invalid budgets fail at adapter construction. Names
accept hyphens or underscores, and are case-insensitive. Imports and existing
adapter constructors continue to work.

Install Gantry plus just the SDK you need:

| `gantry.framework(name)` | Install | Minimal wiring |
|---|---|---|
| `langchain` | `pip install 'agent-gantry[langchain]'` | `model.bind_tools(await adapter.select(query))` — see [LangChain and LangGraph](#langchain-and-langgraph-first-class) |
| `langgraph` | `pip install 'agent-gantry[langgraph]'` | `create_agent(model, middleware=[await adapter.middleware()])` — see [LangChain and LangGraph](#langchain-and-langgraph-first-class) |
| `llamaindex` | `pip install 'agent-gantry[llamaindex]'` | `FunctionAgent(tool_retriever=adapter.live())` — see [LlamaIndex](#llamaindex-first-class) |
| `crewai` | `pip install 'agent-gantry[crewai]'` | `Agent(..., tools=await adapter.select(query))` — see [CrewAI](#crewai-first-class) |
| `pydantic_ai` | `pip install 'agent-gantry[pydantic-ai]'` | `Agent(model, capabilities=[adapter.capability()])` — see [Pydantic AI](#pydantic-ai-first-class) |
| `openai_agents` | `pip install 'agent-gantry[openai-agents]'` | `Agent(..., tools=await adapter.dynamic_tools())` — see [OpenAI Agents SDK](#openai-agents-sdk-first-class) |
| `haystack` | `pip install 'agent-gantry[haystack]'` | `adapter.agent(chat_generator=...)` — see [Haystack](#haystack-first-class) |
| `agno` | `pip install 'agent-gantry[agno]'` | `adapter.agent(model=...)` — see [Agno](#agno-first-class) |
| `google_adk` | `pip install 'agent-gantry[google-adk]'` | `adapter.agent(model=..., name=...)` — see [Google ADK](#google-adk-first-class) |
| `strands` | `pip install 'agent-gantry[strands]'` | `Agent(hooks=[adapter.live()])` — see [Strands Agents](#strands-agents-first-class) |
| `dspy` | `pip install 'agent-gantry[dspy]'` | `dspy.ReAct(signature, tools=await adapter.select(query))` — see [DSPy](#dspy-first-class) |
| `agent_framework` | `pip install agent-gantry agent-framework-core` | `provider = adapter.live(); provider.attach_to(agent)` |

Each row assumes `adapter = gantry.framework(name)` and the framework's native
imports. Model providers may need their own SDK extra and credentials. Framework
dependency constraints can differ; install the frameworks used by each app in
that app's environment. Live integrations refresh at the framework's supported
tier, detailed below. All twelve adapters support `await adapter.select(query)`.
Microsoft Agent Framework also retains `adapter.tool_bridge()` for cached
wrappers, relative thresholds, and its lower-level callable integration.

## Supported native tool objects

| Framework | Adapter class | Native object |
|---|---|---|
| LangChain | `LangChainAdapter` | `langchain_core.tools.StructuredTool` |
| LangGraph | `LangGraphAdapter` | LangChain `StructuredTool` (LangGraph consumes these) |
| LlamaIndex | `LlamaIndexAdapter` | `llama_index.core.tools.FunctionTool` |
| CrewAI | `CrewAIAdapter` | `crewai.tools.BaseTool` subclass |
| Pydantic AI | `PydanticAIAdapter` | `pydantic_ai.tools.Tool` |
| OpenAI Agents SDK | `OpenAIAgentsAdapter` | `agents.FunctionTool` |
| Haystack | `HaystackAdapter` | `haystack.tools.Tool` |
| Agno | `AgnoAdapter` | `agno.tools.function.Function` |
| Google ADK | `GoogleADKAdapter` | `google.adk.tools.FunctionTool` |
| Strands Agents | `StrandsAdapter` | `strands.tools.decorator.DecoratedFunctionTool` |
| DSPy | `DSPyAdapter` | `dspy.Tool` |
| Microsoft Agent Framework | `AgentFrameworkAdapter` | `agent_framework.FunctionTool` |

All third-party imports are **lazy** — `import agent_gantry` never requires any
of these frameworks. A missing framework raises `ImportError` with a
`pip install …` hint only when you call its adapter.

Agent-Gantry supports Python 3.10–3.14. CrewAI's current SDK declares Python
`<3.14`, so its integration requires Python 3.10–3.13. The combined
`agent-frameworks` extra omits CrewAI on 3.14; other adapters remain available.

Microsoft Agent Framework native tools delegate confirmation to AF's
`approval_mode="always_require"`. AF pauses before invoking the tool and resumes
after approval; Gantry still applies its domain, rate and other denial checks.
Calling `FunctionTool.invoke` directly is host-level execution in AF, so the
caller must obtain approval first. Bare bridge callables retain Gantry's own
confirmation gate.

## Usage

```python
from agent_gantry import AgentGantry
from agent_gantry.langchain import LangChainAdapter   # clean per-framework namespace

gantry = AgentGantry()
# ... register tools, await gantry.sync() ...

# Select the top-3 relevant tools and get them as LangChain StructuredTools:
tools = await LangChainAdapter(gantry).select("email the quarterly report to finance", limit=3)
# hand `tools` to a LangChain/LangGraph agent
```

Each framework has a top-level namespace — `from agent_gantry.<framework> import …`
(`agent_gantry.langchain`, `agent_gantry.crewai`, `agent_gantry.llamaindex`,
`agent_gantry.pydantic_ai`, `agent_gantry.openai_agents`,
`agent_gantry.haystack`, `agent_gantry.agno`, `agent_gantry.google_adk`, `agent_gantry.langgraph`,
`agent_gantry.strands`, `agent_gantry.dspy`, `agent_gantry.agent_framework`) re-exporting that framework's `<Framework>Adapter`
class (which carries both the static `.select`/`.convert` and the deep live
methods). Importing `agent_gantry` never pulls these in.

Every `<Adapter>(gantry).select(query, *, limit=None, score_threshold=0.0,
namespaces=None, tools_already_used=None, required=None, always_include=None)`
accepts the same explicit selection knobs as `GantryToolset.select` — `limit`
defaults to the adapter's `default_limit` (5, `DEFAULT_TOOL_LIMIT` in
`base.py`) when omitted. Need one conversion at a time? Use the staticmethod
`<Adapter>.convert(spec)` with specs from `GantryToolset(gantry).select(query)`.

## Guaranteed & pinned tools (`required` / `always_include`)

Ported from the Microsoft Agent Framework provider
(`GantryContextProvider(required=..., always_include=...)`), every adapter's
`select`/`select_or_empty`/`live(...)` now accepts the same two keywords —
shared by `GantryToolset.select` (`base.py`), so all 11 framework adapters
(plus the Microsoft Agent Framework provider it was ported from) get the same
guarantee:

- **`required=[...]`** — bare or `namespace.name`-qualified tool names that
  **must** be present in the result. A name already in the semantic slice
  counts; anything missing is fetched from the registry and appended. If a
  name doesn't resolve against the registry at all, `select` raises
  `MissingRequiredToolError` (`agent_gantry.integrations.frameworks.errors`,
  re-exported from `agent_gantry`, `agent_gantry.integrations`, and
  `agent_gantry.integrations.frameworks`) rather than silently returning an
  incomplete selection.
- **`always_include=[...]`** — same resolution and append behaviour, but a
  name that isn't in the registry logs a `WARNING` and is skipped rather than
  raising.

Both are appended *after* the semantic slice (`required` before
`always_include`), deduplicated against it and against each other, and are
**never counted against `limit`** — `limit` bounds only the semantic
retrieval, so a required tool is never dropped because the semantic slice
already filled the budget, and pins never silently shrink the slice you asked
for. `select_or_empty` resolves pins even on a blank query (they don't depend
on the query's retrieval signal — only the semantic leg is skipped).

```python
from agent_gantry.langchain import LangChainAdapter

tools = await LangChainAdapter(gantry).select(
    "book a flight to Tokyo",
    limit=3,
    required=["cancel_booking"],       # guaranteed present, or MissingRequiredToolError
    always_include=["escalate_to_human"],  # pinned if present, warned+skipped if not
)
```

The live/dynamic paths (`adapter.live(required=..., always_include=...)` and
the bespoke methods it delegates to) re-apply both pins on every re-selection
round, not just the first.

## Schema ownership, updates, and result types

Every selected `ToolSpec` owns a snapshot of its parameter schema. Framework
normalization and application edits to an exported schema cannot change Gantry's
registry validation or another selection's schema.

The Microsoft Agent Framework bridge refreshes cached wrappers when the tool's
schema, description, version, native name, capabilities, or confirmation flag
changes. Runtime health and routing-only metadata keep the existing wrapper.
Its cache retains at most 256 wrappers using least-recently-used eviction;
configure `adapter.tool_bridge(cache_size=64)` or disable reuse with
`cache_size=0`. Existing `cache=False` and `clear_cache()` options remain available.
Tool updates become visible after registration/sync and the next selection.

The Agent Framework and OpenAI Agents text wrappers serialize Pydantic models,
dataclasses, dates, UUIDs, decimals, sets and structured mappings as JSON, matching
Gantry's provider result formatting. String results remain unchanged.

## Namespaces and callable tools

Tools can share a name across namespaces. Native framework exports use distinct,
provider-safe names so each tool survives the framework's name index. Ordinary
names stay unchanged; collisions get a namespace-based alias with a short digest,
and names longer than 64 characters are shortened. Names remain stable across
queries and namespace filters while the registry's tool identities stay the same.

`ToolSpec.name` and `ToolSpec.qualified_name` retain the original identity;
`ToolSpec.native_name` is the name advertised to the framework. Calls still execute
the original tool through Gantry, including its policy and confirmation checks.
Keep using original names for pins, such as `required=["billing.lookup"]`;
`tools_already_used` also accepts the advertised native names.
Generated aliases retain their namespace when computing conversation signals.
An explicitly supplied bare name continues to apply to matching tools in every namespace.

You can register callable objects and partially bound functions directly:

```python
from functools import partial

class Notify:
    async def __call__(self, recipient: str) -> str:
        """Build a notification for a recipient."""
        return f"Notification for {recipient}"

def label(prefix: str, value: int) -> str:
    """Label a numeric value with a fixed prefix."""
    return f"{prefix}: {value}"

gantry.register(Notify(), name="notify")
gantry.register(partial(label, "Report"), name="label_report")
```

Their types and docstrings are preserved. Awaitable results are awaited, even
when a synchronous decorator returns them. Execution configuration supplies the
default timeout and retry budget to every framework; a direct
`ToolCall(..., retry_count=0)` explicitly disables retries for that call. Omit
the fields to inherit configuration. To serialize a call while preserving which
settings were omitted, use `call.model_dump(exclude_unset=True)`.
Plain `model_dump()` writes the model's default values; reloading that payload
treats those values as explicit overrides, including zero retries.
Provider tool-call conversion and `execute_tool_calls(response)` inherit the same
configuration. Use `execute_tool_calls(response, timeout_ms=5000, retry_count=0)`
to override the settings for a model turn.

A timeout stops waiting for a synchronous handler but cannot stop its worker
thread. Retrying may overlap with that still-running call. For tools with side
effects, choose a retry policy deliberately and set timeouts in the underlying
I/O client too. Async handlers and returned awaitables receive cancellation
when they are being awaited.

## Multi-turn re-selection — autonomous *and* conversational

`ToolRefresher` generalizes the Microsoft Agent Framework provider's per-call
retrieval to any framework: re-rank the whole registry on every turn so the
agent can pivot to a different tool as the task changes. It is the
**standalone** utility for hand-rolled agent loops (a raw LLM SDK call in a
`while` loop, no framework underneath) — if you're using one of the 14
frameworks below, prefer `adapter.live(...)` instead, which wires Gantry into
that framework's own lifecycle rather than requiring you to call `refresh()`
by hand between turns.

```python
from agent_gantry.integrations import ToolRefresher

refresher = ToolRefresher(gantry, limit=3, dialect="openai")

# Each turn, pass the running message list; selection is recomputed fresh and
# tools already used are deprioritized so the agent keeps moving forward.
tools = await refresher.refresh(messages)        # dialect schemas
specs = await refresher.refresh_specs(messages)  # ToolSpec objects
```

The default query generator (`agent_gantry.query.latest_activity`) is
**recency-aware**, so one refresher serves both styles with no configuration:

- **Autonomous agents / tool pipelines** — when the newest message is a tool
  result (the agent is chaining tools with no new user input), that result's
  *content* selects the next tool: `fetch → clean → train → evaluate → report`.
- **Conversational agents** — when the newest message is the user's, their new
  request selects the next tool: `weather → flights → hotel → email`.

To force one behaviour, pass `query_generator=` explicitly — `last_user_text`
(always the user), `last_tool_result` (always the last result), or a custom
`fallback_chain(...)`. See `examples/frameworks/multi_turn_refresher_example.py`
for both modes side by side.

The message list can be in any of the common SDK shapes — plain role/content
dicts, OpenAI Chat Completions (tool messages carry only `tool_call_id`),
OpenAI Responses items, Anthropic (tool output arrives as a `user` turn of
`tool_result` blocks), LangChain or Agent Framework messages. Used-tool
tracking reads the tool's name from the assistant's call where the result
message has none (`agent_gantry.query.tool_names_used`).

## Uniform `live_tier` / `live()` entry point

The `<Adapter>.select` methods above are *static*: select once, hand over a fixed
tool list. Every adapter also exposes a **dynamic** re-selection surface that goes
deeper — but historically each framework named and shaped it differently
(`react_agent`, `toolset`, `tool_hook`, `agent_builder`, …), so writing
framework-agnostic code against it meant knowing 14 different method names.

`adapter.live_tier` and `adapter.live(...)` are the uniform entry point on top
of those bespoke methods (which are **not** removed or renamed — they stay the
documented, framework-idiomatic path; `live()` just delegates to one of them):

- **`adapter.live_tier`** — `"per-turn"` or `"per-call"`, the deepest dynamic
  re-selection tier that framework supports (see `LiveTier` /
  `BaseFrameworkAdapter.live_tier` in `base.py`).
  - `"per-turn"` — the framework calls back into Gantry (directly, or via a
    Gantry-built hook/toolset/provider) on every model turn / reasoning step,
    so the tool surface can change *mid-run*.
  - `"per-call"` — the framework fixes its tool list at agent-construction
    time with no native mid-run hook, so the deepest Gantry can do is rebuild
    a fresh agent/tool list before each new top-level call (see
    `live_wrappers.py`).
- **`adapter.live(*, limit=None, score_threshold=0.0, namespaces=None,
  required=None, always_include=None, **framework_kwargs)`** — returns the
  framework-appropriate live object (the hook / toolset / provider / builder
  that the framework consumes). Some frameworks' native hooks are inherently
  bound to an external object you must supply (a chat model, an already-built
  agent, a kernel) — those adapters require it as a named `framework_kwargs`
  entry (see the table below) and raise a clean `TypeError`/`KeyError` if it's
  missing. `required`/`always_include` follow `GantryToolset.select`'s pinning
  contract (see above) and are re-applied on every dynamic re-selection round.

| Framework | `live_tier` | `live()` delegates to | `live()` returns | Plug into |
|---|---|---|---|---|
| LangChain | per-call | `select` (bound alias) | async `query -> list[StructuredTool]` callable | rebuild `AgentExecutor` / `.bind_tools()` before each call |
| LangGraph | per-turn | `react_agent` (requires `model=`) | compiled LangGraph agent (`Pregel`) | call `.ainvoke()` / `.invoke()` directly |
| LlamaIndex | per-turn | `tool_retriever` | `GantryToolRetriever` (`ObjectRetriever`) | `FunctionAgent(tool_retriever=<result>)` |
| CrewAI | per-call | `agent_builder` | `GantryLiveCrewAgent` builder | `await builder.build(query)` per task |
| Pydantic AI | per-turn | `toolset` | `GantryToolset` (`AbstractToolset`) | `Agent(model, toolsets=[<result>])` |
| OpenAI Agents SDK | per-turn | `session` (requires `agent=`) | `GantryAgentSession` | `await session.run(run_input)` per turn |
| Haystack | per-call | `tool_invoker_builder` | `GantryLiveHaystackToolInvoker` builder (ToolInvoker on haystack 2.x, Agent on >=3.0) | `adapter.agent()` re-selects every step instead; `await builder.build(query)` per call |
| Agno | per-call | `agent_builder` | `GantryLiveAgnoAgent` builder | `await builder.build(query)` per run (or one `adapter.agent()` that re-selects inside Agno per run) |
| Google ADK | per-turn | `before_model_callback` | async `(callback_context, llm_request) -> None` | `Agent(tools=[], before_model_callback=<result>)` |
| Strands Agents | per-turn | `tool_hook` | `GantryStrandsToolHook` (`HookProvider`) | `Agent(tools=[], hooks=[<result>])` |
| DSPy | per-call | `agent_builder` (requires `signature=`) | `GantryLiveDSPyReAct` builder | `await builder.build(query)` per task |
| Microsoft Agent Framework | per-turn | `context_provider` (`query_strategy="per_call"`) | `GantryContextProvider` | `Agent(context_providers=[<result>])` + `<result>.as_chat_middleware()`, or `<result>.attach_to(agent)` |

`AgentFrameworkAdapter` uses the shared `select`/`convert` surface and retains
its AF-specific provider and middleware factories — see `agent_framework_adapter.py`.

```python
from agent_gantry.llamaindex import LlamaIndexAdapter

adapter = LlamaIndexAdapter(gantry)
adapter.live_tier                      # "per-turn"
retriever = adapter.live(limit=5)      # same object as .tool_retriever(limit=5)
```

### Bespoke per-framework methods (what `live()` delegates to)

Each framework's own live methods remain available directly — some frameworks
offer more than one (e.g. Google ADK's `before_model_callback()` for a
hand-built agent vs. `agent()` for a fully-assembled one); `live()` always
picks the one requiring the fewest extra arguments. Import them lazily from
`agent_gantry.integrations.frameworks` (importing `agent_gantry` never loads
these — the framework is only required when you use its provider).

| Framework | Bespoke live methods | Native hook |
|---|---|---|
| LlamaIndex | `LlamaIndexAdapter(gantry).tool_retriever()` / `.function_agent(llm)` | `FunctionAgent(tool_retriever=…)` (`ObjectRetriever`) |
| Pydantic AI | `PydanticAIAdapter(gantry).toolset()` / `.deferred_toolset()` / `.tool_search()` / `.capability(mode="per-turn" \| "search")` | `AbstractToolset.get_tools()` + `for_run()`; `ToolSearch(strategy=…)`; `AbstractCapability` |
| Google ADK | `GoogleADKAdapter(gantry).before_model_callback()` / `.agent()` | `Agent(before_model_callback=…)` |
| Strands Agents | `StrandsAdapter(gantry).tool_hook()` / `.agent()` | `Agent(hooks=[…])` — `BeforeModelCallEvent` |
| LangGraph | `LangGraphAdapter(gantry).middleware()` (recommended) / `.areact_agent(model)` / `.react_agent(model)` / `.select_for_state(state)` | `create_agent` `AgentMiddleware.wrap_model_call` (narrows `request.tools` each turn) |
| OpenAI Agents SDK | `OpenAIAgentsAdapter(gantry).dynamic_tools()` (recommended) / `.run(agent, run_input)` / `.session(agent)` / `.run_hooks(agent)` | `FunctionTool.is_enabled` per turn; `RunHooks.on_llm_start` + per-run refresh |

```python
from agent_gantry.llamaindex import LlamaIndexAdapter
agent = LlamaIndexAdapter(gantry).function_agent(llm)   # LlamaIndex agent that re-selects tools each step
```

Frameworks whose tool list is **fixed for a run** (CrewAI, Agno,
DSPy, and Haystack outside `HaystackAdapter.agent()`, which re-selects every
step through Agent hooks) can't re-advertise tools mid-run; their "live" wrappers —
obtained via `CrewAIAdapter(gantry).agent_builder(...)`,
`AgnoAdapter(gantry).agent_builder(...)` (or `AgnoAdapter(gantry).agent(...)`, one
Agent whose callable `tools` factory re-selects for each run),
`DSPyAdapter(gantry).agent_builder(signature, ...)`,
and `HaystackAdapter(gantry).tool_invoker_builder(...)` (or `HaystackAdapter(gantry).live_tools(query)`
for tools alone) — re-select and rebuild the agent on each top-level call —
the deepest those frameworks allow. DSPy's `dspy.ReAct` bakes each tool's
name/description into its instruction prompt at construction and has no
runtime hook (`dspy.utils.callback.BaseCallback`'s `on_tool_start`/
`on_module_start` fire around an already-selected call, not before the model
picks the next tool) — see [DSPy](#dspy-first-class).

## Pydantic AI (first-class)

`pip install 'agent-gantry[pydantic-ai]'` (pins `pydantic-ai-slim>=2.0.0`;
CI runs the adapter tests on 2.0.0 and on the latest release). Gantry plugs into
each of Pydantic AI's own extension points:

| Mode | Build | Plug into | When |
|---|---|---|---|
| Static slice | `await adapter.select(query)` | `Agent(model, tools=[...])` | Fixed task, one selection |
| Per-turn push | `adapter.toolset()` (= `adapter.live()`) | `Agent(model, toolsets=[...])` | Most agents; top-k re-selected each step |
| Tool search | `adapter.deferred_toolset()` + `adapter.tool_search()` | `toolsets=[...]`, `capabilities=[...]` | Large catalogues, prompt-cache-sensitive agents |
| One object | `adapter.capability(mode="per-turn" \| "search", skills=0)` | `Agent(model, capabilities=[...])` | Recommended: either mode, plus per-turn Gantry skills |

```python
from pydantic_ai import Agent, DeferredToolRequests
from agent_gantry.pydantic_ai import PydanticAIAdapter

adapter = PydanticAIAdapter(gantry)
agent = Agent(
    "openai:gpt-5.5",
    deps_type=MyDeps,
    output_type=[str, DeferredToolRequests],   # only needed for approval-gated tools
    capabilities=[adapter.capability(limit=5, skills=2)],
)
```

**Per-turn selection.** Each model step's query is the newest user prompt, or
after a tool call the run's prompt plus the newest tool result (first 2,000
characters), so a chained task surfaces the next tool it needs mid-run. Tools
the model already called feed the router's `tools_already_used` signal.
Selections are cached per run (one retrieval per distinct query, not per
step); with `sticky=True` (default) tools offered earlier in a run stay
offered, up to `max_offered` per step (default `3 * limit`; the current
step's selection and pins are always offered, older tools are dropped least
recently selected first). Run state lives on the copy `for_run()` returns, so concurrent runs
can share one toolset. Give each Gantry toolset on an agent a distinct `id=`.
Tool-result text used in the query is sent to the embedder (a remote service
for OpenAI-style embedders); pass `query_from_results=False` to `toolset()` /
`capability()` when tool output must not leave the process, and selection
then uses the run's prompt.

**Tool search.** `deferred_toolset()` returns the catalogue with
`defer_loading=True` (`required`/`always_include` tools stay visible), and
`tool_search()` is Pydantic AI's `ToolSearch` with Gantry's semantic router as
its `strategy`. The model calls `search_tools` (or the provider's native tool
search) and only discovered tools reach it. The tool list sent each turn is
stable, so provider prompt caches survive, which per-turn re-selection cannot
guarantee. Other deferred tools on the agent are ranked by keyword and fused
with Gantry's semantic ranking (reciprocal rank fusion) under the same `limit`,
so an exact keyword match is not crowded out by an unrelated Gantry hit.
`capability(mode="search")` wires both.

**Errors and approval.** See the [error-handling policy](#error-handling-policy):
`ModelRetry` for argument errors, `ToolFailed` for terminal failures,
`ApprovalRequired` for Gantry's confirmation gate (the tool's
`requires_confirmation` flag or a `SecurityPolicy` pattern). Approve with
Pydantic AI's `DeferredToolResults`; the approved replay runs once through
`gantry.execute` with the gate cleared. `error_mode="raise"` opts out. The
model sees a failure's text (`error_type: message`, capped at 2,000
characters), as it would from a hand-written tool. To redact it, pass
`PydanticAIAdapter(gantry, error_formatter=fn)`: `fn(exc: ToolExecutionError)
-> str` rewrites every model-facing failure, and a formatter that raises is
replaced by a generic message so a bug in it cannot leak the original text.

**Run context.** Gantry handlers can read the run's `RunContext` (`deps`,
`run_id`, `tool_call_id`, retry count) without changing their schema:

```python
from agent_gantry import current_tool_invocation
from agent_gantry.pydantic_ai import current_run_context

@gantry.register
async def get_balance(account: str) -> str:
    """Fetch an account balance for the signed-in user."""
    ctx = current_run_context()          # pydantic_ai.RunContext, or None
    return await ctx.deps.bank.balance(account)
```

`current_tool_invocation()` is the framework-neutral form. It reaches async and
thread-dispatched sync handlers, not out-of-process executors (sandbox, MCP,
A2A, HTTP).

**Tracing.** When Pydantic AI is instrumented (Logfire or any OpenTelemetry
backend), the active tool span's trace and span ids become the `ToolCall`'s
`trace_id`/`parent_span_id`, so Gantry's execution telemetry joins the agent's
trace.

## OpenAI Agents SDK (first-class)

`pip install 'agent-gantry[openai-agents]'` (pins `openai-agents>=0.23.1`; CI
runs the adapter tests on 0.23.1 and on the latest release). The extra cannot
share an environment with `agent-frameworks`: `openai-agents>=0.23` needs
`openai>=3` while CrewAI pins `openai<3`. Older SDKs that arrive transitively
still work for static tools (the adapter feature-detects `FunctionTool`).

| Mode | Build | Use when |
|---|---|---|
| Per-turn (recommended) | `tools = await adapter.dynamic_tools(limit=5)` → `Agent(tools=tools)` | Most agents; Gantry picks each turn's tools |
| Static slice | `await adapter.select(query)` → `Agent(tools=...)` | Fixed task, one selection |
| Per-run helpers | `adapter.session(agent)` / `.run(agent, input)` / `.run_hooks(agent)` | Earlier API; rewrites `agent.tools`, so one run at a time per agent |

**Per-turn selection.** `dynamic_tools()` returns the catalogue as
`FunctionTool`s whose `is_enabled` callbacks share one selector. At the start
of each turn the SDK asks every tool whether it is enabled; the first answer in
a turn runs one Gantry selection and the rest reuse it. The query is the newest
user message in the run's `turn_input`, or after a Gantry tool call that message
plus the tool's output (first 2,000 characters; `query_from_results=False`
keeps tool output out of retrieval). Selections are cached per run, tools stay
offered for the rest of the run (`sticky=True`, capped by `max_offered`,
default `3 * limit`), and state is keyed to the run (the SDK shares one `usage`
object between a run's context and its tool contexts), so concurrent runs of
one agent are independent and `agent.tools` is never rewritten. Tools
registered after `dynamic_tools()` is called are not in the catalogue; call it
again to pick them up.

**Errors and approval.** Failures reach the model as tool output (see the
[error-handling policy](#error-handling-policy)). Gantry's confirmation gate
(the tool's `requires_confirmation` flag or a `SecurityPolicy` pattern, as
reported by `AgentGantry.requires_confirmation()`) sets the SDK's
`needs_approval`, so the run pauses with `result.interruptions`:

```python
result = await Runner.run(agent, "refund order 4471")
if result.interruptions:
    state = result.to_state()
    for item in result.interruptions:
        state.approve(item)          # or state.reject(item)
    result = await Runner.run(agent, state)
```

The approved call runs once through `gantry.execute` with only the confirmation
gate cleared. `OpenAIAgentsAdapter(gantry, error_formatter=fn)` rewrites
failure text before the model sees it; `error_mode="raise"` opts out of both
behaviours.

**Run context and tracing.** Gantry handlers read the call's `ToolContext`
(`context`, `tool_call_id`, `tool_name`) with
`agent_gantry.openai_agents.current_run_context()` or
`agent_gantry.current_tool_invocation()`. When the SDK is tracing, its current
span becomes the `ToolCall`'s `trace_id`/`parent_span_id`; otherwise an active
OpenTelemetry span is used.

## LangChain and LangGraph (first-class)

`pip install 'agent-gantry[langchain]'` or `'agent-gantry[langgraph]'` (both pin
`langchain`, `langchain-core` and, for LangGraph, `langgraph` at `>=1.0.0`; CI
runs the tests on exactly the 1.0.0 releases and on the latest). Both adapters
build `StructuredTool`s; LangGraph adds per-turn selection.

| Mode | Build | Use when |
|---|---|---|
| Per-turn (recommended) | `create_agent(model, middleware=[await adapter.middleware(limit=5)])` | Agents; Gantry picks each model turn's tools |
| One call | `await LangGraphAdapter(gantry).areact_agent(model, limit=5, checkpointer=...)` | The same agent, built for you |
| Static slice | `await adapter.select(query)` → `model.bind_tools(...)` / `ToolNode(...)` | A fixed task or your own graph |

**Per-turn selection.** The middleware carries the in-scope catalogue as its
`tools` (so the agent's `ToolNode` can run any of them) and, in
`wrap_model_call`, narrows the Gantry tools in `request.tools` to this turn's
selection. Tools you pass to `create_agent` yourself are never removed. The
query is the newest activity in `state["messages"]`: the user's request, or the
last successful tool result (`query_from_results=False` keeps tool output out;
a failed result, `status="error"`, never feeds it). State is per run, keyed by
the thread and the newest user message, so concurrent runs are independent and
a run resumed after an approval carries on. Within a run selections are cached,
earlier tools stay offered up to `max_offered` (`sticky=True`, default
`3 * limit`), and a retrieval failure keeps the previous turn's tools plus the
pins.

**Errors and approval.** A failed call raises LangChain's `ToolException` and
the tool has `handle_tool_error=True`, so the model gets an error `ToolMessage`
(`status="error"`) and the run continues. A confirmation-gated call (the tool's
flag or a `SecurityPolicy` pattern, per `AgentGantry.requires_confirmation()`)
pauses the graph with LangGraph's `interrupt()`:

```python
agent = await LangGraphAdapter(gantry).areact_agent(model, checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "t1"}}
result = await agent.ainvoke({"messages": [("user", "refund order 4471")]}, config)
if "__interrupt__" in result:
    request = result["__interrupt__"][0].value   # {"type": "gantry_approval", "tool", "arguments", ...}
    result = await agent.ainvoke(Command(resume=True), config)   # False (or anything else) rejects
```

An approved call runs once through `gantry.execute` with only the confirmation
gate cleared; a rejected one never runs and the model is told so. Approval needs
a checkpointer (an interrupt can only be resumed from a checkpoint); without
one, or outside a graph, the model is told the call needs approval and it is not
run. `error_formatter=fn` rewrites failure text; `error_mode="raise"` opts out
of both behaviours.

**Run context and tracing.** Gantry handlers read the run with
`agent_gantry.langgraph.current_tool_runtime()` (or `agent_gantry.langchain`):
`.context` (the graph's `context=`), `.store`, `.config` and `.tool_call_id`;
`current_tool_invocation().metadata` carries the `thread_id`. An active
OpenTelemetry span, else the current LangSmith run, becomes the `ToolCall`'s
`trace_id`/`parent_span_id`.

## Google ADK (first-class)

`pip install 'agent-gantry[google-adk]'` (pins `google-adk>=2.6.1`; CI runs the
tests on exactly 2.6.1 and on the latest). `agent-frameworks` also includes it,
capped below 2.8 for an opentelemetry trade-off with Agent Framework.

| Mode | Build | Use when |
|---|---|---|
| Per-turn (recommended) | `adapter.agent(model=..., name=...)`, or `Agent(tools=[], before_model_callback=adapter.before_model_callback())` | Agents; Gantry picks each model request's tools |
| Static slice | `await adapter.select(query)` → `Agent(tools=...)` | A fixed task |

**Per-turn selection.** Before every model request the callback selects tools
and injects them with `LlmRequest.append_tools` (which also registers them in
`tools_dict`, so ADK can run them); tools the agent already has are left
alone. The query is the newest user message plus the latest successful tool
result after it (first 2,000 characters; `query_from_results=False` keeps
output out; an `{"error": ...}` response never feeds it). State is keyed by
the ADK invocation, so concurrent runs are independent; within a run
selections are cached, earlier tools stay offered up to `max_offered`
(`sticky=True`, default `3 * limit`), and a retrieval failure keeps the
previous tools plus the pins.

**Errors and approval.** A failed call returns `{"error": "<type>: <message>"}`
as the function response, ADK's convention for telling the model a tool
failed (an exception would end the run). A confirmation-gated call (the tool's
flag or a `SecurityPolicy` pattern, per `AgentGantry.requires_confirmation()`)
sets ADK's `require_confirmation`, so the run emits an
`adk_request_confirmation` function call:

```python
async for event in runner.run_async(user_id=uid, session_id=sid, new_message=msg):
    for part in event.content.parts if event.content else []:
        if part.function_call and part.function_call.name == "adk_request_confirmation":
            pending = part.function_call
reply = types.FunctionResponse(name="adk_request_confirmation", id=pending.id,
                               response={"confirmed": True})   # False rejects
await consume(runner.run_async(user_id=uid, session_id=sid,
                               new_message=types.Content(role="user", parts=[types.Part(function_response=reply)])))
```

A confirmed call runs once through `gantry.execute` with only the confirmation
gate cleared; ADK itself refuses a rejected one. ADK resumes a confirmed call
through the agent's own tools, so `adapter.agent()` includes a small toolset
that supplies the gated Gantry tools only on the turn answering a confirmation;
when you wire `before_model_callback()` into your own `Agent`, add
`tools=[adapter.confirmation_toolset()]` (with the same `namespaces` /
`required` / `always_include` as the callback). `error_formatter=fn` rewrites
failure text; `error_mode="raise"` opts out of both behaviours.

**Run context and tracing.** Gantry handlers read the call's ADK `ToolContext`
(session `state`, `user_id`, `invocation_id`, `function_call_id`, artifacts,
memory) with `agent_gantry.google_adk.current_tool_context()` or
`current_tool_invocation()` (`deps` is the session state; `metadata` carries
`agent_name`, `user_id`, `session_id`). ADK traces with OpenTelemetry, so its
active span becomes the `ToolCall`'s `trace_id`/`parent_span_id`.

## CrewAI (first-class)

`pip install 'agent-gantry[crewai]'` (pins `crewai>=1.15.22` on Python 3.10–3.13;
CI runs the tests on exactly 1.15.22 and on the latest). It cannot share an
environment with `openai-agents` (CrewAI pins `openai<3`).

| Mode | Build | Use when |
|---|---|---|
| Static slice | `Agent(tools=await adapter.select(query))` | Most crews; select per agent |
| Per call | `builder = adapter.agent_builder(role=..., llm=...)` → `await builder.build(query)` per task | Tools should follow each new task |

CrewAI fixes an agent's tools when it is built and has no hook to change them
between model turns, so selection is per call rather than per turn.

**Errors.** A failed call returns CrewAI's `ToolFailure` (`message`, the error
type as `code`, `retryable` for argument errors), so the agent reads the
failure and CrewAI records it on `TaskOutput.tool_failures`; the agent's or
task's `tool_failure_policy` (`ignore` / `warn` / `raise`) decides whether the
run carries on. `error_formatter=fn` rewrites the text; `error_mode="raise"`
lets `ToolExecutionError` propagate into CrewAI's own exception handling.

**Approval.** CrewAI has no pause-and-resume for tool calls; its native gate is
a `before_tool_call` hook. Gantry registers one (once per process) and, for a
confirmation-gated tool (per `AgentGantry.requires_confirmation()`), asks the
adapter's approver:

```python
def approve(request: ApprovalRequest) -> bool:      # tool_name, arguments, agent, task, crew
    return ask_a_human(request)

adapter = CrewAIAdapter(gantry, approve=approve)      # or approve="console" for request_human_input
```

An approved call runs once with only the confirmation gate cleared; a refused
one is blocked by CrewAI. Without an approver the agent is told the call needs
approval and it is not run.

**Run context and tracing.** Gantry handlers read the call's CrewAI
`ToolCallHookContext` (`agent`, `task`, `crew`) with
`agent_gantry.crewai.current_crew_context()` or `current_tool_invocation()`
(`deps` is the crew; `metadata` carries `agent_role`, `task_name`, `task_id`).
An active OpenTelemetry span becomes the `ToolCall`'s trace.

## LlamaIndex (first-class)

`pip install 'agent-gantry[llamaindex]'` (pins `llama-index-core>=0.13.0`; CI runs
the tests on exactly 0.13.0 and on the latest).

| Mode | Build | Use when |
|---|---|---|
| Per step (recommended) | `adapter.function_agent(llm)`, or `FunctionAgent(tool_retriever=adapter.tool_retriever(), llm=llm)` | Agents; Gantry picks each step's tools |
| Static slice | `FunctionAgent(tools=await adapter.select(query), llm=llm)` | A fixed task |

**Errors.** A failed call raises `LlamaIndexToolError` (a `ToolExecutionError`
whose text is `<type>: <message>`, capped); LlamaIndex agents turn it into
`ToolOutput(is_error=True)` for the model and the run continues.
`error_formatter=fn` rewrites the text; `error_mode="raise"` raises the plain
`ToolExecutionError` (LlamaIndex still shows its message to the model).

**Approval.** In a workflow agent, a confirmation-gated call (per
`AgentGantry.requires_confirmation()`) waits with LlamaIndex's
human-in-the-loop events:

```python
handler = agent.run(user_msg="refund order 4471")
async for event in handler.stream_events():
    if isinstance(event, InputRequiredEvent):          # event.gantry_tool, event.gantry_arguments
        handler.ctx.send_event(                        # "yes" approves; anything else rejects
            HumanResponseEvent(response="yes", gantry_approval_id=event.gantry_approval_id)
        )
result = await handler
```

The response must echo the request's `gantry_approval_id`: each call waits only
for a response naming it, so with several gated calls pending an answer never
decides another. Run gated tools in `adapter.function_agent(...)` or
`agent_gantry.llamaindex.GantryFunctionAgent(tools=..., llm=...)` (a
`FunctionAgent` that passes each call's id to its tool): there, two identical
calls in one step get one request each and an approved call runs exactly once
even when LlamaIndex replays the step. In a plain `FunctionAgent` the request is
identified by the tool and its arguments. `"yes"`, `"y"`, `"approve"` or `"approved"` (or an event with `approved=True`)
runs the call once with only the confirmation gate cleared. Outside a workflow
(a direct `tool.call`), the model is told the call needs approval.

**Run context.** The workflow `Context` is injected through a hidden `ctx`
parameter (kept out of the tool schema; direct calls get `None`) and readable
with `agent_gantry.llamaindex.current_workflow_context()` or
`current_tool_invocation()` (`deps` is `ctx.store`).

**Per-step selection.** `tool_retriever()` re-selects for every step's query.
`FunctionAgent` passes only the query, so there is no per-run state to be
sticky with; when retrieval fails the step offers the pinned tools alone.

## Strands Agents (first-class)

`pip install 'agent-gantry[strands]'` (pins `strands-agents>=1.38.0`; CI runs the
tests on exactly 1.38.0 and on the latest).

| Mode | Build | Use when |
|---|---|---|
| Per-turn (recommended) | `adapter.agent(model=...)`, or `Agent(tools=[], hooks=[adapter.tool_hook()])` | Agents; Gantry picks each model call's tools |
| Static slice | `Agent(tools=await adapter.select(query))` | A fixed task |

**Errors.** A failed call returns an error `ToolResult`
(`{"status": "error", "content": [{"text": "<type>: <message>"}]}`, capped), so
the model reads it and the run continues, whether the tool runs in the agent
loop or is called directly. `error_formatter=fn` rewrites the text;
`error_mode="raise"` raises `ToolExecutionError` (Strands' loop then reports
its own `Error: ...` result).

**Approval.** A confirmation-gated call (per `AgentGantry.requires_confirmation()`)
raises a Strands interrupt from inside the tool
(`tool_context.interrupt("gantry-approval", reason={...})`), so the run stops:

```python
result = await agent.invoke_async("refund order 4471")
if result.stop_reason == "interrupt":
    [interrupt] = result.interrupts            # interrupt.reason: tool, arguments, description
    result = await agent.invoke_async(
        [{"interruptResponse": {"interruptId": interrupt.id, "response": "yes"}}]
    )
```

`"yes"`, `"y"`, `"approve"`, `"approved"`, `True` or `{"approved": True}` runs
the call once with only the confirmation gate cleared; anything else rejects it.
A direct call (no agent) reports that it needs approval.

**Run context and tracing.** Gantry handlers read the Strands `ToolContext`
(`tool_use`, `agent`, `invocation_state` — the keyword arguments passed to
`agent(...)`) with `agent_gantry.strands.current_tool_context()` or
`current_tool_invocation()` (`deps` is `invocation_state`, `tool_call_id` the
`toolUseId`). Strands traces with OpenTelemetry, so its span becomes the
`ToolCall`'s trace.

## Agno (first-class)

`pip install 'agent-gantry[agno]'` (pins `agno>=2.6.10`; CI runs the tests on
exactly 2.6.10 and on the latest).

| Mode | Build | Use when |
|---|---|---|
| Per-run (recommended) | `adapter.agent(model=...)` | One agent; Gantry picks each run's tools |
| Static slice | `Agent(tools=await adapter.select(query))` | A fixed task |
| Per-call builder | `await adapter.agent_builder(model=...).build(query)` | A fresh `Agent` per query |

`adapter.agent()` gives the `Agent` a callable `tools` factory (with
`cache_callables=False`) and a pre-hook that records each run's input, so Agno
itself asks Gantry for that run's tools. The selection is kept per `run_id`, so
`continue_run` after an approval is offered the same tools; an agent object
resuming another's paused run reads the input back from the shared `db`. If
retrieval fails the pinned tools are still offered (with a `WARNING`). Tools
use a synchronous entrypoint, so they work under both `agent.run()` and
`agent.arun()`.

**Errors.** A failed call raises `AgnoToolError`, a `ToolExecutionError` whose
`str()` is `"<type>: <message>"` (capped). Agno records it as the tool's
failed result (`tool_call_error=True`) and shows the model that text, so the
run continues. `error_formatter=fn` rewrites the text; `error_mode="raise"`
raises the plain `ToolExecutionError`.

**Approval.** A confirmation-gated tool (per `AgentGantry.requires_confirmation()`)
becomes a `Function(requires_confirmation=True)`, so the run pauses:

```python
run = await agent.arun("refund order 4471")
if run.is_paused:
    for requirement in run.active_requirements:   # .tool_execution: tool_name, tool_args
        requirement.confirm()                     # or requirement.reject()
    run = await agent.acontinue_run(run_response=run)
```

A confirmed call runs once with only the confirmation gate cleared; a
rejected one never runs. A direct call (no agent) reports that it needs
approval. `error_mode="raise"` turns the pause off (gated calls report as not
run).

**Run context and tracing.** Gantry handlers read Agno's `RunContext`
(`run_id`, `session_id`, `user_id`, `session_state`, `dependencies`,
`metadata`) with `agent_gantry.agno.current_run_context()` or
`current_tool_invocation()` (`deps` is `run_context.dependencies`,
`tool_call_id` the model's call id, metadata `agent_name` / `session_id` /
`user_id`). The active OpenTelemetry span becomes the `ToolCall`'s trace.


## Haystack (first-class)

`pip install 'agent-gantry[haystack]'` (pins `haystack-ai>=3.0.0`; CI runs the
tests on exactly 3.0.0 and on the latest).

| Mode | Build | Use when |
|---|---|---|
| Per-step (recommended) | `adapter.agent(chat_generator=..., confirmation_ui=...)` | Agents; Gantry picks each step's tools |
| Static slice | `Agent(tools=await adapter.select(query), hooks={"before_tool": [adapter.confirmation_hook(ui)]})` | A fixed task |

`adapter.agent()` gives the `Agent` a dynamic `Toolset` and `before_run` /
`after_tool` hooks. Gantry selects tools for the latest user message before
the first step and again after each tool step (joining the latest successful
tool result to the query), and Haystack re-flattens the toolset at the start
of every step, so the next model call sees the new selection. `sticky=True`
keeps earlier tools up to `max_offered` (default `3 * limit`); if retrieval
fails the run keeps its previous tools plus pins (with a `WARNING`). Per-run
state travels in a context variable, so concurrent runs of one Agent stay
apart. Your own `hooks` still run.

**Errors.** A failed call raises `HaystackToolError`, a `ToolExecutionError`
whose `str()` is `"<type>: <message>"` (capped). The Agent turns it into an
error tool result with exactly that text (Gantry's `Tool` subclass drops
Haystack's echo of the call's arguments) and the run continues. `error_formatter=fn` rewrites the text; `error_mode="raise"`
raises the plain `ToolExecutionError`.

**Approval.** `adapter.confirmation_hook(confirmation_ui)` is a `before_tool`
hook built on Haystack's `ConfirmationHook`: it asks the `ConfirmationUI`
(`SimpleConsoleUI()`, `RichConsoleUI()` or your own) about confirmation-gated
Gantry tools only, per `AgentGantry.requires_confirmation()`.

```python
from haystack.hooks.human_in_the_loop import SimpleConsoleUI

agent = HaystackAdapter(gantry).agent(chat_generator=generator, confirmation_ui=SimpleConsoleUI())
```

A confirmed call runs once with only the confirmation gate cleared, a
modified one runs once with the new arguments, and a rejected one never runs
(Haystack tells the model). The approval is tied to the exact call on the
run's `State`. Without a confirmation UI, or when called directly, a gated
tool reports that it needs approval.

**Run context and tracing.** Each call gets the Agent's live `State`
(Haystack injects it; the model's schema never shows it), readable with
`agent_gantry.haystack.current_agent_state()` or `current_tool_invocation()`
(`deps` is `state.data`). Tools have a native `async_function`, so
`Agent.run_async` does not block. The active OpenTelemetry span becomes the
`ToolCall`'s trace.

## DSPy (first-class)

`pip install 'agent-gantry[dspy]'` (pins `dspy>=3.5.0`, the redesigned
`dspy.ReAct` with native `tool_calls` and a `submit` tool; CI runs the tests on
exactly 3.5.0 and on the latest).

| Mode | Build | Use when |
|---|---|---|
| Static slice | `dspy.ReAct(signature, tools=await adapter.select(query))` | A fixed task |
| Per-call builder | `await adapter.agent_builder(signature).build(query)` | A fresh `ReAct` per query |

`dspy.ReAct` fixes its tools, and bakes their names into its instructions, at
construction, and DSPy has no hook before each model call, so selection is per
call. The builder offers the pinned tools if retrieval fails (with a
`WARNING`). Tool bodies are synchronous, so they work under both `react(...)`
and `await react.acall(...)` with no DSPy setting.

**Errors.** A failed call raises `DSPyToolError`, a `ToolExecutionError` whose
`str()` is `"<type>: <message>"` (capped). `dspy.ReAct` records it as an error
tool result (`is_error=True`, `"Execution error in <tool>: ..."`) and the model
gets another turn. DSPy adds its own short traceback to every tool error; it
ends with Gantry's text, and the original error is not chained onto it, so
`error_formatter=fn` redaction holds. `error_mode="raise"` raises the plain
`ToolExecutionError`.

**Approval.** DSPy has no human-in-the-loop mechanism, so the adapter takes an
approver for confirmation-gated calls (per `AgentGantry.requires_confirmation()`):

```python
from agent_gantry.dspy import DSPyAdapter

def approve(request):  # ApprovalRequest: tool_name, arguments, description, module
    return input(f"Run {request.tool_name}({request.arguments})? [y/N] ") == "y"

tools = await DSPyAdapter(gantry, approve=approve).select("refund order 4471")
```

`True` runs the call once with only the confirmation gate cleared; anything
else (or an approver that raises) blocks it and tells the model it was
declined. Without an approver a gated call reports that it needs approval.
`error_mode="raise"` never asks the approver.

**Run context and tracing.** Gantry handlers read the DSPy module running the
call (normally the `dspy.ReAct`) with `agent_gantry.dspy.current_dspy_module()`
or `current_tool_invocation().native_context`. DSPy has no run or call ids to
pass on. The active OpenTelemetry span becomes the `ToolCall`'s trace.

## Shared base

`base.py` provides `GantryToolset` (selection, including the `required`/
`always_include` pinning contract above) and `ToolSpec` (a framework-neutral
handle with `.name`, `.description`, `.parameters`, plus `ainvoke`/`invoke`).
`invoke()` is safe to call from synchronous framework code even inside a
running event loop — it runs the coroutine on a worker thread and blocks for
the result. `errors.py` holds the shared `MissingRequiredToolError`, imported
by both `GantryToolset.select` and the Microsoft Agent Framework's
`GantryContextProvider` (`agent_framework_provider.py` re-exports it from
there for backward compatibility — MAF's own `required=[...]` implementation
was left in place rather than delegating to the shared helper, since it's
deeply entangled with skills/`static_tools`/`ContextVar`-scoped retrieval
history that has no equivalent in the plain adapter layer; only the error
type is shared).

## Error-handling policy

Tool failures surface two different ways depending on *what* failed:

- **Tool execution failure** — `gantry.execute()` returns a non-success
  `ToolResult` (or raises). This is the everyday case: a registered tool
  raised, timed out, tripped a circuit breaker, or was denied by policy.
- **Selection failure** — `gantry.retrieve()` itself raises, e.g. the vector
  store or embedder is briefly unavailable. This only applies to the eight
  *per-turn* live providers (`integrations/frameworks/*_live.py`), which call
  back into Gantry autonomously, mid-conversation, with no application code
  between calls to catch anything.

These are deliberately **not** unified — a bad tool call is the model's
problem to react to; a broken retrieval pipe is an infrastructure problem the
agent's *turn* must survive regardless of what the model does next.

### 1. Tool execution failure — default: raises `ToolExecutionError`

`ToolSpec.ainvoke` (and its sync bridge, `ToolSpec.invoke`, safe to call even
from inside a running event loop — it runs the coroutine on a worker thread
via `_run_coroutine_sync` and blocks for the result, verified to propagate the
exception through that bridge intact) raises
`agent_gantry.integrations.frameworks.base.ToolExecutionError(tool_name,
status, error)` on any non-success result. `ToolExecutionError.error` carries
the underlying error string; its message is
`f"Tool {tool_name!r} failed (status={status}): {error or 'no detail'}"` —
stable and pattern-matchable (locked by
`tests/frameworks/test_conformance.py::test_tool_execution_error_message_format`).

Two subclasses distinguish the "never executed by design" outcomes so a
caller can branch without string-matching the status —
`ToolConfirmationRequiredError` (the tool is confirmation-gated: either its
own `requires_confirmation=True` flag or a `SecurityPolicy`
`require_confirmation` pattern matched; approve by re-issuing the call with
`ToolCall(require_confirmation=False)`, which clears both gates while every
denial check still runs) and `ToolPermissionDeniedError` (the security
policy refused it outright). Both are re-exported from
`agent_gantry.integrations.frameworks`; catching `ToolExecutionError` still
catches every outcome.

**Every native adapter except Pydantic AI, the OpenAI Agents SDK, LangChain, LangGraph, Google ADK, CrewAI and LlamaIndex lets this propagate uncaught** from the
native tool object's own invocation entry point (`.func`, `._run`, `.forward`,
`.entrypoint`, `.method`, `.on_invoke_tool`, …) — proven for all of them,
including the sync wrappers, by
`test_conformance.py::test_adapter_tool_failure_matches_documented_error_kind`.
From there, each framework's *own* error handling takes over exactly as it
would for a hand-written native tool (a LangChain `AgentExecutor`'s
`handle_tool_error`, a Haystack `ToolInvoker`'s `raise_on_failure` (haystack 2.x; `Agent`'s
`raise_on_tool_invocation_failure` on >=3.0), an OpenAI
Agents SDK `failure_error_function`, …) — Gantry does not second-guess it.

**Pydantic AI is the one deviation at the `convert()` layer.** Pydantic AI's
own handling of an ordinary exception from a tool is to end the run, while it
has native exceptions meaning exactly what Gantry's outcomes mean. So by
default (`error_mode="model"`) the adapter re-raises, chained from the
`ToolExecutionError`: argument errors (`error_type` `ValidationError` /
`ToolNotFound`) as `ModelRetry`, so the model corrects its call within the
tool's retry budget; `ToolConfirmationRequiredError` as `ApprovalRequired`,
Pydantic AI's human-in-the-loop flow (an approved replay executes once with
the confirmation gate cleared, every denial check still running); and every
other failure as `ToolFailed`, which the model sees as a terminal result (on
2.x releases before `ToolFailed` existed, the failure text is returned as the
tool's result instead, which is equally terminal). The text is capped at 2,000
characters.
`error_mode="raise"` (on the adapter, `select`, `toolset`, `capability`, …)
restores the raise contract above. Locked by the `native_tool_failed` case
of the conformance matrix and `tests/frameworks/test_pydantic_ai.py`.

**The OpenAI Agents SDK is the other.** The SDK has no retry exception, and
an exception from a raw `FunctionTool` ends the run with `UserError` (its
default error handler, used for decorated tools, hides the reason from the
model). So by default (`error_mode="model"`) a failure is returned as the
tool's output, `Error: <error_type>: <message>` (capped, rewritten by
`error_formatter` when set), which the model reads and can act on, including
correcting bad arguments. Confirmation-gated calls never get that far: the
SDK's `needs_approval` pauses the run first (see the section above). A call
stopped by a custom policy the pre-check could not predict is reported as
not run. `error_mode="raise"` restores the raise contract. Locked by the
`native_tool_output` case of the conformance matrix and
`tests/frameworks/test_openai_agents_native.py`.

Three deliberate deviations exist **one layer deeper** than `convert()`/
`select()` — i.e. not in the wrapper this repo builds, but in a framework's
own downstream consumption of it:

1. **Microsoft Agent Framework** (`agent_framework_bridge.py`,
   `_build_callable_for_tool`) returns a JSON `{"error": ...}`
   **string** to the model instead of raising. Deliberate: AF's tool runner
   otherwise replaces an uncaught error with an opaque `"Error: Function
   failed."` string when `include_detailed_errors` is off (the AF default),
   destroying the root cause — returning it as tool output lets the LLM see
   and react to the real error.
2. **AWS Strands Agents** tools return an error `ToolResult`
   (`status="error"`, `<type>: <message>`) for a failed call, both in the
   agent loop (`.stream()`) and when called directly; `error_mode="raise"`
   raises `ToolExecutionError` instead, which Strands' loop then turns into
   its own error result. See
   `tests/frameworks/test_strands_live.py::test_tool_failure_is_an_error_tool_result`.

DSPy tools raise `DSPyToolError` (a `ToolExecutionError`) when called
directly; `dspy.ReAct` itself turns any raising tool into an error tool result
(`is_error=True`) and carries on — see
`tests/frameworks/test_dspy_live.py::test_react_records_tool_failure_as_an_error_result`.

### 2. Selection failure (live/per-turn mode) — always degrades, never raises

A `gantry.retrieve()` failure inside a per-turn live provider must not kill
the agent's turn. Every one of the eight `per-turn` providers now follows one
uniform rule: **catch, log a `WARNING` (with `exc_info=True`), and degrade —
never propagate.** *How* it degrades depends on whether that framework's tool
surface is stateless-per-turn or persists across turns:

- **Stateless per-turn recomputation** (the framework calls the hook fresh
  every turn/step with no notion of "last turn's tools" to fall back to) →
  degrade to **no tools this turn**: Google ADK, LangGraph, LlamaIndex.
- **Pydantic AI** sits between the two: its toolset keeps per-run state (on
  the copy `for_run` returns), so a failed step degrades to **the tools
  already offered earlier in this run** (none on the first step).
- **Stateful in-place mutation** (the framework's tool registry/plugin/list
  persists across turns) → degrade to **leave the previous turn's tools in
  place**, so a transient retrieval blip doesn't strip a working agent of
  tools it already has: OpenAI Agents SDK, Strands.

This does not apply to the five `per-call` adapters (LangChain, CrewAI, Agno,
Haystack) or DSPy's per-call builder: their "live" surface is a
builder whose `.build(query)` the *caller* awaits directly before each new
top-level call — an ordinary Python call the caller can wrap in `try`/`except`
itself, not a framework-internal hook invoked deep inside someone else's loop.

Locked by the `test_conformance.py::test_*_live_selection_failure_degrades_gracefully`
tests (one per per-turn provider, patching `GantryToolset.select` to raise and
asserting no exception + a `WARNING` log record).

### Per-framework summary

| Framework | (a) Tool execution failure | (b) Selection failure (live/per-turn) |
|---|---|---|
| LangChain | **Deviation:** LangChain's `ToolException` (the tool has `handle_tool_error=True`, so the model gets an error `ToolMessage`), chained from `ToolExecutionError`; `error_mode="raise"` restores raising | *(per-call — see above)* |
| LangGraph | **Deviation:** as LangChain | `middleware()`: degrades to this run's previous tools plus pins (per-run state) + `WARNING`; `select_for_state`: no tools + `WARNING` |
| LlamaIndex | Raises `LlamaIndexToolError` (a `ToolExecutionError` whose text is the model-facing failure); agents show it as `ToolOutput(is_error=True)` | Degrades to the pinned tools alone this step (stateless) + `WARNING` |
| CrewAI | **Deviation:** `_run` returns CrewAI's `ToolFailure` (recorded on `TaskOutput.tool_failures`); `error_mode="raise"` restores raising | *(per-call — see above)* |
| Pydantic AI | **Deviation:** `ModelRetry` / `ToolFailed` / `ApprovalRequired` (native), chained from `ToolExecutionError`; `error_mode="raise"` restores raising | Degrades to this run's earlier tools (per-run state) + `WARNING` |
| OpenAI Agents SDK | **Deviation:** returned to the model as `Error: ...` tool output; `error_mode="raise"` restores raising | `dynamic_tools`: degrades to this run's earlier tools (per-run state) + `WARNING`; `session`/`run_hooks`: previous turn's `agent.tools` + `WARNING` |
| Haystack | Raises `HaystackToolError` (a `ToolExecutionError` whose text is the model-facing failure), which the Agent turns into an error tool result; `error_mode="raise"` raises the plain `ToolExecutionError` | Per-step `agent()`: keeps the run's previous tools plus pins + `WARNING` |
| Agno | Raises `AgnoToolError` (a `ToolExecutionError` whose text is the model-facing failure), which Agno records as a failed tool result; `error_mode="raise"` raises the plain `ToolExecutionError` | Per-run `agent()`: degrades to the pinned tools + `WARNING` |
| Google ADK | **Deviation:** `FunctionTool.run_async` returns `{"error": ...}` (ADK's convention; the raw `.func` still raises); `error_mode="raise"` restores raising | Degrades to this run's previous tools plus pins (per-run state) + `WARNING` |
| Strands Agents | **Deviation:** error `ToolResult(status="error")` from both `__call__` and the agent loop; `error_mode="raise"` restores raising | Degrades to previous turn's registered tools (stateful) + `WARNING` — the other precedent |
| DSPy | Raises `DSPyToolError` (a `ToolExecutionError` whose text is the model-facing failure), which `dspy.ReAct` records as an error tool result; `error_mode="raise"` raises the plain `ToolExecutionError` | Per-call builder: degrades to the pinned tools + `WARNING` |
| Microsoft Agent Framework | **Deviation:** JSON `{"error": ...}` string returned to the model, not raised | *(`GantryToolBridge`/`ContextProvider` selection is a separate mechanism from `GantryToolset`, outside this audit's scope)* |
