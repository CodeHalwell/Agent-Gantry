# Native framework tool adapters

Agent-Gantry selects a small, relevant slice of tools from a large registry.
These adapters hand that slice to a concrete agent framework as **native tool
objects** — not just JSON schemas — so the framework can introspect and call
them, while every invocation still routes through `gantry.execute` (retries,
timeouts, circuit breakers, security policy all apply).

## Supported frameworks

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

All third-party imports are **lazy** — `import agent_gantry` never requires any
of these frameworks. A missing framework raises `ImportError` with a
`pip install …` hint only when you call its adapter.

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
shared by `GantryToolset.select` (`base.py`), so all 15 framework integrations
get the same guarantee:

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
| Haystack | per-call | `tool_invoker_builder` | `GantryLiveHaystackToolInvoker` builder (ToolInvoker on haystack 2.x, Agent on >=3.0) | `await builder.build(query)` per call |
| Agno | per-call | `agent_builder` | `GantryLiveAgnoAgent` builder | `await builder.build(query)` per run |
| Google ADK | per-turn | `before_model_callback` | async `(callback_context, llm_request) -> None` | `Agent(tools=[], before_model_callback=<result>)` |
| Strands Agents | per-turn | `tool_hook` | `GantryStrandsToolHook` (`HookProvider`) | `Agent(tools=[], hooks=[<result>])` |
| Microsoft Agent Framework* | per-turn | `context_provider` (`query_strategy="per_call"`) | `GantryContextProvider` | `Agent(context_providers=[<result>])` + `<result>.as_chat_middleware()`, or `<result>.attach_to(agent)` |

\* `AgentFrameworkAdapter` (`agent_gantry.agent_framework`) is not a
`BaseFrameworkAdapter` subclass (no `select`/`convert`) but participates in the
same `live_tier`/`live()` facade — see `agent_framework_adapter.py`.

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
| Pydantic AI | `PydanticAIAdapter(gantry).toolset()` | `AbstractToolset.get_tools()` |
| Google ADK | `GoogleADKAdapter(gantry).before_model_callback()` / `.agent()` | `Agent(before_model_callback=…)` |
| Strands Agents | `StrandsAdapter(gantry).tool_hook()` / `.agent()` | `Agent(hooks=[…])` — `BeforeModelCallEvent` |
| LangGraph | `LangGraphAdapter(gantry).react_agent(model)` / `.areact_agent(model)` / `.select_for_state(state)` | dynamic `model` callable (re-binds tools per turn) |
| OpenAI Agents SDK | `OpenAIAgentsAdapter(gantry).run(agent, run_input)` / `.session(agent)` / `.run_hooks(agent)` | `RunHooks.on_llm_start` + per-run refresh |

```python
from agent_gantry.llamaindex import LlamaIndexAdapter
agent = LlamaIndexAdapter(gantry).function_agent(llm)   # LlamaIndex agent that re-selects tools each step
```

Frameworks whose tool list is **fixed at agent construction** (CrewAI, Agno,
Haystack, DSPy) can't re-advertise tools mid-run; their "live" wrappers —
obtained via `CrewAIAdapter(gantry).agent_builder(...)`,
`AgnoAdapter(gantry).agent_builder(...)`,
`DSPyAdapter(gantry).agent_builder(signature, ...)`,
and `HaystackAdapter(gantry).tool_invoker_builder(...)` (or `HaystackAdapter(gantry).live_tools(query)`
for tools alone) — re-select and rebuild the agent on each top-level call —
the deepest those frameworks allow. DSPy's `dspy.ReAct` bakes each tool's
name/description into its instruction prompt at construction and has no
runtime hook (`dspy.utils.callback.BaseCallback`'s `on_tool_start`/
`on_module_start` fire around an already-selected call, not before the model
picks the next tool) — see the module docstring in
`agent_gantry/integrations/frameworks/dspy.py` for the full analysis.

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

**Every one of the 14 native adapters lets this propagate uncaught** from the
native tool object's own invocation entry point (`.func`, `._run`, `.forward`,
`.entrypoint`, `.method`, `.on_invoke_tool`, …) — proven for all of them,
including the sync wrappers, by
`test_conformance.py::test_adapter_tool_failure_matches_documented_error_kind`.
From there, each framework's *own* error handling takes over exactly as it
would for a hand-written native tool (a LangChain `AgentExecutor`'s
`handle_tool_error`, a Haystack `ToolInvoker`'s `raise_on_failure` (haystack 2.x; `Agent`'s
`raise_on_tool_invocation_failure` on >=3.0), an OpenAI
Agents SDK `failure_error_function`, …) — Gantry does not second-guess it.

Three deliberate deviations exist **one layer deeper** than `convert()`/
`select()` — i.e. not in the wrapper this repo builds, but in a framework's
own downstream consumption of it:

1. **Microsoft Agent Framework** (`agent_framework_bridge.py`,
   `GantryToolBridge._build_tool_execute`) returns a JSON `{"error": ...}`
   **string** to the model instead of raising. Deliberate: AF's tool runner
   otherwise replaces an uncaught error with an opaque `"Error: Function
   failed."` string when `include_detailed_errors` is off (the AF default),
   destroying the root cause — returning it as tool output lets the LLM see
   and react to the real error.
2. **AWS Strands Agents' real `Agent` tool-execution loop**
   (`strands.tools.decorator.DecoratedFunctionTool.stream`, Strands' own code,
   not this adapter's) catches any exception and converts it into an error
   `ToolResult` (`status="error"`). This is Strands' native contract for
   *every* function-based tool, not something `StrandsAdapter` opts into —
   calling the tool directly via `__call__`/`.acall` (bypassing Strands' own
   dispatch) still raises `ToolExecutionError` untouched; only `.stream()`
   (what the real agent loop calls for a model-issued tool call) absorbs it.
   See `tests/frameworks/test_strands_live.py::test_stream_absorbs_tool_failure_into_error_tool_result`.

A fourth case looks similar but is **DSPy's own behaviour, not a Gantry
deviation**: `dspy.Tool.__call__`/`.acall` never swallow the error (verified
in `tests/frameworks/test_dspy_live.py::test_tool_failure_surfaces_as_tool_execution_error`)
— but `dspy.ReAct.forward`/`.aforward` (DSPy's own agentic driver, what
`DSPyAdapter.agent_builder`/`.live()` hand you) wrap each tool call in a bare
`except Exception` and fold it into the trajectory as an `"Execution error in
<tool>: ..."` observation string instead of raising — see
`test_react_forward_absorbs_tool_failure_into_trajectory`. It is listed here
for completeness (it's what most DSPy users actually see) but isn't a
Gantry-adapter contract at all.

### 2. Selection failure (live/per-turn mode) — always degrades, never raises

A `gantry.retrieve()` failure inside a per-turn live provider must not kill
the agent's turn. Every one of the eight `per-turn` providers now follows one
uniform rule: **catch, log a `WARNING` (with `exc_info=True`), and degrade —
never propagate.** *How* it degrades depends on whether that framework's tool
surface is stateless-per-turn or persists across turns:

- **Stateless per-turn recomputation** (the framework calls the hook fresh
  every turn/step with no notion of "last turn's tools" to fall back to) →
  degrade to **no tools this turn**: Google ADK, LangGraph, LlamaIndex,
  Pydantic AI.
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
| LangChain | Raises `ToolExecutionError` | *(per-call — see above)* |
| LangGraph | Raises `ToolExecutionError` | Degrades to no tools this turn (stateless) + `WARNING` |
| LlamaIndex | Raises `ToolExecutionError` | Degrades to no tools this step (stateless) + `WARNING` |
| CrewAI | Raises `ToolExecutionError` | *(per-call — see above)* |
| Pydantic AI | Raises `ToolExecutionError` | Degrades to previous run's tools (stateful cache) + `WARNING` |
| OpenAI Agents SDK | Raises `ToolExecutionError` | Degrades to previous turn's `agent.tools` (stateful) + `WARNING` |
| Haystack | Raises `ToolExecutionError` | *(per-call — see above)* |
| Agno | Raises `ToolExecutionError` | *(per-call — see above)* |
| Google ADK | Raises `ToolExecutionError` | Degrades to no tools this turn (stateless) + `WARNING` — the original precedent |
| Strands Agents (`__call__`/`.acall`) | Raises `ToolExecutionError` | — |
| Strands Agents (real agent loop, `.stream()`) | **Deviation:** error `ToolResult(status="error")`, not raised (Strands' own contract) | Degrades to previous turn's registered tools (stateful) + `WARNING` — the other precedent |
| DSPy (`dspy.Tool.__call__`/`.acall`) | Raises `ToolExecutionError` | *(per-call — see above)* |
| DSPy (`dspy.ReAct.forward`/`.aforward`) | **DSPy's own behaviour** (not a Gantry deviation): folded into the trajectory as an `"Execution error in <tool>: ..."` observation | *(per-call — see above)* |
| Microsoft Agent Framework | **Deviation:** JSON `{"error": ...}` string returned to the model, not raised | *(`GantryToolBridge`/`ContextProvider` selection is a separate mechanism from `GantryToolset`, outside this audit's scope)* |
