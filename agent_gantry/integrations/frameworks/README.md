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
| Smolagents | `SmolagentsAdapter` | `smolagents.Tool` subclass |
| Haystack | `HaystackAdapter` | `haystack.tools.Tool` |
| Agno | `AgnoAdapter` | `agno.tools.function.Function` |
| AutoGen / AG2 | `AutoGenAdapter` (`.select`, `.register`) | callables + `register_function` |
| Semantic Kernel | `SemanticKernelAdapter` (`.select`, `.plugin`) | `KernelFunction` (`@kernel_function`) |
| Google ADK | `GoogleADKAdapter` | `google.adk.tools.FunctionTool` |
| Strands Agents | `StrandsAdapter` | `strands.tools.decorator.DecoratedFunctionTool` |

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
`agent_gantry.pydantic_ai`, `agent_gantry.openai_agents`, `agent_gantry.smolagents`,
`agent_gantry.haystack`, `agent_gantry.agno`, `agent_gantry.autogen`,
`agent_gantry.semantic_kernel`, `agent_gantry.google_adk`, `agent_gantry.langgraph`,
`agent_gantry.strands`, `agent_gantry.agent_framework`) re-exporting that framework's `<Framework>Adapter`
class (which carries both the static `.select`/`.convert` and the deep live
methods). Importing `agent_gantry` never pulls these in.

Every `<Adapter>(gantry).select(query, *, limit=None, score_threshold=0.0,
namespaces=None, tools_already_used=None)` accepts the same explicit selection
knobs as `GantryToolset.select` — `limit` defaults to the adapter's
`default_limit` (5, `DEFAULT_TOOL_LIMIT` in `base.py`) when omitted. Need one
conversion at a time? Use the staticmethod `<Adapter>.convert(spec)` with specs
from `GantryToolset(gantry).select(query)`.

## Multi-turn re-selection — autonomous *and* conversational

`ToolRefresher` generalizes the Microsoft Agent Framework provider's per-call
retrieval to any framework: re-rank the whole registry on every turn so the
agent can pivot to a different tool as the task changes.

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

## Deep per-turn "live" providers (as embedded as Microsoft Agent Framework)

The `<Adapter>.select` methods above are *static*: select once, hand over a fixed
tool list. The **live** methods go deeper — they hook each framework's own
per-turn lifecycle so Gantry re-selects tools on **every turn**, exactly like
`GantryContextProvider` does for Microsoft Agent Framework. Import them lazily
from `agent_gantry.integrations.frameworks` (importing `agent_gantry` never
loads these — the framework is only required when you use its provider).

| Framework | Live entry point | Native hook |
|---|---|---|
| LlamaIndex | `LlamaIndexAdapter(gantry).tool_retriever()` / `.function_agent(llm)` | `FunctionAgent(tool_retriever=…)` (`ObjectRetriever`) |
| Pydantic AI | `PydanticAIAdapter(gantry).toolset()` | `AbstractToolset.get_tools()` |
| AutoGen | `AutoGenAdapter(gantry).workbench()` | `autogen_core.tools.Workbench.list_tools()` |
| Google ADK | `GoogleADKAdapter(gantry).before_model_callback()` / `.agent()` | `Agent(before_model_callback=…)` |
| Strands Agents | `StrandsAdapter(gantry).tool_hook()` / `.agent()` | `Agent(hooks=[…])` — `BeforeModelCallEvent` |
| LangGraph | `LangGraphAdapter(gantry).react_agent(model)` | dynamic `model` callable (re-binds tools per turn) |
| Semantic Kernel | `SemanticKernelAdapter(gantry).function_provider(kernel)` / `.refresh(kernel, query)` | per-invocation plugin refresh |
| OpenAI Agents SDK | `OpenAIAgentsAdapter(gantry).run(agent, run_input)` / `.session(agent)` / `.run_hooks(agent)` | `RunHooks.on_llm_start` + per-run refresh |

```python
from agent_gantry.llamaindex import LlamaIndexAdapter
agent = LlamaIndexAdapter(gantry).function_agent(llm)   # LlamaIndex agent that re-selects tools each step
```

Frameworks whose tool list is **fixed at agent construction** (CrewAI, Agno,
Haystack, Smolagents) can't re-advertise tools mid-run; their "live" wrappers —
obtained via `CrewAIAdapter(gantry).agent_builder(...)`,
`AgnoAdapter(gantry).agent_builder(...)`, `SmolagentsAdapter(gantry).agent_builder(...)`,
and `HaystackAdapter(gantry).tool_invoker_builder(...)` (or `HaystackAdapter(gantry).live_tools(query)`
for tools alone) — re-select and rebuild the agent on each top-level call —
the deepest those frameworks allow.

## Shared base

`base.py` provides `GantryToolset` (selection) and `ToolSpec` (a
framework-neutral handle with `.name`, `.description`, `.parameters`, plus
`ainvoke`/`invoke`). `invoke()` is safe to call from synchronous framework code
even inside a running event loop — it runs the coroutine on a worker thread and
blocks for the result.
