# Native framework tool adapters

Agent-Gantry selects a small, relevant slice of tools from a large registry.
These adapters hand that slice to a concrete agent framework as **native tool
objects** — not just JSON schemas — so the framework can introspect and call
them, while every invocation still routes through `gantry.execute` (retries,
timeouts, circuit breakers, security policy all apply).

## Supported frameworks

| Framework | `for_*` helper | Native object |
|---|---|---|
| LangChain | `for_langchain` | `langchain_core.tools.StructuredTool` |
| LangGraph | `for_langgraph` | LangChain `StructuredTool` (LangGraph consumes these) |
| LlamaIndex | `for_llamaindex` | `llama_index.core.tools.FunctionTool` |
| CrewAI | `for_crewai` | `crewai.tools.BaseTool` subclass |
| Pydantic AI | `for_pydantic_ai` | `pydantic_ai.tools.Tool` |
| OpenAI Agents SDK | `for_openai_agents` | `agents.FunctionTool` |
| Smolagents | `for_smolagents` | `smolagents.Tool` subclass |
| Haystack | `for_haystack` | `haystack.tools.Tool` |
| Agno | `for_agno` | `agno.tools.function.Function` |
| AutoGen / AG2 | `for_autogen`, `register_with_autogen` | callables + `register_function` |
| Semantic Kernel | `for_semantic_kernel`, `gantry_plugin` | `KernelFunction` (`@kernel_function`) |
| Google ADK | `for_google_adk` | `google.adk.tools.FunctionTool` |

All third-party imports are **lazy** — `import agent_gantry` never requires any
of these frameworks. A missing framework raises `ImportError` with a
`pip install …` hint only when you call its adapter.

## Usage

```python
from agent_gantry import AgentGantry
from agent_gantry.integrations.frameworks import for_langchain

gantry = AgentGantry()
# ... register tools, await gantry.sync() ...

# Select the top-3 relevant tools and get them as LangChain StructuredTools:
tools = await for_langchain(gantry, "email the quarterly report to finance", limit=3)
# hand `tools` to a LangChain/LangGraph agent
```

Every `for_<fw>(gantry, query, *, limit=3, **select_kwargs)` accepts the same
selection knobs as `GantryToolset.select` (`score_threshold`, `namespaces`,
`tools_already_used`). Need one conversion at a time? Use `spec_to_<fw>(spec)`
with specs from `GantryToolset(gantry).select(query)`.

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

The `for_<fw>` helpers above are *static*: select once, hand over a fixed tool
list. The **live** providers go deeper — they hook each framework's own
per-turn lifecycle so Gantry re-selects tools on **every turn**, exactly like
`GantryContextProvider` does for Microsoft Agent Framework. Import them lazily
from `agent_gantry.integrations.frameworks` (importing `agent_gantry` never
loads these — the framework is only required when you use its provider).

| Framework | Live entry point | Native hook |
|---|---|---|
| LlamaIndex | `gantry_tool_retriever` / `gantry_function_agent` | `FunctionAgent(tool_retriever=…)` (`ObjectRetriever`) |
| Pydantic AI | `gantry_toolset` | `AbstractToolset.get_tools()` |
| AutoGen | `gantry_workbench` | `autogen_core.tools.Workbench.list_tools()` |
| Google ADK | `gantry_before_model_callback` / `gantry_adk_agent` | `Agent(before_model_callback=…)` |
| LangGraph | `create_gantry_react_agent` | dynamic `model` callable (re-binds tools per turn) |
| Semantic Kernel | `GantryFunctionProvider` / `refresh_kernel_tools` | per-invocation plugin refresh |
| OpenAI Agents SDK | `run_with_gantry` / `GantryAgentSession` / `gantry_run_hooks` | `RunHooks.on_llm_start` + per-run refresh |

```python
from agent_gantry.integrations.frameworks import gantry_function_agent
agent = gantry_function_agent(gantry, llm)   # LlamaIndex agent that re-selects tools each step
```

Frameworks whose tool list is **fixed at agent construction** (CrewAI, Agno,
Haystack, Smolagents) can't re-advertise tools mid-run; their "live" wrappers
(`GantryLiveCrewAgent`, `GantryLiveAgnoAgent`, `gantry_haystack_tools`,
`GantryLiveSmolAgent`) re-select and rebuild the agent on each top-level call —
the deepest those frameworks allow.

## Shared base

`base.py` provides `GantryToolset` (selection) and `ToolSpec` (a
framework-neutral handle with `.name`, `.description`, `.parameters`, plus
`ainvoke`/`invoke`). `invoke()` is safe to call from synchronous framework code
even inside a running event loop — it runs the coroutine on a worker thread and
blocks for the result.
