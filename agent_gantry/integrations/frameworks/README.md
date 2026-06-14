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

## Multi-turn re-selection (any framework)

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

## Shared base

`base.py` provides `GantryToolset` (selection) and `ToolSpec` (a
framework-neutral handle with `.name`, `.description`, `.parameters`, plus
`ainvoke`/`invoke`). `invoke()` is safe to call from synchronous framework code
even inside a running event loop — it runs the coroutine on a worker thread and
blocks for the result.
