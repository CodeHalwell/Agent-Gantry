# Native framework adapters

> Part of the `agent-gantry` skill. Core concepts, debugging and common
> pitfalls are in [`SKILL.md`](../SKILL.md); read those first.

For LangChain, LangGraph, LlamaIndex, CrewAI, Pydantic AI, OpenAI Agents SDK, Haystack, Agno, Strands Agents, DSPy, and Google ADK, use the native `<Framework>Adapter` class. Its `.select(query, limit=...)` method selects the top-K relevant tools and returns the framework's **native tool objects**, with every call still routed through `gantry.execute`.

```python
from agent_gantry import AgentGantry
from agent_gantry.langchain import LangChainAdapter   # clean per-framework namespace

gantry = AgentGantry()
# ... register tools, await gantry.sync() ...

tools = await LangChainAdapter(gantry).select("email the quarterly report to finance", limit=3)
# hand `tools` straight to a LangChain agent / ChatModel.bind_tools(tools)
```

Every adapter shares the identical signature `<Framework>Adapter(gantry).select(query, *, limit=3, **select_kwargs)` and the same per-framework namespace (`agent_gantry.<framework>`):

| Framework | Adapter class | Native object | Namespace import |
|---|---|---|---|
| LangChain | `LangChainAdapter` | `StructuredTool` | `from agent_gantry.langchain import LangChainAdapter` |
| LangGraph | `LangGraphAdapter` | LangChain `StructuredTool` | `from agent_gantry.langgraph import LangGraphAdapter` |
| LlamaIndex | `LlamaIndexAdapter` | `FunctionTool` | `from agent_gantry.llamaindex import LlamaIndexAdapter` |
| CrewAI | `CrewAIAdapter` | `crewai.tools.BaseTool` | `from agent_gantry.crewai import CrewAIAdapter` |
| Pydantic AI | `PydanticAIAdapter` | `pydantic_ai.tools.Tool` | `from agent_gantry.pydantic_ai import PydanticAIAdapter` |
| OpenAI Agents SDK | `OpenAIAgentsAdapter` | `agents.FunctionTool` | `from agent_gantry.openai_agents import OpenAIAgentsAdapter` |
| Haystack | `HaystackAdapter` | `haystack.tools.Tool` | `from agent_gantry.haystack import HaystackAdapter` |
| Agno | `AgnoAdapter` | `agno.tools.function.Function` | `from agent_gantry.agno import AgnoAdapter` |
| Strands Agents | `StrandsAdapter` | `strands.tools.decorator.DecoratedFunctionTool` | `from agent_gantry.strands import StrandsAdapter` |
| DSPy | `DSPyAdapter` | `dspy.Tool` | `from agent_gantry.dspy import DSPyAdapter` |
| Google ADK | `GoogleADKAdapter` | `google.adk.tools.FunctionTool` | `from agent_gantry.google_adk import GoogleADKAdapter` |

Adapters with secondary methods expose them on the same class — there is nothing extra to import.

Need one conversion at a time (you already hold the selected specs)? Use `<Adapter>.convert(spec)` (a staticmethod) with specs from `GantryToolset(gantry).select(query, limit=...)`.

```python
from agent_gantry.integrations.frameworks import GantryToolset
from agent_gantry.crewai import CrewAIAdapter

specs = await GantryToolset(gantry).select("research and writing", limit=4)
crew_tools = [CrewAIAdapter.convert(s) for s in specs]
```

## Deep per-turn "live" providers

The `.select(...)` method is *static* — select once, hand over a fixed list. The **live** adapter methods hook each framework's own per-turn lifecycle so Gantry re-selects tools on **every turn**, matching `GantryContextProvider` depth for Microsoft Agent Framework. Construct the adapter from the per-framework namespace, then call the live method.

```python
from agent_gantry.llamaindex import LlamaIndexAdapter
agent = LlamaIndexAdapter(gantry).function_agent(llm)   # re-selects tools each step
```

| Framework | Live adapter method | Native hook |
|---|---|---|
| LlamaIndex | `LlamaIndexAdapter(gantry).tool_retriever()` / `.function_agent(llm)` | `FunctionAgent(tool_retriever=…)` |
| Pydantic AI | `PydanticAIAdapter(gantry).toolset()` / `.capability()` / `.deferred_toolset()` + `.tool_search()` | `AbstractToolset.get_tools()` + `for_run()`; `AbstractCapability`; `ToolSearch(strategy=…)` |
| Google ADK | `GoogleADKAdapter(gantry).before_model_callback()` / `.agent(model=, name=)` | `Agent(before_model_callback=…)` |
| Strands Agents | `StrandsAdapter(gantry).tool_hook()` / `.agent(...)` | `Agent(hooks=[…])` — `BeforeModelCallEvent` |
| LangGraph | `LangGraphAdapter(gantry).middleware()` (recommended) / `.areact_agent(model)` / `.react_agent(model)` | `create_agent` `AgentMiddleware.wrap_model_call` (narrows `request.tools` per turn) |
| OpenAI Agents SDK | `OpenAIAgentsAdapter(gantry).dynamic_tools()` (recommended) / `.run(agent, run_input)` / `.session(agent)` / `.run_hooks(agent)` | `FunctionTool.is_enabled` per turn; `RunHooks.on_llm_start` + per-run refresh |

The returned live objects keep their classes, importable from the framework's `*_live` module for `isinstance` checks: `GantryToolRetriever` (`llamaindex_live`), `GantryToolset` / `GantryDeferredToolset` (`pydantic_ai_live`, also `agent_gantry.pydantic_ai`), `GantryAgentSession` (`openai_agents_live`) and `GantryStrandsToolHook` (`strands_live`). Google ADK returns a plain `before_model_callback` callable and LangGraph's `react_agent` a compiled agent, so there is no Gantry class to check for; LangGraph's `middleware()` returns a `GantryToolMiddleware` (built lazily, as it subclasses LangChain's `AgentMiddleware`).

Pydantic AI goes further than the other adapters (native retries, approval, `deps`, tool search, a `Capability`) — see [Pydantic AI](pydantic-ai.md).

Frameworks whose tool list is **fixed at agent construction** (CrewAI, Agno, DSPy, and Haystack outside `HaystackAdapter.agent()`) can't re-advertise tools mid-run. Build a self-rebuilding agent with `<Adapter>(gantry).agent_builder(...)` (Haystack: `HaystackAdapter(gantry).tool_invoker_builder(...)`; DSPy: `DSPyAdapter(gantry).agent_builder(signature, ...)`, since `dspy.ReAct` needs a task signature); it re-selects and rebuilds on each top-level call. For a one-shot fresh slice of native tools, call `<Adapter>(gantry).live_tools(query)` (async, not available on DSPy — use `.select(query)` instead).
