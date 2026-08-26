# Agent Framework Integrations

This directory contains examples of how to integrate **Agent-Gantry** with popular agent frameworks.

## Core Concept

Agent-Gantry acts as a **Semantic Tool Router**. Instead of passing all your tools to an agent (which increases token costs and reduces accuracy), you use Agent-Gantry to retrieve only the most relevant tools for the current user query.

The general pattern is:
1. **Register** all your tools with `AgentGantry`.
2. **Select + convert** in one call with a native adapter — e.g.
   `tools = await LangChainAdapter(gantry).select(query, limit=3)` — which retrieves
   the relevant tools, wraps them as that framework's native tool objects, and wires
   execution back through `gantry.execute` (retries, timeouts, circuit breakers,
   security policy all apply). No manual factory or name-based branching.
3. **Hand** the returned native tools to your agent.

Each framework has a clean namespace — `from agent_gantry.<framework> import <Framework>Adapter`
(`agent_gantry.langchain` → `LangChainAdapter`, `agent_gantry.crewai` → `CrewAIAdapter`,
`agent_gantry.llamaindex` → `LlamaIndexAdapter`, `agent_gantry.pydantic_ai` → `PydanticAIAdapter`,
`agent_gantry.openai_agents` → `OpenAIAgentsAdapter`, `agent_gantry.smolagents` → `SmolagentsAdapter`,
`agent_gantry.haystack` → `HaystackAdapter`, `agent_gantry.agno` → `AgnoAdapter`,
`agent_gantry.autogen` → `AutoGenAdapter`, `agent_gantry.semantic_kernel` → `SemanticKernelAdapter`,
`agent_gantry.google_adk` → `GoogleADKAdapter`, `agent_gantry.langgraph` → `LangGraphAdapter`,
`agent_gantry.strands` → `StrandsAdapter`, `agent_gantry.dspy` → `DSPyAdapter`,
`agent_gantry.agent_framework` → `AgentFrameworkAdapter`) — each exposing
`.select(query, limit=...)`, the `.convert(spec)` staticmethod, and that framework's deep
per-turn live methods. See [`../frameworks/`](../frameworks/) for offline, runnable walkthroughs.

For per-step (multi-turn) re-selection within one run, use `ToolRefresher`
(`agent_gantry.integrations`) — see `../frameworks/multi_turn_refresher_example.py`.

## Examples Included

- `agent_framework_provider_example.py` — **recommended**: AF-native integration via `GantryContextProvider`. The provider plugs into `Agent(context_providers=[...])` and dynamically injects the top-k tools for each `agent.run(...)` — no pre-baking, coexists with `SkillsProvider`, and flows through every workflow builder unchanged. Includes a `skills=True` flag for always-on skill-bound tools and an `always_include` pin list.
- `agent_framework_trace_events_example.py` — **per-call routing with built-in observability**. Shows the multi-step pattern (`query_strategy="per_call"`) where the tool surface re-selects every chat round, wired with one `provider.attach_to(agent, trace=True)` call. Demonstrates the library's batteries-included observability: the built-in console **trace** middleware (no hand-rolled `@function_middleware`), the framework-agnostic `gantry.on_tool_call(...)` **event hook**, per-round `provider.selections` history, and `agent_gantry.render_result(...)`. Uses stdlib-only tools, so it runs without extra API keys (beyond your chat client).
- `agent_framework_tui_demo.py` — **Textual TUI dashboard** for screenshots and demos. Three-panel view: full tool registry, per-round semantic selection (via `GantryContextProvider.dry_run_retrieve`), and execution log (`gantry.execute`). Press **D** for a scripted two-step demo (no API key); press **L** for a live AF agent when `OPENAI_API_KEY` is set. From this repo: `uv sync --extra agent-frameworks && uv pip install textual`, then `uv run examples/agent_frameworks/agent_framework_tui_demo.py`. Optional: `--embedder nomic` for stronger retrieval (requires `agent-gantry[nomic]`; embeddings sync before the TUI starts).
- `agent_framework_example.py`: Microsoft Agent Framework **1.5.0** integration via `GantryToolBridge` (still useful when the tool set is *static* and known at agent-construction time). Demonstrates three construction patterns:
  1. `bridge.build_agent(client, query, ...)` — convenience one-liner via `client.as_agent()`.
  2. `bridge.as_agent(client, query, ...)` — direct `Agent(client, ...)` construction; preferred for `WorkflowBuilder` because the result is a first-class `Agent`.
  3. `bridge.build_workflow([specs], edges=[...])` — fan-out/handoff routing where each participating agent receives its own semantically-selected Gantry tool subset.
- `agent_framework_orchestration_example.py`: Multi-agent orchestration patterns (Sequential / Concurrent / Handoff) on AF **1.5.0**, each participating agent receiving its own semantically selected Gantry tool slice. Since AF 1.2.2, the terminal output of sequential/concurrent workflows is an `AgentResponse` (str-coercible) rather than a bare string — extract text with `str(result)` or `result.text`.
- `agent_framework_harness_example.py`: AF background-agent harness integration (see the file's docstring for the AF version gates it exercises).
- `google_adk_example.py`: Google Agent Development Kit (ADK) integration using `Agent` + `Runner`.
- `langchain_example.py`: `LangChainAdapter.select` feeding `langchain.agents.create_agent` (LangChain 1.x; the old `langgraph.prebuilt.create_react_agent` is deprecated and removed in LangGraph 2.0).
- `langgraph_example.py`: Integrating Gantry into a LangGraph workflow.
- `crewai_example.py`: Using Gantry to provide dynamic tools to a CrewAI Agent (using `kickoff_async`).
- `autogen_example.py`: Registering Gantry tools for an AutoGen (AG2) `AssistantAgent`.
- `llamaindex_example.py`: Using Gantry with LlamaIndex's new `AgentWorkflow` based `ReActAgent`.
- `semantic_kernel_example.py`: `SemanticKernelAdapter.refresh(kernel, query)` — Gantry selects the slice and (re)builds the kernel's `gantry` plugin in one call, with auto function calling; `adapter.function_provider(kernel)` is the per-turn variant for chat loops.
- `generic_adapters_example.py`: The framework-neutral `GantryToolset` + `spec.to_*` conversion path, for wiring a framework that has no dedicated adapter yet.
- `dspy_example.py`: `DSPyAdapter` — static `select` → native `dspy.Tool` (schema passed through DSPy's own `convert_input_schema_to_tool_args` bridge), plus the per-call tier via `adapter.agent_builder(signature)` (rebuilds a fresh `dspy.ReAct` per call).
- `strands_example.py`: `StrandsAdapter` — static `select` → native `DecoratedFunctionTool`, plus the genuine per-turn tier via `adapter.tool_hook()` / `adapter.agent()` (Strands' `BeforeModelCallEvent` fires before every model call, so the registry swap lands on the very call about to happen).
- `pydantic_ai_example.py`: `PydanticAIAdapter` — static `select` → native `pydantic_ai.tools.Tool`, plus the deep per-turn tier via `adapter.toolset(...)` (a live `AbstractToolset` that re-selects on every run/step from the `RunContext`). Runs fully offline with `pydantic_ai.models.test.TestModel` — no API key needed anywhere in the file.
- `openai_agents_example.py`: `OpenAIAgentsAdapter` — static `select` → native `agents.FunctionTool`, plus the deep per-turn tier via `adapter.session(agent, ...)` (re-selects and applies `agent.tools` before each `session.run(...)`, installing `adapter.run_hooks(agent)` for intra-run dynamism; `adapter.refresh(agent, ...)` is the underlying one-shot primitive). Tool selection and re-selection run with no API key; the actual `agents.Runner.run(...)` turn is gated behind `OPENAI_API_KEY`.
- `smolagents_example.py`: `SmolagentsAdapter` — static `select` → native `smolagents.Tool`, plus the per-call tier via `adapter.agent_builder(...)` (smolagents fixes an agent's tools at construction, so this rebuilds a fresh agent per call with freshly selected tools). Tool selection runs with no API key; building the model + running the agent is gated behind `OPENAI_API_KEY`.
- `haystack_example.py`: `HaystackAdapter` — static `select` → native `haystack.tools.Tool`, plus the per-call tier via `adapter.live_tools(...)` and `adapter.tool_invoker_builder(...)` (rebuilds a fresh `ToolInvoker` per call on haystack 2.x; on haystack >=3.0, which removed `ToolInvoker`, it builds a per-call `Agent` when given `chat_generator=...`). Tool selection runs with no API key; a live chat + tool-invocation round is gated behind `OPENAI_API_KEY`.
- `agno_example.py`: `AgnoAdapter` — static `select` → native `agno.tools.function.Function`, plus the per-call tier via `adapter.agent_builder(...)` (Agno fixes an agent's tools at construction, so this rebuilds a fresh `agno.agent.Agent` per call with freshly selected tools). Tool selection runs with no API key; building the model + running the agent is gated behind `OPENAI_API_KEY`.

## Installation Requirements

You can install all framework integrations at once or install individual frameworks as needed.

```bash
# Option 1: Install all framework dependencies (uv project)
uv add "agent-gantry[agent-frameworks]"

# Option 2: Install individual frameworks
uv add agent-gantry langchain langchain-openai langgraph
uv add agent-gantry crewai
uv add agent-gantry autogen-agentchat "autogen-ext[openai]"
# etc.

# Option 3: Developing in the agent-gantry repo
uv sync --extra agent-frameworks
```

**pydantic_ai / openai_agents / smolagents / haystack / agno / strands / dspy are
intentionally NOT part of `agent-frameworks`** (or any other project extra) —
several of them can't co-resolve with the combined extra's pinned dependencies
(see the note below the `agent-frameworks` extra in `pyproject.toml` for the
specifics). Install them standalone alongside `agent-gantry`:

```bash
pip install agent-gantry pydantic-ai-slim
pip install agent-gantry openai-agents
pip install agent-gantry smolagents
pip install agent-gantry haystack-ai
pip install agent-gantry agno
pip install agent-gantry strands-agents
pip install agent-gantry dspy
```

For `AutoGenAdapter.register(...)` (the classic AG2 caller/executor API), the
`autogen` module comes from `pip install "ag2[openai]<1"` — AG2 1.x renamed its
import to `ag2` with a new agent API, and `pyautogen` >= 0.10 is a Microsoft
autogen-agentchat shim; neither provides `autogen.register_function`. The
Microsoft AutoGen path (`AutoGenAdapter.workbench()`) uses `autogen-core` /
`autogen-agentchat` from the `agent-frameworks` extra instead.

**Note on Python Version:** These examples are verified on **Python 3.13**, but should work on any supported Agent-Gantry version (**Python 3.10+**).

## Environment Variables

Ensure you have your API keys set in a `.env` file:

```env
OPENAI_API_KEY=sk-...
# Other keys as needed for specific providers
```
