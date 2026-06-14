# Agent Framework Integrations

This directory contains examples of how to integrate **Agent-Gantry** with popular agent frameworks.

## Core Concept

Agent-Gantry acts as a **Semantic Tool Router**. Instead of passing all your tools to an agent (which increases token costs and reduces accuracy), you use Agent-Gantry to retrieve only the most relevant tools for the current user query.

The general pattern is:
1. **Register** all your tools with `AgentGantry`.
2. **Select + convert** in one call with a native adapter — e.g.
   `tools = await for_langchain(gantry, query, limit=3)` — which retrieves the
   relevant tools, wraps them as that framework's native tool objects, and wires
   execution back through `gantry.execute` (retries, timeouts, circuit breakers,
   security policy all apply). No manual factory or name-based branching.
3. **Hand** the returned native tools to your agent.

Native adapters live in `agent_gantry.integrations.frameworks`:
`for_langchain`, `for_langgraph`, `for_llamaindex`, `for_crewai`,
`for_pydantic_ai`, `for_openai_agents`, `for_smolagents`, `for_haystack`,
`for_agno`, `for_autogen`, `for_semantic_kernel`, `for_google_adk` (and a
matching `spec_to_<fw>` for single-tool conversion). See
[`../frameworks/`](../frameworks/) for offline, runnable walkthroughs.

For per-step (multi-turn) re-selection within one run, use `ToolRefresher`
(`agent_gantry.integrations`) — see `../frameworks/multi_turn_refresher_example.py`.

## Examples Included

- `agent_framework_provider_example.py` — **recommended**: AF-native integration via `GantryContextProvider`. The provider plugs into `Agent(context_providers=[...])` and dynamically injects the top-k tools for each `agent.run(...)` — no pre-baking, coexists with `SkillsProvider`, and flows through every workflow builder unchanged. Includes a `skills=True` flag for always-on skill-bound tools and an `always_include` pin list.
- `agent_framework_example.py`: Microsoft Agent Framework **1.3.0** integration via `GantryToolBridge` (still useful when the tool set is *static* and known at agent-construction time). Demonstrates three construction patterns:
  1. `bridge.build_agent(client, query, ...)` — convenience one-liner via `client.as_agent()`.
  2. `bridge.as_agent(client, query, ...)` — direct `Agent(client, ...)` construction; preferred for `WorkflowBuilder` because the result is a first-class `Agent`.
  3. `bridge.build_workflow([specs], edges=[...])` — fan-out/handoff routing where each participating agent receives its own semantically-selected Gantry tool subset.
- `agent_framework_orchestration_example.py`: Multi-agent orchestration patterns (Sequential / Concurrent / Handoff) on AF **1.3.0**, each participating agent receiving its own semantically selected Gantry tool slice. Since AF 1.2.2, the terminal output of sequential/concurrent workflows is an `AgentResponse` (str-coercible) rather than a bare string — extract text with `str(result)` or `result.text`.
- `google_adk_example.py`: Google Agent Development Kit (ADK) integration using `Agent` + `Runner`.
- `langchain_example.py`: Using Gantry with LangChain 0.3+ `create_agent`.
- `langgraph_example.py`: Integrating Gantry into a LangGraph workflow.
- `crewai_example.py`: Using Gantry to provide dynamic tools to a CrewAI Agent (using `kickoff_async`).
- `autogen_example.py`: Registering Gantry tools for an AutoGen (AG2) `AssistantAgent`.
- `llamaindex_example.py`: Using Gantry with LlamaIndex's new `AgentWorkflow` based `ReActAgent`.
- `semantic_kernel_example.py`: Registering Gantry tools as Semantic Kernel plugins with auto-function calling.

## Installation Requirements

You can install all framework integrations at once or install individual frameworks as needed.

```bash
# Option 1: Install all framework dependencies with pip
pip install "agent-gantry[agent-frameworks]"

# Option 2: Install individual frameworks with pip
pip install agent-gantry langchain langchain-openai langgraph
pip install agent-gantry crewai
pip install agent-gantry autogen-agentchat autogen-ext[openai]
# etc.

# Option 3: If using uv for project dependency management
uv add langchain langchain-openai langgraph crewai autogen-agentchat autogen-ext[openai] \
	llama-index-core llama-index-llms-openai semantic-kernel agent-framework google-adk
```

**Note on Python Version:** These examples are verified on **Python 3.13**, but should work on any supported Agent-Gantry version (**Python 3.10+**).

## Environment Variables

Ensure you have your API keys set in a `.env` file:

```env
OPENAI_API_KEY=sk-...
# Other keys as needed for specific providers
```
