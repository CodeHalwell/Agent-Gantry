# Agent Framework Integrations

This directory contains examples of how to integrate **Agent-Gantry** with popular agent frameworks.

## Core Concept

Agent-Gantry acts as a **Semantic Tool Router**. Instead of passing all your tools to an agent (which increases token costs and reduces accuracy), you use Agent-Gantry to retrieve only the most relevant tools for the current user query.

The general pattern is:
1. **Register** all your tools with `AgentGantry`.
2. **Select + convert** in one call with a native adapter — e.g.
   `tools = await gantry.framework("langchain").select(query, limit=3)` — which retrieves
   the relevant tools, wraps them as that framework's native tool objects, and wires
   execution back through `gantry.execute` (retries, timeouts, circuit breakers,
   security policy all apply). No manual factory or name-based branching.
3. **Hand** the returned native tools to your agent.

## Two things that decide whether this works well

**Give every tool `examples=[...]`.** It is the single highest-value field on a
tool definition — the text the router embeds and a selector reads. On our own
benchmark it took the default embedder from 1/5 to 5/5 correct, further than
switching to a larger embedding model did:

```python
@gantry.register(
    tags=["weather"],
    examples=["what's the weather in London", "is it raining in Leeds"],
)
def get_weather(location: str) -> str:
    """Get the current weather in a given location."""
    ...
```

Write the phrases a user would actually type, not a paraphrase of the
description. Every example in this directory does this.

**Leave `score_threshold` alone unless you have measured it.** It defaults to
`0.0` on every adapter, and it is an *absolute* cosine cutoff rather than a
relative one. Longer queries dilute absolute similarity, so a non-zero value
quietly returns fewer tools — or none at all — with no error to tell you.
Use `limit` to control how many tools reach the model.

## Running these without API keys

After installing the chosen SDK, the scripts run without model credentials. Registration,
sync, retrieval and conversion need no credentials, so each file does that work
first and prints what was selected; only a final live model call is gated, and
it names the variable it wants. That makes these useful for seeing what Gantry
picks before you spend anything. The scripts explicitly use `SimpleEmbedder`
for an offline walkthrough; configure a semantic embedder for production routing.
The TUI additionally needs `textual`.

All twelve frameworks support `gantry.framework(name).select(query)` and
`gantry.framework(name).convert(spec)`. `supported_frameworks()` lists the names.
The selected adapter retains its framework-specific live methods and editor completion.

Direct adapter imports also remain available — `from agent_gantry.<framework> import <Framework>Adapter`
(`agent_gantry.langchain` → `LangChainAdapter`, `agent_gantry.crewai` → `CrewAIAdapter`,
`agent_gantry.llamaindex` → `LlamaIndexAdapter`, `agent_gantry.pydantic_ai` → `PydanticAIAdapter`,
`agent_gantry.openai_agents` → `OpenAIAgentsAdapter`,
`agent_gantry.haystack` → `HaystackAdapter`, `agent_gantry.agno` → `AgnoAdapter`,
`agent_gantry.google_adk` → `GoogleADKAdapter`, `agent_gantry.langgraph` → `LangGraphAdapter`,
`agent_gantry.strands` → `StrandsAdapter`, `agent_gantry.dspy` → `DSPyAdapter`,
`agent_gantry.agent_framework` → `AgentFrameworkAdapter`) — each exposing
`.select(query, limit=...)`, the `.convert(spec)` staticmethod, and that framework's deep
per-turn live methods. See [`../frameworks/`](../frameworks/) for offline, runnable walkthroughs.

For per-step (multi-turn) re-selection within one run, use `ToolRefresher`
(`agent_gantry.integrations`) — see `../frameworks/multi_turn_refresher_example.py`.

## Examples Included

- `agent_framework_provider_example.py` — **recommended**: AF-native integration via `GantryContextProvider`. The provider plugs into `Agent(context_providers=[...])` and dynamically injects the top-k tools for each `agent.run(...)` — no pre-baking, coexists with `SkillsProvider`, and flows through every workflow builder unchanged. Includes a `skills=True` flag for always-on skill-bound tools and an `always_include` pin list.
- `agent_framework_trace_events_example.py` — **per-turn routing with built-in observability**. Shows the multi-step pattern (`query_strategy="per_turn"`) where the tool surface re-selects every chat round, wired with one `provider.attach_to(agent, trace=True)` call. Demonstrates the library's batteries-included observability: the built-in console **trace** middleware (no hand-rolled `@function_middleware`), the framework-agnostic `gantry.on_tool_call(...)` **event hook**, per-round `provider.selections` history, and `agent_gantry.render_result(...)`. Uses stdlib-only tools, so it runs without extra API keys (beyond your chat client).
- `agent_framework_tui_demo.py` — **Textual TUI dashboard** for screenshots and demos. Three-panel view: full tool registry, per-round semantic selection (via `GantryContextProvider.dry_run_retrieve`), and execution log (`gantry.execute`). Press **D** for a scripted four-step demo (no API key); press **L** for a live AF agent when `OPENAI_API_KEY` is set. From this repo: `uv sync --extra agent-frameworks && uv pip install textual`, then `uv run examples/agent_frameworks/agent_framework_tui_demo.py`. Optional: `--embedder nomic` for stronger retrieval (requires `agent-gantry[nomic]`; embeddings sync before the TUI starts).
- `agent_framework_example.py`: Microsoft Agent Framework (1.5.0+) integration via `GantryToolBridge` (still useful when the tool set is *static* and known at agent-construction time). Demonstrates three construction patterns:
  1. `bridge.build_agent(client, query, ...)` — convenience one-liner that constructs an AF `Agent`.
  2. `bridge.as_agent(client, query, ...)` — direct `Agent(client, ...)` construction; preferred for `WorkflowBuilder` because the result is a first-class `Agent`.
  3. `bridge.build_workflow([specs], edges=[...])` — fan-out/handoff routing where each participating agent receives its own semantically-selected Gantry tool subset.
- `agent_framework_orchestration_example.py`: Multi-agent orchestration patterns (Sequential / Concurrent / Handoff) on AF 1.5.0+, each participating agent receiving its own semantically selected Gantry tool slice. Since AF 1.2.2, the terminal output of sequential/concurrent workflows is an `AgentResponse` (str-coercible) rather than a bare string — extract text with `str(result)` or `result.text`.
- `agent_framework_harness_example.py`: AF background-agent harness integration (see the file's docstring for the AF version gates it exercises).
- `google_adk_example.py`: runs offline with a scripted ADK model. Per-turn selection through `before_model_callback` (the email tool appears after the CRM lookup), failures returned as `{"error": ...}` responses, a confirmation-gated tool answered with `adk_request_confirmation`, and a tool reading the session state through `current_tool_context()`.
- `langchain_example.py`: `LangChainAdapter.select` feeding `langchain.agents.create_agent` (LangChain 1.x; the old `langgraph.prebuilt.create_react_agent` is deprecated and removed in LangGraph 2.0).
- `langgraph_example.py`: runs offline with a scripted chat model. `LangGraphAdapter.middleware()` re-selects tools every model turn (the email tool appears after the CRM lookup returns), failures come back as error `ToolMessage`s, a confirmation-gated tool pauses with `interrupt()` and runs once on `Command(resume=True)`, and a tool reads the graph's `context=` through `current_tool_runtime()`.
- `crewai_example.py`: runs offline with a scripted CrewAI LLM. A failing tool comes back as a CrewAI `ToolFailure` (recorded on `tool_failures`), a confirmation-gated tool goes through the adapter's approver, and a tool reads the agent and task through `current_crew_context()`.
- `llamaindex_example.py`: runs offline with a scripted function-calling LLM. A `FunctionAgent` re-selects tools each step, a failing tool comes back as `ToolOutput(is_error=True)`, a confirmation-gated tool waits for a `HumanResponseEvent`, and a tool reads the workflow state through `current_workflow_context()`.
- `generic_adapters_example.py`: The framework-neutral `GantryToolset.select` → `ToolSpec` path (`name`, `description`, JSON-Schema `parameters`, `ainvoke`) for hand-wiring a framework that has no dedicated adapter yet. Needs no framework and no key.
- `dspy_example.py`: runs offline with DSPy's scripted `DummyLM` (DSPy 3.5+). `DSPyAdapter.select` feeds a real `dspy.ReAct`, a failing tool comes back as an error tool result (`is_error=True`), a confirmation-gated tool goes through the adapter's approver, a tool reads the running `dspy.ReAct` through `current_dspy_module()`, and `agent_builder(signature)` builds a fresh ReAct per call.
- `strands_example.py`: runs offline with a scripted Strands model. `adapter.agent()` re-selects tools before every model call (`BeforeModelCallEvent`), a failing tool returns an error `ToolResult`, a confirmation-gated tool raises a Strands interrupt that resumes with "yes", and a tool reads the `agent(...)` keyword arguments through `current_tool_context()`.
- `pydantic_ai_example.py`: `PydanticAIAdapter` — static `select` → native `pydantic_ai.Tool`; `adapter.capability()` re-selecting tools mid-run from a tool result; a confirmation-gated tool pausing with `DeferredToolRequests` and running once on approval; `deps` read via `current_run_context()`; and `capability(mode="search")` with Gantry-backed `ToolSearch`. Runs fully offline with `FunctionModel` — no API key needed anywhere in the file.
- `openai_agents_example.py`: `OpenAIAgentsAdapter` — static `select` → native `agents.FunctionTool`; `adapter.dynamic_tools()` re-selecting per turn through `FunctionTool.is_enabled`, surfacing the email tool after a lookup's output; a failing tool returning `Error: ...` to the model; a confirmation-gated tool pausing with `result.interruptions` and running once on approval; and `Runner.run(context=...)` read via `current_run_context()`. Runs fully offline with the SDK's `agents.testing.ScriptedModel`.
- `haystack_example.py`: `HaystackAdapter` — static `select` → native `haystack.tools.Tool`, plus the per-run tier via `adapter.live_tools(...)` and `adapter.tool_invoker_builder(...)` (rebuilds a fresh `ToolInvoker` per call on haystack 2.x; on haystack >=3.0, which removed `ToolInvoker`, it builds a per-run `Agent` when given `chat_generator=...`). Tool selection runs with no API key; a live chat + tool-invocation round is gated behind `OPENAI_API_KEY`.
- `agno_example.py`: `AgnoAdapter` — static `select` → native `agno.tools.function.Function`, plus the per-run tier via `adapter.agent_builder(...)` (Agno fixes an agent's tools at construction, so this rebuilds a fresh `agno.agent.Agent` per call with freshly selected tools). Tool selection runs with no API key; building the model + running the agent is gated behind `OPENAI_API_KEY`.

## Installation Requirements

Install the individual SDK for the framework you use. The combined extra contains
LangChain, LangGraph, CrewAI, LlamaIndex, Google ADK and Microsoft Agent Framework.

```bash
# Combined extra for those six frameworks (uv project)
uv add "agent-gantry[agent-frameworks]"

# Option 2: Install individual frameworks
uv add agent-gantry langchain langchain-openai langgraph
uv add agent-gantry crewai
# etc.

# Option 3: Developing in the agent-gantry repo
uv sync --extra agent-frameworks
```

**pydantic_ai / openai_agents / haystack / agno / strands / dspy are
intentionally NOT part of `agent-frameworks`** — they are kept out of the
main test matrix (see the note below the `agent-frameworks` extra in
`pyproject.toml`). Each has its own extra with a supported version floor
(install `[openai-agents]` in its own environment: it needs `openai>=3`,
CrewAI pins `openai<3`):

```bash
pip install 'agent-gantry[pydantic-ai]'
pip install 'agent-gantry[openai-agents]'
pip install 'agent-gantry[haystack]'
pip install 'agent-gantry[agno]'
pip install 'agent-gantry[strands]'
pip install 'agent-gantry[dspy]'
```

**Note on Python version:** CI exercises the adapter suite on **Python 3.10, 3.11, 3.12, 3.13 and 3.14** (ubuntu, plus a macOS 3.12 cell). Agent-Gantry supports Python 3.10–3.14.

CrewAI's SDK currently requires Python **3.10–3.13**. Use one of those versions
for `crewai_example.py`; the combined framework extra omits CrewAI on 3.14.

## Environment variables

Only needed for the final live-model step of each example; everything before it
runs without them.

```env
OPENAI_API_KEY=sk-...          # most examples
AZURE_OPENAI_ENDPOINT=...      # agent_framework_* examples accept these
AZURE_OPENAI_API_KEY=...       # in place of OPENAI_API_KEY
GOOGLE_API_KEY=...             # to run google_adk_example.py on Gemini (or GEMINI_API_KEY)
AWS_ACCESS_KEY_ID=...          # strands_example.py on Bedrock (it runs offline otherwise),
AWS_SECRET_ACCESS_KEY=...      # so the credential must be entitled to the model
```

`pydantic_ai_example.py` and `dspy_example.py` need no key at all: they run a
full agent loop against a scripted `FunctionModel` and `DummyLM` respectively.

`agent_framework_harness_example.py` needs `agent-framework>=1.7.0` for its
experimental `create_harness_agent`. From agent-gantry 0.17.0 the
`agent-frameworks` extra requires AF >= 1.19.0, so it works out of the box;
only an older environment needs `pip install 'agent-framework>=1.7.0'`
separately.
