# Agent-Gantry Examples

Hands-on examples. **Every one of them runs from a clean checkout with no API
keys set** — the Gantry half (registration, sync, retrieval, conversion) needs
no credentials, so you can see what the library does before spending anything.
Where an example finishes by calling a real model, that last step is gated and
tells you which key it wants.

Each subdirectory has its own README with detail and run commands.

## Start here: your agent framework

Most people arrive with a framework already chosen, so start with yours.

```bash
python examples/agent_frameworks/langchain_example.py    # no key needed to see selection
```

| Framework | Example | Installed via |
|---|---|---|
| LangChain | [`agent_frameworks/langchain_example.py`](agent_frameworks/langchain_example.py) | `agent-gantry[agent-frameworks]` |
| LangGraph | [`agent_frameworks/langgraph_example.py`](agent_frameworks/langgraph_example.py) | `agent-gantry[agent-frameworks]` |
| CrewAI | [`agent_frameworks/crewai_example.py`](agent_frameworks/crewai_example.py) | `agent-gantry[agent-frameworks]` |
| LlamaIndex | [`agent_frameworks/llamaindex_example.py`](agent_frameworks/llamaindex_example.py) | `agent-gantry[agent-frameworks]` |
| Google ADK | [`agent_frameworks/google_adk_example.py`](agent_frameworks/google_adk_example.py) | `agent-gantry[agent-frameworks]` |
| Microsoft Agent Framework | [`agent_frameworks/agent_framework_provider_example.py`](agent_frameworks/agent_framework_provider_example.py) | `agent-gantry[agent-frameworks]` |
| OpenAI Agents SDK | [`agent_frameworks/openai_agents_example.py`](agent_frameworks/openai_agents_example.py) | `pip install openai-agents` |
| Pydantic AI | [`agent_frameworks/pydantic_ai_example.py`](agent_frameworks/pydantic_ai_example.py) | `pip install pydantic-ai-slim` |
| Haystack | [`agent_frameworks/haystack_example.py`](agent_frameworks/haystack_example.py) | `pip install haystack-ai` |
| Agno | [`agent_frameworks/agno_example.py`](agent_frameworks/agno_example.py) | `pip install agno` |
| Strands | [`agent_frameworks/strands_example.py`](agent_frameworks/strands_example.py) | `pip install strands-agents` |
| DSPy | [`agent_frameworks/dspy_example.py`](agent_frameworks/dspy_example.py) | `pip install dspy` |
| *(none yet)* | [`agent_frameworks/generic_adapters_example.py`](agent_frameworks/generic_adapters_example.py) | — |

The six rows marked `pip install ...` — OpenAI Agents SDK, Pydantic AI,
Haystack, Agno, Strands and DSPy — are **deliberately not** part of any project
extra: they cannot co-resolve with the combined `agent-frameworks` set. Install
them standalone alongside `agent-gantry`. See
[`agent_frameworks/README.md`](agent_frameworks/README.md).

No adapter for your framework? [`agent_frameworks/generic_adapters_example.py`](agent_frameworks/generic_adapters_example.py) shows the
framework-neutral `GantryToolset` + `spec.to_*` path, which works anywhere.

## Not using a framework?

```bash
python examples/fast_track_demo.py
```

Upgrades a vanilla OpenAI call to semantic tools in about ten lines, with a
before-and-after comparison.

## The one thing worth knowing before you copy anything

Give every tool `examples=[...]`:

```python
@gantry.register(
    tags=["weather"],
    examples=["what's the weather in London", "is it raining in Leeds"],
)
def get_weather(location: str) -> str:
    """Get the current weather in a given location."""
```

That field is the text the router embeds and the text a selector reads. On our
own benchmark it moved the default embedder from 1/5 to 5/5 correct — a bigger
improvement than switching to a larger embedding model. Write the phrases a
user would actually type, not a restatement of the description.

Every example under `agent_frameworks/` does this, so copying one gets you the
good pattern. Much of the rest of the tree predates the measurement and still
registers tools without `examples`; those are being brought in line (see
issue #434). Copy from `agent_frameworks/` if you are starting fresh.

The matching trap: leave `score_threshold` alone unless you have measured it.
It is an **absolute** cosine cutoff, and longer queries dilute absolute
similarity, so a non-zero value silently returns fewer tools — or none —
with no error.

## Directory map

- `agent_frameworks/` — **per-framework integrations.** One file per framework,
  each showing the static tier (select once) and, where the framework supports
  it, the dynamic tier (re-select every turn). Start here.
- `frameworks/` — **framework-neutral plumbing**, despite the similar name: the
  universal `GantryToolset`/`ToolSpec` core, a cross-framework verification
  harness, the multi-turn `ToolRefresher`, and importing existing framework
  tools *into* Gantry. All offline.
- `fast_track_demo.py` — vanilla OpenAI to semantic tools in ten lines.
- `basics/` — registration, async execution, multi-tool routing, plug-and-play imports.
- `routing/` — semantic routing, custom adapters, health-aware ranking, asymmetric
  embedders, and the Jev selector.
- `execution/` — circuit breakers, batch execution, security policy enforcement.
- `llm_integration/` — end-to-end loops against OpenAI/Anthropic/Google/Groq/Mistral
  using the `@with_semantic_tools` decorator.
- `observability/` — console telemetry and token-savings analysis.
- `protocols/` — MCP and A2A demos, including Claude Desktop config.
- `project_demo/`, `tool_vector_db/` — fuller applications with persistence.
- `testing_limits/` — stress tests for token savings and accuracy at 30 and 100 tools.

## Installing

```bash
# Everything the examples can use
pip install -e ".[example-tools,agent-frameworks,mcp,a2a]"

# Or only what you need
pip install -e ".[agent-frameworks]"   # LangChain, LangGraph, CrewAI, LlamaIndex, ADK, MS AF
pip install -e ".[openai,anthropic]"   # LLM provider examples
pip install -e ".[mcp]"                # MCP / Claude Desktop
pip install -e ".[a2a]"                # Agent-to-Agent protocol
```

## The two integration patterns

**Decorator** — used throughout `llm_integration/`:

```python
@with_semantic_tools(gantry, limit=3)
async def chat(prompt: str, *, tools=None):
    # `tools` is injected: only the relevant ones, already in the provider's dialect
    return await client.chat.completions.create(model="...", tools=tools, ...)
```

**Adapter** — used throughout `agent_frameworks/`, one call for retrieval,
conversion and execution wiring:

```python
tools = await LangChainAdapter(gantry).select(query, limit=3)
agent = create_agent(model=llm, tools=tools)
```

Either way every call routes back through `gantry.execute`, so retries,
timeouts, circuit breakers and the security policy still apply.
