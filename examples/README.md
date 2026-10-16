# Agent-Gantry Examples

Run these examples from the repository root after installing the library and
any optional packages listed for the example. Start with:

```bash
pip install -e ".[dev]"
python examples/frameworks/universal_adapters_example.py
```

Framework examples print their tool selection before making a model call. They
use a local hash embedder, so they need no embedding-model download. Install
the chosen SDK to see its native tools; only live model runs need credentials.
Pydantic AI and DSPy also demonstrate full agent loops using local test models.
The Nomic routing demo explicitly needs the `nomic` extra and a model download;
provider-specific demos describe their credential requirements in their headers.

Each subdirectory has its own README with detail and run commands.

## Start here: your agent framework

Most people arrive with a framework already chosen, so start with yours.

```bash
python examples/agent_frameworks/langchain_example.py    # no key needed to see selection
```

| Framework | Example | Installed via |
|---|---|---|
| LangChain | [`agent_frameworks/langchain_example.py`](agent_frameworks/langchain_example.py) | `pip install langchain langchain-openai` |
| LangGraph | [`agent_frameworks/langgraph_example.py`](agent_frameworks/langgraph_example.py) | `pip install langchain langgraph langchain-openai` |
| CrewAI | [`agent_frameworks/crewai_example.py`](agent_frameworks/crewai_example.py) | `pip install crewai` (Python 3.10–3.13) |
| LlamaIndex | [`agent_frameworks/llamaindex_example.py`](agent_frameworks/llamaindex_example.py) | `pip install llama-index-core llama-index-llms-openai` |
| Google ADK | [`agent_frameworks/google_adk_example.py`](agent_frameworks/google_adk_example.py) | `pip install google-adk` |
| Microsoft Agent Framework | [`agent_frameworks/agent_framework_provider_example.py`](agent_frameworks/agent_framework_provider_example.py) | `pip install agent-framework-core agent-framework-openai agent-framework-orchestrations` |
| OpenAI Agents SDK | [`agent_frameworks/openai_agents_example.py`](agent_frameworks/openai_agents_example.py) | `pip install openai-agents` |
| Pydantic AI | [`agent_frameworks/pydantic_ai_example.py`](agent_frameworks/pydantic_ai_example.py) | `pip install pydantic-ai-slim` |
| Haystack | [`agent_frameworks/haystack_example.py`](agent_frameworks/haystack_example.py) | `pip install haystack-ai` |
| Agno | [`agent_frameworks/agno_example.py`](agent_frameworks/agno_example.py) | `pip install agno` |
| Strands | [`agent_frameworks/strands_example.py`](agent_frameworks/strands_example.py) | `pip install 'strands-agents>=1'` |
| DSPy | [`agent_frameworks/dspy_example.py`](agent_frameworks/dspy_example.py) | `pip install dspy` |
| *(none yet)* | [`agent_frameworks/generic_adapters_example.py`](agent_frameworks/generic_adapters_example.py) | — |

Install only the SDK for the framework you use. The `agent-frameworks` extra
contains the first six frameworks; OpenAI Agents SDK, Pydantic AI, Haystack,
Agno, Strands and DSPy are installed separately because some of their dependency
constraints conflict with that combined extra. See
[`agent_frameworks/README.md`](agent_frameworks/README.md).

No adapter for your framework? [`agent_frameworks/generic_adapters_example.py`](agent_frameworks/generic_adapters_example.py) shows the
framework-neutral path, which works anywhere: `GantryToolset(gantry).select(query, limit=1)`
returns `ToolSpec`s carrying `spec.native_name`, `spec.description` and `spec.parameters` — hand those
to your framework's tool type and point its callback at `spec.ainvoke(...)`.

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

Every example in this tree does this, including the bulk catalogues under
`project_demo/tools/` and `tool_vector_db/`, and `agent-gantry lint` reports
any registered tool that does not (#434). Copy from `agent_frameworks/` if you
are starting fresh.

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
- `basics/` — registration patterns, async execution, multi-tool routing,
  plug-and-play imports, and skills (retrieved guidance, never executed).
- `routing/` — semantic routing, custom adapters, health-aware ranking, asymmetric
  embedders, and the Jev selector.
- `execution/` — circuit breakers, batch execution, security policy enforcement.
- `llm_integration/` — end-to-end loops against OpenAI/Anthropic/Google/Groq/Mistral
  using the `@with_semantic_tools` decorator.
- `observability/` — console telemetry and token-savings analysis. The library
  attaches a `NullHandler`, so console output appears only after
  `enable_console_logging()`; the demos call it, and yours must too.
- `protocols/` — MCP and A2A demos. The Python files run offline (nothing is served
  or connected); `claude_desktop_config.json` launches the real server with
  `agent-gantry serve-mcp --module your.module`.
- `project_demo/`, `tool_vector_db/` — fuller applications with persistence.
  The persistent variants build a gitignored LanceDB cache on first run (`project_demo/tools/.tool_cache/`
  and `gantry_tools.lancedb/` in the repo root), so expect the first run to take
  longer when using a downloaded embedder; later runs reuse the stored embeddings.
- `testing_limits/` — retrieval accuracy at 30 and 100 tools; the 30-tool run adds an
  LLM selection step when a key is set.

## Installing

```bash
# Larger example catalogues and the combined framework set
pip install -e ".[dev,example-tools,agent-frameworks,mcp,a2a]"

# Or only what you need
pip install -e ".[agent-frameworks]"   # LangChain, LangGraph, CrewAI, LlamaIndex, ADK, MS AF
pip install -e ".[openai,anthropic]"   # LLM provider examples
pip install -e ".[mcp]"                # MCP / Claude Desktop
pip install -e ".[a2a]"                # Agent-to-Agent protocol
pip install -e ".[nomic]"              # routing/nomic_tool_demo.py, testing_limits/ (~550 MB model)
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
tools = await gantry.framework("langchain").select(query, limit=3)
agent = create_agent(model=llm, tools=tools)
```

Either way every call routes back through `gantry.execute`, so retries,
timeouts, circuit breakers and the security policy still apply.

## Checking the examples

```bash
python -m pytest tests/examples tests/test_examples_agent_frameworks.py -q
```

The checks parse every Python example for Python 3.10 compatibility and run the
framework entry points without credentials in separate processes. SDK-specific
checks skip when their optional package is absent. The real-package framework
matrix tests native construction and tool execution with all twelve SDKs.
