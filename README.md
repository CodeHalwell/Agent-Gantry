# dotfetch

**Universal Tool Orchestration Platform for LLM-Based Agent Systems**

*Context is precious. Execution is sacred. Trust is earned.*

dotfetch is a Python library (**v0.27.0**) for building agents that can discover, select, and execute the right tools without flooding every prompt with every schema your organization owns. It combines semantic retrieval, provider schema conversion, secure execution, framework bridges, MCP/A2A interoperability, persistence adapters, and observability into one tool orchestration layer.

> **Renamed in 0.27.0.** dotfetch was previously published as `agent-gantry`. Version 0.27.0 renamed the import from `agent_gantry` to `dotfetch` and the class from `AgentGantry` to `DotFetch`. This is a clean break: the old imports no longer work.

## Documentation

The project documentation is now an **Astro + React + TypeScript** site with an implementation journey, interactive tool lifecycle walkthrough, integration matrix, and production operations guidance.

```bash
npm install
npm run dev      # local docs server
npm run build    # type-check and static build
npm run preview  # verify the generated site styling
```

Start with the rich docs in [`src/pages/index.astro`](src/pages/index.astro). The published site lives at [codehalwell.github.io/Agent-Gantry](https://codehalwell.github.io/Agent-Gantry).

## Install

```bash
uv add dotfetch
# or
pip install dotfetch
```

Useful extras:

```bash
uv add "dotfetch[openai]"
uv add "dotfetch[anthropic]"
uv add "dotfetch[google-genai]"
uv add "dotfetch[lancedb,nomic]"
uv add "dotfetch[mcp,a2a]"
uv add "dotfetch[agent-frameworks]"
uv add "dotfetch[pydantic-ai]"
uv add "dotfetch[openai-agents]"   # own environment: needs openai>=3, CrewAI pins <3
uv add "dotfetch[langgraph]"       # LangGraph + LangChain 1.x ([langchain] for LangChain alone)
uv add "dotfetch[google-adk]"      # Google ADK 2.x
uv add "dotfetch[crewai]"          # CrewAI (Python 3.10-3.13; not with openai-agents)
uv add "dotfetch[llamaindex]"      # LlamaIndex (llama-index-core 0.13+)
uv add "dotfetch[strands]"         # Strands Agents 1.38+
uv add "dotfetch[agno]"            # Agno 2.6.10+
uv add "dotfetch[haystack]"        # Haystack 3.x
uv add "dotfetch[dspy]"            # DSPy 3.5+
uv add "dotfetch[all]"
```

## Quick start

```python
from openai import AsyncOpenAI
from dotfetch import DotFetch, set_default_fetcher, with_semantic_tools

client = AsyncOpenAI()
fetcher = DotFetch()
set_default_fetcher(fetcher)

@fetcher.register(tags=["weather"])
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is 72°F and sunny."

@with_semantic_tools(limit=3, dialect="openai")
async def ask_llm(prompt: str, *, tools=None):
    return await client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
    )

await ask_llm("What's the weather in San Francisco?")
```

dotfetch automatically fingerprints registered tools, syncs definitions to the configured vector store, retrieves semantically relevant tools, and converts schemas to the requested provider dialect.

## Framework integration in one line

Use the same registry with any supported framework. Install the framework SDK
you use; dotfetch imports it only when it builds native tools or live hooks.

```python
# Native LangChain tools, ready for model.bind_tools(tools).
tools = await fetcher.framework("langchain").select("What's the weather?", limit=3)

# Pydantic AI: tools reselected on every model step, with native retries,
# approvals and deps (pip install 'dotfetch[pydantic-ai]').
from pydantic_ai import Agent
agent = Agent("openai:gpt-5.5", capabilities=[fetcher.framework("pydantic-ai").capability(limit=3)])

# Discover all twelve integrations without installing their SDKs.
from dotfetch import supported_frameworks
print(supported_frameworks())
```

Framework names preserve adapter types for editor completion. Hyphenated names
such as `google-adk` and `openai-agents` also work. Existing adapter classes and
imports remain available. See the [integration recipes](dotfetch/integrations/frameworks/README.md)
for each framework's install command, static tools, and live selection hook.

## Core capabilities

- **Semantic tool routing:** reduce prompt context by retrieving top-k relevant tools instead of injecting every tool.
- **Register once, run anywhere:** emit schemas for OpenAI (Chat Completions and Responses), Anthropic, Gemini, Mistral, Groq, and Microsoft Agent Framework dialects — plus the framework-native adapters, MCP, and A2A paths below.
- **Secure execution:** run tools through policies, capabilities, timeouts, retries, rate limits, circuit breakers, callbacks, and telemetry. Batch and streaming tool calls are supported via `execute_batch()` and `execute_tool_calls()`.
- **Persistence and retrieval:** use in-memory defaults, LanceDB, Qdrant, Chroma, pgvector, OpenAI/Azure/Nomic/sentence-transformers embeddings, and Cohere / cross-encoder / System One rerankers.
- **Framework coverage:** Microsoft Agent Framework plus LangChain, LangGraph, LlamaIndex, CrewAI, Google ADK, Pydantic AI, OpenAI Agents SDK, Haystack, Agno, Strands Agents, and DSPy.
- **MCP both ways:** consume local (stdio) and remote (Streamable HTTP / SSE) MCP servers, and serve your registry to Claude Desktop, Claude Code or any remote client with `fetcher.serve_mcp()` / `dotfetch serve-mcp --module my_app.tools` — two meta-tools instead of the whole tool list.
- **A2A both ways:** discover and call remote A2A agents as tools with `add_a2a_agent()`, and serve your own registry as an A2A agent (Agent Card at `/.well-known/agent.json`) with `fetcher.serve_a2a()`.
- **Skills, retrieved by meaning:** load any Agent Skills (`SKILL.md`) directory with `fetcher.add_skills_from_directory(...)` and inject only the skills relevant to each prompt.
- **Bundled Claude Skill:** install with `dotfetch install-skill --claude` or target a project-local skills directory.
- **Observability:** emit retrieval/execution telemetry to the console, OpenTelemetry, or Prometheus, and track token savings with the built-in metrics helpers.
- **Selection without embeddings:** point a decision model at the catalogue instead of a vector store. `SystemOneSelector` replaces embed-then-search for tools, skills and MCP servers; `SystemOneReranker` refines the shortlist when the catalogue is too large to send. A selector can return nothing when no tool fits; a reranker only reorders, so it always returns its top-k. Both fail open to semantic routing, and the result says when that happened (`selector_fallback`); `on_selector_fallback="raise"` makes evals fail loudly instead.

dotfetch supports Python 3.10–3.14. CrewAI's current SDK requires Python
3.10–3.13; the `agent-frameworks` extra includes it only on those versions.

## Selecting tools without a vector store

Semantic routing embeds the query and searches. A *selector* asks a decision
model directly, which needs no embedder, no vector store and no sync — and can
return nothing when nothing fits, which top-k cannot express.

```python
from dotfetch import DotFetch, SystemOneSelector

fetcher = DotFetch(selector=SystemOneSelector(threshold=0.3))  # reads TYPESAFE_API_KEY

@fetcher.register(tags=["email"], examples=["show my messages", "any new mail"])
def list_inbox(limit: int = 10) -> list[str]:
    """List the most recent messages sitting in the inbox."""
    return []

@fetcher.register(tags=["email"], examples=["email Bob about the meeting"])
def send_email(to: str, body: str) -> str:
    """Send an email message to a named recipient."""
    return "sent"

# With a key, this honours the negation and returns list_inbox alone.
# Without one, selection fails open and semantic routing answers instead.
tools = await fetcher.retrieve_tools("show my messages, but do not send anything")
```

The same selector covers `retrieve_skills()` and `retrieve_mcp_servers()`. Above
a few hundred entries, keep semantic search and refine it instead:

```python
from dotfetch import DotFetch, SystemOneReranker

fetcher = DotFetch(reranker=SystemOneReranker())   # reorders the vector-search shortlist
```

The two are not interchangeable on one point: a reranker only *orders* the
shortlist, so it still returns its top-k for a query no tool can serve. Only
the selector, which gates each entry at `threshold`, can return nothing.

Both ask a "System One" decision model: TypeSafe's Jev by default
(`pip install 'dotfetch[system-one]'`), or another via `model=`. They were
called `JevSelector` / `JevReranker` before 0.23.0; those names, the `jev`
extra and `type="jev"` still work, with a `DeprecationWarning`.

Write `examples=[...]` on your tools before reaching for either: on our own
benchmark that moved the default embedder from 1/5 to 5/5, more than any model
change did. See `dotfetch/adapters/selectors/README.md` for the measured
trade-offs.

## Manual retrieval and execution

```python
from dotfetch import DotFetch
from dotfetch.schema.execution import ToolCall

fetcher = DotFetch()

@fetcher.register(tags=["finance"])
def calculate_tax(amount: float) -> float:
    """Calculate US sales tax for an amount."""
    return amount * 0.08

tools = await fetcher.retrieve_tools("What is the tax on $100?", limit=5)
result = await fetcher.execute(ToolCall(
    tool_name="calculate_tax",
    arguments={"amount": 100.0},
))
```

## Development

```bash
uv sync --all-extras
uv run pytest
npm install
npm run build
```

## License

MIT
