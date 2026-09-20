# Agent-Gantry

**Universal Tool Orchestration Platform for LLM-Based Agent Systems**

*Context is precious. Execution is sacred. Trust is earned.*

Agent-Gantry is a Python library (**v0.18.0**) for building agents that can discover, select, and execute the right tools without flooding every prompt with every schema your organization owns. It combines semantic retrieval, provider schema conversion, secure execution, framework bridges, MCP/A2A interoperability, persistence adapters, and observability into one tool orchestration layer.

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
uv add agent-gantry
# or
pip install agent-gantry
```

Useful extras:

```bash
uv add "agent-gantry[openai]"
uv add "agent-gantry[anthropic]"
uv add "agent-gantry[google-genai]"
uv add "agent-gantry[lancedb,nomic]"
uv add "agent-gantry[mcp,a2a]"
uv add "agent-gantry[agent-frameworks]"
uv add "agent-gantry[all]"
```

## Quick start

```python
from openai import AsyncOpenAI
from agent_gantry import AgentGantry, set_default_gantry, with_semantic_tools

client = AsyncOpenAI()
gantry = AgentGantry()
set_default_gantry(gantry)

@gantry.register(tags=["weather"])
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

Agent-Gantry automatically fingerprints registered tools, syncs definitions to the configured vector store, retrieves semantically relevant tools, and converts schemas to the requested provider dialect.

## Core capabilities

- **Semantic tool routing:** reduce prompt context by retrieving top-k relevant tools instead of injecting every tool.
- **Register once, run anywhere:** emit schemas for OpenAI (Chat Completions and Responses), Anthropic, Gemini, Mistral, Groq, and Microsoft Agent Framework dialects — plus the framework-native adapters, MCP, and A2A paths below.
- **Secure execution:** run tools through policies, capabilities, timeouts, retries, rate limits, circuit breakers, callbacks, and telemetry. Batch and streaming tool calls are supported via `execute_batch()` and `execute_tool_calls()`.
- **Persistence and retrieval:** use in-memory defaults, LanceDB, Qdrant, Chroma, pgvector, OpenAI/Azure/Nomic/sentence-transformers embeddings, and Cohere / cross-encoder / Jev rerankers.
- **Framework coverage:** Microsoft Agent Framework plus LangChain, LangGraph, LlamaIndex, CrewAI, Google ADK, Pydantic AI, OpenAI Agents SDK, Haystack, Agno, Strands Agents, and DSPy.
- **MCP both ways:** consume local (stdio) and remote (Streamable HTTP / SSE) MCP servers, and serve your registry to Claude Desktop, Claude Code or any remote client with `gantry.serve_mcp()` / `agent-gantry serve-mcp --module my_app.tools` — two meta-tools instead of the whole tool list.
- **A2A both ways:** discover and call remote A2A agents as tools with `add_a2a_agent()`, and serve your own registry as an A2A agent (Agent Card at `/.well-known/agent.json`) with `gantry.serve_a2a()`.
- **Skills, retrieved by meaning:** load any Agent Skills (`SKILL.md`) directory with `gantry.add_skills_from_directory(...)` and inject only the skills relevant to each prompt.
- **Bundled Claude Skill:** install with `agent-gantry install-skill --claude` or target a project-local skills directory.
- **Observability:** emit retrieval/execution telemetry to the console, OpenTelemetry, or Prometheus, and track token savings with the built-in metrics helpers.
- **Selection without embeddings:** point a decision model at the catalogue instead of a vector store. `JevSelector` replaces embed-then-search for tools, skills and MCP servers; `JevReranker` refines the shortlist when the catalogue is too large to send. Both fail open to semantic routing.

## Selecting tools without a vector store

Semantic routing embeds the query and searches. A *selector* asks a decision
model directly, which needs no embedder, no vector store and no sync — and can
return nothing when nothing fits, which top-k cannot express.

```python
from agent_gantry import AgentGantry, JevSelector

gantry = AgentGantry(selector=JevSelector(threshold=0.3))  # reads TYPESAFE_API_KEY

@gantry.register(tags=["email"], examples=["show my messages", "any new mail"])
def list_inbox(limit: int = 10) -> list[str]:
    """List the most recent messages sitting in the inbox."""
    return []

@gantry.register(tags=["email"], examples=["email Bob about the meeting"])
def send_email(to: str, body: str) -> str:
    """Send an email message to a named recipient."""
    return "sent"

# With a key, this honours the negation and returns list_inbox alone.
# Without one, selection fails open and semantic routing answers instead.
tools = await gantry.retrieve_tools("show my messages, but do not send anything")
```

The same selector covers `retrieve_skills()` and `retrieve_mcp_servers()`. Above
a few hundred entries, keep semantic search and refine it instead:

```python
from agent_gantry import AgentGantry, JevReranker

gantry = AgentGantry(reranker=JevReranker())   # reorders the vector-search shortlist
```

Write `examples=[...]` on your tools before reaching for either: on our own
benchmark that moved the default embedder from 1/5 to 5/5, more than any model
change did. See `agent_gantry/adapters/selectors/README.md` for the measured
trade-offs.

## Manual retrieval and execution

```python
from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall

gantry = AgentGantry()

@gantry.register(tags=["finance"])
def calculate_tax(amount: float) -> float:
    """Calculate US sales tax for an amount."""
    return amount * 0.08

tools = await gantry.retrieve_tools("What is the tax on $100?", limit=5)
result = await gantry.execute(ToolCall(
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
