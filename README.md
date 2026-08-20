# Agent-Gantry

**Universal Tool Orchestration Platform for LLM-Based Agent Systems**

*Context is precious. Execution is sacred. Trust is earned.*

Agent-Gantry is a Python library (**v0.11.0**) for building agents that can discover, select, and execute the right tools without flooding every prompt with every schema your organization owns. It combines semantic retrieval, provider schema conversion, secure execution, framework bridges, MCP/A2A interoperability, persistence adapters, and observability into one tool orchestration layer.

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
- **Register once, run anywhere:** emit schemas for OpenAI-compatible APIs, Anthropic, Gemini, framework adapters, MCP, and A2A paths.
- **Secure execution:** run tools through policies, capabilities, timeouts, retries, rate limits, circuit breakers, callbacks, and telemetry.
- **Persistence and retrieval:** use in-memory defaults, LanceDB, Qdrant, Chroma, pgvector, OpenAI/Nomic/sentence-transformers embeddings, and rerankers.
- **Framework coverage:** Microsoft Agent Framework plus LangChain, LangGraph, LlamaIndex, CrewAI, AutoGen, Semantic Kernel, Google ADK, Pydantic AI, OpenAI Agents SDK, Smolagents, Haystack, Agno, Strands Agents, and DSPy.
- **Bundled Claude Skill:** install with `agent-gantry install-skill --claude` or target a project-local skills directory.

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
