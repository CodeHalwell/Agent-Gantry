# Building a tool vector store and wiring it to LLM SDKs

This guide shows how to register many tools with Agent-Gantry, index them in a vector store, and
hand provider-specific tool schemas to popular SDKs (OpenAI Responses API, Google GenAI, and Claude).

## Prerequisites

- Python 3.10+
- `pip install agent-gantry[llm-providers]`
- Provider SDKs installed for the clients you plan to call (`openai`, `google-genai`, `anthropic`)
- For config permutations (remote stores, LanceDB/Nomic) see `configuration.md` and
  `local_persistence_and_skills.md`.

## 1) Register tools and sync them into the vector store

```python
import asyncio
from agent_gantry import AgentGantry
from agent_gantry.schema.config import AgentGantryConfig, VectorStoreConfig

# Point at a persistent store when you have a large tool catalog
config = AgentGantryConfig(
    vector_store=VectorStoreConfig(
        type="qdrant",
        url="http://localhost:6333",
        collection_name="my_tools",
    )
)
gantry = AgentGantry(config=config)

# Register many tools. Decorators automatically build JSON Schema for arguments.
@gantry.register(tags=["math"], examples=["add two numbers"])
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@gantry.register(tags=["knowledge"], examples=["summarize text"])
def summarize(text: str) -> str:
    """Summarize a passage of text."""
    return text[:200]

# When you have dozens or hundreds of tools, sync once to embed and index them.
async def bootstrap() -> None:
    await gantry.sync()

asyncio.run(bootstrap())
```

`sync()` batches embedding and upserts each tool into the configured vector store (in-memory by
default, or remote options like Qdrant/Chroma/PGVector).

## 2) Retrieve relevant tools for the current request

```python
import asyncio
from agent_gantry.schema.query import ConversationContext, ToolQuery

async def pick_tools(question: str):
    context = ConversationContext(query=question)
    query = ToolQuery(context=context, limit=8)
    retrieval = await gantry.retrieve(query)
    return retrieval

retrieval = asyncio.run(pick_tools("summarize quarterly revenue and compute tax"))
```

`retrieval.tools` contains scored `ToolDefinition` objects. Use the helpers below to format them for
each SDK.

## 3) Provide tools to OpenAI (Chat Completions or Responses API)

```python
from openai import OpenAI

tools = [t.tool.to_openai_schema() for t in retrieval.tools]

client = OpenAI(api_key="sk-...")
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Summarize Q1 revenue and compute the tax."}],
    tools=tools,
    tool_choice="auto",
)
# If you prefer the Responses API, swap to `client.responses.create(...)` with the same payload.
```

## 4) Provide tools to Google GenAI (Gemini)

```python
from google import genai
from google.genai import types

gemini_tools = [
    types.Tool(function_declarations=[t.tool.to_gemini_schema() for t in retrieval.tools])
]

client = genai.Client(api_key="...")  # genai SDK >= 0.8.0
result = client.models.generate_content(
    model="gemini-1.5-pro",
    contents="Summarize Q1 revenue and compute the tax.",
    tools=gemini_tools,
)
```

## 5) Provide tools to Claude (Anthropic SDK)

```python
from anthropic import Anthropic

anthropic_tools = [t.tool.to_anthropic_schema() for t in retrieval.tools]

client = Anthropic(api_key="sk-ant-...")
message = client.messages.create(
    model="claude-3.7-sonnet-async",
    messages=[{"role": "user", "content": "Summarize Q1 revenue and compute the tax."}],
    tools=anthropic_tools,
)
```

## Tips for large catalogs

- Use namespaces (`@gantry.register(namespace="finance")`) to group tools and filter by namespace in
  `ToolQuery`.
- Set `limit` on `ToolQuery` to keep only the most relevant tools in the prompt.
- Configure `reranker.enabled=True` in `RerankerConfig` when you need higher precision retrieval.
- For write-heavy workloads, disable `auto_sync` in `AgentGantryConfig` and call `sync()` manually
  after bulk registering tools.
