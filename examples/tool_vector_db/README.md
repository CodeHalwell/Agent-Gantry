# Agent-Gantry: Simple Vector DB Example with Nomic Embeddings

The simplest possible example of using Agent-Gantry with Nomic embeddings to:
1. Register 50 tools
2. Semantically retrieve only the relevant ones
3. Execute them via OpenAI Responses API

## Why This Matters

Without Agent-Gantry, you'd send all 50 tool definitions to the LLM on every request. That's:
- **~15,000+ tokens** per request just for tool definitions
- **Slower responses** due to larger context
- **Higher costs** from token usage

With Agent-Gantry:
- Only **~5 relevant tools** are sent (configurable)
- **~90% token savings** on tool definitions
- **Faster, cheaper** LLM calls

## Files

- `tools.py` - 50 simple tools organized by category (math, text, datetime, utility, conversion)
- `main.py` - Main script showing the complete workflow

## Quick Start

```bash
# Install dependencies (includes sentence-transformers for Nomic)
pip install agent-gantry[nomic] openai python-dotenv

# Set your API key
export OPENAI_API_KEY=your-key-here

# Run the example
python main.py
```

## How It Works

```python
from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.nomic import NomicEmbedder
from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore
from agent_gantry.integrations.semantic_tools import with_semantic_tools
from agent_gantry.schema.execution import ToolCall

# 1. Create gantry with Nomic embeddings (high-quality semantic search)
embedder = NomicEmbedder(dimension=256)
vector_store = InMemoryVectorStore()
tools = AgentGantry(embedder=embedder, vector_store=vector_store)

# 2. Register tools using decorator
@tools.register()
def calculate_mean(numbers: list[float]) -> float:
    """Calculate the average of numbers."""
    return sum(numbers) / len(numbers)

# 3. Sync to vector store (embeds descriptions)
await tools.sync()

# 4. Use @with_semantic_tools to auto-inject relevant tools
@with_semantic_tools(tools, limit=5, dialect="openai_responses")
async def chat(prompt: str, **kwargs):
    return await client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        **kwargs,  # tools injected here automatically!
    )

# 5. Call the function - decorator handles tool retrieval
response = await chat("Calculate the mean of [1, 2, 3]")

# 6. Execute tool calls from the response
for item in response.output:
    if item.type == "function_call":
        call = ToolCall(tool_name=item.name, arguments=json.loads(item.arguments))
        result = await tools.execute(call)
        print(f"{item.name} = {result.result}")
```

## Tool Categories

| Category | Count | Examples |
|----------|-------|----------|
| Math | 10 | add, subtract, calculate_mean, calculate_sqrt |
| Text | 10 | count_words, reverse_string, to_uppercase |
| DateTime | 10 | get_current_date, days_between, is_leap_year |
| Utility | 10 | generate_uuid, hash_text, flip_coin |
| Conversion | 10 | celsius_to_fahrenheit, meters_to_feet |

## Configuration

Change the number of tools retrieved:

```python
@with_semantic_tools(tools, limit=3)   # Only top 3 tools
@with_semantic_tools(tools, limit=10)  # Top 10 tools
```

## Next Steps

- See `examples/project_demo/` for persistent storage with LanceDB
- See `examples/llm_integration/` for more LLM provider examples
- See `docs/` for full documentation
