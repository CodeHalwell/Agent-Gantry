# Agent-Gantry: Simple Vector DB Example with LanceDB

The simplest possible example of using Agent-Gantry with persistent LanceDB storage to:
1. Register 50 tools
2. Semantically retrieve only the relevant ones
3. Execute them via OpenAI Responses API

## Why This Matters

Without Agent-Gantry, you'd send all 50 tool definitions to the LLM on every request. That's:
- **~15,000+ tokens** per request just for tool definitions
- **Slower responses** due to larger context
- **Higher costs** from token usage

With Agent-Gantry:
- Only the **top-k relevant tools** are sent (`limit=10` in `main.py`, configurable)
- **~90% token savings** on tool definitions
- **Faster, cheaper** LLM calls

## Files

- `tools.py` - 50 simple tools organized by category (math, text, datetime, utility, conversion)
- `main.py` - Main script showing the complete workflow

## Quick Start

```bash
# Install dependencies
pip install "agent-gantry[lancedb,openai]" python-dotenv

# Run the example. Registration, sync and retrieval need no key: it prints
# the tools Gantry picked, then stops unless OPENAI_API_KEY is set.
python -m examples.tool_vector_db.main

# Set your API key to run the model call and tool execution too
export OPENAI_API_KEY=your-key-here
python -m examples.tool_vector_db.main
```

## How It Works

```python
from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore
from agent_gantry.integrations.semantic_tools import with_semantic_tools

# 1. Create gantry with persistent LanceDB storage (high-quality semantic search)
embedder = NomicEmbedder(dimension=768)  # Matryoshka truncation (64/128/256/512/768)
vector_store = LanceDBVectorStore(db_path="gantry_tools.lancedb", dimension=768)
tools = AgentGantry(embedder=embedder, vector_store=vector_store)

# 2. Register tools using decorator
@tools.register(examples=["calculate the average of these numbers"])
def calculate_mean(numbers: list[float]) -> float:
    """Calculate the average of numbers."""
    return sum(numbers) / len(numbers)

# 3. Sync to vector store (embeds descriptions)
await tools.sync()

# 4. Use @with_semantic_tools to auto-inject relevant tools
@with_semantic_tools(tools, limit=5, dialect="openai_responses")
async def chat(prompt: str, **kwargs):
    return await client.responses.create(
        model="gpt-5.4-mini",
        input=prompt,
        **kwargs,  # tools injected here automatically!
    )

# 5. Call the function - decorator handles tool retrieval
response = await chat("Calculate the mean of [1, 2, 3]")

# 6. Execute and format all results for the next Responses API turn
outputs = await tools.execute_tool_calls(response, dialect="openai_responses")
if outputs:
    response = await client.responses.create(
        model="gpt-5.4-mini", input=outputs, previous_response_id=response.id,
    )
```

For semantic embeddings, install `agent-gantry[nomic]` and replace the hash
embedder with `NomicEmbedder(dimension=768)`. Use a fresh database path when
changing embedding models, even if the dimensions match.

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
- See [the documentation site](https://codehalwell.github.io/Agent-Gantry/) for full documentation
