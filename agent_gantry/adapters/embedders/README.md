# agent_gantry/adapters/embedders

Embedding adapters turn tool metadata into vectors so the semantic router can score relevance. All
embedders implement the `EmbeddingAdapter` protocol defined in `base.py`.

## Modules

- `base.py`: Defines the adapter interface (`embed_texts`, `model_name`, `dimensions`) and simple
  validation helpers.
- `openai.py`: OpenAI / Azure OpenAI implementations. Supports model selection, Azure endpoints, and
  streaming multiple texts in one request.
- `nomic.py`: Uses the Nomic `nomic-embed-text-v1.5` model for high-accuracy matching with minimal
  configuration. Ideal for local or cost-sensitive setups.
- `simple.py`: A pure-Python deterministic embedder used in tests and demos where external calls are
  undesired.

## Choosing an embedder

| Adapter        | Best for                           | Config hook                               |
|----------------|------------------------------------|-------------------------------------------|
| `OpenAIEmbedder` / `AzureOpenAIEmbedder` | Hosted, low-latency, high-quality embeddings | `EmbedderConfig(provider=\"openai\", ...)` |
| `NomicEmbedder`| Local or cost-optimized accuracy   | `EmbedderConfig(provider=\"nomic\")`       |
| `SimpleEmbedder`| Tests, offline demos, reproducibility | `EmbedderConfig(provider=\"simple\")`    |

## Example

```python
from agent_gantry import AgentGantry
from agent_gantry.schema.config import EmbedderConfig

gantry = AgentGantry(
    embedder=None,  # let the config drive it
    config=AgentGantryConfig(
        embedder=EmbedderConfig(provider="openai", model="text-embedding-3-large")
    ),
)

await gantry.sync()  # embeds registered tools using the configured provider
```

If you need a custom provider, subclass `EmbeddingAdapter` and supply it to `AgentGantry(embedder=..)`;
the router will use it without further changes.*** End Patch" Great! To maintain grammar end.
