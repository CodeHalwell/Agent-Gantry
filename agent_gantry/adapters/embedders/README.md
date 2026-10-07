# agent_gantry/adapters/embedders

Embedding adapters turn tool metadata into vectors so the semantic router can score relevance. All
embedders implement the `EmbeddingAdapter` protocol defined in `base.py`.

## Modules

- `base.py`: Defines the adapter interface (`embed_text`, `embed_batch`, `model_name`, `dimension`)
  and simple validation helpers.
- `openai.py`: OpenAI / Azure OpenAI implementations. Supports model selection, Azure endpoints, and
  streaming multiple texts in one request.
- `sentence_transformers.py`: The default backend: any sentence-transformers model, loaded lazily
  (`all-MiniLM-L6-v2` unless `EmbedderConfig.model` says otherwise).
- `nomic.py`: A subclass of the sentence-transformers embedder that runs `nomic-ai/nomic-embed-text-v1.5`
  with Nomic's task prefixes and optional Matryoshka truncation. Ideal for local or cost-sensitive setups.
- `cached.py`: `CachedEmbedder`, a sqlite-backed wrapper that remembers embeddings so a cold start
  does not re-embed the whole registry with a paid embedder. Wrap any embedder and pass it in.
- `simple.py`: A pure-Python deterministic hash embedder used in tests and demos where external calls
  are undesired; also the fallback when sentence-transformers is not installed.

## Choosing an embedder

| Adapter        | Best for                           | Config hook                               |
|----------------|------------------------------------|-------------------------------------------|
| `OpenAIEmbedder` / `AzureOpenAIEmbedder` | Hosted, low-latency, high-quality embeddings | `EmbedderConfig(type=\"openai\", ...)` |
| `SentenceTransformersEmbedder` | Local embeddings, no key (the default) | `EmbedderConfig(type=\"sentence_transformers\")` |
| `NomicEmbedder`| Local or cost-optimized accuracy   | `EmbedderConfig(type=\"nomic\")`           |
| `SimpleEmbedder`| Tests, offline demos, reproducibility | pass `embedder=SimpleEmbedder()` (also the fallback when sentence-transformers is missing) |

## Example

```python
from agent_gantry import AgentGantry, AgentGantryConfig
from agent_gantry.schema.config import EmbedderConfig

gantry = AgentGantry(
    embedder=None,  # let the config drive it
    config=AgentGantryConfig(
        embedder=EmbedderConfig(type="openai", model="text-embedding-3-large")
    ),
)

await gantry.sync()  # embeds registered tools using the configured provider
```

If you need a custom provider, subclass `EmbeddingAdapter` and supply it to `AgentGantry(embedder=..)`;
the router will use it without further changes.
