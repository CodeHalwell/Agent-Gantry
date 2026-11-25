# dotfetch/adapters/embedders

Embedding adapters turn tool metadata into vectors so the semantic router can score relevance. All
embedders implement the `EmbeddingAdapter` protocol defined in `base.py`.

## Modules

- `base.py`: Defines the adapter interface (`embed_text`, `embed_batch`, `model_name`, `dimension`)
  and simple validation helpers.
- `openai.py`: OpenAI / Azure OpenAI implementations. Supports model selection, Azure endpoints, and
  streaming multiple texts in one request.
- `nomic.py`: Uses the Nomic `nomic-embed-text-v1.5` model for high-accuracy matching with minimal
  configuration. Ideal for local or cost-sensitive setups.
- `simple.py`: A pure-Python deterministic embedder used in tests and demos where external calls are
  undesired.

## Choosing an embedder

| Adapter        | Best for                           | Config hook                               |
|----------------|------------------------------------|-------------------------------------------|
| `OpenAIEmbedder` / `AzureOpenAIEmbedder` | Hosted, low-latency, high-quality embeddings | `EmbedderConfig(type=\"openai\", ...)` |
| `NomicEmbedder`| Local or cost-optimized accuracy   | `EmbedderConfig(type=\"nomic\")`           |
| `SentenceTransformersEmbedder` | Local, no API key; the default | `EmbedderConfig(type=\"sentence_transformers\")` (default; needs the `embeddings` extra) |
| `SimpleEmbedder`| Tests, offline demos, reproducibility | `DotFetch(embedder=SimpleEmbedder())` |

The default needs sentence-transformers. Without it the factory falls back to
`SimpleEmbedder`, which hashes tokens and understands no meaning, and logs a
`WARNING` once per process naming the extra to install. Every result says which
embedder searched (`RetrievalResult.embedder`; `"simple-hash-embedder"` is the
fallback), and `EmbedderConfig(on_unavailable="raise")` raises `ImportError`
instead of falling back.

## Example

```python
from dotfetch import DotFetch
from dotfetch.schema.config import EmbedderConfig

fetcher = DotFetch(
    embedder=None,  # let the config drive it
    config=DotFetchConfig(
        embedder=EmbedderConfig(type="openai", model="text-embedding-3-large")
    ),
)

await fetcher.sync()  # embeds registered tools using the configured provider
```

If you need a custom provider, subclass `EmbeddingAdapter` and supply it to `DotFetch(embedder=..)`;
the router will use it without further changes.
