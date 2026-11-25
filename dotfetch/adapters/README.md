# dotfetch/adapters

Adapter implementations that let dotfetch plug into external systems. Adapters keep the core
router and executor protocol-agnostic by providing common interfaces for embeddings, storage,
re-ranking, and delegated execution.

## Why Adapters Matter

dotfetch's power comes from its flexibility. You can start with simple in-memory components for development and seamlessly upgrade to production-grade systems without changing your application code.

**Development → Production Path:**
```
Simple Embedder → OpenAI → Nomic (local)
In-Memory Store → LanceDB (local) → Qdrant/Chroma (remote)
No Reranker → Cohere Rerank
```

## Sub-packages

- `embedders/`: Embed tool metadata into vectors. Ships with OpenAI/Azure, Nomic, and a lightweight
  development embedder. See `EmbedderConfig` in `schema.config`.
- `vector_stores/`: Persistent or in-memory vector storage. Includes LanceDB, in-memory, and remote
  Chroma/PGVector/Qdrant clients.
- `rerankers/`: Optional re-rankers that take the initial vector search results and reorder them
  (e.g., Cohere Rerank) to improve accuracy on hard queries.
- `executors/`: Backends for dispatching tool invocations outside the local process. Includes an A2A
  executor and an MCP client executor.

All adapters follow a slim interface (`EmbeddingAdapter`, `VectorStoreAdapter`, etc.) so you can
swap implementations without touching the core. Most configuration can be driven from
`DotFetchConfig`/`schema.config` without code changes.

## Easy Adapter Swapping

### Example 1: Upgrade from In-Memory to LanceDB

**Before (Development):**
```python
from dotfetch import DotFetch

# Default: SimpleEmbedder + In-Memory VectorStore
fetcher = DotFetch()

# Register tools...
await fetcher.sync()
```

**After (Production):**
```python
from dotfetch import DotFetch, DotFetchConfig
from dotfetch.schema.config import EmbedderConfig, VectorStoreConfig

# Just change configuration - same code everywhere else
config = DotFetchConfig(
    embedder=EmbedderConfig(type="nomic", model="nomic-embed-text-v1.5"),
    vector_store=VectorStoreConfig(type="lancedb", db_path="./dotfetch.lance"),
)

fetcher = DotFetch(config=config)

# Same code: Register tools...
await fetcher.sync()  # Now persists to disk with better embeddings
```

### Example 2: Add OpenAI Embeddings + Qdrant

```python
from dotfetch import DotFetch, DotFetchConfig
from dotfetch.schema.config import EmbedderConfig, VectorStoreConfig

config = DotFetchConfig(
    embedder=EmbedderConfig(
        type="openai",
        model="text-embedding-3-large",
        dimension=3072
    ),
    vector_store=VectorStoreConfig(
        type="qdrant",
        url="http://localhost:6333",
        collection_name="agent_tools"
    ),
)

fetcher = DotFetch(config=config)
# Everything else stays the same!
```

### Example 3: Add Cohere Reranking

```python
from dotfetch import DotFetch, DotFetchConfig
from dotfetch.schema.config import RerankerConfig

config = DotFetchConfig(
    reranker=RerankerConfig(
        type="cohere",
        model="rerank-english-v3.0",
        top_k=3
    )
)

fetcher = DotFetch(config=config)
# Retrieval now uses two-stage ranking: vector search + reranking
```

## Configuration-Driven Adapter Selection

All adapter configuration can be done via YAML for easy environment management:

**fetcher.yaml:**
```yaml
embedder:
  type: openai
  model: text-embedding-3-large
  dimension: 3072

vector_store:
  type: lancedb
  db_path: ./dotfetch.lance

reranker:
  type: cohere
  model: rerank-english-v3.0
  top_k: 3

routing:
  weights:
    semantic: 0.8
    health: 0.2
```

**Load in code:**
```python
from dotfetch import DotFetch, DotFetchConfig

config = DotFetchConfig.from_yaml("fetcher.yaml")
fetcher = DotFetch(config=config)
```

## Common Adapter Combinations

| Use Case | Embedder | Vector Store | Reranker |
|----------|----------|--------------|----------|
| **Local Development** | SimpleEmbedder | In-Memory | None |
| **Production (Cloud)** | OpenAI | Qdrant/Chroma | Cohere |
| **Production (Local)** | Nomic | LanceDB | CrossEncoder |
| **High Accuracy** | OpenAI (large) | Qdrant | Cohere |
| **Cost-Optimized** | Nomic | LanceDB | None |

## Programmatic Adapter Creation

You can also instantiate adapters directly and pass them to DotFetch:

```python
from dotfetch import DotFetch
from dotfetch.adapters.embedders.nomic import NomicEmbedder
from dotfetch.adapters.vector_stores.lancedb import LanceDBVectorStore

# Create adapters explicitly
embedder = NomicEmbedder(model="nomic-embed-text-v1.5")
vector_store = LanceDBVectorStore(db_path="./my_tools.lance")

# Pass to DotFetch
fetcher = DotFetch(
    embedder=embedder,
    vector_store=vector_store
)
```

## See Also

For deeper explanations, each subdirectory has its own README that documents the adapter surface,
configuration knobs, and usage examples:
- [embedders/README.md](./embedders/README.md)
- [vector_stores/README.md](./vector_stores/README.md)
- [rerankers/README.md](./rerankers/README.md)
- [executors/README.md](./executors/README.md)
