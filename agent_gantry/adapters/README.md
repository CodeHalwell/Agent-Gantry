# agent_gantry/adapters

Adapter implementations that let Agent-Gantry plug into external systems. Adapters keep the core
router and executor protocol-agnostic by providing common interfaces for embeddings, storage,
re-ranking, and delegated execution.

## Why Adapters Matter

Agent-Gantry's power comes from its flexibility. You can start with simple in-memory components for development and seamlessly upgrade to production-grade systems without changing your application code.

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
`AgentGantryConfig`/`schema.config` without code changes.

## Easy Adapter Swapping

### Example 1: Upgrade from In-Memory to LanceDB

**Before (Development):**
```python
from agent_gantry import AgentGantry

# Default: SimpleEmbedder + In-Memory VectorStore
gantry = AgentGantry()

# Register tools...
await gantry.sync()
```

**After (Production):**
```python
from agent_gantry import AgentGantry, AgentGantryConfig
from agent_gantry.schema.config import EmbedderConfig, VectorStoreConfig

# Just change configuration - same code everywhere else
config = AgentGantryConfig(
    embedder=EmbedderConfig(provider="nomic", model="nomic-embed-text-v1.5"),
    vector_store=VectorStoreConfig(provider="lancedb", uri="./gantry.lance"),
)

gantry = AgentGantry(config=config)

# Same code: Register tools...
await gantry.sync()  # Now persists to disk with better embeddings
```

### Example 2: Add OpenAI Embeddings + Qdrant

```python
from agent_gantry import AgentGantry, AgentGantryConfig
from agent_gantry.schema.config import EmbedderConfig, VectorStoreConfig

config = AgentGantryConfig(
    embedder=EmbedderConfig(
        provider="openai",
        model="text-embedding-3-large",
        dimensions=3072
    ),
    vector_store=VectorStoreConfig(
        provider="qdrant",
        url="http://localhost:6333",
        collection_name="agent_tools"
    ),
)

gantry = AgentGantry(config=config)
# Everything else stays the same!
```

### Example 3: Add Cohere Reranking

```python
from agent_gantry import AgentGantry, AgentGantryConfig
from agent_gantry.schema.config import RerankerConfig

config = AgentGantryConfig(
    reranker=RerankerConfig(
        provider="cohere",
        model="rerank-english-v3.0",
        top_n=3
    )
)

gantry = AgentGantry(config=config)
# Retrieval now uses two-stage ranking: vector search + reranking
```

## Configuration-Driven Adapter Selection

All adapter configuration can be done via YAML for easy environment management:

**gantry.yaml:**
```yaml
embedder:
  provider: openai
  model: text-embedding-3-large
  dimensions: 3072

vector_store:
  provider: lancedb
  uri: ./gantry.lance

reranker:
  provider: cohere
  model: rerank-english-v3.0
  top_n: 3

routing:
  weights:
    semantic: 0.8
    health: 0.2
```

**Load in code:**
```python
from agent_gantry import AgentGantry, AgentGantryConfig

config = AgentGantryConfig.from_yaml("gantry.yaml")
gantry = AgentGantry(config=config)
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

You can also instantiate adapters directly and pass them to AgentGantry:

```python
from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.nomic import NomicEmbedder
from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

# Create adapters explicitly
embedder = NomicEmbedder(model="nomic-embed-text-v1.5")
vector_store = LanceDBVectorStore(uri="./my_tools.lance")

# Pass to AgentGantry
gantry = AgentGantry(
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
