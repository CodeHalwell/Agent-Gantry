# agent_gantry/adapters

Adapter implementations that let Agent-Gantry plug into external systems. Adapters keep the core
router and executor protocol-agnostic by providing common interfaces for embeddings, storage,
re-ranking, and delegated execution.

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

## Quick configuration example

```python
from agent_gantry import AgentGantry, AgentGantryConfig
from agent_gantry.schema.config import EmbedderConfig, VectorStoreConfig

config = AgentGantryConfig(
    embedder=EmbedderConfig(provider="openai", model="text-embedding-3-large"),
    vector_store=VectorStoreConfig(provider="lancedb", uri="./gantry.lance"),
)

gantry = AgentGantry(config=config)
await gantry.sync()
```

For deeper explanations, each subdirectory has its own README that documents the adapter surface,
configuration knobs, and usage examples.*** End Patch"|()
