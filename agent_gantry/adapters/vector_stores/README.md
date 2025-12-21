# agent_gantry/adapters/vector_stores

Vector store adapters index tool embeddings and return nearest neighbors for a `ToolQuery`. Each
adapter implements `VectorStoreAdapter` from `base.py`, allowing the `SemanticRouter` to remain
storage-agnostic.

## Modules

- `base.py`: Interface for indexing, searching, and deleting tool vectors, plus helpers for packing
  metadata.
- `memory.py`: In-memory store optimized for tests and demos. Zero dependencies and perfect for
  rapid iteration.
- `lancedb.py`: LanceDB-backed store with local persistence and namespace isolation.
- `remote.py`: Wrappers for Chroma, PGVector, and Qdrant so you can point Agent-Gantry at existing
  hosted or self-managed vector databases.

## Picking a backend

| Adapter               | Best for                       | Config example                                   |
|-----------------------|--------------------------------|--------------------------------------------------|
| `InMemoryVectorStore` | Tests, quick start, notebooks  | `VectorStoreConfig(provider=\"memory\")`         |
| `LanceDBVectorStore`  | Local persistence + fast recall| `VectorStoreConfig(provider=\"lancedb\", uri=\"./db\")` |
| `ChromaVectorStore` / `PGVectorStore` / `QdrantVectorStore` | Managed or self-hosted vector DBs | `VectorStoreConfig(provider=\"remote\", type=\"qdrant\", url=\"http://localhost:6333\")` |

## Example

```python
from agent_gantry import AgentGantry, AgentGantryConfig
from agent_gantry.schema.config import VectorStoreConfig

config = AgentGantryConfig(vector_store=VectorStoreConfig(provider="memory"))
gantry = AgentGantry(config=config)
await gantry.sync()  # pushes tool embeddings into the chosen store
```

Adding a new backend is as simple as implementing `VectorStoreAdapter`; the router will consume it
without further changes.*** End Patch" serializer error? Actually apply_patch expects raw string not JSON. Let's re-run plain.
