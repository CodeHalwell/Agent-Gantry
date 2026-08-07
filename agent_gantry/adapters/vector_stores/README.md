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
without further changes.

## Skills

`InMemoryVectorStore` and `LanceDBVectorStore` also store **skills** (procedural memory retrieved
semantically and injected into prompts — see `gantry.add_skills()` / `gantry.retrieve_skills()`).
Stores without skill support raise `NotImplementedError` from the facade's skill methods.

Notes: skill content is injected into prompts verbatim, so register skills only from trusted
sources. Switching embedding models re-embeds stored skills automatically when the dimension is
unchanged; a dimension change cannot be migrated in place on fixed-schema stores (LanceDB) —
recreate the store. Concurrent gantry instances with *different* embedders sharing one store are
unsupported (each would re-migrate the other's vectors).

## Qdrant quantized search

`QdrantVectorStore(quantization="scalar")` enables int8 scalar quantization (~4x smaller vectors,
kept in RAM, minimal recall loss); `quantization="binary"` compresses ~32x and suits
high-dimensional embeddings. Searches oversample and rescore against the original vectors, so
returned scores stay exact. Applied at collection creation — recreate the collection to change it.
