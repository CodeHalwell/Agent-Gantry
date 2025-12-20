# agent_gantry/adapters/vector_stores

Vector store adapters for indexing and retrieving tool embeddings.

- `__init__.py`: Exposes available vector store adapters.
- `base.py`: Abstract vector store interface plus common helpers.
- `memory.py`: In-memory vector store for fast local development and testing.
- `remote.py`: Wrappers for remote stores (Chroma, PGVector, Qdrant) with unified API.
