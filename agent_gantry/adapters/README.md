# agent_gantry/adapters

Adapter layer that lets Agent-Gantry talk to external systems for embeddings, storage, reranking,
and execution.

- `__init__.py`: Adapter package exports.
- `embedders/`: Embedding providers used to encode tool descriptions.
- `executors/`: Execution backends and clients for delegated tool execution.
- `rerankers/`: Reranking providers that reorder retrieved tools.
- `vector_stores/`: Vector store implementations for indexing and querying tools.
