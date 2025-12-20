# agent_gantry/adapters/embedders

Embedding providers used to convert tool descriptions into vectors.

- `__init__.py`: Export utilities for embedder adapters.
- `base.py`: Base embedder interface definition.
- `openai.py`: OpenAI and Azure OpenAI embedding adapters.
- `simple.py`: Lightweight local embedder for development without external calls.
