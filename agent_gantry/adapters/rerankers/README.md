# agent_gantry/adapters/rerankers

Rerankers take the initial vector search results and reorder them using richer relevance signals.
They are optional but can improve precision on ambiguous or verbose queries by considering more than
cosine similarity alone.

## Modules

- `base.py`: Declares the `RerankerAdapter` protocol; implements small helpers for packaging inputs
  and outputs.
- `cohere.py`: Cohere ReRank implementation that scores tool descriptions (and docstrings) using
  Cohere's hosted models.

## Typical configuration

```python
from agent_gantry import AgentGantry, AgentGantryConfig
from agent_gantry.schema.config import RerankerConfig

config = AgentGantryConfig(
    reranker=RerankerConfig(provider="cohere", model="rerank-english-v3.0", top_n=5)
)
gantry = AgentGantry(config=config)
await gantry.sync()
```

During retrieval the semantic router first performs vector search, then (if configured) passes the
candidates to the reranker and merges those scores with health weighting before returning
`ScoredTool` results.*** End Patch" Magic!
