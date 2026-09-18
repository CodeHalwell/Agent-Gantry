# agent_gantry/adapters/rerankers

Rerankers take the initial vector search results and reorder them using richer relevance signals.
They are optional but can improve precision on ambiguous or verbose queries by considering more than
cosine similarity alone.

## Modules

- `base.py`: Declares the `RerankerAdapter` protocol; implements small helpers for packaging inputs
  and outputs.
- `cohere.py`: Cohere ReRank implementation that scores tool descriptions (and docstrings) using
  Cohere's hosted models.
- `cross_encoder.py`: Local sentence-transformers cross-encoder, no API calls.
- `jev.py`: `JevReranker`, backed by [TypeSafe's Jev](https://docs.typesafe.ai). Asks one yes/no
  question per candidate — "would this tool help?" — and orders by the returned probability. This is
  the recipe TypeSafe publish for retrieval: on their CLERC benchmark, reranking a BM25 top-30
  shortlist moved top-1 accuracy from 5% to 18% and top-10 from 38% to 62%.

## Typical configuration

```python
from agent_gantry import AgentGantry, AgentGantryConfig
from agent_gantry.schema.config import RerankerConfig

config = AgentGantryConfig(
    reranker=RerankerConfig(enabled=True, type="cohere", model="rerank-english-v3.0", top_k=5)
)
gantry = AgentGantry(config=config)
await gantry.sync()
```

During retrieval the semantic router first performs vector search, then (if configured) passes the
candidates to the reranker and merges those scores with health weighting before returning
`ScoredTool` results.

## Reranking vs. selecting

A reranker refines an existing shortlist; a *selector* (`../selectors/`) replaces the search that
produced it. `JevReranker` and `JevSelector` ask the same model the same question — the difference
is whether it sees 30 candidates or the whole catalogue. Use the reranker at any catalogue size; the
selector only where the catalogue is small enough to send on every query.

A reranker also fails open: if the provider is unavailable, the vector-search order is returned
untouched, so losing it costs precision rather than tools.

Worth knowing before you reach for one: a reranker can only reorder what search handed it. On a
12-tool catalogue asked for its single best tool, `JevReranker` took top-1 accuracy from 1/5 to 4/5;
the remaining miss was a tool that never entered the shortlist, so no amount of reranking could
promote it. The router fetches `limit * 4` candidates, so a small `limit` makes a narrow shortlist —
raise it if the reranker seems to be missing obvious answers.
