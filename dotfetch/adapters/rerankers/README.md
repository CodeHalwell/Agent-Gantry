# dotfetch/adapters/rerankers

Rerankers take the initial vector search results and reorder them using richer relevance signals.
They are optional but can improve precision on ambiguous or verbose queries by considering more than
cosine similarity alone.

## Modules

- `base.py`: Declares the `RerankerAdapter` protocol; implements small helpers for packaging inputs
  and outputs.
- `cohere.py`: Cohere ReRank implementation that scores tool descriptions (and docstrings) using
  Cohere's hosted models.
- `cross_encoder.py`: Local sentence-transformers cross-encoder, no API calls.
- `system_one.py`: `SystemOneReranker`, backed by a "System One" decision model —
  [TypeSafe's Jev](https://docs.typesafe.ai) by default, or whichever model `model=` names. (Named
  `JevReranker` before 0.23.0; that name still imports, with a `DeprecationWarning`.) Asks one yes/no
  question per candidate — "would this tool help?" — and orders by the returned probability. This is
  the recipe TypeSafe publish for retrieval: on their CLERC benchmark, reranking a BM25 top-30
  shortlist moved top-1 accuracy from 5% to 18% and top-10 from 38% to 62%.

## Typical configuration

```python
from dotfetch import DotFetch, DotFetchConfig
from dotfetch.schema.config import RerankerConfig

config = DotFetchConfig(
    reranker=RerankerConfig(enabled=True, type="cohere", model="rerank-english-v3.0", top_k=5)
)
fetcher = DotFetch(config=config)
await fetcher.sync()
```

During retrieval the semantic router first performs vector search, then (if configured) passes the
candidates to the reranker and merges those scores with health weighting before returning
`ScoredTool` results.

## Skills and MCP servers

A reranker is written against tools, but it is not limited to them:

```python
skills = await fetcher.retrieve_skills("how do I refund a customer", rerank=True)
servers = await fetcher.retrieve_mcp_servers("charge a card", rerank=True)
```

Both rescore the best candidates with the fetcher's reranker and are off unless asked for, so a
reranker configured for tools does not start charging for the other two. Underneath,
`rerank_texts(reranker, query, texts, top_k)` in `base.py` dresses each text as a tool-shaped
document whose searchable text is the text exactly, so any reranker works as it stands. One that
defines `async def rerank_texts(query, texts, top_k) -> list[tuple[int, float]]` is called
directly instead; `CrossEncoderReranker` does.

## Reranking vs. selecting

A reranker refines an existing shortlist; a *selector* (`../selectors/`) replaces the search that
produced it. `SystemOneReranker` and `SystemOneSelector` ask the same model the same question — the difference
is whether it sees 30 candidates or the whole catalogue. Use the reranker at any catalogue size; the
selector only where the catalogue is small enough to send on every query.

**A reranker cannot decline.** It orders the shortlist and returns the best `limit` of it, however
low the scores: `SystemOneReranker`'s probabilities are never compared with a threshold. Asked
something no tool can serve, a reranker-backed setup still returns tools, exactly as semantic search
alone does. `SystemOneSelector` gates each entry at its `threshold`, so it is the one backend that
can return nothing — "rerank" sounds as though it brings the model's judgement about fit with it,
and it brings only the ordering. See
[Selector or reranker](../selectors/README.md#selector-or-reranker-only-one-of-them-can-say-no).

A reranker also fails open: if the provider is unavailable, the vector-search order is returned
untouched, so losing it costs precision rather than tools.

Worth knowing before you reach for one: a reranker can only reorder what search handed it. The
router fetches `max(limit * 4, 32)` candidates, so the shortlist is never narrower than 32, but a
tool that did not make it cannot be promoted no matter how good the reranker is. Raise `limit` if
the reranker seems to be missing obvious answers on a very large catalogue.

Worth knowing *first*, though: on the catalogue in `../selectors/README.md`, adding `examples=[...]`
to the tools improved the default embedder far more than any reranker or model upgrade did. Fix the
metadata before you add a network round-trip to every retrieval.
