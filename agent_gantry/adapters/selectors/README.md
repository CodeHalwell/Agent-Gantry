# agent_gantry/adapters/selectors

Selectors are the alternative to semantic matching. Instead of embedding the query and searching a
vector store, a selector puts the catalogue to a decision model and asks, per entry, whether it is
relevant. No embeddings, no vector store, no sync step.

## Modules

- `base.py`: Declares the `SelectorAdapter` protocol.
- `jev.py`: `JevSelector`, backed by [TypeSafe's Jev](https://docs.typesafe.ai) — a "System One"
  model that returns typed decisions with calibrated probabilities rather than text.

## When to use one

| | Semantic routing | Selector |
|---|---|---|
| Cost per query | Flat in catalogue size | Linear in catalogue size |
| Setup | Embedder + vector store + sync | An API key |
| Recall of paraphrases | Good | Good |
| Understands negation, scoping, "not X" | Poor | Good |
| Catalogue ceiling | Millions | Tens to a few hundred |

The ceiling is the thing to plan around. Every candidate is sent as input on every query, so a
selector re-reads the catalogue each time. That is affordable because a decision model is priced
roughly two orders of magnitude below a frontier LLM, but it is not sub-linear the way a vector
index is. Above a few hundred entries, prefer semantic retrieval with `JevReranker` over its
shortlist — same model, same question, applied to 30 candidates rather than 3,000.

## Configuration

```python
from agent_gantry import AgentGantry
from agent_gantry.schema.config import AgentGantryConfig, SelectorConfig

config = AgentGantryConfig(
    selector=SelectorConfig(enabled=True, threshold=0.3),
)
gantry = AgentGantry(config=config)  # reads TYPESAFE_API_KEY
```

Or inject one directly, which also lets you tune the question:

```python
from agent_gantry import AgentGantry, JevSelector

gantry = AgentGantry(
    selector=JevSelector(
        question="Would this tool help accomplish the request? Answer about this tool only.",
        threshold=0.3,
    )
)
```

The selector covers all three catalogues — tools (`retrieve`/`retrieve_tools`), Agent Skills
(`retrieve_skills`) and MCP servers (`retrieve_mcp_servers`).

## How it behaves

**It fails open.** A selector never raises because a provider was unavailable, rate-limited or slow.
It reports `fallback=True`, and retrieval quietly takes the semantic path instead. A selection layer
that fails closed hands the agent an empty tool list, which is worse than an unranked one.

**It respects the query's hard constraints.** Deprecation, namespaces, capabilities, sources and
circuit-breaker health are applied by the router's own `filter_tools` before anything is sent, so a
selector can never surface a tool the semantic path would hide.

**It skips the work it does not need.** When the selector answers, `retrieve()` does not sync, does
not embed the query and does not search. `RetrievalResult.selection_time_ms` is set and the
embedding and search timings are `0.0`.

**It narrows before it scores.** Above `group_after` entries the selection runs in two passes —
score the groups (namespace, or originating MCP server), then score only the members of the best
`max_groups`. The group pass is a budget device and is deliberately *not* gated by the threshold;
the threshold gates members, so one cautious answer about a namespace cannot zero out the catalogue
before any individual tool has been looked at.

## Tuning

- **`threshold`** is the knob that matters. 0.30 is TypeSafe's own default for this shape of task.
  Raise it if the agent is being handed tools it does not use; lower it if it is missing ones.
  `SelectionResult.scores` reports *every* candidate's probability, not just the winners, so you can
  see what a threshold is cutting before you change it.
- **`question`** is read literally. jev-1.13 "answers the question you wrote, not the one you
  meant" — scoping words, negations and implied conditions are taken at face value.
- **Descriptions do the work.** A selector reads each entry's name, description and tags, and
  nothing else. It has no embedding to fall back on, so a vague description costs more here than it
  does under semantic search.

## Caveats worth knowing

- **Tool descriptions from MCP servers are third-party text.** jev-1.13 "does not treat [adversarial
  content] as hostile by default", so a hostile description can bias selection. This sits alongside
  Gantry's zero-trust execution model rather than inside it: selection decides what an agent *sees*,
  while policies and capabilities still decide what it may *run*.
- **The SDK is early access.** `typesafe-sdk` shipped on 2026-09-15; the pinned floor and the
  co-resolution notes are in `pyproject.toml` under the `jev` extra.

## Installing

```bash
pip install agent-gantry[jev]
export TYPESAFE_API_KEY=...
```
