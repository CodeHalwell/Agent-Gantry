# Selection without embeddings (System One)

> Part of the `agent-gantry` skill. Core concepts, debugging and common
> pitfalls are in [`SKILL.md`](../SKILL.md); read those first.

Added in 0.16.0. `SystemOneSelector` replaces embed-then-search entirely: it puts the
catalogue to a System One decision model and asks, per entry, whether it is
relevant. No embedder, no vector store, no `sync()`. `SystemOneReranker` instead
refines an ordinary semantic shortlist. Both are optional, both are off by
default, and both live behind the `system-one` extra. The default model is
TypeSafe's Jev (`jev-latest`); `model=` names a different one.

Renamed in 0.23.0 from `JevSelector` / `JevReranker` / the `jev` extra /
`type="jev"`, since Jev is only the default model. The old names still work,
with a `DeprecationWarning`; write the new ones.

**Only the selector can decline.** It gates each entry at `threshold`, so a
query nothing serves can come back empty. The reranker only *orders* the
shortlist and returns the best `limit` of it however low the scores, so a
reranker-backed setup returns tools for an unanswerable query just as plain
semantic search does. If "return nothing when nothing fits" matters, use the
selector.

```bash
uv add "agent-gantry[system-one]"
export TYPESAFE_API_KEY=...
```

```python
from agent_gantry import AgentGantry, SystemOneSelector

gantry = AgentGantry(selector=SystemOneSelector(threshold=0.4))
# register tools as usual; no sync() needed on the selector path
tools = await gantry.retrieve_tools("refund the customer's order", limit=3)
```

It covers all three catalogues, not just tools: `retrieve_tools`,
`retrieve_skills` and `retrieve_mcp_servers` all consult the selector first.

As a reranker instead, keeping semantic retrieval:

```python
from agent_gantry import AgentGantry, SystemOneReranker

gantry = AgentGantry(reranker=SystemOneReranker())
```

**When each one fits.** A selector sends every candidate on every query, so
cost grows with catalogue size rather than staying flat. That suits tens to a
few hundred entries. Above that, keep semantic retrieval and put the reranker
over its shortlist — only the shortlist is sent, so it works at any catalogue
size. `SelectorConfig.group_after` handles the middle ground by narrowing to
the best namespaces before scoring their members.

**What it buys.** Measured against a properly configured embedder, recall is a
tie. The difference is precision: 1–3 spurious tools against 16 on the same
set. That is the context-window saving, not "finds better tools". A selector
can also answer *none of these*, which top-k cannot — with `limit=3` a vector
search always returns three tools whether or not any of them fit.

The sharper win is distinctions an embedder cannot make, because it compares
text rather than reasoning about fit. Measured live against `jev-1.13.0`, with
a `get_weather` tool described as "get the **current** weather":

| catalogue | query | model score |
|---|---|---|
| current only | "the forecast for Leeds **tomorrow**" | `get_weather` **0.21** → nothing selected |
| current only | "the weather in Leeds **right now**" | `get_weather` **0.98** |
| current + forecast | "the forecast for tomorrow" | `get_forecast` **0.94**, `get_weather` 0.25 |
| current + forecast | "the weather right now" | `get_weather` **0.98**, `get_forecast` 0.35 |

"current weather" and "weather forecast" embed almost on top of each other, so
a vector search hands the agent a tool that cannot answer the question. The
selector declined instead. If a user is choosing between the two approaches, that — not
the token count — is the argument.

**Both fail open**, and neither raises into a caller's retrieval path — but
they fail open differently, and the difference matters if you are choosing
between them.

A provider that is unavailable, rate-limited or slow costs precision and never
the catalogue: the selector reports a fallback and retrieval takes the semantic
path, while the reranker returns the vector-search order untouched, truncated
to `top_k`.

A provider that answers for only *some* of the candidates is not the same case.
The selector treats it as a failed pass and falls back — a candidate that was
never scored is indistinguishable from one scored zero, so ranking the answered
subset would be a confident guess. The reranker keeps the pass: answered
candidates are ordered by probability, unanswered ones keep their search rank
*below* all of them, and the result is truncated to `top_k`. That last step is
the caveat — where more candidates were sent than `top_k`, a high-ranked
candidate the model simply did not answer for can be pushed out of the result.
It is a precision loss rather than a correctness one, and it cannot happen on
the selector path.

**Know which backend answered.** A fallback is not silent. `gantry.retrieve(...)`
returns `backend` (`"selector"` or `"semantic"`), `selector_fallback` and
`selector_fallback_reason`; `RetrievalDecision` (`provider.last_selection`,
`provider.selections`, `dry_run_retrieve()`) carries the same three. In an eval
that configures a selector, assert `selector_fallback is False`, or construct
the gantry with `on_selector_fallback="raise"` so a fallback raises
`SelectorFallbackError` instead of passing on semantic routing. That error also
ends an agent run: `with_semantic_tools`, `GantryContextProvider` and the
framework adapters' per-turn hooks survive other retrieval failures but let
this one through. Production should keep the default and fail open.

A selector *declines* when it answers with nothing (a result: `backend ==
"selector"`, no tools); it *falls back* when it does not answer. The log line
for a fallback is `Selector failed for tool selection (<reason>); using
semantic routing`, at `WARNING` once per distinct reason. The reason to watch
is `catalogue of N exceeds max_candidates=512`: a catalogue over the ceiling
falls back on every call, so raise `max_candidates` or use the reranker.
`retrieve_tools()` and the adapters' `select()` return plain lists and cannot
carry the flag; use `retrieve()` or strict mode there.

Query constraints are applied *before* anything is sent — deprecation,
namespaces, capabilities, sources and circuit-breaker health all go through the
router's own filter (`SemanticRouter.filter_tools()`), so a selector can never
surface a tool the semantic path would hide.

**Debugging.** `SelectionResult` carries `scores` for every candidate, not just
the winners, so you can see exactly what a threshold would keep and cut before
committing to it — see `examples/routing/system_one_threshold_tuning_demo.py`, which
sweeps thresholds over a labelled query set.
