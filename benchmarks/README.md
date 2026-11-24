# Benchmarks

Runnable performance and quality benchmarks for Agent-Gantry.

| Script | Measures |
|---|---|
| `benchmark_tool_selection.py` | Semantic selection quality picking **top-3 of 50 tools** — top-1 accuracy, hit@3, MRR, latency p50/p95, token savings. |
| `benchmark_multi_turn.py` | **Multi-turn re-selection** in a single agent run: select a tool, read its result, pivot to a different tool driven by the new sub-task. |
| `benchmark_memory_store.py` | `InMemoryVectorStore.search` latency across registry sizes (10–5000 tools). |
| `benchmark_pgvector.py` | PGVector store add/search throughput (mocked pool). |
| `benchmark_skills.py` | Anthropic skills client message-build throughput. |
| `framework_argument_normalization.py` | Shared framework argument normalization latency and peak allocations; no SDKs or model calls. |

## Running

```bash
pip install -e .                 # core deps (numpy, pydantic, …)
pip install sentence-transformers  # for a real embedder (recommended)

python benchmarks/benchmark_tool_selection.py --embedder sentence-transformers --limit 3
python benchmarks/benchmark_multi_turn.py     --embedder sentence-transformers --limit 3
python benchmarks/benchmark_memory_store.py
python benchmarks/framework_argument_normalization.py --iterations 200
```

The two selection benchmarks are embedder-agnostic (`--embedder auto|sentence-transformers|nomic|simple`).
`SimpleEmbedder` is a hash-based toy for offline smoke-runs only — use a real embedder for meaningful
accuracy numbers.

The framework normalization benchmark needs only core dependencies. On Python
3.12, a 10,000-entry integer mapping took about 4.42 ms before the change and
3.36 ms after it; peak temporary allocations fell from about 311 KB to 112 bytes
for an unchanged mapping. A free-form object can skip traversal entirely.
These are local microbenchmark results for argument normalization, not estimates
of end-to-end agent latency. Payloads that need conversion still allocate their
changed containers.
It is a manual diagnostic, with no timing gate in CI: runner load and Python
versions affect these small measurements. Compare before and after on the same machine.

## Retrieval eval

The scripts above time retrieval or score it on 50 hand-picked tools. The eval
is the one to use for a relevance change: a generated catalogue of 345 tools
(most of them near-duplicates of each other, a sixth of them converted from
three invented MCP servers), 20 skills and 12 MCP servers, with labelled
queries for each and a `kind` on every query saying what it tests.

```bash
python scripts/eval_retrieval.py                                  # hash embedder, as CI runs it
python scripts/eval_retrieval.py --check                          # fail if worse than the baseline
python scripts/eval_retrieval.py --embedder sentence-transformers --by-kind --misses
```

`--check` compares against `tests/retrieval/baseline.json`, and the test suite
runs the same comparison. That baseline is measured with `SimpleEmbedder`
because it is deterministic; it catches a pipeline that broke, not an embedding
that got worse. A change meant to move the numbers rewrites it with
`--update-baseline` in the same commit. To score your own registry, build
`RetrievalCase`s and call `agent_gantry.evaluation.evaluate_retrieval`.

`all-MiniLM-L6-v2`, in-memory store, `k=5`, `score_threshold=0.0`, as of the
commit that added the eval:

| Modality (cases) | recall@1 | recall@5 | declined when nothing fits |
|---|---|---|---|
| tools (59) | 0.765 | 0.873 | 0.000 |
| skills (14) | 0.917 | 1.000 | 0.000 |
| MCP servers (13) | 0.727 | 0.909 | 0.000 |

Tools by kind of query, top-1: paraphrase 21/30, identifier 8/8, parameter name
3/6, MCP server name 5/5, and 2/2 where the answer sits past character 2000 of
a long MCP description. Skills whose answer is only in the body: 4/4. MCP
servers described vaguely, where only their tools say what they do: 2/3. Nothing declines at a threshold of 0.0; pass `--score-threshold`
to see what yours does, or `--calibrate` to have one found from the labelled
queries. Here that gives tools a threshold of 0.26, at which all 8 unanswerable
queries are declined and no right answer is lost, but from a window only 0.01
wide; skills and MCP servers have no threshold that separates the two.

Tried on the same eval and not built: fusing lexical (BM25, k1=1.5, b=0.75,
over each tool's searchable text) with vector results by reciprocal rank
(k=60). Over the 51 answerable tool queries, vector alone gets 39 top-1 and 45
in the top five; lexical alone 25 and 31; the fusion 29 and 44. It loses eight
paraphrases at top-1 (21/30 to 13/30) to gain two parameter-name queries in
the top five. A lexical signal would have to be weighted well below the
vector one to be worth adding.

These are one synthetic catalogue and a few dozen queries per modality. They
are for comparing a change with what came before it, not for quoting as the
library's accuracy.

## Reference results

`sentence-transformers/all-MiniLM-L6-v2`, in-memory store, `score_threshold=0.0`:

- **top-3 of 50:** top-1 0.889, hit@3 1.000, MRR 0.94, p50 ~24ms, token savings ~93%.
- **multi-turn (7 turns):** gold-in-top3 1.000, 7 distinct tools, pivots every turn.
- **store search (after NumPy vectorization):** ~0.05ms at 50 tools, ~0.6ms at 1000 tools
  (≈36–59× faster than the previous pure-Python loop).

See `docs/REVIEW_AGENT_FRAMEWORKS_2026-06-14.md` for the full review and methodology, including the
`score_threshold=0.5` default that silently drops correct tools if not overridden.
