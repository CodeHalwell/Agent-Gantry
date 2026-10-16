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

## Reference results

`sentence-transformers/all-MiniLM-L6-v2`, in-memory store, `score_threshold=0.0`:

- **top-3 of 50:** top-1 0.889, hit@3 1.000, MRR 0.94, p50 ~24ms, token savings ~93%.
- **multi-turn (7 turns):** gold-in-top3 1.000, 7 distinct tools, pivots every turn.
- **store search (after NumPy vectorization):** ~0.05ms at 50 tools, ~0.6ms at 1000 tools
  (≈36–59× faster than the previous pure-Python loop).

See `docs/REVIEW_AGENT_FRAMEWORKS_2026-06-14.md` for the full review and methodology, including the
`score_threshold=0.5` default that silently drops correct tools if not overridden.
