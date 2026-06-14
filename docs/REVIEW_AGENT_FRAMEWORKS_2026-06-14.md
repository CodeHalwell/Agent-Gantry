# Agent-Gantry — Agent-Framework Functionality & Performance Review

*Date: 2026-06-14 · Scope: framework compatibility, semantic tool selection quality,
performance, and multi-turn re-selection · All numbers below were measured in this
repo with `sentence-transformers/all-MiniLM-L6-v2` unless noted.*

This review answers five questions:

1. Current status of functionality with agent frameworks.
2. What it takes to be *fully* functional with every agent framework.
3. Performance improvements (one is implemented and measured here).
4. Benchmark of semantic tool selection — top-3 out of 50 tools.
5. Multi-turn tool selection within a single agent run (pivot to a different tool).

Two reusable benchmarks and one performance fix were added as part of this review:

- `benchmarks/benchmark_tool_selection.py` — top-3-of-50 retrieval quality + latency.
- `benchmarks/benchmark_multi_turn.py` — direction-changing selection within one run.
- `benchmarks/benchmark_memory_store.py` — store search latency (before/after the fix).
- `agent_gantry/adapters/vector_stores/memory.py` — NumPy-vectorized cosine search.

---

## 1. Current status with agent frameworks

| Framework | Integration depth | Multi-turn / per-call | Where |
|---|---|---|---|
| **Microsoft Agent Framework** | **Native, deep** — `ContextProvider`, middleware, bridge, workflow builders | ✅ `query_strategy="per_call"` + chat middleware | `integrations/agent_framework_provider.py`, `_bridge.py`, `_middleware.py` |
| **Anthropic (Claude)** | Dedicated clients: auto-retrieve tools, thinking modes, skills | ✅ per `create_message` | `integrations/anthropic_features.py`, `anthropic_skills.py` |
| **Any LLM SDK** (OpenAI, Azure, Groq, Mistral, OpenRouter, Gemini) | `@with_semantic_tools` decorator with dialect transcoding (`openai`/`anthropic`/`gemini`) | ✅ retrieval re-runs per decorated call | `integrations/semantic_tools.py` |
| **LangChain, LangGraph, AutoGen, CrewAI, LlamaIndex, Semantic Kernel, Google ADK, Strands** | **Shallow** — `fetch_framework_tools()` emits OpenAI-style *schemas*; binding to native tool objects is left to example code | ⚠️ Only if you re-call retrieval yourself each turn | `integrations/framework_adapters.py` + `examples/agent_frameworks/*` |
| **OpenAI Agents SDK, Pydantic AI, Haystack, Smolagents, DSPy, Agno/Phidata** | **None** — no adapter, no example | ❌ | — |

**Key observations**

- The *only* deep, "drop-in" integration is **Microsoft Agent Framework**. It is genuinely
  first-class: per-run and per-call retrieval, `last_selection` introspection, `dry_run_retrieve`,
  workflow/handoff/sequential flow-through, and security/observability middleware.
- `framework_adapters.fetch_framework_tools()` (`framework_adapters.py:35`) validates the framework
  name then, for every non-AF framework, returns the **same** `result.to_openai_tools()` shape. It is
  a schema emitter, not an integration — the user still has to wrap those schemas into LangChain
  `StructuredTool` / LlamaIndex `FunctionTool` / CrewAI `BaseTool` and re-call it each turn manually.
- **Dynamic MCP server selection** is explicitly a **preview/placeholder** (README L621-624); the
  semantic-search-over-servers path is not fully implemented.

**Two correctness traps found while benchmarking** (both reproduce in the repo's own tests):

1. **`score_threshold` default is `0.5` and silently drops correct tools.**
   `ToolQuery.score_threshold` defaults to `0.5` (`schema/query.py`), and `retrieve_tools()`
   (`gantry.py:893`) and `fetch_framework_tools()` (`framework_adapters.py:41`) inherit it — while the
   documented `GantryContextProvider` default is `0.0`. With any real-but-modest embedder (MiniLM
   cosine scores sit ~0.2–0.45), the default filters out the right tool. Measured impact below:
   the **same** benchmark scores **30.6%** hit@3 at threshold 0.5 vs **100%** at 0.0.
2. **A `**kwargs`-only tool becomes un-executable.** Registering `def f(**kwargs)` makes the schema
   builder emit `kwargs` as a *required* property, so `execute(arguments={})` fails with
   `ValidationError: Missing required parameter: kwargs`.

---

## 2. Making it fully functional with every framework

Priority-ordered, concrete work:

**P0 — fix the traps that already break shipped paths**
- Make `0.0` the default `score_threshold` for `retrieve_tools()` / `fetch_framework_tools()` (or at
  least clamp-warn when a non-Simple embedder returns an empty set purely due to threshold). This is
  the single highest-leverage change — it currently makes large registries look broken.
- Treat `**kwargs` / `**VAR_KEYWORD` params as optional (not required) in the schema builder.

**P1 — turn the shallow adapters into real ones.** Each framework needs a *native tool object* exporter
plus a *refresh hook*, mirroring the AF provider. Suggested shape:

```python
# A universal exporter protocol, one thin module per framework.
class GantryToolset:
    async def for_langchain(self, query, *, limit=3) -> list[StructuredTool]: ...
    async def for_llamaindex(self, query, *, limit=3) -> list[FunctionTool]: ...
    async def for_crewai(self, query, *, limit=3) -> list[BaseTool]: ...
    async def for_pydantic_ai(self, query, *, limit=3) -> list[Tool]: ...
    async def for_openai_agents(self, query, *, limit=3) -> list[FunctionTool]: ...
    async def for_smolagents(self, query, *, limit=3) -> list[Tool]: ...
```

Each wrapper closes over `gantry.execute(ToolCall(...))` and carries the JSON-schema so the host
framework can call it. This is the missing piece for OpenAI Agents SDK, Pydantic AI, Smolagents,
Haystack, and Agno.

**P2 — make multi-turn first-class everywhere, not just AF.** The per-call machinery already exists
generically (`query/strategies.py`: `last_tool_result`, `fallback_chain`, `keyword_focused`, plus the
router's `tools_already_used` penalty). Expose a framework-agnostic `refresh(messages)` callback so
LangGraph/CrewAI/etc. can re-rank on every step the way the AF middleware does.

**P3 — finish dynamic MCP server selection** (currently placeholder) and add a **conformance test
matrix**: one parametrized test per framework that registers 10 tools, runs a 3-turn pivot, and
asserts the right tool is bound each turn. That converts "we have an example" into "we guarantee it".

---

## 3. Performance improvements

### Implemented and measured here: vectorized in-memory search

`InMemoryVectorStore.search()` computed cosine similarity in a **pure-Python loop**, one dot-product
per tool, over list-typed embeddings (`memory.py`, old `_cosine_similarity`). It now builds a cached,
L2-normalized `(n, d)` NumPy matrix and scores the whole registry with a single matmul
(`query · M.T`). The cache invalidates on `add_tools`/`delete` and rebuilds lazily on the next search.
NumPy is already a core dependency, so this adds no new requirement.

`benchmarks/benchmark_memory_store.py`, dim=384, 200 searches each:

| Tools | Before (ms/search) | After (ms/search) | Speedup |
|------:|-------------------:|------------------:|--------:|
| 10    | 0.350 | 0.030 | ~12× |
| 50    | 1.705 | 0.047 | **~36×** |
| 100   | 3.414 | 0.070 | ~48× |
| 500   | 17.07 | 0.301 | ~57× |
| 1000  | 35.10 | 0.599 | ~59× |
| 5000  | 181.9 | 5.15  | ~35× |

At the review's headline scenario (50 tools) search drops from **1.7ms to 0.05ms**. The full targeted
test suite (router/retrieval/vector/memory/performance) stays green; the change is behavior-preserving.

### Further opportunities (not yet implemented)

- **Vectorize the router's signal scoring.** The router over-fetches `limit*4` candidates
  (`router.py:267`) and then computes `RoutingSignals` per candidate in Python. With the matrix already
  built, the semantic component and penalties can be computed array-wise.
- **Cache the query embedding within a single run.** In a multi-step agent loop the same derived query
  is often embedded repeatedly; memoize per turn.
- **Default-on `CachedEmbedder` for paid embedders** to kill cold-start re-embedding cost (the class
  exists; it is opt-in today).
- **Optional ANN index (hnswlib/FAISS) for the in-memory store** beyond ~10k tools — though the
  vectorized brute force is already sub-6ms at 5k, so this is low priority.

---

## 4. Benchmark — best top-3 out of 50 tools

`benchmarks/benchmark_tool_selection.py`: 50 tools across 10 domains (weather, finance, email/messaging,
files, database, web/NLP, maps/travel, math/data, devops, productivity), 36 labeled natural-language
queries, retrieve **top-3**, embedder `all-MiniLM-L6-v2`, in-memory store.

```
top1_accuracy ............. 0.889   (32/36 ranked the gold tool #1)
hit@3 ..................... 1.000   (gold tool in top-3 on every query)
MRR ....................... 0.940
latency p50 / p95 (ms) .... 23.6 / 25.8   (includes query embedding + search)
token_savings (top3 vs 50)  0.934   (~93% fewer tool-schema tokens)
```

Notes:
- With the library's **default** `score_threshold=0.5`, the identical run collapses to **30.6%** hit@3 —
  this is the threshold trap from §1, and the reason the benchmark passes `score_threshold=0.0`.
- `SimpleEmbedder` is a hash-based **toy** (its own docstring says so); it is unsuitable for real
  routing and should never be the production default. The benchmark auto-selects the best available
  embedder and falls back to Simple only so the harness still runs offline.

---

## 5. Multi-turn tool selection within one agent run

`benchmarks/benchmark_multi_turn.py` simulates a **single agent run** of 7 turns whose intent lurches
across domains, re-ranking the **whole 50-tool registry** on every turn (per-call retrieval), executing
the chosen tool for real, then feeding the result back into the next turn's query:

```
turn 1  get_current_weather   ↪  turn 2  search_flights   ↪  turn 3  book_hotel
↪ turn 4  send_email          ↪  turn 5  convert_currency ↪  turn 6  create_todo
↪ turn 7  (summarize_text in top-3; model picked send_sms)
```

```
gold_in_top3 ................. 1.000   (right tool surfaced every turn)
pick_accuracy ................ 0.857   (6/7 top-1 picks correct)
distinct_pivots .............. 7       (a different tool every single turn)
distinct_tools_used .......... 7
selection_pivoted_every_turn . true
```

This demonstrates exactly the requested behavior: the agent selects a tool, reads its result, then
**changes direction and selects a completely different tool** driven purely by the semantics of the new
sub-task — not by a fixed pre-bound tool list. The enabling machinery:

- **Per-call retrieval**: `GantryContextProvider(query_strategy="per_call")` for Microsoft AF, or, for
  any framework, re-calling `gantry.retrieve(...)` each turn (as the benchmark does directly).
- **Query generators** (`agent_gantry/query/strategies.py`): `last_user_text`, `last_tool_result`,
  `fallback_chain(...)`, `keyword_focused(...)` derive the next turn's retrieval query from the evolving
  conversation, so the query that drives selection moves with the task.
- **Forward-progress penalty**: the router applies an `already_used_penalty` to tools in
  `tools_already_used` (`router.py:428`), nudging the agent off tools it has already run.

---

## Summary scorecard

| Area | Status |
|---|---|
| Microsoft Agent Framework | ✅ Production-grade, multi-turn native |
| Generic LLM SDKs (OpenAI/Anthropic/Gemini/…) | ✅ Solid via decorator + dialect transcoding |
| LangChain/LlamaIndex/CrewAI/etc. | ⚠️ Schema-only; needs native tool-object adapters |
| OpenAI Agents SDK / Pydantic AI / Smolagents / Haystack / Agno | ❌ Not yet supported |
| Top-3-of-50 selection quality | ✅ 100% hit@3, 0.94 MRR (MiniLM) — once threshold trap avoided |
| Multi-turn pivot in one run | ✅ Demonstrated, 7/7 distinct tools |
| In-memory search performance | ✅ 36–59× faster (vectorized) |
| `score_threshold=0.5` default | ❌ Silently drops correct tools — fix recommended |
| `**kwargs` tools | ❌ Become un-executable — fix recommended |
| Dynamic MCP server selection | ⚠️ Preview/placeholder |
