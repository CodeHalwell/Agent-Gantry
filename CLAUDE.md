# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

dotfetch is a **Universal Tool Orchestration Platform** for LLM-based agent systems (Python library, package name `dotfetch`, importable as `dotfetch`). Core philosophy: *Context is precious. Execution is sacred. Trust is earned.*

It solves three problems:
1. **Context window tax** — semantic routing retrieves only the top-k relevant tools per prompt instead of injecting every tool schema (~90% token reduction).
2. **Tool/protocol fragmentation** — register a tool once, emit schemas for OpenAI, Anthropic, Gemini, ~12 agent frameworks, MCP, and A2A.
3. **Operational blindness** — zero-trust execution with policies, capabilities, timeouts, retries, rate limits, circuit breakers, callbacks, and telemetry.

The project was previously published as `agent-gantry` (import `agent_gantry`, class `AgentGantry`); 0.27.0 renamed it to `dotfetch` / `DotFetch` as a clean break with no alias. Repo-internal environment variables (CI and tests, e.g. `DOTFETCH_REQUIRED_FRAMEWORKS`) are `DOTFETCH_*`; the library itself reads none of its own. Do not reintroduce the old names (the GitHub repository and docs URLs still say `Agent-Gantry`; leave those).

The repo also contains an **Astro + React + TypeScript** documentation site (`src/`, `public/`, `astro.config.mjs`) published to GitHub Pages. This is separate from the Python library.

## Commands

### Python library (primary)
```bash
uv sync --all-extras          # set up dev env (preferred; reproducible via uv.lock)
pip install -e ".[dev]"        # alternative minimal dev install
pip install -e ".[all]"        # everything (heavy; many optional deps)

uv run pytest                  # run all tests
pytest tests/test_tool.py      # single file
pytest tests/test_tool.py::TestToolDefinition::test_create_minimal_tool  # single test
pytest --cov=dotfetch --cov-report=html  # coverage

ruff check dotfetch/ examples/ scripts/ tests/       # lint (same scope as CI)
ruff check --fix dotfetch/ examples/ scripts/ tests/ # lint + autofix
ruff format dotfetch/                                # format
```

Note: there is **no mypy config** in `pyproject.toml` and mypy is not a dev dependency. Ruff is the enforced linter/formatter (line length 100, rules `E,F,I,N,W,UP`, ignoring `E501,E402`). Static typing is still expected by convention but not gated in CI.

### Docs site
```bash
npm install
npm run dev       # local docs server (astro dev)
npm run build     # astro check (type-check) + astro build
npm run preview
```

### CLI (`dotfetch`, entry point `dotfetch.cli:main`)
Subcommands: `list`, `search`, `lint`, `sim` (pairwise similarity), `sync`, `serve-mcp` (expose the registry as an MCP server over stdio, Streamable HTTP or SSE), and `install-skill` (installs the bundled Claude Skill, e.g. `dotfetch install-skill --claude`).

## Architecture

Everything funnels through one facade: **`DotFetch`** in `dotfetch/core/fetcher.py`. It composes the subsystems below via dependency injection (constructed in `_create_*` helpers). Public API is re-exported from `dotfetch/__init__.py` — import from the top-level package, not deep modules.

### Core flow
1. **Register** — `@fetcher.register(tags=[...], examples=[...])` (or `add_tool`, `from_modules`, MCP/A2A discovery) records a `ToolDefinition` in the **registry** (`core/registry.py`).
2. **Sync** — `sync()` fingerprints tools, embeds them, and upserts changed ones into the configured **vector store**. `core/sync_manager.py` handles change detection so re-syncs are incremental.
3. **Retrieve** — `retrieve_tools(query, limit=k)` / `retrieve(ToolQuery)` runs the **SemanticRouter** (`core/router.py`): embed query → vector search → optional rerank → MMR diversity → intent filtering. This is the token-saving step.
4. **Convert** — schemas are transcoded to a provider dialect (`adapters/tool_spec/`) for OpenAI/Anthropic/Gemini/etc.
5. **Execute** — `execute(ToolCall)` runs through the **ExecutionEngine** (`core/executor.py`), gated by **security** (`core/security.py`: policies, capabilities), **rate limiting** (`core/rate_limiter.py`), and circuit breakers; emits telemetry events.

### Module map (`dotfetch/`)
- `core/` — facade (`fetcher.py`), `registry.py`, `router.py`, `executor.py`, `security.py`, `rate_limiter.py`, `sync_manager.py`, `chunks.py` (long tool descriptions and skill bodies are indexed a second time in pieces, as pseudo-entries in reserved `__chunks__<namespace>` namespaces; an entry is scored by the best of its own vector and its pieces', a tool's pieces at `CHUNK_SCORE_WEIGHT`), `a2a_manager.py` (register and route to A2A agents, mirroring the MCP control plane), `context.py` (a small conversation-state tracker that builds `ConversationContext`), `invocation.py` (`current_tool_invocation()`: a contextvar through which framework adapters expose their run context, e.g. Pydantic AI `deps`, to tool handlers), and the MCP control plane (`mcp_manager.py`, `mcp_registry.py`, `mcp_router.py`).
- `schema/` — **Pydantic v2** data models: `tool.py` (ToolDefinition, ToolCapability, ToolCost, ToolHealth, ToolSource), `execution.py` (ToolCall, ToolResult, ToolCallEvent), `query.py`, `config.py`, `mcp.py`, `a2a.py`, `skill.py`, `base.py`. **Schema-first**: define/extend a model here before implementing behavior.
- `adapters/` — pluggable backends behind base protocols: `vector_stores/` (in-memory default, LanceDB, Qdrant, Chroma, pgvector), `embedders/` (OpenAI, Nomic, sentence-transformers), `rerankers/` (Cohere, CrossEncoder, SystemOne), `selectors/` (SystemOne — chooses catalogue entries by asking a decision model instead of embedding, as an alternative to the semantic router; TypeSafe's Jev is the default model, and the pre-0.23.0 `Jev*` names remain as deprecated aliases), `executors/` (direct, sandbox, MCP, HTTP, A2A), `tool_spec/` (provider schema transcoding), `llm_client.py`.
- `integrations/` + framework modules at package root (`langchain.py`, `langgraph.py`, `crewai.py`, `llamaindex.py`, `google_adk.py`, `agno.py`, `haystack.py`, `pydantic_ai.py`, `openai_agents.py`, `strands.py`, `dspy.py`) and LLM provider modules (`openai.py`, `anthropic.py`, `gemini.py`, `groq.py`, `mistral.py`, `vertexai.py`). The Microsoft Agent Framework bridge/provider/middleware live in `integrations/`. `semantic_tools.py` provides `with_semantic_tools` + `set_default_fetcher`.
- `providers/` — import tools from external sources.
- `servers/` — MCP and A2A server implementations (`serve_mcp`, `serve_a2a`).
- `observability/` — telemetry, metrics, console logging (`enable_console_logging`).
- `evaluation/` — the retrieval eval (`RetrievalCase`, `evaluate_retrieval`): recall@1, recall@k and decline-correctness for any retriever. `scripts/eval_retrieval.py` runs it over the labelled catalogue in `tests/retrieval/`; `--check` gates on `baseline.json`, as does the test suite. Relevance changes quote their before/after from it.
- `skills/dotfetch/` — the bundled Claude Skill (`SKILL.md` + `references/`), shipped in the wheel via hatch `shared-data` to `share/claude/skills/`.

## Conventions

- **Async by default.** Core operations (`sync`, `retrieve`, `execute`) are coroutines. Tests must use `@pytest.mark.asyncio` — but note `asyncio_mode = "auto"` is set, so the decorator is optional in practice.
- **Pydantic v2** for all data models. Type hints everywhere; `from __future__ import annotations` at the top of modules. Google-style docstrings.
- **Import style** enforced by ruff isort (`I`): stdlib → third-party → local.
- **Naming** (ruff `N`): PascalCase classes, snake_case functions, UPPER_SNAKE constants, `_` prefix for private.
- **Logging hygiene**: the package attaches a `NullHandler` on import and never configures handlers for consumers — output is opt-in via `enable_console_logging()`.
- **Optional dependencies are isolated**: import third-party SDKs lazily (deferred imports inside functions) so `import dotfetch` works without optional extras installed. See `disable_af_instrumentation` in `__init__.py` for the pattern.

## Optional-dependency landscape (read before touching `pyproject.toml`)

`pyproject.toml` carries extensive, dated comments documenting why specific version floors/pins exist — many encode real cross-package resolution conflicts (e.g. `agent-framework` vs `crewai` on opentelemetry, `google-adk` requiring `langgraph<0.4.8`, the quarantined `mistralai` package routed through the OpenAI SDK, `onnxruntime<1.24` override for Python 3.10 wheels). **Do not bump versions or remove pins without reading the adjacent comment** — they explain failures that will recur. The native-tool-adapter frameworks are kept out of the main test matrix; a dedicated CI job installs their extras in an isolated env. First-class frameworks get their own extra with a tested floor, which the same CI job re-runs on exactly: `pydantic-ai` (`pydantic-ai-slim>=2.0.0`, part of `all`), `langchain` / `langgraph` (`>=1.0.0`), `google-adk` (`>=2.6.1`), `crewai` (`>=1.15.22`) and `llamaindex` (`llama-index-core>=0.13.0`), all also covered by `agent-frameworks`, `strands` (`strands-agents>=1.38.0`), `agno` (`agno>=2.6.10`), `haystack` (`haystack-ai>=3.0.0`) and `dspy` (`dspy>=3.5.0`), none in `agent-frameworks` or `all`, and `openai-agents` (`openai-agents>=0.23.1`, *not* part of `all`: it needs `openai>=3` while CrewAI pins `openai<3`, declared as a `[tool.uv] conflicts` pair so the lock resolves both sides). Shared first-class building blocks (error modes, `error_formatter`, `execute_spec`, sticky `carry_over`) live in `integrations/frameworks/native.py`; `DotFetch.requires_confirmation()` lets adapters route gated calls into a framework's own approval flow.

## Tests

- `tests/` mirrors source; fixtures in `tests/conftest.py` (notably `fetcher` for a fresh `DotFetch` and `sample_tools`). Subdirs: `tests/frameworks/` (real-package adapter guards), `tests/integration/`, `tests/examples/`, `tests/test_modules/`.
- `tests/docker-compose.test.yml` spins up backing services (vector stores) for integration tests.
- Phase test files (`test_phase2.py`…`test_phase6_a2a.py`) map to the historical roadmap (execution → routing → adapters → MCP → A2A).

## Adding things — established patterns

- **New adapter** (vector store / embedder / reranker / executor): implement the base protocol in the matching `adapters/<type>/` subdir, add an optional-deps entry in `pyproject.toml`, mirror an existing adapter's tests, document in `adapters/<type>/README.md`.
- **New framework integration**: add a module wrapping `DotFetch`, add tests (`tests/frameworks/` for real-API guards), add an example under `examples/agent_frameworks/`, update `integrations/README.md`.
- **New LLM provider**: add transcoding in `adapters/tool_spec/`, wire into `with_semantic_tools`, test in `tests/test_llm_sdk_compatibility.py`, add an example under `examples/llm_integration/`.
- **New schema field**: edit the Pydantic model in `schema/`, add backward-compat/migration handling, update `tests/test_tool.py` (or relevant).

## Repo notes

- Version of record is `pyproject.toml` / `dotfetch/__init__.py` `__version__` (currently `0.27.0`). Keep `package.json`, `package-lock.json`, `uv.lock`, `README.md` and the dated changelog in sync too; `scripts/check_release.py` validates release metadata.
- `.jules/` holds accumulated performance-optimization learnings (e.g. vectorized MMR, fast token matching) — useful context when optimizing the router or serialization paths.
- `CHANGELOG.md` should be updated for user-facing changes.
- Releasing/publishing: see `RELEASING.md` and `PUBLISHING.md`.
