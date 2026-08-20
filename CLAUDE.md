# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Agent-Gantry is a **Universal Tool Orchestration Platform** for LLM-based agent systems (Python library, package name `agent-gantry`, importable as `agent_gantry`). Core philosophy: *Context is precious. Execution is sacred. Trust is earned.*

It solves three problems:
1. **Context window tax** — semantic routing retrieves only the top-k relevant tools per prompt instead of injecting every tool schema (~90% token reduction).
2. **Tool/protocol fragmentation** — register a tool once, emit schemas for OpenAI, Anthropic, Gemini, ~12 agent frameworks, MCP, and A2A.
3. **Operational blindness** — zero-trust execution with policies, capabilities, timeouts, retries, rate limits, circuit breakers, callbacks, and telemetry.

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
pytest --cov=agent_gantry --cov-report=html  # coverage

ruff check agent_gantry/       # lint
ruff check --fix agent_gantry/ # lint + autofix
ruff format agent_gantry/      # format
```

Note: there is **no mypy config** in `pyproject.toml` and mypy is not a dev dependency, despite older docs (CONTRIBUTING.md, copilot-instructions.md) mentioning it. Ruff is the enforced linter/formatter (line length 100, rules `E,F,I,N,W,UP`, ignoring `E501,E402`). Static typing is still expected by convention but not gated in CI.

### Docs site
```bash
npm install
npm run dev       # local docs server (astro dev)
npm run build     # astro check (type-check) + astro build
npm run preview
```

### CLI (`agent-gantry`, entry point `agent_gantry.cli:main`)
Subcommands: `list`, `search`, `lint`, `sim` (pairwise similarity), `sync`, and `install-skill` (installs the bundled Claude Skill, e.g. `agent-gantry install-skill --claude`).

## Architecture

Everything funnels through one facade: **`AgentGantry`** in `agent_gantry/core/gantry.py`. It composes the subsystems below via dependency injection (constructed in `_create_*` helpers). Public API is re-exported from `agent_gantry/__init__.py` — import from the top-level package, not deep modules.

### Core flow
1. **Register** — `@gantry.register(tags=[...], examples=[...])` (or `add_tool`, `from_modules`, MCP/A2A discovery) records a `ToolDefinition` in the **registry** (`core/registry.py`).
2. **Sync** — `sync()` fingerprints tools, embeds them, and upserts changed ones into the configured **vector store**. `core/sync_manager.py` handles change detection so re-syncs are incremental.
3. **Retrieve** — `retrieve_tools(query, limit=k)` / `retrieve(ToolQuery)` runs the **SemanticRouter** (`core/router.py`): embed query → vector search → optional rerank → MMR diversity → intent filtering. This is the token-saving step.
4. **Convert** — schemas are transcoded to a provider dialect (`adapters/tool_spec/`) for OpenAI/Anthropic/Gemini/etc.
5. **Execute** — `execute(ToolCall)` runs through the **ExecutionEngine** (`core/executor.py`), gated by **security** (`core/security.py`: policies, capabilities), **rate limiting** (`core/rate_limiter.py`), and circuit breakers; emits telemetry events.

### Module map (`agent_gantry/`)
- `core/` — facade (`gantry.py`), `registry.py`, `router.py`, `executor.py`, `security.py`, `rate_limiter.py`, `sync_manager.py`, `context.py` (contextvars-based async-safe state), and the MCP control plane (`mcp_manager.py`, `mcp_registry.py`, `mcp_router.py`).
- `schema/` — **Pydantic v2** data models: `tool.py` (ToolDefinition, ToolCapability, ToolCost, ToolHealth, ToolSource), `execution.py` (ToolCall, ToolResult, events), `query.py`, `config.py`, `mcp.py`, `a2a.py`, `skill.py`, `base.py`. **Schema-first**: define/extend a model here before implementing behavior.
- `adapters/` — pluggable backends behind base protocols: `vector_stores/` (in-memory default, LanceDB, Qdrant, Chroma, pgvector), `embedders/` (OpenAI, Nomic, sentence-transformers), `rerankers/` (Cohere, CrossEncoder), `executors/` (direct, sandbox, MCP, HTTP, A2A), `tool_spec/` (provider schema transcoding), `llm_client.py`.
- `integrations/` + framework modules at package root (`langchain.py`, `langgraph.py`, `crewai.py`, `autogen.py`, `llamaindex.py`, `semantic_kernel.py`, `google_adk.py`, `agno.py`, `haystack.py`, `pydantic_ai.py`, `openai_agents.py`, `smolagents.py`) and LLM provider modules (`openai.py`, `anthropic.py`, `gemini.py`, `groq.py`, `mistral.py`, `vertexai.py`). The Microsoft Agent Framework bridge/provider/middleware live in `integrations/`. `semantic_tools.py` provides `with_semantic_tools` + `set_default_gantry`.
- `providers/` — import tools from external sources.
- `servers/` — MCP and A2A server implementations (`serve_mcp`, `serve_a2a`).
- `observability/` — telemetry, metrics, console logging (`enable_console_logging`).
- `skills/agent-gantry/` — the bundled Claude Skill (`SKILL.md` + `references/`), shipped in the wheel via hatch `shared-data` to `share/claude/skills/`.

## Conventions

- **Async by default.** Core operations (`sync`, `retrieve`, `execute`) are coroutines. Tests must use `@pytest.mark.asyncio` — but note `asyncio_mode = "auto"` is set, so the decorator is optional in practice.
- **Pydantic v2** for all data models. Type hints everywhere; `from __future__ import annotations` at the top of modules. Google-style docstrings.
- **Import style** enforced by ruff isort (`I`): stdlib → third-party → local.
- **Naming** (ruff `N`): PascalCase classes, snake_case functions, UPPER_SNAKE constants, `_` prefix for private.
- **Logging hygiene**: the package attaches a `NullHandler` on import and never configures handlers for consumers — output is opt-in via `enable_console_logging()`.
- **Optional dependencies are isolated**: import third-party SDKs lazily (deferred imports inside functions) so `import agent_gantry` works without optional extras installed. See `disable_af_instrumentation` in `__init__.py` for the pattern.

## Optional-dependency landscape (read before touching `pyproject.toml`)

`pyproject.toml` carries extensive, dated comments documenting why specific version floors/pins exist — many encode real cross-package resolution conflicts (e.g. `agent-framework` vs `crewai` on opentelemetry, `google-adk` requiring `langgraph<0.4.8`, the quarantined `mistralai` package routed through the OpenAI SDK, `onnxruntime<1.24` override for Python 3.10 wheels). **Do not bump versions or remove pins without reading the adjacent comment** — they explain failures that will recur. The native-tool-adapter frameworks (pydantic-ai, openai-agents, smolagents, haystack-ai, agno) are intentionally *not* a project extra; they're installed standalone by a dedicated CI job because they can't co-resolve with the `agent-frameworks` extra.

## Tests

- `tests/` mirrors source; fixtures in `tests/conftest.py` (notably `gantry` for a fresh `AgentGantry` and `sample_tools`). Subdirs: `tests/frameworks/` (real-package adapter guards), `tests/integration/`, `tests/examples/`, `tests/test_modules/`.
- `tests/docker-compose.test.yml` spins up backing services (vector stores) for integration tests.
- Phase test files (`test_phase2.py`…`test_phase6_a2a.py`) map to the historical roadmap (execution → routing → adapters → MCP → A2A).

## Adding things — established patterns

- **New adapter** (vector store / embedder / reranker / executor): implement the base protocol in the matching `adapters/<type>/` subdir, add an optional-deps entry in `pyproject.toml`, mirror an existing adapter's tests, document in `adapters/<type>/README.md`.
- **New framework integration**: add a module wrapping `AgentGantry`, add tests (`tests/frameworks/` for real-API guards), add an example under `examples/agent_frameworks/`, update `integrations/README.md`.
- **New LLM provider**: add transcoding in `adapters/tool_spec/`, wire into `with_semantic_tools`, test in `tests/test_llm_sdk_compatibility.py`, add an example under `examples/llm_integration/`.
- **New schema field**: edit the Pydantic model in `schema/`, add backward-compat/migration handling, update `tests/test_tool.py` (or relevant).

## Repo notes

- Version of record is `pyproject.toml` / `agent_gantry/__init__.py` `__version__` (currently `0.11.0`). The docs `package.json` and `README.md` may lag — `tests/test_version_consistency.py` guards consistency where it matters.
- `.jules/` holds accumulated performance-optimization learnings (e.g. vectorized MMR, fast token matching) — useful context when optimizing the router or serialization paths.
- `CHANGELOG.md` should be updated for user-facing changes.
- Releasing/publishing: see `RELEASING.md` and `PUBLISHING.md`.
