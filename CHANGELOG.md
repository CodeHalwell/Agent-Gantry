# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`AnthropicFeatures.thinking_display`** — new optional field (`"summarized"` | `"omitted"`)
  that controls thinking visibility in the response. `"summarized"` condenses the thinking
  block; `"omitted"` hides it but preserves the signature for multi-turn continuity.
  Exposed through `create_anthropic_client(thinking_display=...)`.
  *Source: https://platform.claude.com/docs/en/api/messages (thinking parameter)*
  *Risk: safe additive — existing code unaffected; default is None (full thinking shown).*

### Fixed

- **`AnthropicAdapter.format_tool_result` now accepts `is_error: bool = False`** — when
  `True`, the `"is_error": true` field is included in the `tool_result` block so the
  model can distinguish error content from a normal tool result.
  *Source: https://platform.claude.com/docs/en/api/messages (tool_result block)*
  *Risk: safe additive — callers that do not pass `is_error` see no change.*

- **`AnthropicClient.execute_tool_calls` and `SkillsClient.execute_tool_calls` now set
  `"is_error": true` on tool result blocks when execution fails** — previously, failed
  tool calls were represented only by an `"Error: …"` content string with no API-level
  error flag, preventing the model from reliably distinguishing tool errors from tool
  output that happens to mention errors.
  *Risk: safe with shim — callers that re-send the tool results array to `messages.create`
  will now include the `is_error` field; this is additive and backward-compatible.*

### Changed

- **Dependency: `agent-framework` bumped to `>=1.2.2,<2.0.0`** (was `>=1.2.1,<2.0.0`).
  Picks up the observability span-nesting fix during streaming and full conversation-history
  propagation for hosted workflow agents. **Breaking in AF 1.2.2**: sequential-approval and
  concurrent workflow terminal outputs now return as `AgentResponse` rather than a plain
  string. Calling code that does `str(result)` or `print(result)` continues to work; code
  that pattern-matches on the raw string type must be updated.
  *Source: https://pypi.org/pypi/agent-framework/1.2.2/json*
  *Risk: safe with shim — `AgentResponse` is str-coercible; bare-string assumptions break.*

- **Dependency: `langchain` bumped to `>=1.2.16`** (was `>=1.2.15`). Minor patch release.
  *Risk: safe internal.*

- **Docs: `docs/reference/llm_sdk_compatibility.md` header updated from "Late 2025" to
  "April 2026"** — reflects the actual date of the most recent compatibility review.

- **Full Microsoft Agent Framework 1.0 GA integration**:
  - `GantryToolBridge` now emits real `agent_framework.FunctionTool` instances
    via the GA `@tool` decorator. Gantry's `ToolCapability` set is auto-mapped
    to AF's `approval_mode`: destructive caps (`DELETE_DATA`, `WRITE_DATA`,
    `EXECUTE_CODE`, `FINANCIAL`, `PII_ACCESS`) elevate the tool to
    `"always_require"`; read-only tools stay on AF's default. Bridge accepts
    a new `as_function_tool` constructor flag to preserve bare-callable
    behaviour when needed.
  - `GantryToolBridge.build_agent(client, query, *, name, instructions, ...)`
    — one-liner that semantically retrieves tools for a query and constructs
    an AF `Agent(client, instructions, ...)` with optional middleware.
  - New `agent_gantry.integrations.agent_framework_middleware` module with
    `GantryApprovalMiddleware` (routes AF tool execution through Gantry's
    `SecurityPolicy`, raising `MiddlewareTermination` for `require_confirmation`
    patterns and `PermissionDeniedError` for policy-denied calls) and
    `GantryObservabilityMiddleware` (records per-invocation timing onto Gantry
    telemetry).
  - New example `examples/agent_frameworks/agent_framework_orchestration_example.py`
    demonstrating Sequential, Concurrent, and Handoff orchestration patterns
    with each participant agent receiving a distinct Gantry-selected tool slice.
  - 15 new orchestration tests in `tests/test_agent_framework_orchestration.py`
    driving the real `agent-framework` package against a scripted chat client
    to verify single-turn, multi-turn, sequential, concurrent, handoff,
    group-chat, agent-as-tool, workflow, and middleware approval flows all
    execute Gantry-bridged tools correctly.
- **`GantryToolBridge.build_sequential_workflow()`** — convenience helper that
  constructs a sequential multi-agent pipeline via `SequentialBuilder` without
  needing to wrap agents in `AgentExecutor` manually.
- **`GantryToolBridge.build_handoff_workflow()`** — convenience helper that
  constructs a handoff-style multi-agent workflow via `HandoffBuilder`, supporting
  named handoff edges with descriptions.
- **`_require_af_installed()` private helper** extracts the repeated
  `ImportError`-with-guidance pattern from five bridge methods into a single
  module-level function, reducing maintenance surface.

### Fixed
- **`GantryToolBridge.build_agent()` used non-existent `client.as_agent()` method.**
  AF 1.x chat clients do not expose `as_agent()`; the standard constructor is
  `Agent(client, instructions, ...)`. `build_agent()` now uses the correct
  constructor pattern, matching the existing `as_agent()` bridge method.
  *Risk: safe internal — behaviour is identical for callers.*

- **`GantryToolBridge.build_workflow()` passed bare `Agent` objects to `WorkflowBuilder`**
  instead of the required `AgentExecutor` wrappers. `WorkflowBuilder` in AF 1.x
  accepts `AgentExecutor` nodes, not `Agent` instances. Additionally, the
  `add_chain()` shorthand is not part of the public `WorkflowBuilder` API; the
  correct pattern is sequential `add_edge()` calls. Both issues have been corrected.
  *Risk: safe internal — callers pass the same `agent_specs` dict list.*

- **`GantryToolBridge.build_workflow()` silently dropped conditional edge conditions.**
  3-tuple edges `(source, target, condition)` were accepted by the type system but
  the condition was never forwarded to `WorkflowBuilder.add_edge()`, causing all
  routes to behave as unconditional fan-out. The condition is now passed through
  when present. The type signature is updated to `list[tuple[str, str] | tuple[str, str, Any]]`.
  *Risk: safe — existing 2-tuple callers are unaffected; 3-tuple callers now work correctly.*

- **`GantryApprovalMiddleware` / `GantryObservabilityMiddleware` imported
  `FunctionMiddleware` from `agent_framework`**, which may be absent in some AF 1.x
  point releases. The middleware module now falls back to `ChatMiddlewareLayer`
  when `FunctionMiddleware` is not importable.
  *Risk: safe with shim — functional behaviour unchanged.*

- Example `agent_framework_example.py` updated to reflect correct AF API patterns:
  `SequentialBuilder` replaces the invalid `WorkflowBuilder.add_chain()` call;
  conditional-edge 3-tuples restored in `build_workflow(edges=[...])` example.

### Changed
- **Agent Framework 1.0 GA support**: Bumped minimum `agent-framework` to `>=1.0.0` and updated integration example to use the renamed `OpenAIChatClient` (the RC-era `OpenAIResponsesClient` was renamed to `OpenAIChatClient` in 1.0 GA; the old `OpenAIChatClient` is now `OpenAIChatCompletionClient`). Docstrings and adapter class docs refer to "1.0 GA" instead of "RC+".
- **Dependency lower bounds bumped** (non-breaking for existing installs):
  - `agent-framework>=1.2.1,<2.0.0` (was `>=1.0.0,<2.0.0`): picks up
    `GeminiChatClient` (1.1.0), `HandoffBuilder` fixes (1.1.1), functional
    workflow API and AF→A2A bridge (1.2.0). Compatible with `crewai==1.6.1`
    (the current lock); `crewai>=1.12.0` introduces `opentelemetry-api<1.35`
    which conflicts with `agent-framework>=1.2.1` (`>=1.39.0`) — see the
    comment in `pyproject.toml` for the standalone-environment workaround.
  - `openai>=2.33.0` (was `>=2.0.0`): latest stable; both Chat Completions and
    Responses API remain fully supported.
  - `langchain>=1.2.15` (was `>=1.2.0`), `langgraph>=1.1.10` (was `>=1.1.9`)
  - `llama-index-core>=0.14.21` (was `>=0.14.10`),
    `llama-index-llms-openai>=0.7.7` (was `>=0.6.12`)
  - `anthropic>=0.97.0` (was `>=0.96.0`)
  - `crewai>=1.6.1` (retained; `>=1.12.0` conflicts with agent-framework opentelemetry)
  - `groq>=1.2.0` (was `>=1.0.0`)
  - `langchain-openai>=1.2.1` (was `>=1.1.14`)
- **`mistralai` upper bound retained at `<2.0.0`**: mistralai 2.x changes the
  async client to a context-manager pattern (`async with Mistral(...)`). Migration
  of `LLMClient.classify_intent` is documented as a pending task; a code comment
  has been added to `agent_gantry/adapters/llm_client.py` to guide the migration.
- **Documentation** (`docs/reference/llm_sdk_compatibility.md`):
  - Anthropic model corrected to `claude-sonnet-4-6` (was `claude-sonnet-4-20250514`).
  - Gemini model corrected to `gemini-2.5-flash` (was `gemini-2.0-flash`, ×6).
  - Install pins updated: `anthropic>=0.97.0`, `openai>=2.33.0`, `groq>=1.2.0`.
  - Prompt caching migrated from `client.beta.prompt_caching.messages.create()`
    to the standard `client.messages.create()` with `cache_control`.
  - Anthropic integration example now uses `to_dialect("anthropic")` instead of
    manual field-mapping from OpenAI format.
  - Gemini integration example now uses `to_dialect("gemini")` + `FunctionDeclaration`.
- **`GeminiChatClient` (AF 1.1.0)** documented in `GantryToolBridge.build_agent()`
  and `as_agent()` docstrings. Orchestration example updated to reference AF 1.2.
- **`build_sequential_workflow()` `workflow_name` parameter removed**: `SequentialBuilder`
  in AF 1.x does not accept a name argument; the parameter was ignored and has been
  removed from the signature to avoid misleading callers.
- Anthropic SDK minimum version assertion in `tests/test_llm_sdk_compatibility.py`
  updated from `0.94.0` to `0.97.0` to match `pyproject.toml`.

### Performance
- **Vectorized MMR** (PR #97): Replaced the pure-Python nested loop in `SemanticRouter._apply_mmr` with a vectorized `numpy` implementation using pre-normalized embeddings and matrix-vector dot products, drastically reducing CPU overhead during tool reranking.

### Fixed (accessibility)
- **External link a11y** (PR #99 + follow-up on `navigation.js`): Added `aria-hidden="true"` to decorative SVGs and visually-hidden "(opens in a new tab)" text for screen readers.

## [0.1.4] - 2026-03-11

### Added
- **Microsoft Agent Framework Integration** (PR #91): First-class support for Microsoft Agent Framework RC with `GantryToolBridge` for seamless tool bridging
- **MCP Server Fingerprinting** (PR #73): Capability-aware fingerprinting for MCP servers including `requires_confirmation` in computation
- **A2A Structured JSON Input Parsing** (PR #80): More reliable inter-agent communication via structured JSON input parsing
- **PGVector `include_embeddings` Flag** (PR #88): Skip embedding retrieval for performance optimization in PGVector queries
- **Comprehensive Code Quality Improvements** (PR #90): 17 tasks across 5 phases from code review plan

### Fixed
- **UUID Privacy Leak** (PR #63): Replaced UUIDv1 (which embeds MAC addresses) with privacy-safe alternatives
- **Executor Argument Validation** (PR #70, #86): Recursive argument validation using `jsonschema`
- **Example Code** (PR #91): Fixed examples to use `GantryToolBridge` instead of removed legacy code paths

### Changed
- **Slimmed Core Dependencies**: Moved example-only packages (matplotlib, pandas, scikit-learn, pillow, etc.) from core dependencies to `example-tools` optional extra, significantly reducing install size
- **Removed `ty` from Dependencies**: Removed unused Astral type checker from runtime dependencies
- **Refactored LanceDB into Domain Mixins** (PR #72): Split monolithic `LanceDBVectorStore` into focused mixins
- **Unified Schema Conversions** (PR #87): Single source of truth for tool schema conversions
- **Refactored OpenAI Embedders** (PR #75): Common base class reducing duplication
- **Refactored SemanticRouter.route** (PR #66): Extracted signal computation and filtering into separate methods
- **Refactored ExecutionEngine.execute** (PR #65): Simplified execution engine main method
- **Refactored AgentGantry.sync** (PR #69): Broke down into focused helper methods
- **Batch Embedding for MMR** (PR #89): `_apply_mmr` now uses batch embedding for missing cache entries

### Security
- **Command Injection Fix** (PR #77): Patched command injection vulnerability in `run_shell_command`
- **Arbitrary Code Execution Fix** (PR #78): Removed unsafe `eval()` usage in demo script
- **Enforced `allowed_domains` Policy** (PR #81): SecurityPolicy now properly enforces domain restrictions
- **Enforced Rate Limiting** (PR #74): SecurityPolicy now properly enforces `max_requests_per_minute`

### Performance
- **Concurrent Anthropic Tool Execution** (PR #76): Tools executed concurrently when using Anthropic provider
- **NumPy-Optimized Cosine Similarity** (PR #83): Replaced pure-Python with NumPy vector operations
- **Async I/O for LanceDB** (PR #82): Blocking I/O calls optimized to async
- **PGVector Batch Insert** (PR #71): Fixed N+1 query problem using `executemany`
- **Anthropic String Concatenation** (PR #67): Optimized string building in skills loop

## [0.1.3] - 2026-01-02

### Added
- **Professional GitHub Pages Documentation Site**: Modern, responsive documentation with search, navigation, and beautiful styling
  - Created comprehensive landing page (`docs/index.md`)
  - Added step-by-step getting started tutorial (`docs/getting-started.md`)
  - Complete API reference documentation (`docs/reference/api-reference.md`)
  - Architecture overview and design patterns (`docs/architecture/overview.md`)
  - Production best practices guide (`docs/architecture/best-practices.md`)
  - Troubleshooting guide with FAQ (`docs/troubleshooting.md`)
  - Modern HTML/CSS/JS layout with responsive design
  - Client-side search functionality
  - Mobile-friendly navigation with hamburger menu
  - Syntax-highlighted code blocks with copy buttons
  - WCAG AA accessible design

### Fixed
- **Type Safety Improvements** (6 critical/high priority fixes):
  - Fixed optional string handling in `mcp_router.py:129` (critical type safety issue)
  - Corrected return type annotation in `llm_client.py:175`
  - Added proper vector store return type casts in `gantry.py` (4 occurrences)
  - Moved function-level imports to module level in `gantry.py:1055-1056`
  - Resolved line length violations in `mcp_router.py:106` and `openai.py:57,212`
- All code now passes strict mypy type checking with zero errors in modified files
- All code passes ruff linting checks

### Changed
- **Examples Modernization** - Updated 3 LLM integration examples to use latest API patterns:
  - `examples/llm_integration/google_genai_demo.py` - Added `set_default_gantry()` and `dialect="gemini"`
  - `examples/llm_integration/groq_demo.py` - Modernized to use context-local gantry pattern
  - `examples/llm_integration/mistral_demo.py` - Updated decorator to clean syntax
- **Documentation Cleanup** - Removed development artifacts from docs/ folder:
  - Removed `phase2.md` (development planning document)
  - Cleaned up internal code review and sweep reports
  - Organized docs by user journey (Getting Started → Features → Reference → Help)

### Documentation
- Complete documentation site ready for GitHub Pages at `https://codehalwell.github.io/Agent-Gantry/`
- All user guides enhanced with modern styling and improved examples
- Added 6 new comprehensive documentation files covering installation through production deployment
- Improved cross-referencing between documentation files
- Enhanced code examples with syntax highlighting and copy buttons

### Quality Improvements
- Test suite: 350+ tests passing (100% pass rate on core functionality)
- Code quality grade: A (96/100) - Production ready
- Examples coverage: 50+ production-quality examples across 10 categories
- Documentation coverage: 100% of features documented with tutorials and API reference

## [0.1.2] - 2026-01-02

### Added
- **Dynamic MCP Server Selection**: Semantic routing for MCP servers with lazy loading
  - `register_mcp_server()` - Register MCP servers with rich metadata (no immediate connection)
  - `sync_mcp_servers()` - Sync server metadata for semantic search
  - `retrieve_mcp_servers()` - Find relevant servers using vector similarity
  - `discover_tools_from_server()` - Connect and load tools on-demand from selected servers
  - Health tracking for MCP servers with automatic availability monitoring
  - Capability-based server filtering
  - Namespace organization for multi-tenant scenarios

### Fixed
- Type safety improvements across core modules (6 fixes in `mcp_router.py`, `gantry.py`, `llm_client.py`, `openai.py`)
- Enhanced `InMemoryVectorStore` with dimension property and fingerprinting for consistency
- Improved vector store protocol compliance for better adapter compatibility

### Changed
- MCP servers now stored as pseudo-tools in vector store for semantic search (implementation detail)
- Vector store interface enhanced to support multi-entity storage patterns

### Documentation
- Added comprehensive [Dynamic MCP Selection guide](docs/dynamic_mcp_selection.md)
- Updated README.md with Dynamic MCP Server Selection section
- Improved code examples throughout documentation
- Enhanced installation instructions

## [0.1.0] - 2025-12-23

### Added
- Core foundation with semantic routing and tool orchestration
- Multi-protocol support (OpenAI, Anthropic, Google GenAI, Vertex AI, Mistral, Groq)
- Vector store adapters (In-Memory, Qdrant, Chroma)
- Embedder adapters (Sentence Transformers, OpenAI)
- Reranker support (Cohere, Cross-Encoder)
- Execution engine with retries, timeouts, and circuit breakers
- Zero-trust security with capability-based permissions and policies
- MCP (Model Context Protocol) client and server support
- A2A (Agent-to-Agent) protocol implementation
- Health tracking and observability
- OpenTelemetry integration
- CLI interface for tool management
- Comprehensive documentation and examples

### Features
- **Semantic Routing**: Intelligent tool selection using vector similarity
- **Context Window Optimization**: Reduce token usage by ~90%
- **Circuit Breakers**: Automatic failure detection and recovery
- **Argument Validation**: Defensive validation against tool schemas
- **Async-Native**: Full async support for tools and execution
- **Schema Transcoding**: Automatic conversion between tool formats
- **Intent Classification**: Enhanced routing with intent matching
- **MMR Diversity**: Maximal Marginal Relevance for diverse tool selection

### Documentation
- Comprehensive README with quick start guide
- MCP integration examples
- A2A integration examples
- Phase documentation (Phase 2-6)
- LLM SDK compatibility guide
- Architecture diagrams

[Unreleased]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.1.0...v0.1.2
[0.1.0]: https://github.com/CodeHalwell/Agent-Gantry/releases/tag/v0.1.0
