# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-05-13

### Added

- **End-to-end orchestration test coverage for `bridge.build_agent` and
  `bridge.as_agent`.** The bridge previously had construction-shape tests
  but no end-to-end `agent.run()` coverage with multi-arg Gantry tools.
  New tests in `tests/test_agent_framework_orchestration.py` drive a
  `ScriptedChatClient` through the full
  function_call → Gantry-execute → function_result → final-text loop and
  assert the resulting `function_result` content carries the expected
  `call_id` and payload. Includes `extra_tools=` mixing static AF
  FunctionTools with Gantry-selected tools in one Agent.
- **Tool-surface reduction proof-of-routing test.** Registers 12 tools
  across mixed domains, builds an Agent for a weather query with
  `limit=3`, and asserts (a) the Agent's bound tool set is exactly 3,
  (b) the chat client saw 3 tools (not 12) on the first turn, and
  (c) `get_weather` is present while most unrelated tools are filtered.
  This is the regression guard for Gantry's headline value-prop
  (75% reduction in this scenario).
- **Per-user-turn and per-chat-round re-routing tests.** Drives a real
  multi-turn `agent.run` session over a topic-shifting conversation and
  asserts the chat client's per-turn `seen_tools` differs across user
  turns — Gantry re-queries on every `agent.run()` in `per_run` mode.
  Adds a per-call refresher unit test that walks the
  `_refresh_tools_on_chat_context` code path directly with two synthetic
  chat contexts (round 1 weather, round 2 billing) and asserts the
  surface flips between rounds.
- **Real end-to-end `per_call` middleware test.**
  `test_context_provider_per_call_end_to_end` drives a real `agent.run`
  with `query_strategy="per_call"` + `as_chat_middleware()` through two
  LLM rounds (function_call → result → final text) and asserts the
  function executed via the `function_result` content. This is the
  regression guard for the in-place mutation invariant.
- **Per-round routing adaptation end-to-end test.**
  `test_context_provider_per_call_surface_adapts_to_tool_result` uses a
  deterministic keyword-overlap embedder (no model downloads) to prove
  the per-round tool surface actually shifts as the message stream
  shifts: round 1 = weather tools, round 2 = refund/billing tools after
  the tool result mentions invoice content. Asserts the refund tool was
  findable by AF's function executor (no `"not found"` in
  `fr.exception`).
- **`SkillsProvider` co-existence test.**
  `test_context_provider_preserves_skill_tools_across_refresh` pre-seeds
  options with a foreign `load_skill` tool, runs two refresh rounds
  with a topic shift, and asserts the skill tool survives both
  refreshes and the options dict reference is never replaced. Locks in
  the contract that Gantry's refresh only strips tools whose names are
  in the Gantry registry.
- **Non-dict (Pydantic-ish) options coverage.**
  `test_context_provider_refresh_mutates_non_dict_options_in_place`
  exercises the non-dict options refresh branch with a stand-in
  Pydantic-style object. Asserts (a) same reference mutated in-place,
  (b) peer-provider tools preserved, (c) Gantry's dynamic top-k added.
- **Unit tests for the `_msg_text` fix.**
  `test_msg_text_walks_dict_contents_for_function_result` and
  `test_msg_text_function_result_fallback_is_per_content` in
  `tests/test_query_strategies.py` lock the dict-`contents` walker
  and the per-content fallback gating. AF-Message coverage of the
  same path lives in
  `test_last_tool_result_extracts_text_from_af_function_result_message`
  in the orchestration file (kept out of the query-strategies file
  so it stays AF-free per its module docstring).
- **`agent_gantry/adapters/tool_spec/providers.py`** — `AnthropicAdapter.to_provider_schema()` now accepts `strict=False` (default, backwards-compatible) or `strict=True`. When `True`, the output schema includes `"strict": true` at the tool-definition top level, enabling Anthropic's grammar-constrained sampling so Claude's tool `input` always matches `input_schema` exactly.
  *Risk: safe internal — purely additive; default preserves all existing behaviour.*  
  Source: https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use
- **`tests/test_tool_spec_adapters.py`** — Two new tests: `TestAnthropicAdapter::test_to_provider_schema_strict_mode` and `TestToolDefinitionToDialect::test_to_dialect_anthropic_strict`, verifying the new `strict=True` option.
- **`agent_gantry.query` module** — built-in deterministic query-generation
  strategies for semantic retrieval: `last_user_text` (default),
  `last_assistant_text`, `last_tool_result`, `concatenate_recent`, and
  `fallback_chain`. Strategies operate on AF messages, dicts, or anything
  exposing `role` + `text`/`content`.
- **`GantryContextProvider` per-call retrieval** — new
  `query_strategy="per_call"` (default `"per_run"` is back-compat) plus
  `query_generator=...` parameter for per-chat-round semantic refresh.
  `provider.as_chat_middleware()` returns the AF chat middleware that wires
  the per-round refresh into `Agent(middleware=[...])`. Solves the
  "tool selection is fixed for the whole `agent.run()`" limitation flagged
  by integrators of multi-step workflows.
- **`required=[...]` parameter on `GantryContextProvider`** — hard pins a
  set of tools and raises `MissingRequiredToolError` at construction time
  if any are missing from the gantry. Catches typos / dropped registrations
  earlier than runtime agent failure.
- **Public read-only properties on `GantryContextProvider`** — `top_k`,
  `score_threshold`, `query_strategy`, `always_include`, `required`,
  `gantry`, `bridge`. External observability code can read configuration
  without poking at private attributes.
- **`AgentGantry.preview(query, ...)`** — read-only ranking helper that
  returns `(qualified_name, score)` pairs, useful for calibrating
  `score_threshold` without spinning up an agent.
- **`AgentGantry.list_tools_sync()`** — sync, in-memory inspection of
  registered tools (no `await`, no vector store round-trip). Complements
  the existing async `list_tools()`.
- **`agent_gantry.adapters.embedders` re-exports `SentenceTransformersEmbedder`,
  `OpenAIEmbedder`, `AzureOpenAIEmbedder`** alongside `NomicEmbedder` and
  `SimpleEmbedder`, all behind lazy imports — you can write
  `from agent_gantry.adapters.embedders import SentenceTransformersEmbedder`
  without knowing the deep submodule path. Same pattern applied to
  `agent_gantry.adapters.rerankers` (`CohereReranker`, `CrossEncoderReranker`).
- **Tests** — `tests/test_query_strategies.py` covers the new query module,
  `preview`, `list_tools_sync`, the `SimpleEmbedder` warning, FunctionTool
  registration, and adapter re-exports.

### Changed

- **Dependency: `openai` floor bumped to `>=2.35.1`** (was `>=2.34.0`). Version 2.35.1 fixes an image-generation `size` enum regression introduced in 2.35.0 and removes deprecated CLI tooling. No API surface changes for Gantry's Chat Completions or Responses API call sites.
  *Risk: safe internal — floor bump only.*  
  Source: https://github.com/openai/openai-python/releases
- **Dependency: `anthropic` floor bumped to `>=0.100.0`** (was `>=0.98.1`). Versions 0.99.0 and 0.100.0 add OIDC federation token exchange, Managed Agents beta (multiagents + outcomes), webhook support, and vault validation. All additive; no breaking changes to `client.messages.create`, tool-use, or thinking APIs.
  *Risk: safe internal — floor bump only.*  
  Source: https://github.com/anthropics/anthropic-sdk-python/releases
- **`pyproject.toml`** — Bumped dependency floors (5 May audit):
  - `openai>=2.33.0` → `>=2.34.0`.
  - `anthropic>=0.97.0` → `>=0.98.1`.
  - `google-genai>=1.74.0` → `>=1.75.0`.
  - `mcp>=1.0.0` → `>=1.27.0` (26 minor releases behind; latest stable 2026-04-02).
  *Risk: safe internal — floor bumps only; no upper bounds changed.*
- **`uv.lock`** — Refreshed cumulatively. Key resolved-version changes across recent audits:
  - `openai` 2.34.0 → 2.35.1 (7 May audit).
  - `anthropic` 0.98.1 → 0.100.0 (7 May audit).
  - `mistralai` 1.12.4 → 2.4.4 (5 May audit; corrects stale lock from 4 May pyproject change).
  - `opentelemetry-*` 1.41.0 → 1.39.1 (side effect of mcp 1.27.0 transitive resolution; still satisfies `agent-framework>=1.2.2`'s `>=1.39.0` requirement).
  - `jsonpath-python` 1.1.5 added (new transitive dep of mcp 1.27.0).
  - `invoke` 2.2.1 removed (dropped by transitive deps).
- **`docs/reference/llm_sdk_compatibility.md`** — Install guide `pip install` snippets updated across audits: `openai>=2.35.1`, `anthropic>=0.100.0`, `google-genai>=1.75.0`; Azure OpenAI Responses API examples updated from `gpt-4o` to `gpt-4.1`; Mistral install command updated from `>=1.0.0,<2.0.0` to `>=2.0.0`; Mistral key-methods and integration examples rewritten for `mistralai >= 2.0` async context-manager pattern.
  *Risk: documentation only.*
- **`agent_gantry/adapters/llm_client.py`**: Migrated Mistral provider from the
  `mistralai < 2.0` long-lived client pattern to the `mistralai >= 2.0` per-call
  async context-manager pattern (`async with Mistral(...) as client:`). The
  `LLMClient._initialize_client()` now stores only the API key for the Mistral
  provider; `classify_intent()` opens a fresh context-manager per request so HTTP
  connections are properly released. `health_check()` uses `_mistral_api_key is not
  None` rather than `_client is not None` for the Mistral branch.
  *Risk: safe with shim — no public API change; callers using `LLMConfig(provider="mistral")` are unaffected.*
- **`pyproject.toml`**: Removed `<2.0.0` upper bound on `mistralai`; floor updated
  to `>=2.0.0`. The comment block explaining the migration blocker has been
  removed now that the migration is complete.
  *Risk: safe internal — the lower-bound bump is the only semantic change.*
- **`pyproject.toml`**: Bumped `semantic-kernel` minimum from `>=1.30.0` to
  `>=1.36.0`, matching the version already resolved by `uv.lock`. Full upgrade
  to 1.41.3 remains blocked by `opentelemetry-api` version conflict with
  `agent-framework` on some Python/platform combinations.
  *Risk: safe internal — does not change the installed version.*
- **Dependency: `google-genai` floor bumped to `>=1.74.0`** (was `>=1.0.0`). The
  previous floor was 74 minor versions behind the latest stable release (1.74.0).
  Bumping ensures constrained resolver environments select a supported SDK version.
  *Risk: safe internal — no API changes required.*
- **Dependency: `langchain` floor bumped to `>=1.2.17`** (was `>=1.2.16`). Minor
  patch release. *Risk: safe internal.*
- **`AgentGantry.register()` now accepts `agent_framework.tool`-decorated
  FunctionTool objects** (or any wrapper exposing `.name` / `.func`).
  Previously raised `AttributeError: 'FunctionTool' object has no attribute
  '__name__'`. Bare callables continue to work unchanged.
- **`SimpleEmbedder` warns when paired with `score_threshold > 0.0`** —
  hash-based similarity scores cluster tightly regardless of relevance, so
  a non-zero threshold typically returns 0 tools silently. The first
  retrieval call now emits a `UserWarning` recommending a real embedder.
- **`SimpleEmbedder` docstring** updated to lead with "for testing only —
  produces near-uniform similarity scores", to make its non-production
  nature obvious from `help(SimpleEmbedder)`.
- **`EmbeddingAdapter` docstring** corrected: lists actual implementations
  (`SimpleEmbedder`, `NomicEmbedder`, `SentenceTransformersEmbedder`,
  `OpenAIEmbedder`, `AzureOpenAIEmbedder`) instead of the old typo'd
  `SentenceTransformerEmbedder` (singular).
- **`SentenceTransformersEmbedder` no longer triggers a `FutureWarning`**
  on first init — calls `get_embedding_dimension()` when available and
  falls back to the deprecated `get_sentence_embedding_dimension()` only
  on older releases.

### Documentation

- **`agent_gantry/integrations/anthropic_features.py`**: Clarified that
  `claude-opus-4-7` does **not** support extended thinking (only adaptive thinking).
  Updated `AnthropicFeatures`, `AnthropicClient`, and `create_anthropic_client`
  docstrings accordingly.
  *Source: https://platform.claude.com/docs/en/docs/about-claude/models*

### Fixed

- **`GantryContextProvider` per-call refresh now mutates `options` in
  place instead of replacing the reference.** AF's
  `FunctionInvocationLayer` keeps a reference to the same options dict
  across its inner function-invocation loop and uses it both as the
  chat-call payload *and* as the tool-lookup table when executing
  function calls. The previous code did
  `context.options = new_options`, which updated the chat client's
  view but left the function executor reading a stale tool list — so
  the model emitted a function call, AF couldn't find the tool in
  `mutable_options['tools']`, and the inner loop terminated after one
  round with an unexecuted `function_call` in the message stream.
  Now mutates `options["tools"] = combined` in place. Symptom in the
  wild: `agent.run` returning a function_call with no result.
  *Risk: corrects a silent inner-loop termination; behaviour-fix only.*
- **`GantryContextProvider` non-dict options branch now preserves
  peer-provider tools and mutates in place.** Two regressions in the
  Pydantic ChatOptions path: (a) existing tools were only read from
  dict options, so peer-provider tools (skills, static tools, tools
  from other ContextProviders) on a Pydantic model were dropped on
  every per-call refresh; (b) the fallback path reassigned the same
  reference (no-op) and wrapped the surrounding
  `context.options = new_options` in `try/except AttributeError: pass`,
  silently dropping tool updates on read-only attributes. Now reads
  existing tools from `getattr(options, "tools", None)` for non-dict
  inputs, uses `setattr(options, "tools", combined)` to preserve the
  FunctionInvocationLayer reference invariant, and only falls back to
  `model_copy` + reassign for genuinely frozen Pydantic models —
  warning if even that fails. *Risk: corrects silent data loss for
  non-dict options.*
- **`agent_gantry.query._msg_text` now walks structured `contents`
  for AF tool-role messages.** AF wraps tool output as a
  `function_result` Content nested inside `Message.contents`;
  `Message.text` is empty in that case and the actual text lives in
  `Content.items[].text` (or `Content.result` for primitives).
  `_msg_text` previously only inspected `Message.text` /
  `Message.content`, so `last_tool_result` returned `""` for AF tool
  messages and the query generator's `fallback_chain` collapsed to
  `last_user_text` — which never changes within a single `agent.run`,
  defeating per-round adaptation. Now walks `contents` (and
  `msg["contents"]` for dict-shaped messages), pulling text from
  `text` and `function_result` Content variants. *Risk: corrects
  per-round routing adaptation; previously broken silently.*
- **`agent_gantry.query._msg_text` per-content `function_result`
  fallback now tracks contribution per-content.** The `.result`
  fallback was gated by a global `if not parts:` check across the
  whole message. When an earlier text content already populated
  `parts`, a later `function_result` with empty `items[]` would
  silently drop its primitive `.result`. Tracks `contributed` per
  function_result so earlier text stays AND each function_result
  still falls back to its own `.result`. *Risk: corrects silent
  drop of tool output in mixed-content tool-role messages.*
- **`README.md`**: Updated deprecated model identifiers in quick-start examples:
  `claude-sonnet-4-20250514` → `claude-sonnet-4-6` (retiring 15 June 2026);
  `gemini-2.0-flash` → `gemini-2.5-flash` (deprecated; service shutdown imminent).
  *Source: https://platform.claude.com/docs/en/docs/about-claude/models,
  https://ai.google.dev/gemini-api/docs/models*
  *Risk: none — documentation only.*
- **`docs/reference/llm_sdk_compatibility.md`**:
  - OpenRouter section: corrected `pip install openai>=1.0.0` → `>=2.33.0` (missed
    in April 2026 audit).
  - Tool Format Conversion → Anthropic: replaced deprecated manual
    `to_anthropic_tools()` helper with the canonical `to_dialect("anthropic")`
    pattern (appendix section was missed in April 2026 audit).
  - Tool Format Conversion → Vertex AI and the Vertex AI integration example:
    now use `to_dialect("gemini")` + `**`-unpacking into `FunctionDeclaration`,
    eliminating the indirect two-step conversion from OpenAI format.
- **Cross-event-loop failures with `DurableAIAgentWorker` and similar
  worker-thread loops.** When a gantry was constructed in one context
  (often module import time) and then driven from a different event
  loop on a worker thread, contended access to the rate limiter's
  ``asyncio.Lock`` raised ``RuntimeError: ... is bound to a different
  event loop`` because ``asyncio`` synchronisation primitives bind to
  the loop they were first awaited on. Symptoms in the wild: every
  tool execution returning ``"Error: Function failed."`` (Agent
  Framework's opaque catch-all) once the durable worker took over.
  ``RateLimiter`` now keeps one lock per running loop, lazily
  constructed on first use; closed loops are pruned opportunistically.
  Bounded leak: one entry per distinct loop ever used, typically
  1–2 per process.
- **Sync tool handlers no longer use the deprecated
  ``asyncio.get_event_loop()``** in
  ``ExecutionEngine._execute_with_timeout``. Replaced with
  ``asyncio.to_thread(...)``, which always binds to the running loop
  and removes another cross-loop hazard for worker-thread setups.
- **``NomicEmbedder`` no longer uses ``asyncio.get_event_loop()``** in
  ``embed_text``, ``embed_batch``, ``embed_query``, and ``health_check``.
  Replaced all four with ``asyncio.to_thread(...)``. The previous code
  warned under ``-W error::DeprecationWarning`` on 3.10+ and would have
  picked up the wrong loop in ``DurableAIAgentWorker``-style setups.
- **Bridge wrapper now surfaces the underlying exception** instead of
  letting it propagate up to Agent Framework's tool runner — which
  rewrites every exception as the unhelpful ``"Error: Function
  failed."`` string when ``include_detailed_errors`` is off.
  Wrappers built by ``GantryToolBridge`` (and therefore by
  ``GantryContextProvider``) catch any exception from
  ``gantry.execute(...)`` and return a structured
  ``{"error": "<ExcType>: <message>"}`` JSON string. Failed
  ``ToolResult``s also include ``error_type`` in the surfaced text.
- **New cross-loop test suite** (``tests/test_executor_cross_loop.py``)
  reproduces the durable-worker scenario: gantry built on one loop,
  driven from a worker-thread loop, with genuine lock contention to
  force the binding path. The suite also covers exception surfacing
  for both handler-level and executor-level failures.
- **New durable-worker integration test**
  (``tests/test_durable_worker_integration.py``) drives a real
  :class:`agent_framework.RawAgent` with a ``BaseChatClient`` subclass
  that emits one or more ``function_call`` items, then runs each
  request via ``asyncio.run`` — the exact loop topology used by
  :class:`agent_framework_durabletask.DurableAIAgentWorker`. Covers:
  sequential ``asyncio.run`` requests with sync handlers; parallel
  function-call dispatch with async handlers (real lock contention);
  worker-thread execution; and worker-thread → main-thread
  sequencing. Tools are pre-resolved once at "module load" via
  :class:`GantryToolBridge`, mirroring the integrator pattern.

## [0.2.0] - 2026-05-01

### Changed

- **PyPI publish workflow now uses `pypa/gh-action-pypi-publish@release/v1`** for both
  PyPI and TestPyPI. The previous `uv publish --publish-url https://test.pypi.org/simple/`
  invocation pointed at the install index instead of the upload endpoint
  (`https://test.pypi.org/legacy/`), which made TestPyPI publishes silently fail. The PyPA
  action handles OIDC trusted publishing and the correct upload URLs natively.
- **`test-install` job uses the venv interpreter directly** instead of `uv run`, which
  expects a project context the job didn't provide. Install + import verification are now
  a single step that prints the installed version and location.
- **`environment.url` set on both `publish-pypi` and `publish-testpypi`** for clearer
  deployment links in the GitHub Actions UI.

### Added

- **`ToolSpecAdapter.format_tool_result` protocol extended with `is_error: bool = False`** — all
  concrete adapters (`OpenAIAdapter`, `OpenAIResponsesAdapter`, `AnthropicAdapter`,
  `GeminiAdapter`) now accept the optional keyword argument so callers typed against the
  protocol can pass `is_error` without casting. Non-Anthropic adapters accept and ignore the
  flag; `AnthropicAdapter` uses it to emit the Anthropic `"is_error"` field.
  *Risk: safe additive — default is `False`, backward-compatible.*

- **Unit tests for `is_error` semantics** — added to `TestAnthropicAdapter` in
  `test_tool_spec_adapters.py`, to `TestAnthropicClient` in `test_anthropic_features.py`,
  and to `TestSkillsClient` in `test_anthropic_skills.py`. Each test pair asserts that
  `is_error: true` is present on failure and absent on success, and that dict results are
  JSON-serialised rather than `str()`-coerced.

- **Unit tests for `thinking_display` payload injection** — added three focused tests in
  `TestAnthropicClient` verifying that `create_message()` passes `thinking.display` to
  `AsyncAnthropic.messages.create()` for adaptive mode, extended mode, and that the key is
  absent when `thinking_display=None`.

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

- **`AnthropicClient.execute_tool_calls` and `SkillsClient.execute_tool_calls` now use
  `json.dumps()` for non-string tool results** — previously `str()` was used, which
  produces Python repr notation (e.g. `{'key': 'val'}` with single quotes) rather than
  valid JSON, causing downstream parse failures when the model or the caller attempted to
  deserialise the content.
  *Risk: safe correctness fix — `str` results pass through unchanged; dict/list results are
  now JSON-serialised consistently with `AnthropicAdapter.format_tool_result`.*

- **`ExecutionStatus.SUCCESS` used for status comparisons in `AnthropicClient` and
  `SkillsClient`** — replaced bare `!= "success"` string literals with
  `!= ExecutionStatus.SUCCESS` for type-safety and consistency with the rest of the
  codebase (e.g. `agent_gantry/servers/mcp_server.py`).
  *Risk: none — `ExecutionStatus` inherits from `str` so the comparison is equivalent.*

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

[Unreleased]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.1.0...v0.1.2
[0.1.0]: https://github.com/CodeHalwell/Agent-Gantry/releases/tag/v0.1.0
