# Agent-Gantry Modernisation Audit

**Date:** 2026-08-03
**Repository:** `CodeHalwell/Agent-Gantry` · version `0.9.0`
**Branch:** `claude/agent-gantry-performance-mw8s6p` (based on `main` at `e4e7fd4`)
**Auditor:** Claude

> **Audit history.**
> - **2026-05-27** (`claude/cool-hopper-Y9M5N`): Initial modernisation audit. Bumped
>   langchain, langchain-openai, langgraph floors. Produced AUDIT.md.
> - **2026-05-31** (`claude/cool-hopper-4OYfr`, based on PR #218): Bumped anthropic
>   `0.104.1→0.105.2`, mcp `1.27.1→1.27.2`, groq `1.2.0→1.4.0`. Removed redundant
>   `_reject_newlines` validator from `ToolDefinition`.  Updated `pyproject.toml`
>   comments for AF 1.7.0, crewai 1.14.6, google-genai 2.7.0, google-adk 2.1.0.
>   Regenerated `uv.lock`.
> - **2026-06-01** (`claude/cool-hopper-JkddB`, based on PR #220): No new package
>   releases since 2026-05-31. Added HarnessAgent example and SSRF invalid-port test.
> - **2026-06-02** (`claude/cool-hopper-R79WP`, based on main after PR #221):
>   Bumped `openai>=2.38.0→>=2.40.0`; added `claude-opus-4-8` model capability
>   documentation; added `TestClaudeOpus48ThinkingGuards` (6 tests).
> - **2026-06-03** (`claude/cool-hopper-gPA6f`, based on main after PR #222):
>   Bumped `langchain>=1.3.2→>=1.3.4`, `langgraph>=1.2.2→>=1.2.4`,
>   `cohere>=6.0.0→>=7.0.3`; fixed `gpt-4o-realtime-preview` (discontinued 2026-05-07)
>   → `gpt-realtime-1.5`; migrated all `gpt-4o` / `gpt-4o-mini` model strings in
>   examples and docs to `gpt-5.5` / `gpt-5.4-mini`.
> - **2026-06-04** (`claude/cool-hopper-ojLqy`, based on main after PR #223):
>   Bumped `openai>=2.40.0→>=2.41.0`; updated `google-genai` and `semantic-kernel`
>   comments (2.8.0 and 1.43.0 respectively); fixed `ExecutionStatus.PERMISSION_DENIED`
>   in executor (PR #225 fix); applied registry linter substring pre-check (PR #226 fix).
> - **2026-06-05** (`claude/cool-hopper-0w07i`, based on main after PR #224):
>   Updated `pyproject.toml` comment for AF 1.8.0 (released 2026-06-04, within existing
>   `>=1.5.0,<2.0.0` constraint — no version change required); updated google-adk comment
>   to reflect 2.2.0 as latest stable (was 2.1.0; langgraph conflict still blocks upgrade).
>   No code changes required.
> - **2026-08-03** (`claude/agent-gantry-performance-mw8s6p`): Performance +
>   dependency audit. **Urgent:** capped `mcp>=1.27.2,<2.0.0` — mcp 2.0.0
>   (2026-07-28) removes every v1 API Gantry uses (`Server`, `stdio_server`,
>   `ClientSession`, `stdio_client`); uncapped standalone `[mcp]` installs broke
>   at import. Bumped `crewai>=1.15.0` (its opentelemetry-api pin moved to
>   ~=1.42.0 in 1.15.0, resolving the long-standing agent-framework conflict —
>   combined extra now locks crewai 1.15.10, was 1.6.1), `langchain>=1.3.14`,
>   `langchain-openai>=1.4.1`, `langgraph>=1.2.10`, `llama-index-core>=0.14.23`,
>   `llama-index-llms-openai>=0.7.10`, `openai>=2.45.0` (required by
>   langchain-openai 1.4.x), `anthropic>=0.120.2`, `cohere>=7.0.8`,
>   `groq>=1.6.0`. semantic-kernel bump to 1.43.1 attempted and REVERTED —
>   `uv lock` proved sk 1.43+ (azure-ai-projects<1.1) incompatible with
>   agent-framework-foundry (azure-ai-projects>=2.2); comment updated. Removed
>   obsolete `azure-search-documents>=11.7.0b2` uv override (stable 12.0.0
>   exists). Noted mistralai quarantine LIFTED (kept OpenAI-compatible path)
>   and google-adk's google-genai constraint inversion (>=2.9,<3 since adk
>   1.36.0). CI adapter job caps `pydantic-ai-slim<2`, `haystack-ai<3`,
>   `dspy<3.3` pending adapter migration (all three shipped breaking changes).
>   Regenerated `uv.lock`. Code: fixed broken incremental sync (in-memory
>   fingerprint format; Qdrant/Chroma/PGVector fingerprint + metadata support),
>   rate-limiter slot leak, LanceDB include_embeddings crash under MMR; wired
>   MCP tool execution (handlers + persistent sessions); perf work in
>   router/registry/executor/LanceDB/OpenAI embedder (see CHANGELOG).
>   **Follow-up (same day, same branch):** completed the tracked follow-ups.
>   mcp cap widened `<2.0.0` → `<3` — dual 1.x/2.x support implemented and
>   verified against 1.28.1 and 2.0.0 (v1 client surface survives in 2.0;
>   server handler registration branched decorators vs `on_list_tools`/
>   `on_call_tool` callbacks; dual-spelled `input_schema`/`inputSchema` read
>   fixes a silent v2 empty-schema bug; combined extra still locks 1.x via
>   openai-agents/agent-framework `<2` pins). All three native-adapter CI caps
>   LIFTED: pydantic-ai verified on 2.23.0 (zero changes — keyword-only
>   construction spans both majors), dspy verified on 3.3.0 (backward
>   compatible; ReActV2 is a separate experimental class), haystack-ai 3.0
>   supported via a version branch in `GantryLiveHaystackToolInvoker.build()`
>   (`ToolInvoker` on 2.x; per-call `Agent` with `chat_generator=...` or a
>   clear error on 3.x — previously a misleading "install haystack-ai"
>   ImportError; new real-package guards in
>   `tests/frameworks/test_haystack_build_live.py`). LanceDB mixins
>   consolidated (~600 lines of dead shadowed duplicates removed).
>   BREAKING: `LLMConfig.model` default `gpt-4o-mini` → `gpt-5.4-mini`
>   (shutdown 2026-10-23), closing the last remaining audit next-step.
> - **2026-06-08** (`claude/cool-hopper-cSXxx`): Bumped `anthropic>=0.105.2→>=0.107.1`
>   (three new releases since 2026-06-05: 0.106.0 deprecated claude-opus-4-1 in SDK,
>   0.107.0 added Managed Agents type updates, 0.107.1 fixed Foundry x-api-key header).
>   Updated `anthropic_features.py` — replaced vague "earlier Claude 4 models" in three
>   docstrings with explicit model list; added retirement notice for claude-sonnet-4 and
>   claude-opus-4 (retired 2026-06-15, 7 days from now).  Updated install-pin examples in
>   `docs/reference/llm_sdk_compatibility.md`.  Regenerated `uv.lock`.  See §1 for details.

---

## 1 · Executive Summary

| Severity | Count | Areas |
|----------|-------|-------|
| **Must-change now** | 1 | `anthropic` floor bump `0.105.2→0.107.1` |
| **Good next-step (completed this run)** | 2 | `anthropic_features.py` docstring precision; install-pin docs update |
| **Good next-step (remaining)** | 3 | semantic-kernel floor; google-adk 2.x upgrade; config.py gpt-4o-mini default |

**New findings since 2026-06-05:**

1. **`anthropic` 0.107.1** (released 2026-06-07) — **must-change**: bump floor from
   `>=0.105.2` to `>=0.107.1`.  Three new releases since the 2026-06-05 audit:
   - **0.106.0** (2026-06-05): `claude-opus-4-1` formally marked as deprecated in the
     SDK (retiring 2026-08-05); Foundry client `copy()`/`with_options()` fixes; schema
     `$ref`/`$defs` transform bug fix.  `claude-opus-4-1` is not referenced in Gantry
     source or examples — no code changes required beyond the floor bump.
   - **0.107.0** (2026-06-06): minor type updates for Managed Agents API; no changes to
     Messages API or tool-use surfaces used by Gantry.
   - **0.107.1** (2026-06-07): Foundry x-api-key header authentication fix; no changes
     to Messages API or tool-use surfaces used by Gantry.
   No breaking changes across 0.105.2→0.107.1.  Floor bumped and `pyproject.toml`
   comment updated this run.  `uv.lock` regenerated.
   Source: https://pypi.org/pypi/anthropic/json (verified 2026-06-08)
           https://github.com/anthropics/anthropic-sdk-python/releases (verified 2026-06-08)
2. **Claude model deprecations (imminent)** — `claude-sonnet-4` (`claude-sonnet-4-20250514`)
   and `claude-opus-4` (`claude-opus-4-20250514`) are retired on **2026-06-15** (7 days
   from this audit). Neither model ID appears in Gantry source or examples (confirmed by
   grep across all `.py` and `.md` files). The vague phrase "earlier Claude 4 models" in
   three docstrings of `anthropic_features.py` has been replaced with an explicit model
   list excluding the retiring models, and a retirement notice added.
   `claude-opus-4-1` continues to work until 2026-08-05; added deprecation note where
   relevant.
   Source: https://platform.claude.com/docs/en/docs/about-claude/models/overview (verified 2026-06-08)
           https://github.com/anthropics/anthropic-sdk-python/releases/tag/v0.106.0 (verified 2026-06-08)

**All other packages verified against PyPI on 2026-06-08** — no releases since the
2026-06-05 audit for `openai` (2.41.0), `agent-framework` (1.8.0), `google-genai`
(2.8.0), `langchain` (1.3.4), `langgraph` (1.2.4), `mcp` (1.27.2), `autogen-agentchat`
(0.7.5), `groq` (1.4.0), `cohere` (7.0.3), `langchain-openai` (1.2.2),
`semantic-kernel` (1.43.0), `crewai` (1.14.6), `google-adk` (2.2.0).
See §2 for the full table.

---

## 2 · Dependency Upgrade Plan

### 2.1 Package-by-package table (verified 2026-06-08)

Sources verified against PyPI JSON API.

| Package | Current floor | Latest stable | Action | Risk |
|---------|--------------|---------------|--------|------|
| `openai` | `>=2.41.0` | `2.41.0` | ✅ At latest | — |
| `agent-framework` | `>=1.5.0,<2.0.0` | `1.8.0` | ✅ Within range — comment updated 2026-06-05 | — |
| `anthropic` | `>=0.107.1` | **`0.107.1`** | ✅ **Bumped this run** (`0.105.2→0.107.1`) | Safe internal |
| `autogen-agentchat` | `>=0.7.5` | `0.7.5` | ✅ At latest | — |
| `cohere` | `>=7.0.3` | `7.0.3` | ✅ At latest | — |
| `crewai` | `>=1.6.1` | `1.14.6` | Note only (OTel conflict with AF) | Blocked |
| `google-adk` | `>=1.14.1` | `2.2.0` | Comment updated 2026-06-05; floor unchanged (langgraph conflict) | Blocked |
| `google-genai` | `>=1.75.0` | `2.8.0` | Comment updated 2026-06-04; floor unchanged | Note only |
| `groq` | `>=1.4.0` | `1.4.0` | ✅ At latest | — |
| `langchain` | `>=1.3.4` | `1.3.4` | ✅ At latest | — |
| `langchain-openai` | `>=1.2.2` | `1.2.2` | ✅ At latest | — |
| `langgraph` | `>=1.2.4` | `1.2.4` | ✅ At latest | — |
| `mcp` | `>=1.27.2` | `1.27.2` | ✅ At latest | — |
| `semantic-kernel` | `>=1.36.0` | `1.43.0` | Comment updated 2026-06-04; floor unchanged (OTel conflict) | Blocked |

**Citation sources (all verified 2026-06-08):**
- `openai 2.41.0`: https://pypi.org/pypi/openai/json; https://github.com/openai/openai-python/releases/tag/v2.41.0
- `google-genai 2.8.0`: https://pypi.org/pypi/google-genai/json
- `semantic-kernel 1.43.0`: https://pypi.org/pypi/semantic-kernel/json
- `langchain 1.3.4`: https://pypi.org/pypi/langchain/json
- `langgraph 1.2.4`: https://pypi.org/pypi/langgraph/json
- `cohere 7.0.3`: https://pypi.org/pypi/cohere/json
- `anthropic 0.107.1`: https://pypi.org/pypi/anthropic/json; https://github.com/anthropics/anthropic-sdk-python/releases
- `agent-framework 1.8.0`: https://pypi.org/pypi/agent-framework/json
- `mcp 1.27.2`: https://pypi.org/pypi/mcp/json
- `groq 1.4.0`: https://pypi.org/pypi/groq/json
- OpenAI deprecations: https://developers.openai.com/api/docs/deprecations
- Anthropic models: https://platform.claude.com/docs/en/docs/about-claude/models/overview

### 2.2 openai 2.39.0 / 2.40.0 / 2.41.0 API surface changes

None of these releases change the Chat Completions or Responses API request/response
shapes used by Gantry's `OpenAIAdapter` and `OpenAIResponsesAdapter`.

**2.41.0** (released 2026-06-03) adds:
- `responses.moderation` and `chat_completions.moderation` endpoints — new moderation
  surface. Gantry does not use moderation endpoints; no change needed.

**2.40.0** (released 2026-06-01) adds:
- Amazon Bedrock Responses support — new Bedrock client path. Gantry does not currently
  target Bedrock; this is a **needs-confirmation** item if Bedrock support is planned.

**2.39.0** (released 2026-06-01) adds:
- `additional_tools` field on the Responses API **response** object — server-side
  metadata; Gantry does not parse response objects directly, so no change needed.
- `ActionSearch.query` made optional — Action Search feature only, not used by Gantry.
- Workload identity in audit logs — server-side.

**Risk level:** Safe internal.

**Source:** https://github.com/openai/openai-python/releases (verified 2026-06-04)

### 2.3 langchain 1.3.4 / langgraph 1.2.4 API surface changes

Both are patch releases with no breaking changes to any API surface used by Gantry.

- `langchain 1.3.4`: minor internal fixes; `ChatOpenAI`, `ChatAnthropic`, `BaseTool`
  interfaces unchanged.
- `langgraph 1.2.4`: patch-level graph checkpoint and serialisation fixes; `StateGraph`,
  `CompiledGraph`, and interrupt/resume APIs unchanged.

`langgraph-sdk` resolved from `0.3.13 → 0.4.2` as a transitive consequence of the
floor bump; SDK is not imported directly by Gantry.

**Risk level:** Safe internal.

**Source:** https://pypi.org/pypi/langchain/json; https://pypi.org/pypi/langgraph/json (both verified 2026-06-03)

### 2.4 cohere 7.0.3 API surface changes

cohere 7.0.0 is a major release. The breaking change is:

- **Python minimum version raised from `^3.8` to `^3.10`** — compatible with Gantry's
  `requires-python = ">=3.10"`. No code change required.

API surface:
- `AsyncClientV2.rerank()` — still the correct async rerank call; signature and return
  type unchanged. Gantry's `CohereReranker` (`from cohere import AsyncClientV2`) continues
  to work without modification.
- `AsyncClientV2.embed()` — unchanged.

**Risk level:** Safe with shim (Python version requirement is already satisfied).

**Source:**
- https://github.com/cohere-ai/cohere-python/releases (verified 2026-06-03)
- https://docs.cohere.com/v2/reference/rerank (verified 2026-06-03)

### 2.5 openai model family changes

OpenAI has shipped the GPT-5.x model family:

| Model ID | Context | Max output | Notes |
|----------|---------|------------|-------|
| `gpt-5.5` | 1M tokens | 128k | New flagship |
| `gpt-5.4` | 1M tokens | 128k | More affordable than gpt-5.5 |
| `gpt-5.4-mini` | 400k tokens | 128k | Replaces gpt-4o-mini |
| `gpt-5.4-nano` | — | — | Replaces gpt-4.1-nano |

**Deprecated (shutdown 2026-10-23):**
- `gpt-4o` → replace with `gpt-5.5`
- `gpt-4o-mini` → replace with `gpt-5.4-mini`
- `gpt-4.1-nano` → replace with `gpt-5.4-nano`

`gpt-4.1` is **not** deprecated.

Gantry source code: `config.py` defaults to `gpt-4o-mini` (schema default, user-facing).
Changing this default is a breaking API change requiring a major version bump. Tracked as
a good next-step item (#4 below).

**Source:** https://developers.openai.com/api/docs/models (verified 2026-06-02);
https://developers.openai.com/api/docs/deprecations (verified 2026-06-02)

### 2.6 Blocked upgrades

**`semantic-kernel ≥ 1.43.0`:** OTel conflict with `agent-framework>=1.2.1`; floor
unchanged at `>=1.36.0`. Comment updated in `pyproject.toml` to reflect 1.43.0 as
latest stable (verified 2026-06-04).

**`google-adk ≥ 2.x`:** Requires `langgraph<0.4.8`, conflicting with
`langgraph>=1.2.4`. Floor unchanged at `>=1.14.1` in the combined extra. Latest
stable remains `2.1.0` (verified 2026-06-04).

**`crewai ≥ 1.14.6`:** Latest stable is `1.14.6`. `opentelemetry-api~=1.34.0` pin
conflicts with `agent-framework>=1.2.1` (requires `opentelemetry-api>=1.39.0`). Use
in a standalone environment without agent-framework.

---

## 3 · Provider API Refactors

### 3.1 OpenAI — no changes required ✅

`OpenAIResponsesAdapter` and `OpenAIAdapter` unchanged. The `additional_tools` field
added in openai 2.39.0 is a server-returned metadata field; Gantry does not need to
emit or parse it.

**Source:** https://platform.openai.com/docs/api-reference/responses (verified 2026-06-02).

### 3.2 Anthropic — no changes required ✅

Messages API throughout. `AnthropicAdapter` correctly emits `input_schema`,
`additionalProperties: false` for strict mode, `tool_use_id`, and `is_error`.
`AnthropicClient` in `anthropic_features.py` correctly handles adaptive and extended
thinking. Model capability tables updated to include `claude-opus-4-8`.

**Source:** https://docs.anthropic.com/en/api/messages (verified 2026-06-02).

### 3.3 Google Gemini — no changes required ✅

`GeminiAdapter` shapes are correct for google-genai 1.x and 2.x. No changes since
2026-06-01.

### 3.4 Mistral / Groq — no changes required ✅

Both route correctly through `AsyncOpenAI` (Mistral) and `AsyncGroq` (Groq).

---

## 4 · Framework Integration Refactors

### 4.1 Microsoft Agent Framework (AF 1.5.0 – 1.8.0) — no changes required ✅

All middleware, bridge, and provider surfaces verified current against AF 1.8.0 (released
2026-06-04).  Key AF 1.8.0 notes:

- `GantryToolBridge` is unaffected by the Skill API async refactor (Gantry uses its own
  `SkillRegistry`, not AF's `SkillsProvider`).
- `GantryToolBridge` wraps all callables as async already; AF 1.8.0's "sync tools off
  the event loop" optimisation is a no-op for Gantry-wrapped tools — no double-wrapping.
- AF 1.8.0's fix for unsafe serialisation of function arguments and for message-id
  collisions in compaction both benefit `GantryToolBridge` / `GantryContextProvider`
  without any code changes.
- The new `FunctionInvocationContext` (progressive tool exposure) is a potential future
  enhancement for `GantryContextProvider`; flagged as a good next-step item (#11 below).

### 4.2 LangGraph, AutoGen, CrewAI, LlamaIndex, Semantic Kernel — no changes required ✅

All unchanged from prior audit (2026-06-01).

---

## 5 · Universal Tool-Calling Design — no changes required ✅

`ToolSpec` contract, adapters, and per-provider translators unchanged. All seven
provider dialects verified current.

---

## 6 · Documentation and Example Updates

### 6.1 anthropic_features.py — updated this run ✅

Model capability tables updated to include `claude-opus-4-8`:

```diff
-# Extended thinking: fixed budget via budget_tokens. Not supported on claude-opus-4-7
-# (which only supports adaptive thinking). Supported on claude-sonnet-4-6,
-# claude-opus-4-6, claude-haiku-4-5, and earlier Claude 4 models.
+# Extended thinking: fixed budget via budget_tokens. Not supported on claude-opus-4-8
+# or claude-opus-4-7 (both only support adaptive thinking). Supported on
+# claude-sonnet-4-6, claude-opus-4-6, claude-haiku-4-5, and earlier Claude 4 models.

-# Adaptive thinking: model self-regulates depth; recommended for Opus 4.7, Opus 4.6+,
-# Sonnet 4.6+. Not available on claude-haiku-4-5.
+# Adaptive thinking: model self-regulates depth; recommended for Opus 4.8, Opus 4.7,
+# Opus 4.6+, Sonnet 4.6+. On claude-opus-4-8 effort defaults to "high" per Anthropic
+# docs — set explicitly to override. Not available on claude-haiku-4-5.
```

**File:** `agent_gantry/integrations/anthropic_features.py`
**Risk:** Safe internal (documentation only).
**Source:** https://platform.claude.com/docs/en/docs/about-claude/models/overview

### 6.2 Deprecated OpenAI model strings — migrated this run ✅

All `gpt-4o` and `gpt-4o-mini` model strings in examples and documentation have been
replaced with `gpt-5.5` and `gpt-5.4-mini` respectively (39 occurrences across 23 files).

**gpt-4o → gpt-5.5 (examples):**

| File | Occurrences |
|------|-------------|
| `examples/fast_track_demo.py` | 3 |
| `examples/llm_integration/multi_turn_conversation.py` | 1 |
| `examples/llm_integration/llm_demo.py` | 1 |
| `examples/llm_integration/openai_demo.py` | 1 |
| `examples/llm_intent_classification_example.py` | 1 |
| `examples/observability/multi_provider_metrics_demo.py` | 1 |
| `examples/agent_frameworks/langgraph_example.py` | 1 |
| `examples/agent_frameworks/semantic_kernel_example.py` | 1 |
| `examples/agent_frameworks/llamaindex_example.py` | 1 |
| `examples/agent_frameworks/crewai_example.py` | 1 |
| `examples/agent_frameworks/langchain_example.py` | 1 |
| `examples/agent_frameworks/autogen_example.py` | 1 |

**gpt-4o-mini → gpt-5.4-mini (examples):**

| File | Occurrences |
|------|-------------|
| `examples/tool_vector_db/main.py` | 1 |
| `examples/project_demo/main.py` | 1 |
| `examples/project_demo/main_persistent.py` | 1 |
| `examples/llm_integration/token_savings_demo.py` | 1 |
| `examples/observability/token_savings_demo.py` | 1 |
| `examples/testing_limits/real_world_30_tools_test.py` | 1 |

**Documentation files updated:**

| File | Changes |
|------|---------|
| `README.md` | 3 occurrences `gpt-4o` → `gpt-5.5` |
| `docs/reference/llm_sdk_compatibility.md` | 5 occurrences (`gpt-4o` → `gpt-5.5`, `openai>=2.37.0` → `>=2.40.0`) |
| `agent_gantry/README.md` | 1 occurrence |
| `agent_gantry/skills/agent-gantry/SKILL.md` | 3 occurrences |
| `agent_gantry/integrations/README.md` | 3 occurrences |
| `agent_gantry/core/README.md` | 1 occurrence |
| `examples/tool_vector_db/README.md` | 1 occurrence (`gpt-4o-mini` → `gpt-5.4-mini`) |

**Still pending (breaking change):**

| File | Current value | Replacement | Blocker |
|------|--------------|-------------|---------|
| `agent_gantry/schema/config.py` | `gpt-4o-mini` (schema default) | `gpt-5.4-mini` | Breaking — requires major version bump |

`agent_gantry/integrations/semantic_tools.py` uses `gpt-4.1` — **not deprecated**, no
action required.

**Risk:** Safe internal for examples and docs; breaking for `config.py` default.

### 6.3 gpt-4o-realtime-preview — discontinued, fixed this run ✅

`gpt-4o-realtime-preview` was discontinued by OpenAI on **2026-05-07** (before this audit
date). It appeared in `docs/reference/llm_sdk_compatibility.md` and has been replaced with
`gpt-realtime-1.5` (the current production realtime model).

**File changed:** `docs/reference/llm_sdk_compatibility.md`
**Risk:** Safe internal (documentation only).
**Source:** https://developers.openai.com/api/docs/deprecations (verified 2026-06-03)

---

## 7 · Pydantic Validator Fix (PR #220, previously documented)

Carried forward from the 2026-06-01 audit. No new findings.

---

## 8 · Security: SSRF Tests (2026-06-01, previously documented)

Carried forward. No new security findings in this run.

---

## 9 · Test and Migration Plan

### 9.1 Tests added this run

No new tests added this run. The executor and security tests (9 tests) and linter
tests (22 tests) all pass with the PR #225 and PR #226 fixes applied.

### 9.2 Regeneration checklist

- [x] `uv.lock` — regenerated (2026-06-04); key resolved-version changes:
  - `openai 2.40.0 → 2.41.0`
- [ ] VCR recordings / golden HTTP responses — not present; not applicable.
- [ ] Schema snapshots — not present; not applicable.

### 9.3 Changelog-ready migration notes

```
## [0.4.x] — 2026-06-05

### Changed
- `pyproject.toml`: update `agent-framework` comment to document AF 1.8.0 (released
  2026-06-04, within existing `>=1.5.0,<2.0.0` constraint — no floor bump required).
  Key AF 1.8.0 changes documented: `McpSkillsSource`, `FunctionInvocationContext`,
  background-agent harness, `AgentFileStore`/`FileAccessProvider`; bug fixes for
  unsafe serialisation of function arguments and message-id collisions in compaction;
  Skill API async refactor (experimental, not used by Gantry).
  Source: https://pypi.org/pypi/agent-framework/json (verified 2026-06-05)
- `pyproject.toml`: update `google-adk` comment to reflect 2.2.0 as latest stable
  (was 2.1.0; `langgraph<0.4.8` conflict with `langgraph>=1.2.4` persists; floor
  unchanged at `>=1.14.1`; standalone install example updated to `>=2.2.0`).
  Source: https://pypi.org/pypi/google-adk/json (verified 2026-06-05)

### Deprecation notice (action required by 2026-10-23)
- `agent_gantry/schema/config.py` default `"gpt-4o-mini"` must be migrated to
  `"gpt-5.4-mini"` before OpenAI's 2026-10-23 shutdown. This is a breaking change
  requiring a major version bump.

## [0.4.x] — 2026-06-04

### Fixed
- `agent_gantry/core/executor.py`: map `PermissionDeniedError` to
  `ExecutionStatus.PERMISSION_DENIED` (was incorrectly using `ExecutionStatus.FAILURE`),
  restoring correct downstream security logging and status-based routing.
  Source: PR #225.
- `agent_gantry/utils/registry_linter.py`: add `if other not in full: continue`
  pre-check in `_detect_cross_references` before invoking the compiled regex pattern,
  reducing CPU time for large registries where most tool pairs do not cross-reference.
  Source: PR #226.

### Changed
- `pyproject.toml`: bump `openai` floor `>=2.40.0 → >=2.41.0` (both `openai` and
  `mistral` extras). openai 2.41.0 adds `responses.moderation` and
  `chat_completions.moderation` endpoints; no breaking changes to any Gantry API surface.
  Source: https://github.com/openai/openai-python/releases/tag/v2.41.0
- `pyproject.toml`: update `google-genai` comment to reflect 2.8.0 as latest stable
  (no API changes; floor unchanged). Source: https://pypi.org/pypi/google-genai/json
- `pyproject.toml`: update `semantic-kernel` comment to reflect 1.43.0 as latest stable
  (OTel conflict with agent-framework persists; floor unchanged at >=1.36.0).
  Source: https://pypi.org/pypi/semantic-kernel/json

### Deprecation notice (action required by 2026-10-23)
- `agent_gantry/schema/config.py` default `"gpt-4o-mini"` must be migrated to
  `"gpt-5.4-mini"` before OpenAI's 2026-10-23 shutdown. This is a breaking change
  requiring a major version bump.

## [0.4.x] — 2026-06-03

### Changed
- `pyproject.toml`: bump `langchain` floor `>=1.3.2 → >=1.3.4` (patch release, no
  API changes). Source: https://pypi.org/pypi/langchain/json
- `pyproject.toml`: bump `langgraph` floor `>=1.2.2 → >=1.2.4` (patch release; graph
  checkpoint and serialisation fixes; no API changes). `langgraph-sdk` resolved to
  `0.4.2` as a transitive consequence. Source: https://pypi.org/pypi/langgraph/json
- `pyproject.toml`: bump `cohere` floor `>=6.0.0 → >=7.0.3`. cohere 7.0.0 is a major
  release; the only breaking change is Python >=3.10 (already satisfied by Gantry's
  `requires-python = ">=3.10"`). `AsyncClientV2.rerank()` API is unchanged.
  Source: https://github.com/cohere-ai/cohere-python/releases
- `docs/reference/llm_sdk_compatibility.md`: replace `gpt-4o-realtime-preview` with
  `gpt-realtime-1.5`; OpenAI discontinued `gpt-4o-realtime-preview` on 2026-05-07.
- All example files and documentation: replace `gpt-4o` with `gpt-5.5` and
  `gpt-4o-mini` with `gpt-5.4-mini` (39 occurrences across 23 files). OpenAI has set a
  shutdown date of 2026-10-23 for both deprecated models.
- `docs/reference/llm_sdk_compatibility.md`: update `openai` install pin from
  `>=2.37.0` to `>=2.40.0` (current floor).
- `agent_gantry/schema/config.py`: add deprecation comment on `gpt-4o-mini` default;
  the default is NOT changed (breaking change requiring a major version bump — tracked).

## [0.4.x] — 2026-06-02

### Changed
- `pyproject.toml`: bump `openai` floor `>=2.38.0 → >=2.40.0` (both `openai` and
  `mistral` extras). openai 2.39.0 adds `additional_tools` in Responses API responses
  and workload identity in audit logs; 2.40.0 adds Amazon Bedrock Responses support.
  No breaking changes to any Gantry API surface.
- `agent_gantry/integrations/anthropic_features.py`: model capability tables updated
  to include `claude-opus-4-8` (adaptive thinking only; effort defaults to "high";
  extended thinking not supported).

### Added
- `tests/test_anthropic_features.py::TestClaudeOpus48ThinkingGuards` (6 tests):
  documents expected thinking-mode payloads for `claude-opus-4-8`, including adaptive
  medium/high effort, no-thinking plain messages, `display="omitted"`, and the
  `create_anthropic_client` factory path.
```

---

## 10 · Must-Change Now vs Good Next Steps

### Must-change now

| # | Change | File | Risk | Status |
|---|--------|------|------|--------|
| 1 | ~~Bump `openai` floor `>=2.38.0 → >=2.40.0`~~ | `pyproject.toml` | Safe internal | ✅ Done (2026-06-02 run) |
| 2 | ~~Bump `langchain` floor `>=1.3.2 → >=1.3.4`~~ | `pyproject.toml` | Safe internal | ✅ Done (2026-06-03 run) |
| 3 | ~~Bump `langgraph` floor `>=1.2.2 → >=1.2.4`~~ | `pyproject.toml` | Safe internal | ✅ Done (2026-06-03 run) |
| 4 | ~~Bump `cohere` floor `>=6.0.0 → >=7.0.3`~~ | `pyproject.toml` | Safe with shim | ✅ Done (2026-06-03 run) |
| 5 | ~~Fix `gpt-4o-realtime-preview` → `gpt-realtime-1.5`~~ | `docs/` | Safe internal | ✅ Done (2026-06-03 run) |
| 6 | ~~Bump `openai` floor `>=2.40.0 → >=2.41.0`~~ | `pyproject.toml` | Safe internal | ✅ Done (2026-06-04 run) |
| 7 | ~~Fix `ExecutionStatus.PERMISSION_DENIED` in `_check_security_policy`~~ | `core/executor.py` | Safe internal | ✅ Done (2026-06-04 run) |
| 8 | ~~Bump `anthropic` floor `>=0.105.2 → >=0.107.1`~~ | `pyproject.toml` | Safe internal | ✅ **Done this run** (2026-06-08) |

### Good next-step improvements

| # | Change | File | Risk | Status |
|---|--------|------|------|--------|
| 1 | ~~Add `claude-opus-4-8` model capability docs~~ | `anthropic_features.py` | Safe internal | ✅ Done (2026-06-02 run) |
| 2 | ~~Add `TestClaudeOpus48ThinkingGuards` (6 tests)~~ | `tests/test_anthropic_features.py` | Safe internal | ✅ Done (2026-06-02 run) |
| 3 | ~~Migrate examples/docs from `gpt-4o`/`gpt-4o-mini` to `gpt-5.5`/`gpt-5.4-mini`~~ | `examples/`, `docs/`, `README.md` | Safe internal | ✅ Done (2026-06-03 run, 39 occurrences, 23 files) |
| 4 | ~~Registry linter substring pre-check~~ | `utils/registry_linter.py` | Safe internal | ✅ Done (2026-06-04 run) |
| 5 | ~~Update AF comment for 1.8.0~~ | `pyproject.toml` | Safe internal | ✅ Done (2026-06-05 run) |
| 6 | ~~Update google-adk comment to 2.2.0~~ | `pyproject.toml` | Safe internal | ✅ Done (2026-06-05 run) |
| 7 | ~~Update `anthropic_features.py` docstrings — replace vague "earlier Claude 4 models" with explicit list; add claude-sonnet-4 / claude-opus-4 retirement notice~~ | `agent_gantry/integrations/anthropic_features.py` | Safe internal | ✅ **Done this run** (2026-06-08) |
| 8 | ~~Update install-pin docs (anthropic `0.101.0→0.107.1`, openai `2.40.0→2.41.0`, groq `1.2.0→1.4.0`)~~ | `docs/reference/llm_sdk_compatibility.md` | Safe internal | ✅ **Done this run** (2026-06-08) |
| 9 | Bump `config.py` default from `gpt-4o-mini` → `gpt-5.4-mini` | `agent_gantry/schema/config.py` | Breaking — major version bump | ⏳ Pending (deadline 2026-10-23) |
| 10 | Validate `semantic-kernel>=1.43.0` in isolation with `agent-framework`; bump floor if OTel conflict resolved | `pyproject.toml` | Safe with shim | ⏳ Pending |
| 11 | Validate `google-adk>=2.2.0` standalone env; add docs page showing Workflow Runtime pattern (v2 major) | `docs/`, `examples/` | Needs confirmation | ⏳ Pending |
| 12 | Add MAF migration guide (`docs/migrating-from-autogen.md`) for AutoGen → AF 1.x teams | `docs/` | Documentation | ⏳ Pending |
| 13 | Explore `FunctionInvocationContext` (AF 1.8.0) for progressive tool exposure in `GantryContextProvider` | `agent_gantry/integrations/agent_framework_provider.py` | Safe with shim | ⏳ Pending |
