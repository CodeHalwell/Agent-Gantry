# Agent-Gantry Modernisation Audit

**Date:** 2026-06-02
**Repository:** `CodeHalwell/Agent-Gantry` · version `0.4.0`
**Branch:** `claude/cool-hopper-R79WP` (based on `main` after PR #221 — 2026-06-01 audit run)
**Auditor:** Claude (claude-sonnet-4-6)

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
> - **2026-06-02** (this run, `claude/cool-hopper-R79WP`, based on main after PR #221):
>   Bumped `openai>=2.38.0→>=2.40.0`; added `claude-opus-4-8` model capability
>   documentation; added `TestClaudeOpus48ThinkingGuards` (6 tests). See §1 for details.

---

## 1 · Executive Summary

| Severity | Count | Areas |
|----------|-------|-------|
| **Must-change now** | 1 | openai floor bump `>=2.38.0 → >=2.40.0` |
| **Good next-step (completed this run)** | 2 | claude-opus-4-8 model docs; Opus 4.8 guard tests |
| **Good next-step (remaining)** | 5 | semantic-kernel floor; google-adk 2.x docs; MAF migration guide; gpt-4o/gpt-4o-mini migration; gpt-5.x example updates |

**New findings since 2026-06-01:**

1. **`openai` 2.40.0 and 2.39.0** (both released 2026-06-01) — floor bumped to `>=2.40.0`.
   No breaking changes to Chat Completions or Responses API surfaces.
2. **`claude-opus-4-8` (NextOpus)** — new flagship Anthropic model. Supports adaptive
   thinking only (no extended thinking). Effort defaults to `"high"`. Documentation
   and guard tests added.
3. **`gpt-4o` and `gpt-4o-mini` deprecated** — OpenAI announced shutdown on
   **2026-10-23**; replacements are `gpt-5.5` and `gpt-5.4-mini` respectively.
   Examples using these models should be migrated before that date.
4. **OpenAI GPT-5.x model family** — `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`
   are now the flagship models. No Gantry code change required; examples should be
   updated where appropriate.
5. **All other packages** — unchanged since 2026-06-01. No further bumps required.

---

## 2 · Dependency Upgrade Plan

### 2.1 Package-by-package table (verified 2026-06-02)

Sources verified against PyPI JSON API.

| Package | Current floor | Latest stable | Action | Risk |
|---------|--------------|---------------|--------|------|
| `openai` | `>=2.38.0` → **`>=2.40.0`** | `2.40.0` | ✅ **Bumped this run** | Safe internal |
| `agent-framework` | `>=1.5.0,<2.0.0` | `1.7.0` | ✅ Within range — no change | Safe internal |
| `anthropic` | `>=0.105.2` | `0.105.2` | ✅ At latest | — |
| `autogen-agentchat` | `>=0.7.5` | `0.7.5` | ✅ At latest | — |
| `cohere` | `>=6.0.0` | (stable, unchecked) | No action | — |
| `crewai` | `>=1.6.1` | `1.14.6` | Note only (OTel conflict with AF) | Blocked |
| `google-adk` | `>=1.14.1` | `2.1.0` | Note only (langgraph conflict) | Blocked |
| `google-genai` | `>=1.75.0` | `2.7.0` | ✅ Accessible standalone | Note only |
| `groq` | `>=1.4.0` | `1.4.0` | ✅ At latest | — |
| `langchain` | `>=1.3.2` | `1.3.2` | ✅ At latest | — |
| `langchain-openai` | `>=1.2.2` | `1.2.2` | ✅ At latest | — |
| `langgraph` | `>=1.2.2` | `1.2.2` | ✅ At latest | — |
| `mcp` | `>=1.27.2` | `1.27.2` | ✅ At latest | — |
| `semantic-kernel` | `>=1.36.0` | `1.42.0` | OTel conflict with AF; floor unchanged | Blocked |

**Citation sources (all verified 2026-06-02):**
- `openai 2.40.0`: https://pypi.org/pypi/openai/json
- `anthropic 0.105.2`: https://pypi.org/pypi/anthropic/json
- `agent-framework 1.7.0`: https://pypi.org/pypi/agent-framework/json
- `mcp 1.27.2`: https://pypi.org/pypi/mcp/json
- `groq 1.4.0`: https://pypi.org/pypi/groq/json
- OpenAI deprecations: https://developers.openai.com/api/docs/deprecations
- Anthropic models: https://platform.claude.com/docs/en/docs/about-claude/models/overview

### 2.2 openai 2.39.0 / 2.40.0 API surface changes

Neither release changes the Chat Completions or Responses API request/response shapes
used by Gantry's `OpenAIAdapter` and `OpenAIResponsesAdapter`.

**2.39.0** adds:
- `additional_tools` field on the Responses API **response** object — this is server-side
  metadata about hosted tools used; Gantry does not parse response objects directly,
  so no change needed.
- `ActionSearch.query` made optional — affects the Action Search feature only, not used
  by Gantry.
- Workload identity in audit logs — server-side.

**2.40.0** adds:
- Amazon Bedrock Responses support — new Bedrock client path. Gantry does not currently
  target Bedrock; this is a **needs-confirmation** item if Bedrock support is planned.

**Risk level:** Safe internal.

**Source:** https://github.com/openai/openai-python/releases (verified 2026-06-02)

### 2.3 openai model family changes

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

### 2.4 Blocked upgrades

**`semantic-kernel ≥ 1.42.0`:** OTel conflict with `agent-framework>=1.2.1`; floor
unchanged at `>=1.36.0`.

**`google-adk ≥ 2.x`:** Requires `langgraph<0.4.8`, conflicting with
`langgraph>=1.2.2`. Floor unchanged at `>=1.14.1` in the combined extra.

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

### 4.1 Microsoft Agent Framework (AF 1.5.0 – 1.7.0) — no changes required ✅

Unchanged from 2026-06-01 audit. All middleware, bridge, and provider surfaces remain
current against AF 1.7.0.

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

### 6.2 Examples using deprecated OpenAI models — pending migration

The following examples use `gpt-4o` or `gpt-4o-mini` (shutdown: 2026-10-23).
Migration is tracked as a good next-step item; no breaking change until that date.

| File | Model used | Replacement |
|------|-----------|------------|
| `examples/fast_track_demo.py` | `gpt-4o` | `gpt-5.5` |
| `examples/llm_integration/multi_turn_conversation.py` | `gpt-4o` | `gpt-5.5` |
| `examples/llm_integration/llm_demo.py` | `gpt-4o` | `gpt-5.5` |
| `examples/agent_frameworks/langgraph_example.py` | `gpt-4o` | `gpt-5.5` |
| `examples/agent_frameworks/semantic_kernel_example.py` | `gpt-4o` | `gpt-5.5` |
| `examples/agent_frameworks/llamaindex_example.py` | `gpt-4o` | `gpt-5.5` |
| `examples/agent_frameworks/crewai_example.py` | `gpt-4o` | `gpt-5.5` |
| `examples/agent_frameworks/langchain_example.py` | `gpt-4o` | `gpt-5.5` |
| `examples/agent_frameworks/autogen_example.py` | `gpt-4o` | `gpt-5.5` |
| `examples/tool_vector_db/main.py` | `gpt-4o-mini` | `gpt-5.4-mini` |
| `examples/project_demo/main.py` | `gpt-4o-mini` | `gpt-5.4-mini` |
| `examples/project_demo/main_persistent.py` | `gpt-4o-mini` | `gpt-5.4-mini` |
| `examples/llm_integration/token_savings_demo.py` | `gpt-4o-mini` | `gpt-5.4-mini` |
| `agent_gantry/schema/config.py` | `gpt-4o-mini` (default) | `gpt-5.4-mini` (breaking) |

`agent_gantry/integrations/semantic_tools.py` uses `gpt-4.1` — **not deprecated**, no
action required.

---

## 7 · Pydantic Validator Fix (PR #220, previously documented)

Carried forward from the 2026-06-01 audit. No new findings.

---

## 8 · Security: SSRF Tests (2026-06-01, previously documented)

Carried forward. No new security findings in this run.

---

## 9 · Test and Migration Plan

### 9.1 Tests added this run

| Test class | Tests | File | Description |
|------------|-------|------|-------------|
| `TestClaudeOpus48ThinkingGuards` | 6 | `tests/test_anthropic_features.py` | Documents adaptive/extended/plain thinking behaviour for claude-opus-4-8 |

**28 tests pass** in `test_anthropic_features.py` (22 pre-existing + 6 new).

### 9.2 Regeneration checklist

- [x] `uv.lock` — regenerated; `openai 2.38.0 → 2.40.0`.
- [ ] VCR recordings / golden HTTP responses — not present; not applicable.
- [ ] Schema snapshots — not present; not applicable.

### 9.3 Changelog-ready migration notes

```
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

### Deprecation notice (action required by 2026-10-23)
- OpenAI has deprecated `gpt-4o` (→ `gpt-5.5`) and `gpt-4o-mini` (→ `gpt-5.4-mini`)
  with a shutdown date of **2026-10-23**. Examples and the `config.py` default of
  `"gpt-4o-mini"` must be migrated before that date. Changing `config.py`'s default
  is a breaking change requiring a version bump.
```

---

## 10 · Must-Change Now vs Good Next Steps

### Must-change now

| # | Change | File | Risk | Status |
|---|--------|------|------|--------|
| 1 | Bump `openai` floor `>=2.38.0 → >=2.40.0` | `pyproject.toml` | Safe internal | ✅ **Done this run** |

### Good next-step improvements

| # | Change | File | Risk | Status |
|---|--------|------|------|--------|
| 1 | ~~Add `claude-opus-4-8` model capability docs~~ | `anthropic_features.py` | Safe internal | ✅ **Done this run** |
| 2 | ~~Add `TestClaudeOpus48ThinkingGuards` (6 tests)~~ | `tests/test_anthropic_features.py` | Safe internal | ✅ **Done this run** |
| 3 | Migrate examples from `gpt-4o`/`gpt-4o-mini` to `gpt-5.5`/`gpt-5.4-mini` | `examples/` | Safe internal | ⏳ Pending (deadline 2026-10-23) |
| 4 | Bump `config.py` default from `gpt-4o-mini` → `gpt-5.4-mini` | `agent_gantry/schema/config.py` | Breaking — major version bump | ⏳ Pending |
| 5 | Validate `semantic-kernel>=1.42.0` in isolation with `agent-framework`; bump floor if OTel conflict resolved | `pyproject.toml` | Safe with shim | ⏳ Pending |
| 6 | Validate `google-adk>=2.1.0` standalone env; add docs page showing Workflow Runtime pattern (v2 major) | `docs/`, `examples/` | Needs confirmation | ⏳ Pending |
| 7 | Add MAF migration guide (`docs/migrating-from-autogen.md`) for AutoGen → AF 1.x teams | `docs/` | Documentation | ⏳ Pending |
