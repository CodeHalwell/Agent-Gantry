# Agent-Gantry Modernisation Audit

**Date:** 2026-05-29 (supersedes 2026-05-27 run)
**Repository:** `CodeHalwell/Agent-Gantry` · version `0.4.0`
**Branch:** `claude/cool-hopper-34gYr` (based on `main`)
**Auditor:** Claude (Sonnet 4.6)

> **PR history incorporated.** This audit builds on the 2026-05-27 run which
> already incorporated security fix PR #207 (SSRF bypass), accessibility fix PR #208
> (search combobox Escape focus), audit+lock-bump commit #209 (langchain/langgraph),
> and docs CSS commit #210 (prefers-reduced-motion). The 2026-05-29 run adds
> agent-framework 1.7.0, anthropic 0.105.0, and claude-opus-4-8 model awareness.

---

## 1 · Executive Summary

| Severity | Count | Areas |
|----------|-------|-------|
| **Must-change now** | 3 | AF floor bump (1.5.0→1.7.0), anthropic floor bump (0.104.1→0.105.0), claude-opus-4-8 model awareness in anthropic_features.py |
| **Good next-step** | 4 | google-genai comment refresh (2.7.0), HarnessAgent exploration, google-adk 2.x standalone docs, MAF migration guide |

The repository remains in excellent overall health for `v0.4.0`. All three provider API
surfaces are current and no Assistants API or deprecated model IDs were found.

**New findings this run (2026-05-29):**

1. **`agent-framework` 1.7.0** (released 2026-05-28) — lock was pinned to 1.5.0. The
   1.7.0 breaking change (removed Python-only declarative actions in
   `agent-framework-declarative`) does **not** affect Gantry, which uses only core
   primitives. Floor bumped to `>=1.7.0,<2.0.0`; lock updated.

2. **`anthropic` 0.105.0/0.105.2** (released 2026-05-28/29) — adds `claude-opus-4-8`
   model support, mid-conversation system blocks, and `usage.output_tokens_details`.
   Floor bumped to `>=0.105.0`; lock updated.

3. **`claude-opus-4-8`** is now Anthropic's primary/latest Opus model. Adaptive thinking
   only (no extended thinking). `anthropic_features.py` updated to include it.

4. **Model retirement warning** — `claude-sonnet-4-20250514` and
   `claude-opus-4-20250514` retire on **June 15, 2026** (17 days from today). A grep
   confirms neither deprecated ID appears anywhere in the codebase. ✅

5. **`google-genai` 2.7.0** — no breaking changes to function calling. Comment updated;
   floor unchanged at `>=1.75.0` (google-adk conflict in combined extra).

**Actionable items completed in this audit run:**
1. Bump `agent-framework` floor `>=1.5.0,<2.0.0 → >=1.7.0,<2.0.0`.
2. Bump `anthropic` floor `>=0.104.1 → >=0.105.0`.
3. Update `pyproject.toml` comments for `agent-framework` 1.7.0, `google-genai` 2.7.0.
4. Update `anthropic_features.py` to add `claude-opus-4-8` to model support tables.
5. Regenerate `uv.lock` (agent-framework 1.5.0→1.7.0, anthropic 0.104.1→0.105.2).

**Items completed in the 2026-05-27 run (preserved):**
6. Bump `langchain>=1.3.2`, `langchain-openai>=1.2.2`, `langgraph>=1.2.2`.
7. Regenerated `uv.lock` for those three packages.
8. Fixed SSRF bypass (PR #207), accessibility (PR #208).

---

## 2 · Dependency Upgrade Plan

### 2.1 Package-by-package table

Sources verified against PyPI JSON API (cited URLs below).

| Package | Previous floor | Current floor | Latest stable | Action | Risk |
|---------|---------------|---------------|---------------|--------|------|
| `agent-framework` | `>=1.5.0,<2.0.0` | `>=1.7.0,<2.0.0` | `1.7.0` | **Bumped ✅** | Safe internal — breaking change in `declarative` sub-package, not used by Gantry |
| `anthropic` | `>=0.104.1` | `>=0.105.0` | `0.105.2` | **Bumped ✅** | Safe internal |
| `google-genai` | `>=1.75.0` | `>=1.75.0` | `2.7.0` | Comment updated ✅ | Floor unchanged (google-adk conflict) |
| `openai` | `>=2.38.0` | `>=2.38.0` | `2.38.0` | ✅ at latest | — |
| `langchain` | `>=1.3.2` | `>=1.3.2` | `1.3.2` | ✅ at latest | — |
| `langchain-openai` | `>=1.2.2` | `>=1.2.2` | `1.2.2` | ✅ at latest | — |
| `langgraph` | `>=1.2.2` | `>=1.2.2` | `1.2.2` | ✅ at latest | — |
| `autogen-agentchat` | `>=0.7.5` | `>=0.7.5` | `0.7.5` | ✅ at latest | — |
| `autogen-ext[openai]` | `>=0.7.5` | `>=0.7.5` | `0.7.5` | ✅ at latest | — |
| `crewai` | `>=1.6.1` | `>=1.6.1` | `1.14.5` | Note only — OTel conflict | Blocked |
| `google-adk` | `>=1.14.1` | `>=1.14.1` | `2.1.0` | Note only — langgraph conflict | Blocked |
| `mcp` | `>=1.27.1` | `>=1.27.1` | `1.27.1` | ✅ at latest | — |
| `groq` | `>=1.2.0` | `>=1.2.0` | `1.2.0` | ✅ at latest | — |
| `semantic-kernel` | `>=1.36.0` | `>=1.36.0` | `1.42.0` | Note only — OTel conflict | Blocked |
| `llama-index-core` | `>=0.14.22` | `>=0.14.22` | `0.14.22` | ✅ at latest | — |

**Citation sources (verified 2026-05-29):**
- `agent-framework 1.7.0`: https://pypi.org/pypi/agent-framework/json · https://github.com/microsoft/agent-framework/releases
- `anthropic 0.105.2`: https://pypi.org/pypi/anthropic/json · https://github.com/anthropics/anthropic-sdk-python/releases
- `google-genai 2.7.0`: https://pypi.org/pypi/google-genai/json · https://github.com/googleapis/python-genai/releases
- `openai 2.38.0`: https://pypi.org/pypi/openai/json
- `langchain 1.3.2`: https://pypi.org/pypi/langchain/json
- `langgraph 1.2.2`: https://pypi.org/pypi/langgraph/json
- `mcp 1.27.1`: https://pypi.org/pypi/mcp/json

### 2.2 Commands applied in this run

```bash
# Bumped floors in pyproject.toml, then regenerated the lock:
uv lock --upgrade-package agent-framework --upgrade-package anthropic

# Output:
# Updated agent-framework v1.5.0 -> v1.7.0
# Updated agent-framework-core v1.5.0 -> v1.7.0
# Updated anthropic v0.104.1 -> v0.105.2
```

No solver conflicts were raised. The two packages updated cleanly within the existing
constraint set.

### 2.3 agent-framework 1.7.0 — breaking change analysis

**Breaking change:** Python-only declarative actions removed; alias kinds renamed
in `agent-framework-declarative`.

**Impact on Gantry:** None. Gantry's AF integration (`agent_framework_bridge.py`,
`agent_framework_provider.py`, `agent_framework_middleware.py`) uses:
- `agent_framework.Agent`
- `agent_framework.AgentExecutor`, `WorkflowAgent`, `WorkflowBuilder`
- `agent_framework.orchestrations.SequentialBuilder`, `HandoffBuilder`
- `agent_framework.ContextProvider`, `chat_middleware`, `FunctionMiddleware`
- `agent_framework.telemetry.disable_instrumentation`

None of these symbols reside in `agent-framework-declarative`. Classification: **safe internal**.

**New features in 1.7.0 (not yet used by Gantry — good next-step):**
- `HarnessAgent` / background-agents harness provider
- `A2AAgentSession` with referenced task IDs and input-required support

Source: https://github.com/microsoft/agent-framework/releases (verified 2026-05-29)

### 2.4 Blocked upgrade: `google-adk ≥ 2.x`

`google-adk 2.1.0` (latest stable) requires `langgraph<0.4.8`, incompatible with
`langgraph>=1.2.2`. Floor stays at `>=1.14.1`. Standalone install works:
`pip install "agent-gantry[google-genai]" "google-adk>=2.1.0"`.

### 2.5 Blocked upgrade: `semantic-kernel ≥ 1.42.0`

`opentelemetry-api` range conflict with `agent-framework>=1.2.1`. Floor stays at
`>=1.36.0`. Install in a separate environment without `agent-framework` to use 1.42.0.

---

## 3 · Provider API Refactors

### 3.1 OpenAI

**Finding:** No Assistants API usage. Responses API (`client.responses.create`) is
the primary path. Chat Completions retained as a supported secondary path. ✅

**Tool schema** (`OpenAIResponsesAdapter`): flat `{type: "function", name: ...}` shape —
correct per the Responses API specification. ✅

**Assistants API sunset:** August 26, 2026. No code touches the Assistants API. ✅

Source: https://platform.openai.com/docs/api-reference/responses (verified 2026-05-27)

**Risk level:** Safe internal — no changes required.

### 3.2 Anthropic

**Finding:** Messages API is used throughout with correct shapes. ✅

**New model in 0.105.0: `claude-opus-4-8`**

`claude-opus-4-8` is now Anthropic's primary/latest Opus model (added in anthropic
SDK 0.105.0, released 2026-05-28). Capabilities confirmed against the official model
table (https://platform.claude.com/docs/en/docs/about-claude/models/overview,
verified 2026-05-29):

| Feature | claude-opus-4-8 | claude-opus-4-7 | claude-sonnet-4-6 | claude-haiku-4-5 |
|---------|-----------------|-----------------|-------------------|-----------------|
| Extended thinking | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| Adaptive thinking | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| Context window | 1M tokens | 1M tokens | 1M tokens | 200k tokens |
| Max output | 128k tokens | 128k tokens | 64k tokens | 64k tokens |

**Changes applied to `agent_gantry/integrations/anthropic_features.py`:**
- `AnthropicFeatures` dataclass comment: added `claude-opus-4-8` to models that
  do not support extended thinking.
- `AnthropicFeatures` adaptive thinking comment: added `claude-opus-4-8` to the
  list of supported models.
- `AnthropicClient` docstring: updated extended/adaptive thinking model lists.
- `create_anthropic_client` docstring: updated all model references.

**New field: `usage.output_tokens_details`** — returned automatically by the API
when using 0.105.0+. No code change needed; users can access it via
`response.usage.output_tokens_details` from the SDK object. Noted here for
observability integrators.

**⚠️ Model retirement — June 15, 2026 (17 days from today):**
`claude-sonnet-4-20250514` (alias `claude-sonnet-4-0`) and
`claude-opus-4-20250514` (alias `claude-opus-4-0`) are **deprecated and retire on
June 15, 2026**. A full grep of the codebase confirms neither deprecated model ID
appears in any source file or example. ✅ No action required.

Source: https://platform.claude.com/docs/en/docs/about-claude/models/overview (verified 2026-05-29)

**Risk level:** Safe internal — no breaking changes.

### 3.3 Google Gemini / Gen AI

**Finding:** Correct `functionDeclarations` / `functionResponse` shapes used throughout. ✅

**google-genai 2.7.0** — no breaking changes to `generateContent` or function calling.
New features (computer_use field for Vertex, Reinforcement Tuning) are not used by
Gantry and require no code changes. ✅

`gemini-2.5-flash` is the current stable model used in examples. ✅

Source: https://github.com/googleapis/python-genai/releases (verified 2026-05-29)

**Risk level:** Safe internal — no changes required.

### 3.4 Mistral

`mistralai` quarantined on PyPI (2026-05-12). Correctly migrated to
`AsyncOpenAI(base_url="https://api.mistral.ai/v1")`. ✅

### 3.5 Groq

`AsyncGroq` with `chat.completions.create()`. `GroqAdapter` inherits `OpenAIAdapter`.
All shapes correct. ✅

---

## 4 · Framework Integration Refactors

### 4.1 Microsoft Agent Framework (AF 1.7.0) — Priority review

Gantry's AF integration aligns with `agent-framework>=1.7.0,<2.0.0`.

#### 4.1.1 Agent construction and configuration

`GantryToolBridge.build_agent` and `as_agent` use:

```python
Agent(client, instructions, name=name, tools=tools, middleware=middleware)
```

Idiomatic AF 1.x constructor signature. ✅

#### 4.1.2 Workflow subsystem

| Builder | Status |
|---------|--------|
| `SequentialBuilder` | ✅ `build_sequential_workflow` |
| `HandoffBuilder` | ✅ `build_handoff_workflow` |
| `WorkflowBuilder` + `AgentExecutor` | ✅ `build_workflow` |

All imported from `agent_framework.orchestrations`. ✅

#### 4.1.3 AF 1.7.0 — new features (not yet integrated)

- **`HarnessAgent`**: New agent class for background/harness execution patterns.
  Potential use: long-running Gantry tool pipelines that should run in a background
  harness with progress reporting. Flagged as "good next-step" below.
- **`A2AAgentSession`**: Referenced task IDs and input-required support for A2A
  agent-to-agent communication. Could complement Gantry's A2A integration
  (`agent_gantry/integrations/a2a_bridge.py`).

#### 4.1.4 AF 1.6.0 ContextVar concurrency bug — workaround in place

The workaround introduced in the 2026-05-27 run remains correct for AF 1.7.0:
`disable_af_instrumentation()` calls
`agent_framework.telemetry.disable_instrumentation()` when AF ≥ 1.6.0 is detected.
Sequential workflows are unaffected.

Source: https://pypi.org/pypi/agent-framework/json (verified 2026-05-29)

#### 4.1.5 Middleware pipeline

`GantryApprovalMiddleware`, `GantryObservabilityMiddleware`,
`GantryToolChoiceMiddleware` — all correct. Lazy AF import preserved. ✅

#### 4.1.6 Context provider

`GantryContextProvider` subclasses `ContextProvider` dynamically. Per-run and
per-call strategies implemented. `source_id` keying co-operates with
`SkillsProvider`. ✅

**Risk level:** Safe internal — no changes required for 1.7.0.

### 4.2 LangGraph

`create_react_agent` from `langgraph.prebuilt` and `ChatOpenAI` from
`langchain_openai`. Current LangGraph 1.x idioms. ✅

### 4.3 AutoGen (AG2 0.7.5)

`autogen_agentchat.agents.AssistantAgent` and
`autogen_ext.models.openai.OpenAIChatCompletionClient` — current AG2 v0.4 API.
Migration note to MAF included. ✅

### 4.4 CrewAI / Semantic Kernel / LlamaIndex

Unchanged. See §2.5 for blocked upgrade paths.

---

## 5 · Universal Tool-Calling Design

### 5.1 `ToolSpec` contract

The provider-agnostic contract is implemented via
`agent_gantry/adapters/tool_spec/base.py` (`ToolCallPayload`, `ToolSpecAdapter`
Protocol) and `agent_gantry/adapters/tool_spec/providers.py`.

| Adapter | Dialect | Status |
|---------|---------|--------|
| `OpenAIAdapter` | `openai` | ✅ Chat Completions |
| `OpenAIResponsesAdapter` | `openai_responses` | ✅ Responses API |
| `AnthropicAdapter` | `anthropic` | ✅ Messages API |
| `GeminiAdapter` | `gemini` | ✅ functionDeclarations |
| `MistralAdapter` | `mistral` | ✅ Inherits OpenAIAdapter |
| `GroqAdapter` | `groq` | ✅ Inherits OpenAIAdapter |
| `AgentFrameworkAdapter` | `agent_framework` | ✅ OpenAI-compatible + metadata |

### 5.2 Tool name normalisation

Validated at registration; passed verbatim through all adapters. ✅

### 5.3 Required vs optional arguments

`AnthropicAdapter.to_provider_schema(strict=True)` injects `additionalProperties: false`. ✅
`OpenAIResponsesAdapter.to_provider_schema(strict=True)` sets `schema["strict"] = True`. ✅

### 5.4 Tool call IDs

| Adapter | ID field |
|---------|----------|
| `OpenAIAdapter` | `payload["id"]` |
| `OpenAIResponsesAdapter` | `payload["call_id"]` |
| `AnthropicAdapter` | `payload["id"]` |
| `GeminiAdapter` | `payload.get("id")` |
| `AgentFrameworkAdapter` | `payload.get("id") or payload.get("call_id")` |

All IDs echoed back in `format_tool_result` for parallel tool call correlation. ✅

### 5.5 Parallel tool calls

`GeminiAdapter.format_tool_result` correctly places `id` inside `functionResponse`
(matches `FunctionResponse` proto schema). ✅

---

## 6 · Documentation and Example Updates

All examples use current model families:
- OpenAI: `gpt-4.1` (Responses API), `gpt-4o` (Chat Completions) ✅
- Anthropic: `claude-sonnet-4-6` ✅ (not deprecated)
- Gemini: `gemini-2.5-flash` ✅

No deprecated model IDs (`claude-sonnet-4-20250514`, `claude-opus-4-20250514`,
`claude-sonnet-4-0`, `claude-opus-4-0`) appear anywhere in the codebase. ✅

---

## 7 · Security Audit

### 7.1 SSRF bypass — malformed URL ports (PR #207)

Fixed. `SecurityPolicy._extract_domains` correctly handles `ValueError` from
`urllib.parse.urlparse(...).port`. URLs that lack a valid hostname after port-parsing
raises are blocked as `<invalid_domain>`. Test coverage in
`tests/test_security.py::test_ssrf_port_bypass`. ✅

---

## 8 · Test and Migration Plan

### 8.1 Tests to add or update

| Test | Status | Notes |
|------|--------|-------|
| `test_security.py::test_ssrf_port_bypass` | ✅ Exists | Covers SSRF bypass vector |
| `test_tool_spec_adapters.py` | ✅ Exists | Covers all provider adapters |
| `test_agent_framework_integration.py` | ✅ Exists | Covers AF bridge, provider, middleware |
| AF 1.7.0 regression | ✅ No API changes to core | No new test needed |
| `anthropic 0.105.0` regression | ✅ No breaking changes | No new test needed |
| `test_anthropic_features.py` claude-opus-4-8 | ⚠️ **Recommended** | Add assertions for `claude-opus-4-8` in thinking-mode guards |
| `test_ssrf_general_invalid_port` | ⚠️ Optional | Document `http://example.com:evil.com` expected behaviour |

### 8.2 Regeneration checklist

- [x] `uv.lock` regenerated: agent-framework 1.5.0→1.7.0, anthropic 0.104.1→0.105.2.
- [ ] VCR recordings / golden HTTP responses — none present; not applicable.
- [ ] Schema snapshots — none present; not applicable.

### 8.3 Changelog-ready migration notes

```
## [0.4.x] — 2026-05-29

### Changed
- `agent-framework` floor bumped `>=1.5.0,<2.0.0 → >=1.7.0,<2.0.0` (HarnessAgent,
  A2AAgentSession improvements; declarative breaking change is in
  agent-framework-declarative which Gantry does not use).
- `anthropic` floor bumped `>=0.104.1 → >=0.105.0` (claude-opus-4-8 model support,
  mid-conversation system blocks, usage.output_tokens_details).
- `anthropic_features.py` updated to document claude-opus-4-8 thinking-mode
  restrictions (adaptive only; no extended thinking).
- `pyproject.toml` comments updated for google-genai (2.7.0).

### Deprecated (external)
- `claude-sonnet-4-20250514` and `claude-opus-4-20250514` retire on **June 15, 2026**.
  No code in this repository uses those IDs; no migration needed.

### Previously (2026-05-27)
- `langchain`, `langchain-openai`, `langgraph` floor bumps (patch releases).
- SSRF bypass via malformed URL ports fixed in `SecurityPolicy._extract_domains`.
```

---

## 9 · Must-Change Now vs Good Next Steps

### Must-change now (completed in this and previous run)

| # | Change | File | Status |
|---|--------|------|--------|
| 1 | Bump `agent-framework>=1.7.0,<2.0.0` | `pyproject.toml` | ✅ Done |
| 2 | Bump `anthropic>=0.105.0` | `pyproject.toml` | ✅ Done |
| 3 | Add claude-opus-4-8 to thinking-mode docs | `anthropic_features.py` | ✅ Done |
| 4 | Regenerate `uv.lock` | `uv.lock` | ✅ Done |
| 5 | Bump `langchain>=1.3.2` | `pyproject.toml` | ✅ Done (2026-05-27) |
| 6 | Bump `langchain-openai>=1.2.2` | `pyproject.toml` | ✅ Done (2026-05-27) |
| 7 | Bump `langgraph>=1.2.2` | `pyproject.toml` | ✅ Done (2026-05-27) |

### Good next-step improvements

| # | Change | File | Risk |
|---|--------|------|------|
| 1 | Add `claude-opus-4-8` thinking-mode guard tests | `tests/test_anthropic_features.py` | Safe internal |
| 2 | Explore `HarnessAgent` for long-running Gantry pipelines | `agent_framework_bridge.py` | Safe with shim |
| 3 | Validate `google-adk>=2.1.0` standalone; add Workflow Runtime example | `examples/agent_frameworks/` | Safe with shim |
| 4 | Validate `semantic-kernel>=1.42.0` in isolation with `agent-framework` | `pyproject.toml` | Safe with shim |
| 5 | Add MAF migration guide for teams moving from AutoGen/AG2 | `docs/` | Documentation |
