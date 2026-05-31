# Agent-Gantry Modernisation Audit

**Date:** 2026-05-31  
**Repository:** `CodeHalwell/Agent-Gantry` · version `0.4.0`  
**Branch:** `claude/cool-hopper-4OYfr` (based on `sentinel/fix-newline-injection-18155041855923376733` — PR #218)  
**Auditor:** Claude (claude-sonnet-4-6)

> **Note on PR history.** This audit incorporates:
> - Security fix PR #207 (SSRF bypass via malformed URL ports)
> - UX fix PR #208 (search combobox `Escape` focus retention)
> - Performance fix PR #211 (fast reverse iteration in `RateLimiter.get_stats`)
> - Docs fix PR #212 (smooth scrolling after dynamic anchor generation)
> - Security fix PR #213 (SSRF bypass via `file://` scheme)
> - Security fix PR #218 (regex bypass via newline injection in tool identifiers)
>
> The previous modernisation audit run (2026-05-27, `claude/cool-hopper-Y9M5N`) bumped
> `langchain`, `langchain-openai`, and `langgraph` floors and produced the initial AUDIT.md.
> This run (2026-05-31) is a follow-up incorporating PR #218 and four days of new package
> releases.

---

## 1 · Executive Summary

| Severity | Count | Areas |
|----------|-------|-------|
| **Must-change now** | 4 | Dependency floor bumps (anthropic, mcp, groq); duplicate validator removal |
| **Good next-step** | 5 | AF 1.7.0 HarnessAgent example; semantic-kernel/google-adk isolation docs; invalid-port SSRF test; MAF migration guide |

The repository remains in excellent health for `v0.4.0`. All provider API surfaces are current
and all framework integrations align with the latest stable SDKs.

**Actionable items completed in this audit run:**

1. Bump `anthropic` floor `0.104.1 → 0.105.2` (released 2026-05-29; no breaking changes).
2. Bump `mcp` floor `1.27.1 → 1.27.2` (released 2026-05-29; patch release).
3. Bump `groq` floor `1.2.0 → 1.4.0` (released 2026-05-28).
4. Remove redundant `_reject_newlines` validator from `ToolDefinition` — `validate_identifiers`
   already provides identical protection (see §7).
5. Update `pyproject.toml` comments for `agent-framework` (1.7.0), `crewai` (1.14.6),
   `google-genai` (2.7.0), `google-adk` (re-verified 2026-05-31).
6. Regenerate `uv.lock` (three packages updated, no solver conflicts).

---

## 2 · Dependency Upgrade Plan

### 2.1 Package-by-package table

Sources verified against PyPI JSON API (cited URLs below).

| Package | Previous floor | Current floor | Latest stable | Action | Risk |
|---------|---------------|--------------|---------------|--------|------|
| `agent-framework` | `>=1.5.0,<2.0.0` | `>=1.5.0,<2.0.0` | `1.7.0` | Within range ✅ — comment updated | Safe internal |
| `anthropic` | `>=0.104.1` | **`>=0.105.2`** | `0.105.2` | **Bumped** ✅ | Safe internal |
| `autogen-agentchat` | `>=0.7.5` | `>=0.7.5` | `0.7.5` | ✅ at latest | — |
| `autogen-ext[openai]` | `>=0.7.5` | `>=0.7.5` | `0.7.5` | ✅ at latest | — |
| `cohere` | `>=6.0.0` | `>=6.0.0` | (unchecked) | No action | — |
| `crewai` | `>=1.6.1` | `>=1.6.1` | `1.14.6` | Comment updated (latest 1.14.6) | Note only |
| `google-adk` | `>=1.14.1` | `>=1.14.1` | `2.1.0` | Comment re-verified; floor unchanged (langgraph conflict) | Blocked |
| `google-genai` | `>=1.75.0` | `>=1.75.0` | `2.7.0` | Comment updated (latest 2.7.0, 2026-05-28) | Note only |
| `groq` | `>=1.2.0` | **`>=1.4.0`** | `1.4.0` | **Bumped** ✅ | Safe internal |
| `langchain` | `>=1.3.2` | `>=1.3.2` | `1.3.2` | ✅ at latest | — |
| `langchain-openai` | `>=1.2.2` | `>=1.2.2` | `1.2.2` | ✅ at latest | — |
| `langgraph` | `>=1.2.2` | `>=1.2.2` | `1.2.2` | ✅ at latest | — |
| `llama-index-core` | `>=0.14.22` | `>=0.14.22` | `0.14.22` | ✅ at latest | — |
| `mcp` | `>=1.27.1` | **`>=1.27.2`** | `1.27.2` | **Bumped** ✅ | Safe internal |
| `openai` | `>=2.38.0` | `>=2.38.0` | `2.38.0` | ✅ at latest | — |
| `semantic-kernel` | `>=1.36.0` | `>=1.36.0` | `1.42.0` | OTel conflict with AF; floor unchanged | Blocked |

**Citation sources (all verified 2026-05-31):**
- `anthropic 0.105.2`: https://pypi.org/pypi/anthropic/json
- `groq 1.4.0`: https://pypi.org/pypi/groq/json
- `mcp 1.27.2`: https://pypi.org/pypi/mcp/json
- `google-genai 2.7.0`: https://pypi.org/pypi/google-genai/json
- `crewai 1.14.6`: https://pypi.org/pypi/crewai/json
- `agent-framework 1.7.0`: https://pypi.org/pypi/agent-framework/json + https://github.com/microsoft/agent-framework/releases
- `openai 2.38.0`: https://pypi.org/pypi/openai/json (unchanged)
- `langchain 1.3.2`, `langchain-openai 1.2.2`, `langgraph 1.2.2`: all unchanged from 2026-05-27

### 2.2 Commands applied in this run

```bash
uv lock --upgrade-package anthropic \
        --upgrade-package mcp \
        --upgrade-package groq

# Output:
# Updated anthropic v0.104.1 -> v0.105.2
# Updated groq v1.2.0 -> v1.4.0
# Updated mcp v1.27.1 -> v1.27.2
```

No solver conflicts. The three packages updated cleanly within the existing constraint set.

### 2.3 Agent Framework 1.7.0 — impact assessment

AF 1.7.0 (released 2026-05-28) includes one breaking change:

> **"Remove Python-only declarative actions and rename alias kinds to C# canonical names."**

Gantry does **not** use declarative actions (`@agent_framework.action()`). All Gantry tool
wrapping uses `@agent_framework.tool()` (a `FunctionTool`, not a declarative action), so this
breaking change does **not** affect any Gantry code.

AF 1.7.0 also adds `HarnessAgent` and the background-agents harness provider — additive, and
an interesting future integration path (see §9). The checkpoint restoration fix for
`MessageRole` values benefits `HandoffBuilder` workflows using serialised checkpoints.

**Source:** https://github.com/microsoft/agent-framework/releases (verified 2026-05-31).

### 2.4 Blocked upgrade: `semantic-kernel ≥ 1.42.0`

Unchanged from prior audit. Floor stays at `>=1.36.0` due to OTel conflict with
`agent-framework>=1.2.1`.

### 2.5 Blocked upgrade: `google-adk ≥ 2.x`

Unchanged from prior audit. Floor stays at `>=1.14.1` due to `langgraph<0.4.8` conflict.

---

## 3 · Provider API Refactors

### 3.1 OpenAI

**Status:** No changes required. ✅

No Assistants API usage is present. The Responses API (`client.responses.create`) is the
primary pattern in `examples/llm_integration/openai_demo.py` (Scenario A with `gpt-4.1`).
`OpenAIResponsesAdapter` produces the correct flat schema and `function_call_output` result
format. `OpenAIAdapter` (Chat Completions with `gpt-4o`) is retained as a secondary path.

Assistants API sunset: **26 August 2026**. No Gantry code touches it.

**Source:** https://platform.openai.com/docs/api-reference/responses (verified 2026-05-31).

### 3.2 Anthropic

**Status:** No API refactors required. Floor bumped to `0.105.2`. ✅

Messages API throughout with correct shapes. `AnthropicAdapter.to_provider_schema` produces
correct `input_schema` with optional `additionalProperties: false` for `strict=True`.
`format_tool_result` correctly emits `tool_use_id` and `is_error=True`.

Three thinking modes (interleaved, extended, adaptive) in `anthropic_features.py` remain
correctly implemented.

**Source:** https://docs.anthropic.com/en/api/messages (verified 2026-05-31).

### 3.3 Google Gemini / Gen AI

**Status:** No changes required. ✅

`GeminiAdapter` shapes are correct. `id` placed inside `functionResponse` correctly. Async
`client.aio.models.generate_content` is correct. google-genai 2.7.0 has no breaking changes
affecting function calling.

**Source:** https://ai.google.dev/gemini-api/docs/function-calling (verified 2026-05-31).

### 3.4 Mistral

**Status:** No changes required. ✅

Routes correctly through `AsyncOpenAI(base_url="https://api.mistral.ai/v1")`.

### 3.5 Groq

**Status:** No API refactors required. Floor bumped to `1.4.0`. ✅

`AsyncGroq` with `chat.completions.create()` remains the correct pattern.

---

## 4 · Framework Integration Refactors

### 4.1 Microsoft Agent Framework (AF 1.5.0 – 1.7.0) — Priority review

No refactors required. AF 1.7.0 compatibility confirmed (§2.3).

| Surface | Status |
|---------|--------|
| Agent construction (`Agent`, `AgentExecutor`) | ✅ Correct; unaffected by 1.7.0 |
| Workflow subsystem (`SequentialBuilder`, `HandoffBuilder`, `WorkflowBuilder`) | ✅ Correct; `MessageRole` checkpoint fix benefits `HandoffBuilder` |
| AF 1.6.0/1.7.0 ContextVar instrumentation shim | ✅ Still required and correctly implemented |
| Middleware pipeline (`FunctionMiddleware`, `ChatMiddlewareLayer` fallback) | ✅ Both symbols present in AF 1.7.0 |
| `GantryContextProvider` | ✅ Unaffected by 1.7.0 |
| Human-in-the-loop approval gates | ✅ Unaffected by 1.7.0 |

**Risk level:** Safe internal — no changes required.

### 4.2 LangGraph, AutoGen, CrewAI, LlamaIndex, Semantic Kernel

All unchanged from prior audit (2026-05-27). No refactors required.

---

## 5 · Universal Tool-Calling Design

No changes required. The `ToolSpec` contract (`ToolCallPayload` + `ToolSpecAdapter` protocol)
is sound and covers all current providers:

| Adapter | Dialect | Status |
|---------|---------|--------|
| `OpenAIAdapter` | `openai` | ✅ Chat Completions format |
| `OpenAIResponsesAdapter` | `openai_responses` | ✅ Responses API flat format |
| `AnthropicAdapter` | `anthropic` | ✅ `input_schema`, `is_error`, `tool_use_id` |
| `GeminiAdapter` | `gemini` | ✅ `functionDeclarations`, `functionResponse.id` |
| `MistralAdapter` | `mistral` | ✅ Inherits `OpenAIAdapter` |
| `GroqAdapter` | `groq` | ✅ Inherits `OpenAIAdapter` |
| `AgentFrameworkAdapter` | `agent_framework` | ✅ Inherits `OpenAIAdapter`, metadata opt |

Tool name normalisation, call ID round-tripping, parallel tool call correlation, and
`format_tool_result` shapes are all correct and verified against current provider
documentation. ✅

---

## 6 · Documentation and Example Updates

All examples are current. No rewrites required.

| File | Model | Status |
|------|-------|--------|
| `examples/llm_integration/openai_demo.py` | `gpt-4.1` (Responses API), `gpt-4o` (CC) | ✅ |
| `examples/llm_integration/anthropic_demo.py` | `claude-sonnet-4-6` | ✅ |
| `examples/llm_integration/google_genai_demo.py` | `gemini-2.5-flash` | ✅ |
| `examples/agent_frameworks/agent_framework_example.py` | AF 1.x `Agent`/`AgentExecutor` | ✅ |
| `examples/agent_frameworks/autogen_example.py` | AG2 v0.4 + MAF migration note | ✅ |
| `examples/agent_frameworks/langgraph_example.py` | `create_react_agent` + LG 1.x | ✅ |

---

## 7 · Security: Duplicate Validator Consolidation

### 7.1 Finding

PR #218 added `_reject_newlines` to `ToolDefinition` to close a regex bypass: Pydantic v2's
Rust regex engine treats `$` as end-of-line rather than end-of-string, so the field
constraint `pattern=r"^[a-z][a-z0-9_]*$"` on `name` would accept `"valid_name\n"`.

However, `validate_identifiers` — a pre-existing `@field_validator` on the same three fields
(`name`, `version`, `namespace`) — already performs an identical explicit `"\n" in v or "\r" in v`
check. Both validators run in Pydantic v2's default `mode="after"`. The result is two
validators doing the exact same work on every model construction and assignment.

### 7.2 Fix applied

**File:** `agent_gantry/schema/tool.py`

```diff
-    @field_validator("name", "version", "namespace", mode="after")
-    @classmethod
-    def _reject_newlines(cls, v: Any) -> Any:
-        if isinstance(v, str) and ("\n" in v or "\r" in v):
-            raise ValueError("Newlines are not allowed")
-        return v
-
     # Capabilities & permissions
     capabilities: list[ToolCapability] = Field(default_factory=list)
     ...
 
     @field_validator("name", "version", "namespace")
     @classmethod
     def validate_identifiers(cls, v: str) -> str:
-        """Validate that identifiers do not contain newlines."""
+        """Reject newlines in identifier fields.
+
+        Pydantic v2 (Rust regex engine) treats $ as end-of-line rather than
+        end-of-string, so the pattern=r"^[a-z][a-z0-9_]*$" on `name` would
+        accept "valid_name\\n". Explicit character checks close that bypass for
+        all three identifier fields.
+        """
         if "\n" in v or "\r" in v:
             raise ValueError("Value cannot contain newline characters")
         return v
```

**Security posture unchanged.** `validate_identifiers` blocks newlines identically to the
removed `_reject_newlines`. The pattern constraint on `name` still prevents most malformed
names at the Rust-engine level; `validate_identifiers` closes the `\n`-at-end bypass.

**Risk level:** Safe internal.

### 7.3 Existing security tests

| Test | Status |
|------|--------|
| `test_security.py::test_ssrf_port_bypass` | ✅ Passes |
| `test_security.py` (file:// SSRF) | ✅ Passes |
| `test_tool.py` (ToolDefinition validation) | ✅ Covers newline rejection via `validate_identifiers` |

---

## 8 · Test and Migration Plan

### 8.1 Tests to add or update

| Test | Status | Notes |
|------|--------|-------|
| `test_tool.py` — newline rejection | ✅ Existing | Unaffected by removing `_reject_newlines` |
| `test_security.py::test_ssrf_invalid_port_with_hostname` | ⚠️ Optional | Documents `http://example.com:evil.com` → correct hostname extraction |
| AF 1.7.0 `HarnessAgent` example test | ⚠️ Good next step | If/when a `HarnessAgent` example is added |

### 8.2 Regeneration checklist

- [x] `uv.lock` regenerated (`anthropic`, `groq`, `mcp` updated).
- [ ] VCR recordings / golden HTTP responses — not present; not applicable.
- [ ] Schema snapshots — not present; not applicable.

### 8.3 Changelog-ready migration notes

```
## [0.4.x] — 2026-05-31

### Changed
- `anthropic` floor bumped `>=0.104.1 → >=0.105.2` (patch series; no API changes).
- `mcp` floor bumped `>=1.27.1 → >=1.27.2` (patch; no API changes).
- `groq` floor bumped `>=1.2.0 → >=1.4.0` (no breaking changes).
- `pyproject.toml` comments updated for `agent-framework` (1.7.0 impact analysis),
  `crewai` (1.14.6), `google-genai` (2.7.0).

### Fixed (internal)
- Removed redundant `_reject_newlines` validator from `ToolDefinition`; consolidated
  newline-rejection logic into the pre-existing `validate_identifiers` validator with
  an explanatory docstring. Security posture unchanged.
```

---

## 9 · Must-Change Now vs Good Next Steps

### Must-change now (completed in this run)

| # | Change | File | Status |
|---|--------|------|--------|
| 1 | Bump `anthropic>=0.105.2` | `pyproject.toml` | ✅ Done |
| 2 | Bump `mcp>=1.27.2` | `pyproject.toml` | ✅ Done |
| 3 | Bump `groq>=1.4.0` | `pyproject.toml` | ✅ Done |
| 4 | Remove `_reject_newlines`; expand `validate_identifiers` docstring | `agent_gantry/schema/tool.py` | ✅ Done |
| 5 | Update pyproject comments (AF 1.7.0, crewai 1.14.6, google-genai 2.7.0) | `pyproject.toml` | ✅ Done |
| 6 | Regenerate `uv.lock` | `uv.lock` | ✅ Done |

### Good next-step improvements

| # | Change | File | Risk |
|---|--------|------|------|
| 1 | Add `HarnessAgent` example showing background-agent pattern (new in AF 1.7.0) | `examples/agent_frameworks/` | Safe internal |
| 2 | Validate `semantic-kernel>=1.42.0` in isolation with `agent-framework`; bump floor if OTel conflict resolved | `pyproject.toml` | Safe with shim |
| 3 | Validate `google-adk>=2.1.0` standalone env; add docs page showing Workflow Runtime pattern | `docs/`, `examples/` | Needs confirmation |
| 4 | Add `test_ssrf_invalid_port_with_hostname` to document expected hostname extraction for `http://example.com:evil.com` | `tests/test_security.py` | Safe internal |
| 5 | Add MAF migration guide (`docs/migrating-from-autogen.md`) for AutoGen→AF 1.x teams | `docs/` | Documentation |
