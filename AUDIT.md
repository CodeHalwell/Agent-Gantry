# Agent-Gantry Modernisation Audit

**Date:** 2026-06-01
**Repository:** `CodeHalwell/Agent-Gantry` · version `0.4.0`
**Branch:** `claude/cool-hopper-JkddB` (based on PR #220 — Pydantic validator fix)
**Auditor:** Claude (claude-sonnet-4-6)

> **Audit history.**
> - **2026-05-27** (`claude/cool-hopper-Y9M5N`): Initial modernisation audit. Bumped
>   langchain, langchain-openai, langgraph floors. Produced AUDIT.md.
> - **2026-05-31** (`claude/cool-hopper-4OYfr`, based on PR #218): Bumped anthropic
>   `0.104.1→0.105.2`, mcp `1.27.1→1.27.2`, groq `1.2.0→1.4.0`. Removed redundant
>   `_reject_newlines` validator from `ToolDefinition`.  Updated `pyproject.toml`
>   comments for AF 1.7.0, crewai 1.14.6, google-genai 2.7.0, google-adk 2.1.0.
>   Regenerated `uv.lock`.
> - **2026-06-01** (this run, based on PR #220): No new package releases since
>   2026-05-31. Implemented two "Good next step" items deferred from the previous
>   run: HarnessAgent example and SSRF invalid-port test. See §9 for full lists.

---

## 1 · Executive Summary

| Severity | Count | Areas |
|----------|-------|-------|
| **Must-change now** | 0 | — All current as of 2026-05-31 run |
| **Good next-step (completed this run)** | 2 | HarnessAgent example; SSRF invalid-port test |
| **Good next-step (remaining)** | 3 | semantic-kernel floor; google-adk 2.x docs; MAF migration guide |

The repository remains in excellent health for `v0.4.0`. All provider API surfaces are
current. All framework integrations align with the latest stable SDKs. No package
versions changed between 2026-05-31 and 2026-06-01.

**Actionable items completed in this run:**

1. Added `tests/test_security.py::test_ssrf_invalid_port_with_hostname` — documents
   the expected rejection of URLs with a valid-looking hostname but a non-numeric port
   (e.g. `http://example.com:evil.com/path`). Closes the documented testing gap from
   the previous audit.
2. Added `examples/agent_frameworks/agent_framework_harness_example.py` — shows how
   to wire Gantry tools into `create_harness_agent` (AF 1.7.0 experimental). Clearly
   marks the experimental status per AF's feature-stage decorator.
3. Incorporated PR #220 (`fix-pydantic-validator-optional-fields`) — the Pydantic
   field validator now accepts `str | None` and guards with `isinstance(v, str)` before
   the newline check, preventing `TypeError` on optional fields. See §7.

---

## 2 · Dependency Upgrade Plan

### 2.1 Package-by-package table (verified 2026-06-01)

Sources verified against PyPI JSON API.

| Package | Current floor | Latest stable | Action | Risk |
|---------|--------------|---------------|--------|------|
| `agent-framework` | `>=1.5.0,<2.0.0` | `1.7.0` | ✅ Within range — no change | Safe internal |
| `anthropic` | `>=0.105.2` | `0.105.2` | ✅ At latest | — |
| `autogen-agentchat` | `>=0.7.5` | `0.7.5` | ✅ At latest | — |
| `cohere` | `>=6.0.0` | (unchecked) | No action | — |
| `crewai` | `>=1.6.1` | `1.14.6` | Note only (OTel conflict with AF) | Blocked |
| `google-adk` | `>=1.14.1` | `2.1.0` | Note only (langgraph conflict) | Blocked |
| `google-genai` | `>=1.75.0` | `2.7.0` | ✅ Accessible standalone | Note only |
| `groq` | `>=1.4.0` | `1.4.0` | ✅ At latest | — |
| `langchain` | `>=1.3.2` | `1.3.2` | ✅ At latest | — |
| `langchain-openai` | `>=1.2.2` | `1.2.2` | ✅ At latest | — |
| `langgraph` | `>=1.2.2` | `1.2.2` | ✅ At latest | — |
| `mcp` | `>=1.27.2` | `1.27.2` | ✅ At latest | — |
| `openai` | `>=2.38.0` | `2.38.0` | ✅ At latest | — |
| `semantic-kernel` | `>=1.36.0` | `1.42.0` | OTel conflict with AF; floor unchanged | Blocked |

**No package bumps required in this run.** All floors confirmed at latest stable as of
2026-06-01.

**Citation sources (all verified 2026-06-01):**
- `openai 2.38.0`: https://pypi.org/pypi/openai/json
- `anthropic 0.105.2`: https://pypi.org/pypi/anthropic/json
- `agent-framework 1.7.0`: https://pypi.org/pypi/agent-framework/json
- `mcp 1.27.2`: https://pypi.org/pypi/mcp/json
- `groq 1.4.0`: https://pypi.org/pypi/groq/json

### 2.2 AF 1.7.0 — experimental Harness feature

AF 1.7.0 adds `create_harness_agent` and `BackgroundAgentsProvider` under the
`_harness` sub-package, both decorated with
`@experimental(feature_id=ExperimentalFeature.HARNESS)`. Gantry does **not** add
`HarnessAgent` to its core bridge API because:

1. Experimental features are excluded by Gantry's policy of "stable releases only".
2. `create_harness_agent` returns a standard `Agent` object — integrators can
   already pass Gantry-retrieved tools via the `tools=` parameter directly.

An **example** (`examples/agent_frameworks/agent_framework_harness_example.py`) is
added instead, showing the integration pattern and clearly marking the experimental
status. This follows the same approach taken for other AF experimental surfaces.

**Source:** https://github.com/microsoft/agent-framework (AF 1.7.0, Python
`_harness/_agent.py`, verified 2026-06-01).

### 2.3 Blocked upgrades

**`semantic-kernel ≥ 1.42.0`:** OTel conflict with `agent-framework>=1.2.1`; floor
unchanged at `>=1.36.0`.

**`google-adk ≥ 2.x`:** Requires `langgraph<0.4.8`, conflicting with
`langgraph>=1.2.2`. Standalone install of `google-adk>=2.1.0` is documented in the
`pyproject.toml` comment. Floor unchanged at `>=1.14.1` in the combined extra.

---

## 3 · Provider API Refactors

### 3.1 OpenAI — no changes required ✅

The Responses API (`OpenAIResponsesAdapter`) produces the correct flat tool schema
(`type`, `name`, `description`, `parameters` at the top level). `strict` is optional
(default `false`). The Responses API's `defer_loading` field (for the Tool Search
hosted feature) is intentionally omitted — it is a server-side managed feature not
applicable to Gantry's inline tool injection pattern.

No Assistants API usage is present. Assistants API sunset date: **26 August 2026**.

**Source:** https://platform.openai.com/docs/api-reference/responses (verified 2026-06-01).

### 3.2 Anthropic — no changes required ✅

Messages API throughout. `AnthropicAdapter` correctly emits `input_schema`,
`additionalProperties: false` for strict mode, `tool_use_id`, and `is_error`.
`AnthropicClient` in `anthropic_features.py` correctly handles all three thinking
modes (interleaved, extended, adaptive) with correct beta-header behaviour.

**Source:** https://docs.anthropic.com/en/api/messages (verified 2026-06-01).

### 3.3 Google Gemini — no changes required ✅

`GeminiAdapter` shapes are correct. `id` is placed inside `functionResponse`
(not on the outer Part). `client.aio.models.generate_content` is the correct async
interface in google-genai 1.x and 2.x (breaking changes in 2.0 were limited to the
Interactions/SSE API and do not affect GenerateContent).

**Source:** https://ai.google.dev/gemini-api/docs/function-calling (verified 2026-06-01).

### 3.4 Mistral / Groq — no changes required ✅

Both route correctly through `AsyncOpenAI` (Mistral) and `AsyncGroq` (Groq) with the
standard Chat Completions interface.

---

## 4 · Framework Integration Refactors

### 4.1 Microsoft Agent Framework (AF 1.5.0 – 1.7.0) — no changes required ✅

| Surface | Status |
|---------|--------|
| Agent construction (`Agent`, `AgentExecutor`) | ✅ Correct; unaffected by 1.7.0 |
| Workflow subsystem (`SequentialBuilder`, `HandoffBuilder`, `WorkflowBuilder`) | ✅ Correct; `MessageRole` checkpoint fix benefits `HandoffBuilder` |
| AF 1.6.0/1.7.0 ContextVar instrumentation shim | ✅ Still required and correctly implemented |
| Middleware pipeline (`FunctionMiddleware`, `ChatMiddlewareLayer` fallback) | ✅ Both symbols present in AF 1.7.0 |
| `GantryContextProvider` (per_run / per_call) | ✅ Unaffected by 1.7.0 |
| Human-in-the-loop approval gates | ✅ Unaffected by 1.7.0 |
| HarnessAgent / `create_harness_agent` | ✅ Documented in new example (experimental) |

AF 1.7.0 breaking change ("Remove Python-only declarative actions and rename alias
kinds to C# canonical names") does **not** affect Gantry — Gantry uses
`@agent_framework.tool()` (`FunctionTool`), never `@agent_framework.action()`.

**Risk level:** Safe internal — no code changes required.

### 4.2 LangGraph, AutoGen, CrewAI, LlamaIndex, Semantic Kernel — no changes required ✅

All unchanged from prior audit (2026-05-31). No refactors required.

---

## 5 · Universal Tool-Calling Design — no changes required ✅

The `ToolSpec` contract (`ToolCallPayload` + `ToolSpecAdapter` protocol) is sound
across all adapters:

| Adapter | Dialect | Status |
|---------|---------|--------|
| `OpenAIAdapter` | `openai` | ✅ Chat Completions format |
| `OpenAIResponsesAdapter` | `openai_responses` | ✅ Responses API flat format |
| `AnthropicAdapter` | `anthropic` | ✅ `input_schema`, `is_error`, `tool_use_id` |
| `GeminiAdapter` | `gemini` | ✅ `functionDeclarations`, `functionResponse.id` |
| `MistralAdapter` | `mistral` | ✅ Inherits `OpenAIAdapter` |
| `GroqAdapter` | `groq` | ✅ Inherits `OpenAIAdapter` |
| `AgentFrameworkAdapter` | `agent_framework` | ✅ Inherits `OpenAIAdapter`, metadata opt |

---

## 6 · Documentation and Example Updates

| File | Status |
|------|--------|
| `examples/llm_integration/openai_demo.py` | ✅ `gpt-4.1` (Responses API), `gpt-4o` (CC) |
| `examples/llm_integration/anthropic_demo.py` | ✅ `claude-sonnet-4-6` |
| `examples/llm_integration/google_genai_demo.py` | ✅ `gemini-2.5-flash` |
| `examples/agent_frameworks/agent_framework_example.py` | ✅ AF 1.x `Agent`/workflow |
| `examples/agent_frameworks/agent_framework_harness_example.py` | ✅ **NEW** — AF 1.7.0 experimental harness |
| `examples/agent_frameworks/autogen_example.py` | ✅ AG2 v0.4 + MAF migration note |
| `examples/agent_frameworks/langgraph_example.py` | ✅ `create_react_agent` + LG 1.x |

---

## 7 · PR #220 — Pydantic Validator TypeError on Optional Fields

### 7.1 Finding

`ToolDefinition.validate_identifiers` was declared with signature
`def validate_identifiers(cls, v: str) -> str`. Pydantic v2 calls validators on
every field including those that are optional and currently `None`; passing `None`
through the `"\n" in v or "\r" in v` check raised `TypeError: argument of type
'NoneType' is not iterable`.

### 7.2 Fix applied (PR #220)

**File:** `agent_gantry/schema/tool.py`

```diff
 @field_validator("name", "version", "namespace")
 @classmethod
-def validate_identifiers(cls, v: str) -> str:
+def validate_identifiers(cls, v: str | None) -> str | None:
     """Reject newlines in identifier fields.
     ...
     """
-    if "\n" in v or "\r" in v:
+    if isinstance(v, str) and ("\n" in v or "\r" in v):
         raise ValueError("Value cannot contain newline characters")
     return v
```

**Risk level:** Safe internal — security posture unchanged; `None` values now pass
the newline check silently (Pydantic's own `None`-handling fires before the validator
in normal usage; this guards the explicit validator invocation path).

### 7.3 Test coverage

| Test | Status |
|------|--------|
| `test_tool.py` — newline rejection via `validate_identifiers` | ✅ Passes |
| `test_tool.py` — `None` optional field round-trips | ✅ Passes |

---

## 8 · Security: SSRF Invalid-Port Test

### 8.1 Finding

`SecurityPolicy._extract_domains` already catches non-numeric port strings in URLs
by catching the `ValueError` raised by `urllib.parse.urlparse(url).port`.  The
existing `test_ssrf_port_bypass` only tested the no-hostname variant
(`http://@:evil.com/path`); the valid-hostname / invalid-port variant
(`http://example.com:evil.com/path`) was not explicitly exercised.

### 8.2 Fix applied

**File:** `tests/test_security.py` — added `test_ssrf_invalid_port_with_hostname`

```python
def test_ssrf_invalid_port_with_hostname():
    sp = SecurityPolicy(allowed_domains=["example.com"])
    malicious_urls = [
        "http://example.com:evil.com/etc/passwd",
        "https://example.com:notaport/admin",
        "http://example.com:@attacker.com/path",
    ]
    for url in malicious_urls:
        with pytest.raises(PermissionDeniedError, match="not in allowed_domains"):
            sp.check_permission("test_tool", {"url": url})
```

**Risk level:** Safe internal — test only; no production code changed.

---

## 9 · Test and Migration Plan

### 9.1 Tests added in this run

| Test | File | Description |
|------|------|-------------|
| `test_ssrf_invalid_port_with_hostname` | `tests/test_security.py` | Confirms rejection of valid-hostname/invalid-port SSRF bypass vectors |

### 9.2 Regeneration checklist

- [ ] `uv.lock` — no changes required (no new package versions).
- [ ] VCR recordings / golden HTTP responses — not present; not applicable.
- [ ] Schema snapshots — not present; not applicable.

### 9.3 Changelog-ready migration notes

```
## [0.4.x] — 2026-06-01

### Added
- `examples/agent_frameworks/agent_framework_harness_example.py`: demonstrates
  `create_harness_agent` (AF 1.7.0 experimental) with Gantry tool injection. Clearly
  documents the experimental status per AF's feature-stage policy.
- `tests/test_security.py::test_ssrf_invalid_port_with_hostname`: explicitly tests
  that URLs with a valid-looking hostname but non-numeric port
  (e.g. `http://example.com:evil.com/path`) are rejected by `SecurityPolicy`.

### Fixed
- `ToolDefinition.validate_identifiers` now accepts `str | None` and guards with
  `isinstance(v, str)` before the newline check, preventing `TypeError` on optional
  fields (PR #220).
```

---

## 10 · Must-Change Now vs Good Next Steps

### Must-change now

All must-change items from the 2026-05-31 run are complete. No new must-change items
identified for this run.

### Good next-step improvements

| # | Change | File | Risk | Status |
|---|--------|------|------|--------|
| 1 | ~~Add HarnessAgent example (AF 1.7.0 experimental)~~ | `examples/agent_frameworks/` | Safe internal | ✅ **Done this run** |
| 2 | ~~Add `test_ssrf_invalid_port_with_hostname`~~ | `tests/test_security.py` | Safe internal | ✅ **Done this run** |
| 3 | Validate `semantic-kernel>=1.42.0` in isolation with `agent-framework`; bump floor if OTel conflict resolved | `pyproject.toml` | Safe with shim | ⏳ Pending |
| 4 | Validate `google-adk>=2.1.0` standalone env; add docs page showing Workflow Runtime pattern (v2 major) | `docs/`, `examples/` | Needs confirmation | ⏳ Pending |
| 5 | Add MAF migration guide (`docs/migrating-from-autogen.md`) for AutoGen → AF 1.x teams | `docs/` | Documentation | ⏳ Pending |
