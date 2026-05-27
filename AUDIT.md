# Agent-Gantry Modernisation Audit

**Date:** 2026-05-27  
**Repository:** `CodeHalwell/Agent-Gantry` · version `0.4.0`  
**Branch:** `claude/cool-hopper-Y9M5N` (based on `sentinel/fix-ssrf-port-bypass-15808601177667804830`)  
**Auditor:** Claude (claude-sonnet-4-6)

> **Note on PR history.** This audit incorporates security fix PR #207 (SSRF bypass via
> malformed URL ports, merged via `sentinel/fix-ssrf-port-bypass-15808601177667804830`) and
> accessibility fix PR #208 (search combobox `Escape` focus retention). The audit branch
> starts from the sentinel security branch so all current fixes are in scope.

---

## 1 · Executive Summary

| Severity | Count | Areas |
|----------|-------|-------|
| **Must-change now** | 3 | Dependency floor bumps (langchain, langchain-openai, langgraph) |
| **Good next-step** | 5 | Comment refresh (crewai, google-adk, google-genai, agent-framework 1.6.0 notes), AutoGen→MAF migration page |

The repository is in excellent overall health for `v0.4.0`. All three provider API surfaces are
current:

- **OpenAI**: Responses API (`client.responses.create`) is the primary path. No Assistants API
  code is present. Chat Completions is retained as a supported secondary path.
- **Anthropic**: Messages API with correct `input_schema`, `is_error`, and all three thinking
  modes (interleaved, extended, adaptive).
- **Google Gemini**: `google.genai` SDK with `functionDeclarations` and correct
  `functionResponse` correlation.

The Microsoft Agent Framework integration is comprehensive and aligned with AF 1.5.0/1.6.0:
`Agent`, `AgentExecutor`, `WorkflowBuilder`, `SequentialBuilder`, `HandoffBuilder`,
`ContextProvider`, `FunctionMiddleware`, and `chat_middleware` are all used correctly. A
documented workaround exists for the AF 1.6.0 ContextVar concurrency bug.

Security: the SSRF bypass via malformed URL ports (`http://@:evil.com/…`) has been fixed in
`core/security.py` and is covered by `test_ssrf_port_bypass`. The fix correctly blocks URLs
that lack a valid hostname after port-parsing raises `ValueError`.

**Actionable items completed in this audit run:**
1. Bump `langchain` floor `1.3.1 → 1.3.2` (latest stable).
2. Bump `langchain-openai` floor `1.2.1 → 1.2.2` (latest stable).
3. Bump `langgraph` floor `1.2.1 → 1.2.2` (latest stable).
4. Update `pyproject.toml` comments for `crewai`, `google-adk`, `google-genai`.
5. Regenerate `uv.lock` (three packages updated, no solver conflicts).

---

## 2 · Dependency Upgrade Plan

### 2.1 Package-by-package table

Sources verified against PyPI JSON API (cited URLs below).

| Package | Current floor | Latest stable | Action | Risk |
|---------|--------------|---------------|--------|------|
| `agent-framework` | `>=1.5.0,<2.0.0` | `1.6.0` | Within range ✅ — update comment | Safe internal |
| `anthropic` | `>=0.104.1` | `0.104.1` | ✅ at latest | — |
| `autogen-agentchat` | `>=0.7.5` | `0.7.5` | ✅ at latest | — |
| `autogen-ext[openai]` | `>=0.7.5` | `0.7.5` | ✅ at latest | — |
| `cohere` | `>=6.0.0` | (unchecked) | No action | — |
| `crewai` | `>=1.6.1` | `1.14.5` | Comment updated (latest is 1.14.5) | Note only |
| `google-adk` | `>=1.14.1` | `2.1.0` | Comment updated; floor stays at 1.14.1 (langgraph conflict) | Blocked — see §2.4 |
| `google-genai` | `>=1.75.0` | `2.6.0` | Comment updated (latest is 2.6.0); floor stays 1.75.0 | Note only |
| `groq` | `>=1.2.0` | `1.2.0` | ✅ at latest | — |
| `langchain` | `>=1.3.1` | `1.3.2` | **Bumped to >=1.3.2** ✅ | Safe internal |
| `langchain-openai` | `>=1.2.1` | `1.2.2` | **Bumped to >=1.2.2** ✅ | Safe internal |
| `langgraph` | `>=1.2.1` | `1.2.2` | **Bumped to >=1.2.2** ✅ | Safe internal |
| `llama-index-core` | `>=0.14.22` | `0.14.22` | ✅ at latest | — |
| `llama-index-llms-openai` | `>=0.7.7` | (unchecked) | No action | — |
| `mcp` | `>=1.27.1` | `1.27.1` | ✅ at latest | — |
| `openai` | `>=2.38.0` | `2.38.0` | ✅ at latest | — |
| `semantic-kernel` | `>=1.36.0` | `1.42.0` | Comment only — OTel conflict with AF; see §2.3 | Blocked |

**Citation sources (all verified 2026-05-27):**
- `langchain 1.3.2`: https://pypi.org/pypi/langchain/json
- `langchain-openai 1.2.2`: https://pypi.org/pypi/langchain-openai/json
- `langgraph 1.2.2`: https://pypi.org/pypi/langgraph/json
- `crewai 1.14.5`: https://pypi.org/pypi/crewai/json
- `google-adk 2.1.0`: https://pypi.org/pypi/google-adk/json
- `google-genai 2.6.0`: https://pypi.org/pypi/google-genai/json
- `agent-framework 1.6.0`: https://pypi.org/pypi/agent-framework/json
- `anthropic 0.104.1`: https://pypi.org/pypi/anthropic/json
- `openai 2.38.0`: https://pypi.org/pypi/openai/json
- `mcp 1.27.1`: https://pypi.org/pypi/mcp/json
- `llama-index-core 0.14.22`: https://pypi.org/pypi/llama-index-core/json
- `groq 1.2.0`: https://pypi.org/pypi/groq/json

### 2.2 Commands applied in this run

```bash
# Bumped floors in pyproject.toml, then regenerated the lock:
uv lock --upgrade-package langchain \
        --upgrade-package langchain-openai \
        --upgrade-package langgraph

# Output:
# Updated langchain v1.3.1 -> v1.3.2
# Updated langchain-openai v1.2.1 -> v1.2.2
# Updated langgraph v1.2.1 -> v1.2.2
```

No solver conflicts were raised. The three packages updated cleanly within the existing
constraint set.

### 2.3 Blocked upgrade: `semantic-kernel ≥ 1.42.0`

`semantic-kernel 1.42.0` (latest stable) is blocked in the `agent-frameworks` combined extra
because `agent-framework>=1.2.1` requires `opentelemetry-api>=1.39.0`, while older
`semantic-kernel` versions pin a narrower range. The floor stays at `>=1.36.0`.

**Resolution path:** install `semantic-kernel` in a dedicated virtual environment without
`agent-framework`, or wait for a `semantic-kernel` release that widens its `opentelemetry-api`
range to include `>=1.39.0`.

### 2.4 Blocked upgrade: `google-adk ≥ 2.x`

`google-adk 2.1.0` is the latest stable release (Workflow Runtime + Task API, major version
bump). It is blocked in the combined `agent-frameworks` extra because:

1. `google-adk 2.x` (and `1.34.x` before it) requires `langgraph<0.4.8` via its extensions
   extra, which is incompatible with `langgraph>=1.2.2` now required by this project.
2. Potential transitive `pydantic` / `opentelemetry-api` conflicts in the 2.x internals
   require validation in an isolated environment first.

**Standalone usage:** `pip install "agent-gantry[google-genai]" "google-adk>=2.1.0"` works in
an environment without LangChain/LangGraph.

Source: https://pypi.org/pypi/google-adk/json (verified 2026-05-27).

---

## 3 · Provider API Refactors

### 3.1 OpenAI

**Finding:** No Assistants API usage found in the codebase. The Responses API
(`client.responses.create`) is already the primary pattern. ✅

**Responses API usage** (`examples/llm_integration/openai_demo.py`): Correct.
- `input=` (flat message array), `tools=` (flat schema), `previous_response_id`,
  and `function_call_output` items are all used correctly.
- Uses `gpt-4.1` (current model family).

**Tool schema** (`agent_gantry/adapters/tool_spec/providers.py`, `OpenAIResponsesAdapter`):
produces the correct flat shape:

```python
{"type": "function", "name": "...", "description": "...", "parameters": {...}}
```

per the Responses API specification. ✅

Source: https://platform.openai.com/docs/api-reference/responses (verified 2026-05-27).

**Tool result format** (`OpenAIResponsesAdapter.format_tool_result`): correctly produces:

```python
{"type": "function_call_output", "call_id": "...", "output": "..."}
```
✅

**Chat Completions** (`OpenAIAdapter`): still supported for the `dialect="openai"` path.
Correct format. ✅

**Assistants API sunset:** August 26 2026. No code touches the Assistants API. ✅

**Risk level:** Safe internal — no changes required.

### 3.2 Anthropic

**Finding:** Messages API is used throughout with correct shapes. ✅

**Tool schema** (`AnthropicAdapter.to_provider_schema`): produces correct `input_schema` field
with optional `additionalProperties: false` injection for `strict=True` mode (documented in the
code with a citation to `platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use`).

**Tool result** (`AnthropicAdapter.format_tool_result`): correctly emits `tool_use_id` and
`is_error=True` when the tool failed. ✅

Source: https://docs.anthropic.com/en/api/messages (verified 2026-05-27).

**Thinking modes** (`agent_gantry/integrations/anthropic_features.py`): three modes implemented:

| Mode | Implementation | Status |
|------|----------------|--------|
| `interleaved` | `anthropic-beta: interleaved-thinking-2025-05-14` header | ✅ |
| `extended` | `thinking: {type: "enabled", budget_tokens: N}` | ✅ |
| `adaptive` | `thinking: {type: "adaptive", effort: "low|medium|high"}` | ✅ |

`claude-opus-4-7` correctly excludes `extended` thinking (not supported on that model).
`adaptive` is correctly marked as not available on `claude-haiku-4-5`.

Source: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking (verified 2026-05-27).

**Model names:** `claude-sonnet-4-6` and `claude-opus-4-7` are current production aliases. ✅

**Risk level:** Safe internal — no changes required.

### 3.3 Google Gemini / Gen AI

**Finding:** Correct `functionDeclarations` / `functionResponse` shapes used throughout. ✅

**Tool schema** (`GeminiAdapter.to_provider_schema`): produces:
```python
{"name": "...", "description": "...", "parameters": {...}}
```
which maps directly to Gemini `FunctionDeclaration`. ✅

**Tool result** (`GeminiAdapter.format_tool_result`): places `id` inside `functionResponse`
(correct — `id` is a field on the `FunctionResponse` proto, not on `Part`):
```python
{"functionResponse": {"name": "...", "response": {...}, "id": "..."}}
```
✅

Source: https://ai.google.dev/gemini-api/docs/function-calling (verified 2026-05-27).

**Model names** (`google_genai_demo.py`): `gemini-2.5-flash` is current stable. ✅

**Async interface** (`client.aio.models.generate_content`): correct for google-genai ≥1.x.
The SDK's 2.x breaking changes were limited to the Interactions API surface (SSE event renames,
`response_format` restructuring); `generate_content` and function calling are unaffected. ✅

**Risk level:** Safe internal — no changes required.

### 3.4 Mistral

**Finding:** `mistralai` was quarantined on PyPI on 2026-05-12. The repo correctly migrates to
`AsyncOpenAI(base_url="https://api.mistral.ai/v1")` and `MistralAdapter` inherits
`OpenAIAdapter` for schema and result formats. ✅

**Risk level:** Safe internal — no changes required.

### 3.5 Groq

**Finding:** `AsyncGroq` with `chat.completions.create()`. `GroqAdapter` inherits
`OpenAIAdapter`. All shapes are correct. ✅

**Risk level:** Safe internal — no changes required.

---

## 4 · Framework Integration Refactors

### 4.1 Microsoft Agent Framework (AF 1.5.0 / 1.6.0) — Priority review

All AF integration modules align with the `agent-framework>=1.5.0,<2.0.0` GA surface.

#### 4.1.1 Agent construction and configuration

`GantryToolBridge.build_agent` and `as_agent`
(`agent_gantry/integrations/agent_framework_bridge.py`) construct agents via:

```python
Agent(client, instructions, name=name, tools=tools, middleware=middleware)
```

This is the idiomatic AF 1.x constructor signature. ✅

`_build_callable_for_tool` uses `@agent_framework.tool(wrapper, name=..., description=...,
approval_mode=...)` when AF is installed, producing a real `FunctionTool` with GA metadata. ✅

#### 4.1.2 Workflow subsystem

| Builder | Status |
|---------|--------|
| `SequentialBuilder` | ✅ `build_sequential_workflow` |
| `HandoffBuilder` | ✅ `build_handoff_workflow` — `.participants()`, `.with_start_agent()`, `.add_handoff()` |
| `WorkflowBuilder` + `AgentExecutor` | ✅ `build_workflow` |

All three builders are imported from `agent_framework.orchestrations` ✅

#### 4.1.3 Hosting / AF 1.6.0 ContextVar bug

AF 1.6.0 (released 2026-05-22) enables `asyncio.ContextVar`-based instrumentation by
default. Concurrent `Agent.run()` coroutines via `asyncio.gather()` / `TaskGroup` raise:

```
ValueError: <Token …> was created in a different Context
```

The repo already has a complete workaround:

1. `disable_af_instrumentation()` in `agent_gantry/__init__.py` and
   `agent_framework_bridge.py` — calls `agent_framework.telemetry.disable_instrumentation()`
   when AF ≥ 1.6.0 is detected.
2. `GantryToolBridge(disable_af_instrumentation=True)` — convenience flag to apply the
   shim at bridge construction time.
3. Documentation in `GantryObservabilityMiddleware` docstring.

Sequential workflows (`WorkflowAgent`, `SequentialBuilder`, `HandoffBuilder`) are **not**
affected.

Source: https://pypi.org/pypi/agent-framework/json (1.6.0 release notes, verified 2026-05-27).

#### 4.1.4 Middleware pipeline

`GantryApprovalMiddleware`, `GantryObservabilityMiddleware`, `GantryToolChoiceMiddleware` all
reside in `agent_gantry/integrations/agent_framework_middleware.py`. They:

- Use `FunctionMiddleware` (preferred) with `ChatMiddlewareLayer` as a fallback for older
  1.x point releases. ✅
- Import AF lazily so the module remains usable without AF installed. ✅
- `MiddlewareTermination` raised correctly for approval-required tools. ✅

#### 4.1.5 Tools / skills / providers — `GantryContextProvider`

`GantryContextProvider`
(`agent_gantry/integrations/agent_framework_provider.py`) subclasses `ContextProvider`
dynamically (lazy import). Per-run and per-call retrieval strategies are both implemented,
including the `as_chat_middleware()` factory for per-call refresh. ✅

`source_id` keying ensures co-operation with `SkillsProvider` and other context providers. ✅

`MissingRequiredToolError` is raised at construction time when `required=[...]` names are
absent from the gantry registry. ✅

#### 4.1.6 Human-in-the-loop

`_APPROVAL_REQUIRED_CAPS` maps `WRITE_DATA`, `DELETE_DATA`, `EXECUTE_CODE`, `FINANCIAL`, and
`PII_ACCESS` capabilities to AF's `approval_mode="always_require"`, causing AF to surface an
approval event before invocation. ✅

`GantryApprovalMiddleware` raises `MiddlewareTermination` for tools flagged
`ConfirmationRequiredError` by `SecurityPolicy`. ✅

**Risk level:** Safe internal — no changes required.

### 4.2 LangGraph

`examples/agent_frameworks/langgraph_example.py` uses `create_react_agent` from
`langgraph.prebuilt` and `ChatOpenAI` from `langchain_openai`. Both are the canonical
LangGraph 1.x patterns. ✅

`fetch_framework_tools(gantry, query, framework="langgraph")` in `framework_adapters.py`
returns OpenAI-style schemas (which LangGraph accepts natively). ✅

**Risk level:** Safe internal — no changes required.

### 4.3 AutoGen (AG2 0.7.5)

`examples/agent_frameworks/autogen_example.py` uses `autogen_agentchat.agents.AssistantAgent`
and `autogen_ext.models.openai.OpenAIChatCompletionClient` — the current AG2 v0.4 API. ✅

The example includes a migration note flagging MAF as the successor direction and pointing to
the MAF example for teams evaluating migration. ✅

**Risk level:** Safe internal — no changes required.

### 4.4 CrewAI

`crewai>=1.6.1` is pinned because 1.6.1 is the highest version that co-installs with
`agent-framework` (due to `opentelemetry-api` range conflicts in crewai ≥ 1.7.0). Standalone
CrewAI environments can use 1.14.5 (latest stable). ✅ (comment updated in this run)

### 4.5 LlamaIndex / Semantic Kernel

Both are pinned at current floors (`llama-index-core>=0.14.22`, `semantic-kernel>=1.36.0`)
and have no actionable changes at this time.

---

## 5 · Universal Tool-Calling Design

### 5.1 `ToolSpec` contract

The provider-agnostic contract is implemented via:

**`agent_gantry/adapters/tool_spec/base.py`**
- `ToolCallPayload` (Pydantic model): `tool_name`, `tool_call_id`, `arguments`, `raw_payload`.
- `ToolSpecAdapter` (Protocol): `dialect_name`, `to_provider_schema`, `from_provider_payload`,
  `to_tool_call`, `format_tool_result`.

**`agent_gantry/adapters/tool_spec/providers.py`**

| Adapter | Dialect | Notes |
|---------|---------|-------|
| `OpenAIAdapter` | `openai` | Chat Completions — `{type: "function", function: {...}}` |
| `OpenAIResponsesAdapter` | `openai_responses` | Responses API — flat `{type: "function", name: ...}` |
| `AnthropicAdapter` | `anthropic` | `{name, description, input_schema}` |
| `GeminiAdapter` | `gemini` | `{name, description, parameters}` |
| `MistralAdapter` | `mistral` | Inherits `OpenAIAdapter` |
| `GroqAdapter` | `groq` | Inherits `OpenAIAdapter` |
| `AgentFrameworkAdapter` | `agent_framework` | Inherits `OpenAIAdapter`, adds `metadata` opt |

### 5.2 Tool name normalisation

Tool names are validated at registration via `validate_tool_name` (lowercase alphanumeric +
underscore, 1–128 chars). All provider adapters pass the name through verbatim — no
normalisation needed at the adapter layer. ✅

### 5.3 Required vs optional arguments

`AnthropicAdapter.to_provider_schema(strict=True)` injects `additionalProperties: false`
into the schema copy (without mutating the shared `ToolDefinition`). ✅

`OpenAIResponsesAdapter.to_provider_schema(strict=True)` sets `schema["strict"] = True`. ✅

### 5.4 Tool call IDs

| Adapter | ID field | Source |
|---------|----------|--------|
| `OpenAIAdapter` | `payload["id"]` | Chat Completions `tool_calls[].id` |
| `OpenAIResponsesAdapter` | `payload["call_id"]` | Responses API `output[].call_id` |
| `AnthropicAdapter` | `payload["id"]` | `tool_use` block `id` |
| `GeminiAdapter` | `payload.get("id")` | Present on ≥1.x parallel calls |
| `AgentFrameworkAdapter` | `payload.get("id") or payload.get("call_id")` | Dual-field lookup |

All IDs are echoed back in `format_tool_result` to support parallel tool calls. ✅

### 5.5 Parallel tool calls

`GeminiAdapter.format_tool_result` places `id` inside `functionResponse`:
```python
{"functionResponse": {"name": "...", "response": {...}, "id": "..."}}
```
This is correct per the `FunctionResponse` proto schema. ✅

OpenAI (both adapters) and Anthropic echo back `tool_call_id` / `call_id` / `tool_use_id`
to the provider, enabling parallel call correlation. ✅

### 5.6 Tool-result round-tripping

| Adapter | Result shape |
|---------|-------------|
| `OpenAIAdapter` | `{role: "tool", content: "...", name: "...", tool_call_id: "..."}` |
| `OpenAIResponsesAdapter` | `{type: "function_call_output", call_id: "...", output: "..."}` |
| `AnthropicAdapter` | `{type: "tool_result", content: "...", tool_use_id: "...", is_error: bool}` |
| `GeminiAdapter` | `{functionResponse: {name: "...", response: {...}, id: "..."}}` |

All shapes are verified against current provider documentation. ✅

---

## 6 · Documentation and Example Updates

### 6.1 `examples/fast_track_demo.py`

Uses `client.chat.completions.create()` (Chat Completions). This is intentional — the demo
is a minimal "before/after" and Chat Completions remains fully supported. The Responses API
demo is in `examples/llm_integration/openai_demo.py` (Scenario A). No change needed.

### 6.2 `examples/llm_integration/openai_demo.py`

Correctly shows Responses API as Scenario A with `gpt-4.1`. Chat Completions retained as
Scenario B with `gpt-4o`. Both model names are current. ✅

### 6.3 `examples/llm_integration/anthropic_demo.py`

Uses `claude-sonnet-4-6` — current production model. ✅

### 6.4 `examples/llm_integration/google_genai_demo.py`

Uses `gemini-2.5-flash` — current stable model. ✅
Uses `types.Part.from_function_response()` with `id` kwarg for parallel call correlation. ✅
Uses `client.aio.models.generate_content()` async path. ✅

### 6.5 `examples/agent_frameworks/autogen_example.py`

Includes a migration note to MAF. The AG2 0.7.5 API usage is correct. ✅

### 6.6 `examples/agent_frameworks/langgraph_example.py`

Uses `create_react_agent` and `langchain_core.tools.tool` — both current LangGraph 1.x
idioms. ✅

---

## 7 · Security Audit

### 7.1 SSRF bypass — malformed URL ports (PR #207)

**Fixed.** `SecurityPolicy._extract_domains` in `agent_gantry/core/security.py` now
correctly handles URLs where `urllib.parse.urlparse(...).port` raises `ValueError`:

```python
try:
    _ = parsed.port
except ValueError:
    if not parsed.hostname:
        domains.add("<invalid_domain>")
        continue
```

When `parsed.port` raises `ValueError` **and** `parsed.hostname` is `None` / empty (e.g.,
`http://@:evil.com/etc/passwd`), the URL is blocked by adding `<invalid_domain>` to the
domain set, which is never in the allowed list.

**Test coverage:** `tests/test_security.py::test_ssrf_port_bypass` — passes. ✅

**Clarification on the general case:** For URLs with a valid hostname but an invalid port
(e.g., `http://example.com:evil.com`), `parsed.hostname = "example.com"` and the code
falls through to the normal hostname-extraction path, adding `"example.com"` to the
domain set. This is correct behaviour: the true network target of such a URL is
`example.com` (the malformed port prevents connection, so there is no SSRF risk), and
`"example.com"` is correctly validated against the allowed list.

### 7.2 Search combobox accessibility (PR #208)

`Escape` on the search combobox now calls `e.preventDefault()` instead of
`searchInput.blur()`, retaining focus on the input field. ✅

---

## 8 · Test and Migration Plan

### 8.1 Tests to add or update

| Test | Status | Notes |
|------|--------|-------|
| `test_security.py::test_ssrf_port_bypass` | ✅ Exists | Covers the specific bypass vector |
| `test_tool_spec_adapters.py` | ✅ Exists | Covers all provider adapters |
| `test_agent_framework_integration.py` | ✅ Exists | Covers AF bridge, provider, middleware |
| LangGraph 1.2.2 regression | ✅ No API changes | No new test needed |
| `test_ssrf_general_invalid_port` | ⚠️ **Needs confirmation** | Optional: add test for `http://example.com:evil.com` to document the expected `"example.com"` extraction behaviour |

### 8.2 Regeneration checklist

- [x] `uv.lock` regenerated (`langchain`, `langchain-openai`, `langgraph` updated).
- [ ] VCR recordings / golden HTTP responses — none present in this repo; not applicable.
- [ ] Schema snapshots — not present; not applicable.

### 8.3 Changelog-ready migration notes

```
## [0.4.x] — 2026-05-27

### Changed
- `langchain` floor bumped `>=1.3.1 → >=1.3.2` (patch; no API changes).
- `langchain-openai` floor bumped `>=1.2.1 → >=1.2.2` (patch; no API changes).
- `langgraph` floor bumped `>=1.2.1 → >=1.2.2` (patch; no API changes).
- `pyproject.toml` comments updated for `crewai` (1.14.5), `google-adk` (2.1.0),
  and `google-genai` (2.6.0) to reflect current latest stable versions.

### Fixed
- SSRF bypass via malformed URL ports (`http://@:evil.com/…`) in
  `SecurityPolicy._extract_domains` — URLs that lack a valid hostname after a
  port-parsing `ValueError` are now blocked as `<invalid_domain>`.
```

---

## 9 · Must-Change Now vs Good Next Steps

### Must-change now (completed in this run)

| # | Change | File | Status |
|---|--------|------|--------|
| 1 | Bump `langchain>=1.3.2` | `pyproject.toml` | ✅ Done |
| 2 | Bump `langchain-openai>=1.2.2` | `pyproject.toml` | ✅ Done |
| 3 | Bump `langgraph>=1.2.2` | `pyproject.toml` | ✅ Done |
| 4 | Regenerate `uv.lock` | `uv.lock` | ✅ Done |

### Good next-step improvements

| # | Change | File | Risk |
|---|--------|------|------|
| 1 | Validate `semantic-kernel>=1.42.0` in isolation with `agent-framework`; bump floor if compatible | `pyproject.toml` | Safe with shim |
| 2 | Validate `google-adk>=2.1.0` in standalone env; add docs page showing `google-adk 2.x` Workflow Runtime pattern | `docs/`, `examples/` | Safe with shim |
| 3 | Add optional test for `http://example.com:evil.com` → documents expected hostname extraction behaviour | `tests/test_security.py` | Safe internal |
| 4 | Add a MAF migration guide (`docs/migrating-from-autogen.md`) for teams moving from AutoGen/AG2 to AF 1.x | `docs/` | Documentation |
| 5 | Assess `google-adk 2.x` Workflow Runtime as a possible replacement for custom `WorkflowAgent` wrappers in `examples/agent_frameworks/google_adk_example.py` | `examples/` | Needs confirmation |
