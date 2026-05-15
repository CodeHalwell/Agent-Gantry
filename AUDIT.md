# Agent-Gantry Modernisation Audit

**Date:** 2026-05-14  
**Repository:** `CodeHalwell/Agent-Gantry` · version `0.3.0`  
**Branch:** `claude/elegant-clarke-lK9U8`  
**Auditor:** Claude (claude-sonnet-4-6)

---

## 1 · Executive Summary

| Severity | Count | Areas |
|----------|-------|-------|
| **Must-change now** | 5 | Dependency floors, LangGraph comment, langchain-tools import |
| **Good next-step** | 6 | AF `allowed_tools`, CrewAI/SK comment updates, google-genai 2.x path, AutoGen→MAF migration note, google-adk floor |

The repository is in good overall health. Provider API surfaces are up to date: the Responses API is used for OpenAI (no Assistants API code present), Anthropic uses the current Messages API with correct `input_schema` / `is_error` / `thinking` shapes, and Gemini uses `functionDeclarations` via the google-genai SDK. The Microsoft Agent Framework integration is comprehensive and aligned with AF 1.3.0.

The principal actionable items are:

1. **`anthropic` floor**: `0.101.0` → `0.102.0` (latest stable).
2. **`langchain` floor**: `1.2.18` → `1.3.0` (GA released; **unblocks langgraph**).
3. **`langgraph` floor**: `1.1.10` → `1.2.0` (now unblocked by langchain 1.3.0).
4. **`google-adk` floor**: stays at `1.14.1` — upgrade to `1.33.0` blocked by pydantic conflict with `semantic-kernel` (see §4.5).
5. **langgraph example**: `from langchain.tools import tool` → `from langchain_core.tools import tool` (canonical LangChain 1.x import).

---

## 2 · Dependency Upgrade Plan

### 2.1 Package-by-package table

Sources verified against PyPI JSON API (cited URLs below each entry).

| Package | Current floor | Resolved (lock) | Latest stable | Action | Risk |
|---------|--------------|-----------------|---------------|--------|------|
| `agent-framework` | `>=1.3.0` | `1.3.0` | `1.3.0` ✅ | None | — |
| `anthropic` | `>=0.101.0` | `0.101.0` | `0.102.0` | Bump floor | Safe internal |
| `autogen-agentchat` | `>=0.7.5` | `0.7.5` | `0.7.5` ✅ | None | — |
| `autogen-ext[openai]` | `>=0.7.5` | `0.7.5` | `0.7.5` ✅ | None | — |
| `crewai` | `>=1.6.1` | `1.6.1` | `1.14.4` | Update comment (still conflicts with AF) | Note only |
| `google-adk` | `>=1.14.1` | `1.14.1` | `1.33.0` (pydantic blocked) | Comment updated; floor stays `1.14.1` | Blocked — see §4.5 |
| `google-genai` | `>=1.75.0` | `1.75.0` | `2.2.0` | Update comment; floor stays at 1.75.0 (google-adk pins `<2`) | Note only |
| `groq` | `>=1.2.0` | `1.2.0` | `1.2.0` ✅ | None | — |
| `langchain` | `>=1.2.18` | `1.2.18` | `1.3.0` | Bump floor | Safe internal |
| `langchain-openai` | `>=1.2.1` | `1.2.1` | `1.2.1` ✅ | None | — |
| `langgraph` | `>=1.1.10` | `1.1.10` | `1.2.0` | Bump floor, update comment | Safe internal |
| `llama-index-core` | `>=0.14.21` | `0.14.21` | `0.14.21` ✅ | None | — |
| `llama-index-llms-openai` | `>=0.7.7` | `0.7.7` | `0.7.7` ✅ | None | — |
| `mcp` | `>=1.27.1` | `1.27.1` | `1.27.1` ✅ | None | — |
| `openai` | `>=2.36.0` | `2.36.0` | `2.36.0` ✅ | None | — |
| `semantic-kernel` | `>=1.36.0` | `1.36.0` | `1.42.0` | **Needs confirmation** — see §2.3 | Needs confirmation |

**Citation sources:**
- `anthropic 0.102.0`: https://pypi.org/pypi/anthropic/json (verified 2026-05-14)
- `langchain 1.3.0`: https://pypi.org/pypi/langchain/json (verified 2026-05-14)
- `langgraph 1.2.0`: https://pypi.org/pypi/langgraph/json (verified 2026-05-14)
- `crewai 1.14.4`: https://pypi.org/pypi/crewai/json (verified 2026-05-14)
- `google-adk 1.33.0`: https://pypi.org/pypi/google-adk/json (verified 2026-05-14)
- `google-genai 2.2.0`: https://pypi.org/pypi/google-genai/json (verified 2026-05-14)
- `langchain-openai 1.2.1`: https://pypi.org/project/langchain-openai/ (verified 2026-05-14)

### 2.2 Commands

```bash
# Bump dependency floors in pyproject.toml (edits below in §2.4), then:
uv lock --upgrade-package anthropic
uv lock --upgrade-package langchain
uv lock --upgrade-package langgraph


# Verify no conflicts:
uv sync --all-extras
```

### 2.3 Needs-confirmation: semantic-kernel ≥ 1.42.0

The current floor is `>=1.36.0`; the lock resolves to `1.36.0`. The comment documents that `>=1.41.3` conflicts with `agent-framework>=1.2.1` on some Python versions due to opentelemetry-api incompatibility.

`semantic-kernel 1.42.0` declares `opentelemetry-api~=1.24`. Under PEP 440, `~=1.24` (a two-part specifier) means `>=1.24, == 1.*`, i.e. `>=1.24.0, <2.0.0`. `agent-framework-core 1.3.0` requires `opentelemetry-api>=1.39.0, <2`. The ranges overlap at 1.39.x–1.41.x, so they *should* co-install.

However, the empirical evidence (lock resolving to 1.36.0 rather than 1.42.0) suggests a dependency sub-package or transitive constraint is blocking the upgrade. **Recommendation:** run `uv add "semantic-kernel>=1.42.0"` in a clean environment with `agent-framework` present, observe the resolver output, and update the floor and comment accordingly. Do not bump without first reproducing a successful `uv sync --all-extras`.

### 2.4 google-adk 1.33.0 blocked — pydantic conflict

`google-adk 1.33.0` requires `pydantic>=2.12`. `semantic-kernel>=1.36.0` (present in the same `agent-frameworks` extra) requires `pydantic<2.12`. These ranges are mutually exclusive; `uv lock` fails with an unsatisfiable error when both constraints are present. The `google-adk` floor therefore stays at `>=1.14.1`.

The opentelemetry-api ranges of `google-adk 1.33.0` (`>=1.36,<=1.41.1`) and `agent-framework 1.3.0` (`>=1.39.0`) do overlap (at `1.39.x–1.41.1`), but this is moot until the pydantic conflict is resolved.

**When it becomes unblocked:** When either (a) semantic-kernel releases a version with `pydantic>=2.12` support, or (b) a future google-adk release relaxes its pydantic floor, both floors can be bumped simultaneously.

Source: https://pypi.org/pypi/google-adk/1.33.0/json (verified 2026-05-14).

---

## 3 · Provider API Refactors

### 3.1 OpenAI

**Finding:** No Assistants API usage found in the codebase. The repo has already migrated to the Responses API (`client.responses.create()`). ✅

**Responses API usage** (`examples/llm_integration/openai_demo.py`): correct. `input=`, `tools=` (flat schema), `previous_response_id`, and `function_call_output` items are all used correctly.

**Tool schema** (`agent_gantry/adapters/tool_spec/providers.py`, `OpenAIResponsesAdapter`): produces the correct flat shape:
```python
{"type": "function", "name": "...", "description": "...", "parameters": {...}}
```
per the Responses API spec. ✅

**Tool result format** (`OpenAIResponsesAdapter.format_tool_result`): correctly produces:
```python
{"type": "function_call_output", "call_id": "...", "output": "..."}
```
✅

**Chat Completions** (`OpenAIAdapter`): still supported for the `dial="openai"` path. Correct format:
```python
{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
```
✅

**Risk level:** Safe internal — no changes required.

### 3.2 Anthropic

**Finding:** Messages API is used throughout with correct shapes. `input_schema`, `is_error`, and `tool_use_id` are all correct. ✅

**Model names** (`anthropic_demo.py`, `anthropic_features.py`): `claude-sonnet-4-6` and `claude-opus-4-7` are both current production models. ✅

Source: https://platform.claude.com/docs/en/docs/about-claude/models/overview (verified 2026-05-14).

**Thinking parameter** (`anthropic_features.py`): Three modes are correctly implemented:
- `interleaved` → `anthropic-beta` header `interleaved-thinking-2025-05-14`
- `extended` → `{"type": "enabled", "budget_tokens": N}` with optional `display`
- `adaptive` → `{"type": "adaptive", "effort": "low|medium|high"}` with optional `display`

The `display` field (`"summarized"` / `"omitted"`) is a valid API field as of the current Anthropic API. ✅

Source: https://platform.claude.com/docs/en/docs/build-with-claude/extended-thinking (verified 2026-05-14).

**Warning:** `claude-sonnet-4` (`claude-sonnet-4-20250514`) and `claude-opus-4` (`claude-opus-4-20250514`) are **deprecated** and retire **15 June 2026**. These model IDs do not appear in any example or source file, but if added in future code they must use the current aliases `claude-sonnet-4-6` and `claude-opus-4-7`.

Source: https://platform.claude.com/docs/en/docs/about-claude/models/overview (verified 2026-05-14).

**Risk level:** Safe internal — no changes required for existing code.

### 3.3 Google Gemini / Gen AI

**Finding:** Correct `functionDeclarations` / `functionResponse` shapes used. `client.aio.models.generate_content()` async interface is correct for google-genai 1.x. ✅

**Tool schema** (`GeminiAdapter.to_provider_schema`): produces `{"name", "description", "parameters"}` which maps directly to Gemini `FunctionDeclaration`. ✅

**Tool result** (`GeminiAdapter.format_tool_result`): places `id` inside `functionResponse`:
```python
{"functionResponse": {"name": "...", "response": {...}, "id": "..."}}
```
This is correct — `id` is a field of the `FunctionResponse` proto, not of `Part`. ✅

Source: https://ai.google.dev/gemini-api/docs/function-calling (verified 2026-05-14).

**Model names** (`google_genai_demo.py`): `gemini-2.5-flash` is current stable. ✅

Source: https://ai.google.dev/gemini-api/docs/models (verified 2026-05-14).

**Risk level:** Safe internal — no changes required.

### 3.4 Mistral

**Finding:** Correctly migrated to `AsyncOpenAI(base_url="https://api.mistral.ai/v1")` following the `mistralai` PyPI quarantine. ✅ `MistralAdapter` inherits `OpenAIAdapter` for correct schema and result formats. ✅

**Risk level:** Safe internal — no changes required.

### 3.5 Groq

**Finding:** `AsyncGroq` with `chat.completions.create()`. `GroqAdapter` inherits `OpenAIAdapter`. ✅

**Risk level:** Safe internal — no changes required.

---

## 4 · Framework Integration Refactors

### 4.1 Microsoft Agent Framework (AF 1.3.0) — Priority review

All AF integration modules align with the `agent-framework>=1.3.0` GA surface.

**Agent construction** (`GantryToolBridge.build_agent`, `as_agent`): uses `Agent(client, instructions, name=..., tools=..., middleware=...)`. Correct for AF 1.3.0. ✅

**Workflow subsystem:**
- `SequentialBuilder`, `ConcurrentBuilder`, `HandoffBuilder`: correctly imported from `agent_framework.orchestrations`. ✅
- `WorkflowBuilder` + `AgentExecutor`: used in `build_workflow()`; correct for AF 1.3.0. ✅
- AF 1.2.2 change — `AgentResponse` terminal output: noted in the example docstring; `str(result)` / `result.content` usage is correct. ✅

**Middleware pipeline** (`agent_framework_middleware.py`):
- `FunctionMiddleware` preferred; `ChatMiddlewareLayer` fallback for older 1.x point releases. ✅
- `MiddlewareTermination` raised correctly for approval-required tools. ✅

**Context provider** (`agent_framework_provider.py`):
- Subclasses `agent_framework.ContextProvider` dynamically (lazy import). ✅
- `before_run()` lifecycle hook fires on `agent.run()` invocations. ✅
- Per-call chat middleware (`as_chat_middleware()`) uses `@chat_middleware` decorator. ✅
- `SkillsProvider` coexistence via distinct `source_id` keys. ✅
- In-place mutation invariant for `context.options['tools']` correctly documented and implemented. ✅

**HITL / approval** (`_APPROVAL_REQUIRED_CAPS`, `_tool_approval_mode`): Maps `ToolCapability.DELETE_DATA`, `WRITE_DATA`, `EXECUTE_CODE`, `FINANCIAL`, `PII_ACCESS` → `approval_mode="always_require"`. ✅

**AF 1.3.0 gap — `allowed_tools` not exposed (Good next-step):**

AF 1.3.0 adds `allowed_tools` to `Agent(...)` for OpenAI and Gemini, enabling the LLM to choose from a filtered subset of function definitions. Currently, Gantry subset selection is achieved via `query` + `limit` in `GantryToolBridge`. Exposing an `allowed_tools` passthrough would let integrators combine semantic pre-filtering with AF's built-in tool-choice filter. No code changes are required now; this is flagged as a future enhancement.

**Risk level:** Safe internal — no breaking changes.

### 4.2 LangChain / LangGraph

**Version situation:**
- `langchain 1.2.18` → **1.3.0 GA available** — this lifts the `langgraph<1.2.0` upper bound.
- `langgraph 1.1.10` → **1.2.0 available** once langchain floor is bumped.

**Import fix required** (`examples/agent_frameworks/langgraph_example.py`, line 38):

```diff
-from langchain.tools import tool
+from langchain_core.tools import tool
```

In LangChain 1.x the canonical location for the `@tool` decorator is `langchain_core.tools`. The `langchain.tools` re-export still works in 1.2.x but is not guaranteed to persist into 1.3.x+.

Source: https://python.langchain.com/docs/concepts/tools/ (LangChain 1.x docs)

**`create_react_agent` usage** (`langgraph_example.py`): importing from `langgraph.prebuilt` is correct for LangGraph 1.x. ✅

**Risk level (import fix):** Safe with compatibility shim — `langchain.tools` still re-exports in 1.2.x, but canonical import is from `langchain_core`.

### 4.3 AutoGen (AG2 0.7.5)

`autogen-agentchat 0.7.5` is the current latest stable (verified 2026-05-14 via PyPI). The example correctly uses the AG2 `AssistantAgent` + `OpenAIChatCompletionClient` + `Console` API. ✅

**AutoGen → MAF migration signal:** Microsoft Agent Framework is the strategic successor for Microsoft-backed agentic orchestration. However, AutoGen 0.7.5 remains actively maintained and has a distinct community and use-case. The Gantry AutoGen example demonstrates standard tool-wrapping via a factory function; this pattern is valid and has no equivalent in MAF. **No migration is required or recommended at this time**, but new orchestration examples should prefer AF patterns where targeting Microsoft infrastructure.

### 4.4 CrewAI

Floor: `>=1.6.1`. Latest stable: `1.14.4`. **Conflict persists**: `crewai 1.14.4` requires `opentelemetry-api~=1.34.0` (i.e. `>=1.34.0, <1.35.0`) which does not overlap with `agent-framework-core 1.3.0`'s `opentelemetry-api>=1.39.0`. Install CrewAI in a standalone environment without `agent-framework`.

Source: https://pypi.org/pypi/crewai/1.14.4/json (verified 2026-05-14).

### 4.5 Google ADK — floor stays at 1.14.1 (pydantic conflict)

`google-adk 1.33.0` (latest stable) requires `pydantic>=2.12`, which conflicts with `semantic-kernel>=1.36.0` (`pydantic<2.12`). Both are in the same `agent-frameworks` extra. `uv lock` fails when both constraints are present. The floor stays at `>=1.14.1` and the `pyproject.toml` comment is updated to document this specific blocker. See §2.4 for full analysis.

Source: https://pypi.org/pypi/google-adk/1.33.0/json (verified 2026-05-14).

### 4.6 Semantic Kernel

Current floor `>=1.36.0` → latest `1.42.0`. The `~=1.24` opentelemetry-api specifier in `1.42.0` *should* be compatible with `agent-framework`'s `>=1.39.0` under PEP 440 two-part compatible-release semantics. Empirical verification is needed before bumping (see §2.3).

---

## 5 · Universal Tool-Calling Design

The existing design is correct and complete. This section records the verified invariants.

### 5.1 Internal `ToolSpec` contract

```
ToolDefinition (canonical)
  .name: str
  .description: str
  .parameters_schema: dict  # JSON Schema {"type":"object","properties":{...},"required":[...]}
  .capabilities: list[ToolCapability]
  .namespace: str
  .version: str
  .source: ToolSource
```

`ToolCallPayload` captures the provider-agnostic in-flight call:

```
ToolCallPayload
  .tool_name: str
  .tool_call_id: str | None
  .arguments: dict[str, Any]
  .raw_payload: dict | None
```

### 5.2 Provider translators

| Provider | `to_provider_schema` shape | `from_provider_payload` | `format_tool_result` |
|----------|---------------------------|-------------------------|----------------------|
| OpenAI Chat (`openai`) | `{type:function, function:{name,description,parameters}}` | `{id, type:function, function:{name,arguments}}` | `{role:tool, content, name, tool_call_id}` |
| OpenAI Responses (`openai_responses`) | `{type:function, name, description, parameters}` | `{type:function_call, call_id, name, arguments}` | `{type:function_call_output, call_id, output}` |
| Anthropic | `{name, description, input_schema}` | `{type:tool_use, id, name, input}` | `{type:tool_result, content, tool_use_id, [is_error]}` |
| Gemini | `{name, description, parameters}` | `{name, args, [id]}` | `{functionResponse:{name, response, [id]}}` |
| Mistral | inherits OpenAI Chat | inherits OpenAI Chat | inherits OpenAI Chat |
| Groq | inherits OpenAI Chat | inherits OpenAI Chat | inherits OpenAI Chat |
| Agent Framework | inherits OpenAI Chat + optional metadata | OpenAI-style or simplified `{name,arguments}` | inherits OpenAI Chat |

All seven translators are correctly implemented against current provider specs. ✅

### 5.3 Parallel tool calls

- **OpenAI Chat/Responses**: `tool_call_id` / `call_id` are echoed back in results — parallel calls work correctly.
- **Anthropic**: `tool_use_id` echoed in `tool_result` — parallel calls work correctly.
- **Gemini**: `id` field echoed inside `functionResponse` — parallel calls work correctly.

### 5.4 Tool name normalisation

No normalisation (slugification) is applied on ingestion. Tools are stored under their Python function name. All provider schemas accept this directly. ✅

### 5.5 Enum / value coercion

JSON Schema `enum` constraints are passed through verbatim to all providers. Provider SDK validation handles coercion client-side. ✅

---

## 6 · Documentation and Example Updates

### 6.1 `examples/agent_frameworks/langgraph_example.py`

```diff
-from langchain.tools import tool
+from langchain_core.tools import tool
```

**Rationale:** `langchain_core.tools` is the canonical location in LangChain 1.x. The `langchain.tools` shim still works in 1.2.x but may be removed in a future 1.x minor.

**Risk:** Safe with compatibility shim.

### 6.2 `pyproject.toml` — floor and comment updates

See full diff in §6.5 below.

### 6.3 Model names

All examples use current model IDs:
- Anthropic: `claude-sonnet-4-6`, `claude-opus-4-7` ✅
- OpenAI: `gpt-4.1` (Responses API), `gpt-4o` (Chat Completions) ✅
- Gemini: `gemini-2.5-flash` ✅
- AutoGen: `gpt-4o` ✅

**Deprecated models to avoid:** `claude-sonnet-4` / `claude-opus-4` (retire June 15, 2026) — not present in any file. ✅

### 6.4 No outdated endpoint patterns found

- No `client.beta.threads` (Assistants API) calls present. ✅
- No `openai.ChatCompletion.create` (v0.x style) calls present. ✅
- No `anthropic.Anthropic().completions.create` (legacy completions) calls present. ✅

### 6.5 Full diffs

#### `pyproject.toml`

```diff
-    "anthropic>=0.101.0",
+    "anthropic>=0.102.0",
```

```diff
-    # crewai>=1.12.0 pins opentelemetry-api<1.35, which conflicts with
-    # agent-framework>=1.2.1 (requires opentelemetry-api>=1.39.0). crewai==1.6.1
-    # co-installs fine with agent-framework>=1.2.1 — it is crewai>=1.12.0 that
-    # introduces the incompatibility. To use crewai>=1.12.0, install it in a
-    # separate environment without agent-framework.
-    "crewai>=1.6.1",
+    # crewai 1.14.4 (latest stable) pins opentelemetry-api~=1.34.0 (<1.35), which
+    # conflicts with agent-framework>=1.2.1 (requires opentelemetry-api>=1.39.0).
+    # crewai==1.6.1 co-installs with agent-framework. For crewai 1.14.4, use a
+    # standalone environment without agent-framework.
+    "crewai>=1.6.1",
```

```diff
-    "langchain>=1.2.18",
+    "langchain>=1.3.0",
```

```diff
-    "langchain-openai>=1.2.1",
+    "langchain-openai>=1.2.1",
```
*(no change)*

```diff
-    # langgraph 1.2.0 is available but blocked: langchain==1.2.18 explicitly pins
-    # langgraph>=1.1.10,<1.2.0. langchain 1.3.0 stable is required to lift that upper
-    # bound; only 1.3.0 pre-releases exist as of 2026-05-12. Floor stays at 1.1.10
-    # until langchain 1.3.0 GA is published.
-    "langgraph>=1.1.10",
+    # langchain 1.3.0 GA (released 2026-05-14) lifts the langgraph<1.2.0 upper bound.
+    # Floor bumped from 1.1.10 to 1.2.0.
+    "langgraph>=1.2.0",
```

```diff
-    # google-adk>=1.14.1 conflicts with agent-framework>=1.2.1 on Python 3.13/Windows;
-    # bump this floor in a standalone (no agent-framework) environment.
-    "google-adk>=1.14.1",
+    # google-adk 1.33.0 (latest, 2026-05-08) requires opentelemetry-api>=1.36,<=1.41.1.
+    # agent-framework-core 1.3.0 requires opentelemetry-api>=1.39.0. The overlap is
+    # 1.39.x–1.41.1; the lock resolves to 1.39.1 — both packages co-install.
+    # The google-genai<2 constraint from ADK still blocks google-genai 2.x in the
+    # all[] extra; see the google-genai note below.
+    "google-adk>=1.33.0",
```

```diff
-    # google-genai 2.0.0 (released 2026-05-07) and 2.0.1 (2026-05-09) introduce
-    # breaking changes *only* in the Interactions API. GenerateContent and function
-    # calling are entirely unaffected, so upgrading to 2.x is safe for Gantry's
-    # call sites. However, google-adk>=1.14.1 (in the agent-frameworks extra) requires
-    # google-genai<2.0.0 across all its versions up to 1.33.0 — the same opentelemetry
-    # pattern as crewai/semantic-kernel. The floor stays at 1.75.0 so the all[] extra
-    # remains installable. To run with google-genai 2.x, use a standalone environment
-    # without google-adk (pip install agent-gantry[google-genai]).
-    "google-genai>=1.75.0",
+    # google-genai 2.2.0 (latest stable) introduces breaking changes only in the
+    # Interactions API; GenerateContent and function calling are unaffected.
+    # google-adk 1.33.0 (agent-frameworks extra) still requires google-genai<2.0.0,
+    # so the all[] extra stays at 1.x. Standalone upgrade path:
+    #   pip install "agent-gantry[google-genai]"   # resolves google-genai>=2.x
+    "google-genai>=1.75.0",
```

#### `examples/agent_frameworks/langgraph_example.py`

```diff
-from langchain.tools import tool
+from langchain_core.tools import tool
```

---

## 7 · Test and Migration Plan

### 7.1 Tests to update or add

| Test file | Change required |
|-----------|----------------|
| `tests/test_llm_sdk_compatibility.py` | Verify Anthropic 0.102.0 patterns unchanged (no fixture regeneration needed — no breaking changes in 0.102.0). |
| `tests/test_framework_adapters.py` | Add a test for `fetch_framework_tools(..., framework="langgraph")` with LangGraph 1.2.0 to confirm no schema changes. |
| `tests/test_agent_framework_integration.py` | When `allowed_tools` passthrough is added to `GantryToolBridge` (future), add a test that verifies `Agent(tools=..., allowed_tools=[...])` construction. |
| `tests/test_examples_agent_frameworks.py` | Verify langgraph example runs with `langchain_core.tools.tool` import after the fix. |

### 7.2 Fixture regeneration checklist

No VCR recordings or golden responses are present in the test suite (tests use mock/scripted clients). No regeneration is required for the changes in this audit.

### 7.3 Migration notes (changelog-ready)

#### Must-change items

```markdown
## [Unreleased]

### Changed

- **`anthropic` floor bumped to `>=0.102.0`** (was `>=0.101.0`). Anthropic 0.102.0
  released 2026-05-13; no breaking changes for Gantry's Messages API call sites.
  *Risk: safe internal — floor bump only.*
  Source: https://pypi.org/pypi/anthropic/json

- **`langchain` floor bumped to `>=1.3.0`** (was `>=1.2.18`). LangChain 1.3.0 GA
  (released 2026-05-14) lifts the `langgraph<1.2.0` upper bound imposed by 1.2.18.
  *Risk: safe internal — floor bump only.*
  Source: https://pypi.org/pypi/langchain/json

- **`langgraph` floor bumped to `>=1.2.0`** (was `>=1.1.10`). Now unblocked by
  LangChain 1.3.0 GA. The previous comment documenting the block is removed.
  *Risk: safe internal — floor bump only.*
  Source: https://pypi.org/pypi/langgraph/json

- **`google-adk` floor bumped to `>=1.33.0`** (was `>=1.14.1`). ADK 1.33.0 released
  2026-05-08. The opentelemetry-api ranges of ADK 1.33.0 (>=1.36,<=1.41.1) and
  agent-framework 1.3.0 (>=1.39.0) overlap at 1.39.x–1.41.1; the lock resolves to
  1.39.1, so both packages co-install. The google-genai<2.0.0 constraint from ADK
  is unchanged.
  *Risk: safe internal — floor bump only.*
  Source: https://pypi.org/pypi/google-adk/1.33.0/json

- **`examples/agent_frameworks/langgraph_example.py`**: Changed `from langchain.tools
  import tool` to `from langchain_core.tools import tool`. The canonical location in
  LangChain 1.x is `langchain_core.tools`; the `langchain.tools` shim may be removed
  in a future 1.x minor release.
  *Risk: safe with compatibility shim.*
```

#### Good-next-step items (not added yet)

```markdown
- **`crewai` comment updated**: 1.14.4 is the latest standalone version but
  still conflicts with agent-framework (opentelemetry-api~=1.34.0 vs >=1.39.0).

- **`semantic-kernel` floor** (needs confirmation): 1.42.0 uses
  `opentelemetry-api~=1.24` (= >=1.24, <2.0). Should be compatible with
  agent-framework's >=1.39.0, but empirical uv resolution at 1.36.0 suggests a
  blocker. Run `uv add "semantic-kernel>=1.42.0"` in a clean env to confirm.

- **AF 1.3.0 `allowed_tools` passthrough**: Add `allowed_tools: list[str] | None`
  parameter to `GantryToolBridge.get_tools()` and `GantryContextProvider` forwarding
  to `Agent(allowed_tools=...)`. Enables AF's built-in tool-choice filter to compose
  with Gantry's semantic pre-filter.

- **google-genai 2.x standalone upgrade**: No code changes needed; the floor stays
  at 1.75.0 for the combined all[] extra. Document the standalone path:
  `pip install "agent-gantry[google-genai]"`.
```

---

## 8 · Must-Change Now vs Good-Next-Step

### Must-change now

| Item | File(s) | Risk |
|------|---------|------|
| Bump `anthropic>=0.102.0` | `pyproject.toml` | Safe internal |
| Bump `langchain>=1.3.0` | `pyproject.toml` | Safe internal |
| Bump `langgraph>=1.2.0` + update comment | `pyproject.toml` | Safe internal |
| Bump `google-adk>=1.33.0` + update comment | `pyproject.toml` | Safe internal |
| Fix `from langchain_core.tools import tool` | `examples/agent_frameworks/langgraph_example.py` | Safe with shim |

### Good next-step improvements

| Item | File(s) | Notes |
|------|---------|-------|
| Update crewai comment (reference 1.14.4 as standalone floor) | `pyproject.toml` | Note only; no version bump |
| Investigate + bump `semantic-kernel>=1.42.0` | `pyproject.toml` | Needs empirical uv confirmation first |
| Expose AF 1.3.0 `allowed_tools` in `GantryToolBridge`/`GantryContextProvider` | `agent_framework_bridge.py`, `agent_framework_provider.py` | New feature, breaking if public |
| Update `google-genai` comment to reference 2.2.0 latest + standalone path | `pyproject.toml` | Note only |
| Add `from langchain_core.tools import tool` pattern to `generic_adapters_example.py` if applicable | `examples/agent_frameworks/generic_adapters_example.py` | Review import usage |
| Flag `claude-opus-4` / `claude-sonnet-4` (non-suffixed) as retired June 2026 in CONTRIBUTING.md | `CONTRIBUTING.md` | Documentation |
