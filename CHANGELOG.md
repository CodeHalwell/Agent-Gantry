# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.12.0] - 2026-08-26

Tool schemas now carry what your functions actually declare. Everything a
tool advertises — to every LLM provider dialect and all ~16 agent-framework
adapters — is derived from one schema, and that schema previously threw away
most of what the function said: no parameter descriptions ever, enums
flattened to bare strings, `dict` parameters mislabelled (and then rejected
by the executor), defaults dropped, nested models stringly typed. This
release fixes that at the source and makes the executor, the framework
adapters, and the provider dialects agree with it.

### Added

- **`@gantry.register` captures what the function declares.**
  `build_parameters_schema` now reads docstring parameter sections (Google
  `Args:`, NumPy `Parameters`, Sphinx `:param:`) and `Annotated[T, "…"]`
  metadata into `description`; maps `Literal`/`Enum` to `enum` with an
  inferred type; `dict`/`Mapping` to `object` (with typed
  `additionalProperties` when parameterized); `set`/`frozenset`/`tuple` to
  arrays (`uniqueItems` for sets, homogeneous tuple item types); Pydantic
  models, dataclasses and `TypedDict`s to inlined nested object schemas via
  Pydantic, with local `$ref`s resolved; records JSON-safe defaults (`Enum`
  defaults unwrapped to their values); handles PEP 604 unions in either
  order; and adds string `format`s for `datetime`/`date`/`time`/`UUID`.
- **Signature-introspecting frameworks stop flattening that schema.** The
  adapters whose framework re-derives its own LLM schema from the wrapped
  callable now pass the Gantry schema through natively: LlamaIndex gets an
  explicit `fn_schema` args model, Google ADK gets a declaration carrying
  `parameters_json_schema` verbatim (fixing ADK 1.x advertising *every*
  parameter as required; falls back to signature introspection on genai
  builds without the field), and Semantic Kernel reads per-parameter
  descriptions from `Annotated` metadata. CrewAI's args model is rebuilt on
  a shared JSON-schema→Pydantic bridge (`Literal` enums, typed array items,
  nested objects, defaults, descriptions, nullability), AG2 registration
  carries `Annotated` descriptions, and the Microsoft Agent Framework bridge
  passes the schema to `agent_framework.tool(schema=…)`.
  `ToolSpec.python_signature` gains typed `list[T]` annotations,
  schema-declared defaults, and opt-in `annotated_descriptions`.
- **Typed non-success errors and a working approve/resume path.**
  `ToolSpec.ainvoke` raises `ToolConfirmationRequiredError` /
  `ToolPermissionDeniedError` (both subclasses of the existing
  `ToolExecutionError`) for confirmation-gated and policy-denied calls, and
  both executor confirmation paths populate `error` with the reason and the
  re-issue hint — previously the tool-flag path surfaced as
  `status=pending_confirmation): no detail`. That hint is now true for both
  gates: `ToolCall(require_confirmation=False)` also clears the
  `SecurityPolicy` `require_confirmation` pattern gate, via a new
  `check_permission(confirmation_approved=…)` keyword. Previously a
  pattern-gated tool had no per-call approval mechanism at all.
- **`unsupported_strict_paths()`** (`adapters.tool_spec.schema_utils`)
  reports the locations in a schema that OpenAI strict mode cannot express.

### Changed

- **Approval clears the confirmation gate and nothing else.** Every denial
  check — rate limit, allowed domains — runs whether or not a call is
  approved, on both the executor's `require_confirmation=False` path and the
  Agent Framework middleware's native approval replay. A `SecurityPolicy`
  replacement that predates the `confirmation_approved` keyword stays closed
  on approval rather than being bypassed.
- **A confirmation prompt no longer costs rate-limit budget.** A call that
  comes back `PENDING_CONFIRMATION` never executed, and the approved replay
  that follows is the same logical call, so only the replay is counted. Both
  the `SecurityPolicy` window and the executor's `RateLimiter` previously
  charged both, which at a limit of 1 left confirmation-gated tools
  permanently unexecutable. Denied calls still consume quota.
- **Framework args models reject undeclared arguments.** The generated
  CrewAI/LlamaIndex model takes its `extra` configuration from the schema's
  own `additionalProperties` instead of Pydantic's default `extra="ignore"`,
  so a misspelled or hallucinated argument surfaces as an error rather than
  being silently dropped inside the framework — matching what the executor
  does with the same key. A *typed* `additionalProperties` also constrains
  the extras' value type, so the framework won't accept a string where the
  schema demands an integer.
- **Asking for OpenAI strict mode on a schema it cannot express no longer
  breaks the request.** Strict mode has no representation for an object with
  arbitrary keys (a `dict[str, int]` parameter, an untyped `dict`), and
  OpenAI rejects such a request outright rather than ignoring the shape.
  Affected tools are now emitted unmodified and *without* `strict: true`,
  with a warning naming the tool and parameter, so one tool stays
  unconstrained instead of the whole request failing. Forcing
  `additionalProperties: false` onto such an object is deliberately not done
  — it would produce an object accepting no keys and silently discard the
  parameter's data.

### Fixed

- **A non-finite float default produced a schema that isn't valid JSON.** A
  parameter defaulting to `float("nan")` or `float("inf")` had that value
  embedded in the schema's `default`, and `json.dumps` emits the bare tokens
  `NaN`/`Infinity` for them — so a provider parsing strict JSON rejects the
  request. Non-finite floats are no longer treated as JSON-safe, at the top
  level and inside nested container defaults, so the default is simply
  omitted.
- **`const` was not enforced during argument validation.** Pydantic emits
  `const` rather than `enum` for a single-value `Literal`, so it appears
  throughout the nested model and `TypedDict` schemas introspection now
  inlines — and validation checked only `type` and `enum`, letting any value
  through. `const` equality is checked alongside `enum` membership now.
- **A `None`-annotated parameter advertised a string.** `def f(x: None)`
  resolves to `NoneType`, which isn't in the scalar map and fell through to
  the string fallback, so the schema demanded a string for a parameter that
  admits only `null`. Now maps to `{"type": "null"}`.
- **An optional `const` couldn't express omission under strict mode.**
  `const` is an independent constraint that no `type` widening satisfies —
  a single-value `Literal` (`{"type": "string", "const": "fixed"}`) still
  forbade `null` after widening, and strict mode makes every property
  required, so the model had to send the constant rather than let the
  handler's default apply. Such a property is now wrapped as
  `anyOf: [<original>, {"type": "null"}]`, keeping the constant intact.
- **`prefixItems` was never validated.** Pydantic emits it for a
  heterogeneous `tuple[int, str]`, so it arrives inside the nested schemas
  now inlined — but the array branch only looked at `items`, so
  `["bad", 42]` passed validation. Each position is checked against its own
  entry, with `items` covering the positions past the prefix.
- **The framework bridge also skipped combinators when a `type` was
  present**, the same gate as the executor below. A constraint-only branch
  needs the parent's type pushed into it to mean anything, so
  `{"type": "integer", "anyOf": [{"minimum": 10}, {"maximum": 0}]}` becomes
  a union of two constrained ints rather than a bare `int`.
- **Combinators were only enforced when the schema had no `type`.** JSON
  Schema applies `anyOf`/`oneOf`/`allOf` independently of `type`, so
  `{"type": "integer", "allOf": [{"minimum": 1}]}` — a shape merged and
  imported schemas do produce — accepted `0`. They now run regardless.
- **A `TypedDict` subclass lost its inherited required keys.** The
  rebuild used for Python < 3.12 replayed a single `total=` flag over the
  merged annotations, so `class Child(Base, total=False)` made *Base*'s
  required keys optional too. Requiredness is now rebuilt per key from
  `__required_keys__`.
- **A malformed confirmation probe consumed no quota.** A call whose
  arguments fail validation is terminal even on a confirmation-gated tool —
  it returns a `ValidationError`, never a pending prompt — but it was
  taking the probe's rate-limit exemption, so malformed calls were
  unlimited. Arguments are now validated before that decision is made; the
  result is still returned after the policy check, so a denial outranks a
  validation error exactly as before.
- **Constraints didn't reach recursive framework annotations.**
  `list[Annotated[int, Field(gt=0)]]` carries `exclusiveMinimum` on the
  *item* schema, but constraints were only applied to the outer property,
  so array items, typed `additionalProperties` values, and combinator
  branches were all unconstrained in the generated model.
- **Strict-mode nulls were only normalized at the top level.** Widening
  applies to *every* object in the schema, so an optional *nested* property
  also comes back as an explicit `null` meaning "not provided" — but
  normalization looked only at top-level arguments, so
  `{"payload": {"nickname": null}}` survived and validation then rejected it
  against the canonical non-null schema. Normalization now recurses through
  nested objects and array items. A *required* nested `null` is still
  preserved, so its error stays accurate.
- **The framework bridge lagged the executor on four schema keywords.**
  Every one of them is a rule the executor enforces at dispatch, so the
  CrewAI/LlamaIndex args model advertised and accepted calls the engine then
  refused: `prefixItems` yielded a bare `list` (each position is now checked
  against its own entry, with `items` covering the tail, and the value stays
  a `list` because the executor's validator requires a JSON array);
  `uniqueItems` — which Gantry's own introspection emits for every `set` and
  `frozenset` parameter — wasn't enforced at all; and a float `const`
  (`{"type": "number", "const": 0.5}`) fell through to an unconstrained
  `float`, though the adjacent `enum` path already admitted floats.
- **Normalization stopped at a combinator node.** An optional nested model is
  `{"anyOf": [{object…}, {"type": "null"}]}` — what Pydantic emits for
  `Payload | None` — and normalization read only a node's own `properties`,
  so every value beneath such a property kept its strict-mode nulls and
  failed validation. It now resolves the branch that declares the shape, for
  nested objects and array items alike, and leaves genuinely ambiguous unions
  (two branches declaring `properties`, with no single `required` list to
  decide against) untouched. Positions typed by `prefixItems` are
  normalized too: strict mode widens the optional properties of a positional
  *object* exactly as it does anywhere else, so a `tuple[Payload, int]`
  parameter kept its nested nulls while only `items` was consulted.
- **A meaningfully-nullable parameter forbade `null`.** `T | None` maps to
  `T`'s schema, which is right when the default is `None` — "omitted" and
  "null" mean the same thing there, and a bare type is what provider dialects
  handle best. Two cases break that equivalence and now spell `null` out:
  a parameter with *no* default (`def f(x: int | None)`) is required, so
  omission cannot express `None` and the tool was uncallable with the value
  its own annotation names; and one with a non-`None` default
  (`x: int | None = 5`) treats an explicit null as a distinct choice, which
  was being dropped as "not provided" and handing the handler `5`. An
  optional `Literal` gains `None` in its `enum` too, since widening the type
  alone would buy nothing. `x: int | None = None` — the common case — is
  unchanged.
- **`patternProperties` was never validated.** Pydantic emits it for a mapping
  with constrained keys, and the object branch ignored it entirely: a matching
  key's value went unchecked, and with `additionalProperties: false` *every*
  key was rejected because none counted as declared. Matched keys are now
  validated against their pattern's schema and treated as declared by both
  paths. An uncompilable pattern fails open with a warning, as `pattern` does.
- **A tuple-valued `Enum` reached its handler as a raw array.** Canonicalizing
  those members to JSON arrays (so the schema matches what a provider sends)
  left Pydantic unable to match `[0, 0]` back to the member value `(0, 0)`, so
  reconstruction fell through and the handler got a `list`. Members are now
  recovered by JSON identity — the same equality the executor's `enum` check
  uses — applied to the whole annotation so it reaches `list[Point]` and
  `Point | None` too. A value matching no member is still rejected.
- **A sibling `type` didn't govern combinator nullability.** JSON Schema
  applies a property's own `type` alongside its combinator, so
  `{"type": "string", "anyOf": [{"type": "null"}, {}]}` does *not* admit null
  — but the branches were read in isolation, so a strict-mode placeholder was
  preserved for canonical validation to reject.
- **A container of typed values still reached its handler as raw JSON.**
  `list[Payload]` has origin `list` and isn't a bare class, so it fell through
  every check in the reconstruction test and `def f(items: list[Payload])`
  received a list of dicts — the same failure as below, one container level up.
  Parameterized generics now recurse into their members.
- **`pydantic_model_from_schema` could return something that isn't a model.**
  A *top-level* schema declaring `patternProperties` — reachable from an
  imported or hand-written `ToolDefinition` — returned an `Annotated` alias
  rather than a `type[BaseModel]`, which CrewAI's `args_schema` field rejects
  at tool construction instead of falling back. The pattern check is attached
  as a model validator now, so a real class comes back either way.
- **`$ref` inlining was bounded by depth but not by node count.** A model
  recursing on more than one field per level doubles the expansion each time,
  so the depth-16 cap permitted ~2^16 nodes: an ordinary binary-tree model
  produced a **15.8 MB** schema in 3.4 s. A companion budget on total
  expansions brings that to 62 KB in 61 ms.
- **A typed parameter reached its handler as raw JSON.** Once the schema
  advertises a nested Pydantic model, a dataclass, a `set`/`tuple` or a
  `datetime`/`UUID`/`Enum`, a provider sends that type's JSON form — and the
  executor forwarded it unchanged, so `def f(p: Payload): return p.x` failed
  with `'dict' object has no attribute 'x'` on *every* schema-valid call, and
  a `set[str]` parameter received a `list`. Arguments are now rebuilt into the
  types the signature names, immediately before dispatch. Only parameters
  whose declared type genuinely differs from its JSON form are touched —
  scalars, `list`, `dict`, `TypedDict` and `Any` already arrive as themselves
  and are passed through byte-for-byte — and a conversion failure passes the
  original value through, since validation has already run against the
  canonical schema. This was carried as a known limitation through most of
  this release's development and is now closed.
- **`schema_declares_null` ignored an `enum` that forbids null.** An optional
  `Literal` that strict mode pre-widened arrives as
  `{"type": ["string", "null"], "enum": ["fast", "slow"]}` — nullable by its
  type list, but not by its enum. Both the executor's normalization and
  `ToolSpec.ainvoke` therefore preserved a strict-mode placeholder the
  canonical schema then rejected, instead of dropping it so the handler's
  default applies. `const` is honoured the same way.
- **`patternProperties` went unenforced beside declared `properties`.** That
  shape merely allowed every extra, so a pattern key's value went unchecked
  and a key matching no pattern passed a closed object — both of which the
  executor rejects. The pattern validator is now attached to the generated
  model too, with declared properties exempt from the "matches no pattern"
  check.
- **`ToolSpec.ainvoke` dropped a `null` the schema declares.** It strips
  `None` for optional parameters because frameworks materialize every unset
  optional field that way — but once `x: int | None = 5` advertises
  `["integer", "null"]`, an explicit null is a distinct choice, and dropping it
  handed the handler `5` where the caller asked for `None`. The executor's own
  normalization already made that distinction; both paths now share one
  `schema_declares_null` predicate so they cannot disagree.
- **`patternProperties` was ignored in framework args models, in both
  directions.** An open mapping became an unrestricted `dict` that accepted
  values the executor rejects; worse, one with `additionalProperties: false`
  built a model permitting *no* keys, so a valid matching key was rejected and
  the call was impossible. Matching keys are now validated against their
  pattern's schema, and a key matching none is rejected only when the object is
  closed.
- **A composite `enum` lost its constraint in framework args models.** A
  tuple-valued `Enum` canonicalizes to `[[0, 0], [1, 1]]`; those aren't valid
  `Literal` members and such an enum has no inferred `type`, so the annotation
  fell through to an unconstrained `Any`. Membership is now checked directly,
  by JSON identity, the same way the executor checks it.
- **draft-04's boolean exclusivity was ignored.** Modern JSON Schema gives
  `exclusiveMinimum` a *number*; draft-04 — which is what OpenAPI 3.0 emits,
  and OpenAPI/MCP import is a supported way to register a tool — gives it a
  *boolean* that promotes `minimum` to an exclusive bound. Only the modern
  form was read, so `{"minimum": 5, "exclusiveMinimum": true}` accepted `5`.
  Both dialects now resolve through one shared helper, so a bound is read the
  same way at dispatch and in the framework args model.
- **A parent `type` was dropped when its combinator branches declared their
  own.** JSON Schema applies both, and the executor does — but the framework
  args model inherited the parent type only into *typeless* branches, so
  `{"type": "integer", "anyOf": [{"type": "number"}, {"type": "string"}]}`
  became `float | str` and admitted values the engine rejects. The union is
  now intersected with the parent type, which is a no-op for branches that
  already inherited it. An integer still satisfies a `number` parent, and a
  nullable parent still admits `null`.
- **A nullable parent type didn't reach its combinator branches.** The
  framework args model inherited a parent's type into typeless branches only
  when it was a bare string, so `{"type": ["integer", "null"], "anyOf":
  [{"minimum": 10}, {"maximum": 0}]}` — a shape imported schemas produce —
  left both branches as an unconstrained `Any` and accepted values the
  executor rejects. A genuine multi-type list still inherits nothing, since no
  single type applies to every branch.
- **One `execute()` call could emit two telemetry events.** The validation
  result is computed before the security policy runs, so the rate-limit
  exemption decision is accurate — but it does not always win: a denial
  outranks it, and a pending confirmation is discarded in favour of it.
  Recording where it was built therefore reported an outcome that was never
  returned, inflating call counts and status metrics. Each result is now
  recorded at the point it is returned, so exactly one event is emitted per
  call.
- **Malformed calls to a policy-gated tool consumed no quota.** Deferring the
  rate-limit charge past the confirmation gate is right only when an approved
  replay can follow. Since a call whose arguments fail validation is terminal,
  a match on a `SecurityPolicy.require_confirmation` pattern meant nothing was
  ever counted and such calls were unlimited — the same exemption abuse the
  tool-flag gate was fixed for, reachable through the pattern gate instead.
  The policy is now told whether the arguments validated. A genuinely pending,
  valid probe keeps its exemption, so its approved replay still fits the
  window.
- **LlamaIndex forwarded the caller's raw values, not the validated ones.**
  It checks a call against `fn_schema` but passes the original arguments on, so
  a model answering `"1"` for an `integer` parameter passed the tool's own
  validation and was then rejected by the executor, which holds the caller to
  the advertised schema. CrewAI forwards the validated values and never had
  this gap. The adapter now runs the same args model before dispatch, so what
  the engine sees is what the framework approved. Pinned by a test against the
  real package — the forwarding difference is the whole finding, and a fake
  would only pin the assumption.
- **A `None`-annotated parameter was advertised as a string in signatures.**
  `python_signature`'s type mapping falls back to `str` for anything it doesn't
  recognize, so `{"type": "null"}` — which introspection now emits — reached
  Semantic Kernel, AG2 and Google ADK's fallback path as a string, and the
  string the model produced was rejected by the executor. It maps to
  `NoneType` now.
- **`minProperties`/`maxProperties` were never enforced.** A Pydantic `dict`
  field constrained with `Field(min_length=1)` emits them, so they arrive
  inside the inlined mapping schemas — but the constraint check had branches
  only for numbers, strings and arrays, so an empty or oversized mapping
  reached the handler. Enforced in the executor and mirrored in the framework
  args model.
- **`enum` and `const` compared with Python equality.** `True == 1` in Python,
  so a boolean satisfied a numeric `Literal[1, 1.5]` — which emits an `enum`
  with no single `type`, leaving membership the only constraint — and reached
  the handler. Both now compare by JSON identity, the same helper the
  `uniqueItems` check uses, in the executor and the framework args model
  alike. A boolean enum still accepts booleans; a string enum keeps a bare
  `Literal` with no guard attached.
- **A composite `Enum` value was stored as a Python tuple.** A member valued
  `(0, 0)` is JSON-safe, but a tuple *serializes* to an array — so the
  canonical schema held a value no provider would ever send back, and the
  executor compared the returned `[0, 0]` against the stored `(0, 0)` and
  rejected every valid call. Enum members and defaults are now canonicalized,
  so the stored schema is byte-identical to the document the provider sees.
- **`allOf` was silently ignored in framework args models.** An `allOf`-only
  property became a bare `Any` that accepted values the executor rejects, and
  a typed one kept only its bare type. There is no faithful Python annotation
  for an intersection, so each branch is now checked against the caller's raw
  value. A type declared by any branch is pushed into the branches that
  declare none — `{"allOf": [{"type": "integer", "minimum": 1}, {"maximum":
  3}]}` enforces both bounds rather than leaving the second unconstrained.
- **An `allOf` beside a `type` didn't become nullable under strict mode.**
  `allOf` intersects, so *every* branch must admit null; widening only the
  outer `type` left `{"type": "string", "allOf": [{"const": "fixed"}]}`
  required with a const branch that still rejects it. It is wrapped whole now,
  as `oneOf` already was.
- **A schema carrying both `anyOf` and `oneOf` lost the `oneOf`.** They are
  independent assertions a value must satisfy together, but the framework
  args-model translation returned on whichever it found first — so `oneOf`'s
  exactly-one rule was never applied and the model accepted a value the
  executor then rejected. Both are honoured now: the union comes from `anyOf`,
  and `oneOf` contributes its exclusivity check on top.
- **A malformed call to a policy-gated tool reported "pending confirmation".**
  The invariant that a call whose arguments don't match the schema is terminal
  held for the tool's own `requires_confirmation` flag but not for
  `SecurityPolicy`'s `require_confirmation` *pattern* gate, whose result was
  returned before the validation result was consulted — putting a schema
  violation in front of a human to approve, and failing it anyway once
  approved. Validation now wins over a *pending* policy result; a *denial*
  still outranks validation, so nothing leaks about a tool the caller may not
  invoke at all.
- **`uniqueItems` conflated booleans with numbers.** Python says `True == 1`,
  but JSON Schema compares types before values, so `[1, true]` is two distinct
  items — it was being rejected as a duplicate at dispatch, and by the
  framework args model too. Both now key items by JSON identity via one shared
  `json_identity_key` helper, so they cannot drift; mathematically equal
  numbers (`1` and `1.0`) stay a duplicate, as the spec requires, and the
  keys being hashable makes the check linear rather than quadratic.
- **A nullable `type` list was dropped below a model field.** The framework
  args model recovered `null` for a top-level field via `_is_nullable`, but an
  array item, an `additionalProperties` value and a combinator branch never
  pass through there — so `{"type": "array", "items": {"type": ["string",
  "null"]}}` became `list[str]` and rejected a schema-valid `[null]` before
  Gantry could execute it. An `enum` that excludes `null` still wins, exactly
  as at the top level.
- **Signature-derived schemas lost every `enum`.** `python_signature` backs
  the frameworks that rebuild their own LLM schema from the callable rather
  than from `parameters_schema` — Semantic Kernel, AG2, and Google ADK's
  fallback path — and its annotations read only `type`/`items`. A
  `Literal["fast", "slow"]` parameter therefore reached the model as
  unconstrained `str` on exactly those frameworks, undoing this release's own
  fidelity work for them. `enum`/`const` now become `Literal`, array items are
  annotated recursively (`list[Literal["a", "b"]]`), and a typed mapping keeps
  its value type (`dict[str, int]`). An object with declared `properties` still
  degrades to a bare `dict` — see "Known limitation".
- **Anthropic strict mode had no safety gate.** `to_provider_schema(strict=True)`
  set `additionalProperties: false` on the root unconditionally, so a schema
  that *needs* arbitrary keys — a typed mapping, a `**kwargs` handler — was
  silently closed to accepting none, discarding the parameter's data rather
  than merely leaving it unconstrained. It now uses the same
  `unsupported_strict_paths()` gate as the OpenAI adapters: such a tool is
  emitted without `strict: true`, with a warning naming the offending path.
- **Schema-depth truncation is no longer silent.** `$ref` inlining stops at
  depth 16 and the args-model builder at depth 8 — necessary against
  self-referential schemas, but a legitimately deep acyclic one lost fidelity
  with no signal. Both now log at debug level, matching the multi-member union
  collapse.
- **`SecurityPolicy` signature inspection is cached.** `execute()` asks
  whether a policy's `check_permission` accepts each optional keyword on every
  call, and `inspect.signature` is not cheap. The answer is a property of the
  bound method, so it is now memoized per callable.
- **A multi-member union dropped its members silently.** A genuine `int | str`
  keeps only the first member — deliberate, since most provider dialects reject
  union-typed parameters — but nothing said so. It now logs at debug level, and
  the behaviour is pinned by a test rather than left as an accident. `T | None`
  loses nothing and stays silent.
- **Strict-mode fallback warnings named the wrong dialect.** `MistralAdapter`
  and `GroqAdapter` inherit the OpenAI path, which passed a hardcoded
  `"OpenAI"`, so their warnings pointed at the wrong provider. Each adapter
  now reports its own `dialect_name`.
- **An uncompilable `pattern` was silently unenforced.** A schema declaring an
  ECMA-262-only construct (`\p{L}`) fails Python's `re`, and validation
  correctly fails open rather than rejecting every value — but gave the schema's
  author no signal the constraint was dead. It now logs a warning.
- **A null branch appended inside `anyOf`/`oneOf` didn't make the property
  nullable.** A sibling assertion applies independently of a combinator, so
  `{"type": "integer", "anyOf": [{"minimum": 10}, {"maximum": 0}]}` kept
  failing its untouched `type: integer` on `null` — and strict mode makes the
  property required, so no value satisfied both. Such a schema is now wrapped
  whole as `anyOf: [<original>, {"type": "null"}]`. `oneOf` is wrapped
  unconditionally: it demands *exactly* one match, and `null` passes most
  constraint-only branches vacuously, so an appended branch would make it
  match several. A bare `anyOf` still gains a flat null branch.
- **An empty combinator branch was dropped from framework args models.**
  `{}` is the always-valid JSON Schema, not an absent branch — so
  `anyOf: [{}, {"type": "integer"}]`, which admits every value, generated a
  model that rejected strings, and `oneOf: [{}, {"type": "integer"}]` accepted
  an integer that in fact matches both branches. The validator already counted
  the empty branch as always-matching; the bridge now does too.
- **A pre-widened nullable enum still forbade `null` under strict mode.**
  `_make_nullable` returned early when `null` was already in a property's
  `type` list, so the enum was never widened alongside it: a schema arriving
  as `{"type": ["string", "null"], "enum": ["fast", "slow"]}` looked nullable
  but wasn't, and strict mode then made the property required with no value
  satisfying both constraints.
- **A typed `additionalProperties` alongside declared properties wasn't
  detected as strict-incompatible.** `unsupported_strict_paths()` returned
  early once an object declared any properties, so an object with both
  declared properties and a schema-valued `additionalProperties` was passed
  to strict mode, which forced `additionalProperties: false` and silently
  dropped the typed extras. A `true`/`{}` value alongside properties is
  still treated as the intentional `**kwargs` narrowing.
- **Float enum members were dropped in framework args models.** A
  float-valued `Enum`/`Literal` parameter — which introspection emits —
  failed the `Literal` member guard, so the whole enum fell through to an
  unconstrained `float` that accepted non-members.
- **A fractional `multipleOf` rejected valid numbers.** The check used `%`,
  and binary floats make `0.3 % 0.1` ≈ `0.0999…`, so a property declared
  `multipleOf: 0.1` refused `0.3`. Compared as `Decimal`s via `str`, which
  gives the value as it was written.
- **`oneOf` accepted values matching *no* branch.** The generated model's
  exclusivity check rejected only multiple matches, leaving zero-match values
  to the union — whose coercion then accepted them (`"1"` matches neither a
  strict `number` nor a strict `integer`, but coerces into one). It now
  requires exactly one match in both directions.
- **Constraint keywords never reached the generated framework fields.** The
  executor enforces numeric bounds, string length/pattern and array length,
  but the CrewAI/LlamaIndex args model advertised and accepted values
  violating them — rejected only later at dispatch. They're now folded into
  the generated annotation, on the inner type so a constrained field can
  still be optional and carry its default.
- **The OpenAI Agents adapter didn't use the strict-mode safety gate.**
  `FunctionTool.strict_json_schema` defaults to `True`, and the SDK then runs
  its own `ensure_strict_json_schema`, which raises `UserError` on an object
  with arbitrary keys. `strict_json_schema()` deliberately leaves such a
  schema alone, so exporting a tool with a `dict[str, int]` parameter through
  `OpenAIAgentsAdapter` handed the SDK exactly what it refuses. The adapter
  now consults `unsupported_strict_paths()` like the provider adapters do and
  builds the tool with `strict_json_schema=False`, with a warning.
- **An empty combinator branch was ignored.** `{}` validates every value, so
  `{"anyOf": [{}, {"type": "integer"}]}` means "anything" — but the empty
  branch was filtered out, turning it into an integer-only constraint that
  rejected valid payloads.
- **Two more schema-aliasing sites.** The Agent Framework bridge passed
  `parameters_schema` straight into `agent_framework.tool(schema=...)`, and
  the Google ADK declaration used a shallow `dict()` that left every nested
  subschema shared. Both now deep-copy, matching the provider adapters.
- **A broken `agent-framework` install could crash
  `disable_af_instrumentation()`.** Its import guard caught only
  `ImportError`, so a version mismatch raising anything else propagated to
  the caller. Other exceptions now degrade to a warning — at warning level
  rather than debug, since a silent broad `except` is what hid this helper's
  original bug.
- **Framework args models disagreed with the executor in three more shapes.**
  A closed object with no `properties` key at all fell through to a bare
  `dict`; a `type` list with several real members collapsed to the first,
  rejecting values the executor accepts; and a nullable-typed property whose
  `enum` omits `null` was widened to accept `None`, which the executor then
  correctly rejected at dispatch.
- **Constraint keywords were never enforced during argument validation.**
  Numeric bounds (`minimum`/`maximum`/their exclusive variants,
  `multipleOf`), string `minLength`/`maxLength`/`pattern`, and array
  `minItems`/`maxItems`/`uniqueItems` were all ignored, so a value violating
  them reached the handler. These arrive from any Pydantic-constrained field
  (`Annotated[int, Field(gt=0)]` becomes `exclusiveMinimum: 0`) inside the
  nested schemas now inlined — and `uniqueItems` is emitted by Gantry itself
  for a `set` parameter. Booleans are exempt from numeric bounds, since
  `bool` is an `int` subclass.
- **An `allOf` was treated as nullable when only one branch admitted null.**
  `allOf` intersects its branches, so the combined schema admits `null` only
  when *every* branch does — `[{"type": ["string","null"]}, {"type":
  "string"}]` does not. Treating it as nullable preserved a synthetic null
  the schema forbids, so validation rejected the call instead of letting the
  handler's default apply.
- **An enum listing `null` was dropped entirely in framework args models.**
  `{"enum": ["auto", null]}` is how a nullable choice is expressed, but the
  bridge required every enum member to be a `str`/`int`/`bool`, so the whole
  enum fell through to an unconstrained `Any`. `None` is a valid `Literal`
  member and is now kept.
- **`const` and `oneOf` exclusivity were lost in framework args models.**
  The CrewAI/LlamaIndex bridge ignored `const`, advertising an
  unconstrained scalar for a field the schema pins to one value; it now
  becomes a single-value `Literal`, mirroring how `enum` is handled. And a
  `oneOf` translated to a Python union carries *`anyOf`* semantics — `1`
  satisfies both a `number` and an `integer` branch, which `oneOf` forbids
  — so the union now also carries an "exactly one branch matches" check.
  `anyOf` is unaffected.
- **A null-only property widened to `Any` in framework args models.** A
  property typed `{"type": "null"}` permits only `null`, but the
  CrewAI/LlamaIndex bridge fell through to an unconstrained `Any` that
  accepted strings and numbers the canonical schema forbids.
- **A tool gated by `requires_confirmation=True` could be made unexecutable
  by the rate limit.** That flag is enforced by the executor, not by
  `SecurityPolicy`, so the policy had no way to know a call would stop at
  it: with no matching `require_confirmation` *pattern*, the policy recorded
  the probe against its window, and the approved replay was then denied for
  the rest of the minute. The executor now tells the policy when a call will
  stop at a gate it can't see, via a `pending_confirmation` keyword — every
  check still runs (a probe that would be denied says so before a human is
  asked), only the recording is deferred to the replay that actually
  executes.
- **`oneOf` was validated with `anyOf` semantics.** `oneOf` requires
  *exactly* one matching branch, so a value matching several — `1` against
  `[{"type": "number"}, {"type": "integer"}]` — violates the schema but was
  accepted. Matching branches are counted now; `anyOf` is unaffected.
- **`Sequence`/`Iterable`/`Set` parameters were advertised as scalars.**
  `typing.get_origin(Sequence[int])` returns the `collections.abc` class —
  neither the `typing` alias nor a `list` subclass — so the container branch
  missed it and fell through to a "use the first type argument" fallback,
  emitting `{"type": "integer"}` for a parameter that takes a list. The
  executor then rejected every valid payload. All the `collections.abc`
  container origins are matched now, with mappings checked first (a Mapping
  is also a Collection, so the reverse order would classify `dict[str, int]`
  as an array).
- **An optional enum parameter could not express "not provided" in strict
  mode.** Widening a property's `type` to admit `null` left its `enum`
  untouched, and `enum` is an independent constraint — so a
  `Literal["fast", "slow"] | None = None` parameter advertised
  `type: ["string", "null"]` alongside `enum: ["fast", "slow"]`. Strict mode
  makes every property required, so the model's constrained grammar could
  not emit `null` and had to invent `"fast"` or `"slow"` rather than let the
  handler apply its `None` default. Widening now adds `null` to the enum too.
- **A combinator-typed field became `Any` in framework args models.**
  `{"anyOf": [{"type": "integer"}, {"type": "null"}]}` — what Pydantic emits
  for `int | None`, so it appears throughout the nested models now inlined —
  has no top-level `type`, so the CrewAI/LlamaIndex bridge widened it to an
  unconstrained `Any` that accepted values the executor rejects after
  dispatch. Supported `anyOf`/`oneOf` branches are translated recursively
  into a union instead.
- **A confirmation-gated A2A tool executed remotely without ever being
  gated.** The special-source dispatch branch returned before the
  confirmation check ran, so `requires_confirmation=True` on an A2A tool was
  silently ineffective — the remote agent was invoked and its side effects
  happened before anyone was asked. The check now runs before dispatch by
  any mechanism, making "pending confirmation means nothing ran" true for
  every tool source (and closing a path by which a caller could set
  `require_confirmation=True` to run A2A calls past both the per-minute and
  concurrency limits).
- **Argument validation understands the schemas Gantry itself emits.**
  Explicit `None` for a declared optional parameter is treated as omitted
  (models legitimately send `null` under the strict-mode widened schemas
  Gantry advertises, and several frameworks materialize unset optionals as
  `None`) — that workaround previously lived only in `ToolSpec.ainvoke`, so
  the `execute_tool_calls` provider path rejected its own schema's output.
  A property that explicitly declares `null` in its type keeps a
  caller-supplied `None` as the meaningful value it is — whether it declares
  `null` through its `type` or through an `anyOf`/`oneOf` branch, the shape
  Pydantic and OpenAPI emit for `str | None`. Undeclared keys are
  admitted when `additionalProperties` permits them (`true` or a subschema,
  including the empty schema `{}`, which JSON Schema treats as `true`) and
  refused when it is `false` or absent; a `dict[str, int]`-shaped schema
  validates every value against its subschema instead of accepting anything;
  a closed empty object (`{"properties": {}, "additionalProperties": false}`)
  rejects any payload rather than being treated as free-form; list-typed
  `type` (`["string", "null"]`) validates against any member; a schema that
  constrains a value purely through `anyOf`/`oneOf`/`allOf` with no `type`
  of its own — what Pydantic emits for `int | None`, including inside the
  nested models now inlined — has its branches enforced rather than being
  waved through; and `enum` membership is enforced as the independent
  constraint JSON Schema says it is, including for a nullable property whose
  enum does not itself list `null`.
- **Emitted provider schemas no longer alias the registered tool.** Every
  adapter pass-through path put `ToolDefinition.parameters_schema` itself
  into the returned payload, so a caller that augmented the payload
  corrupted the registered tool, every later conversion of it, and the
  executor's validation, which reads the same object. The Anthropic strict
  path's `{**schema}` spread left every nested property dict shared. All
  paths now deep-copy.
- **`disable_af_instrumentation()` never worked.** It imported
  `agent_framework.telemetry` — a module that has never existed in any AF
  release — and swallowed the `ImportError`, so the documented workaround
  for AF ≥1.6.0's concurrent-`asyncio.gather` ContextVar crash was a silent
  no-op in every install. It now calls the real switch,
  `agent_framework.observability.disable_instrumentation()` (verified on AF
  1.5.0 and 1.15.0), no longer requires the `agent-framework` *meta*-package
  metadata to be present, and only downgrades to debug-logging when there is
  genuinely nothing to disable.
- **AF approval middleware no longer discards its own verdict.** On a
  confirmation-gated tool it sets a `function_approval_request` Content on
  the context before terminating — the shape AF's native approval flow
  requires; a bare termination reached the model as a null function result
  with the reason lost. Replays carrying the human's decision are honoured.
  A policy *denial* returns an explicit "Permission denied by security
  policy: …" result instead of raising — AF converts middleware exceptions
  into an opaque `"Error: Function failed."`, which read as a tool crash and
  hid the reason.
- **AF bridge crash on out-of-order schemas.** `properties` order carries no
  required-first guarantee (MCP servers, OpenAPI imports), and an optional
  property listed before a required one made the synthesized
  `inspect.Signature` raise `ValueError: non-default argument follows
  default argument` — killing the whole run inside `before_run`. Parameters
  are now ordered required-first.
- **Qualified `required=[…]` pins resolve at request time.**
  `GantryContextProvider` accepted `"namespace.name"` pins at construction
  but resolved them per-round with bare-name lookups only, so a qualified
  pin was warned-and-skipped on every round — inverting the guarantee.
  Request-time lookup now mirrors the executor's namespace-aware resolution.
- **`AutoGenAdapter.register`'s install hint pointed at the wrong package.**
  `pip install pyautogen` now delivers a Microsoft autogen-agentchat shim
  (≥0.10) with no `autogen.register_function`, and AG2 1.x renamed its
  import to `ag2` with a new agent API. The hint and docs now name the
  classic line that actually provides the API (`pip install "ag2[openai]<1"`,
  verified end-to-end against AG2 0.14).
- **`delete_tool` deletes everywhere.** It only removed the vector-store
  entry, so a deleted tool kept appearing in `list_tools_sync()`, kept
  resolving as a `required=` pin, and kept executing. It now also purges the
  registry, the handler map, the facade's own `_tool_handlers` (which
  `tool_count` reads), and any pending-sync entry.
- **Non-JSON `Literal`/`Enum` values degrade instead of producing an invalid
  schema.** `Literal` admits `bytes` and an `Enum` member can wrap an
  arbitrary object; `_enum_schema` falls back to a plain string schema (no
  `enum`) when any value isn't JSON-representable.
- `SecurityPolicy.check_permission`'s `arguments` parameter is typed
  `dict[str, Any]` (was `dict[str, str]`) to match what callers actually
  pass — nested dicts/lists and non-string values, which
  `_extract_all_strings` already handles.
- **A heterogeneous tuple advertised an untyped array.** `tuple[int, str]`
  has no single item type, so the emitted schema was a bare
  `{"type": "array"}` — validation accepted `["bad", 1]`, reconstruction then
  couldn't build the tuple, and the fallback handed the handler the raw list,
  the exact failure reconstruction exists to prevent. It now emits
  `prefixItems` plus `minItems`/`maxItems`, both of which the executor and
  the framework bridge already enforce. A variadic `tuple[int, ...]` keeps
  its homogeneous `items`.
- **`oneOf` was read as `anyOf` when deciding nullability.** `oneOf` means
  *exactly* one branch matches, so `{"oneOf": [{"type": "null"}, {}]}` makes
  `null` match twice and is therefore invalid — but the shared any-branch
  reading called it nullable and preserved a strict-mode placeholder that
  validation immediately rejected. Branches admitting `null` are now counted,
  and exactly one is required; `anyOf` is unchanged.
- **A declared property escaped its matching `patternProperties` schema.**
  JSON Schema requires a key to satisfy its `properties` schema *and* every
  matching pattern schema, but the framework bridge skipped declared keys
  outright, so `n_fixed: {"type": "integer"}` beside
  `{"^n_": {"minimum": 5}}` dropped the minimum. Declared keys are now exempt
  only from the closed-object "matches no pattern" check. Fixing it exposed
  the underlying cause: a constraint-only schema with no `type` — what a
  pattern branch looks like — lost its constraint entirely, because every
  keyword-family gate reads `type` first. The executor's JSON-constraint
  checker moved to `agent_gantry.schema.base` as `check_json_constraints` and
  the bridge now applies it to such schemas, so the two agree by
  construction rather than by parallel maintenance.
- **A composite `const` advertised an unconstrained container.**
  `{"type": "array", "const": [1, 2]}` can't be a `Literal` member, so it
  fell through to a plain `list` that accepted anything while the executor
  enforced the constant. It is now checked by JSON identity, exactly as a
  composite `enum` already was.
- **An optional member of a container lost its `null`.** The widening that
  re-admits `null` for `T | None` ran only on a top-level parameter, where
  omission already expresses "no value" — so `list[int | None]` was emitted as
  `{"type": "array", "items": {"type": "integer"}}` and validation rejected
  `[1, None, 2]` for a handler whose own annotation accepts it. An array item,
  a mapping value and a tuple position have no "omitted", so nullability is
  now threaded through the recursion. The top level is deliberately unchanged.
- **Top-level `patternProperties` was never validated.** The construct was
  handled inside a nested object but not at the top level of a tool schema, so
  an identical declaration one level up both rejected a schema-valid matching
  key as `Unknown parameter` and skipped the pattern's own constraint when the
  schema was open. Both paths now share one implementation.
- **A composite `enum` was advertised as a *string*.** A tuple-valued `Enum`
  emits `{"enum": [[0, 0], [1, 1]]}` with no `type`, since its members share
  no scalar kind — so with no `Literal` to build and no `type` to read, the
  signature path fell through to its `str` fallback and told Semantic Kernel,
  AG2 and ADK's fallback path that an array-valued parameter was a string. The
  members now name the container type when the schema declares none; an
  explicit `type` still outranks them.
- **The framework bridge read an absent `additionalProperties` as open.** For
  an object that declares properties, absent means closed — Gantry's own
  documented default, and what the executor enforces — so the generated
  CrewAI/LlamaIndex model accepted a key the engine rejects at dispatch. The
  bridge now mirrors the executor's rule including its asymmetry: an object
  declaring *no* properties stays free-form, that being the shape a plain
  `dict` parameter emits.
- **A homogeneous fixed tuple advertised an array of any length.** The arity
  bounds rode along with `prefixItems`, which only a *heterogeneous* tuple
  needs — so `tuple[int, int]`, having one shared item type, took the `items`
  branch and lost them. `[1]` then validated, reconstruction failed, and the
  fallback handed the handler a raw list. Every fixed-length tuple pins its
  arity now; a variadic `tuple[int, ...]` still has none to pin.
- **`const` was skipped when the value was `null`.** A `type` list naming
  `null` returns early once the value is null, and that shortcut checked
  `enum` but not `const` — so `{"type": ["string", "null"], "const": "fixed"}`
  accepted `null`, which the constant independently forbids. Both keywords now
  survive the shortcut.
- **A constraint-only `oneOf` branch also matches `null`.** `{}` was not the
  only way for null to match twice: `{"minimum": 5}` declares no type, and
  numeric keywords assert nothing about null, so
  `{"oneOf": [{"type": "null"}, {"minimum": 5}]}` is ambiguous and therefore
  not nullable. Branches are counted with a new `null_validates_against`
  — what *matches*, which is what exclusivity means — while `anyOf` keeps
  asking what the author *declared*.
- **A mapping keyed by anything but `str` reached the handler unconverted.**
  JSON object keys are always strings, so `dict[int, str]` arrived as
  `{"1": "value"}` and every lookup or arithmetic on those keys failed.
  Reconstruction looked only at the *value* type; it now flags a non-string
  key annotation too.
- **A bare `set`/`frozenset`/`tuple`/`bytes` annotation was never rebuilt.**
  `typing.get_origin` is `None` for an unparameterized container, so those
  spellings missed the generic branch entirely — while introspection still
  advertised them as JSON arrays, leaving the handler a `list` (and a `str`
  for `bytes`). The parameterized forms were already covered.
- **A malformed `date-time`/`date`/`time`/`uuid` string was passed through as
  a string.** Validation read only the JSON *type*, so it accepted the value;
  reconstruction then failed and the fallback handed the raw `str` to a
  handler annotated `datetime` — reported as a **success**. Those four formats
  are now enforced with the same parser reconstruction uses, so the two agree
  by construction. Deliberately only those four: `format` is an annotation by
  default in JSON Schema, and enforcing `email`/`uri` on an imported schema
  that uses them loosely would reject calls that work.
- **The same formats reached the framework adapters as a bare `str`.**
  Semantic Kernel, AG2 and Google ADK's fallback path rebuild their provider
  schema from `python_signature`, so a `datetime` parameter was advertised to
  them as a free-form string and the model could answer with one the handler
  can't take. The CrewAI/LlamaIndex bridge had the same gap. Both now read one
  shared `RECONSTRUCTED_STRING_FORMATS` table.
- **A required nullable parameter was advertised as non-nullable.** Required
  means the value must be *present*, not that it must be non-null — but only
  *optional* properties were unioned with `None`, so `def f(x: int | None)`
  (now emitted as `{"type": ["integer", "null"]}` in `required`) reached
  Semantic Kernel and AG2 as a bare `int` and the model could not produce the
  null the executor accepts. Google ADK's fallback path is left alone, since
  it rejects union annotations outright.
- **A combinator-only property was advertised as a *string*.**
  `{"anyOf": [{"type": "integer"}, {"type": "null"}]}` is what Pydantic and
  OpenAPI emit for `int | None`, so it arrives from every imported schema and
  inlined nested model — and with no `type` to read, the signature path fell
  through to its `str` fallback. The non-null branch now supplies the type.
- **The framework bridge widened a required `const` to `None`.** Its private
  nullability check tested `enum` but not `const`, so
  `{"type": ["string", "null"], "const": "fixed"}` in `required` accepted
  `null` in the CrewAI/LlamaIndex model while the executor rejected it. That
  check was a weaker duplicate of the shared `schema_declares_null` and has
  been deleted in favour of it.
- **The framework bridge checks a string `format` instead of applying it.**
  Annotating a `date-time` field `datetime` made the generated model hand a
  `datetime` *object* back — CrewAI forwards its validated kwargs and
  LlamaIndex's `_coerced` dumps in Python mode — while the canonical schema
  still types the property as a JSON string, so the executor rejected every
  *valid* formatted call. The value now stays JSON-native across the dispatch
  boundary and only its shape is asserted. `ToolSpec.ainvoke` serializes such
  a value defensively too, for frameworks that parse the model's answer from
  the `python_signature` annotation before handing it back.
- **A failed argument reconstruction is terminal.** It used to pass the raw
  value through, reasoning that validation had already run against the
  canonical schema. That reasoning doesn't hold: the coercer exists precisely
  *because* the handler declares a type the JSON form isn't, so the raw value
  is the one thing it cannot take — a handler annotated `Payload` raised
  `AttributeError` deep inside the tool, or misbehaved silently. Reachable for
  invariants JSON Schema cannot express at all (a Pydantic `field_validator`,
  a mapping key that doesn't parse), which is exactly where validation cannot
  have ruled the value out first. Now a `ValidationError`, not retried.
- **One rule now decides whether an explicit `null` is kept:** keep it iff the
  executor would accept it. `schema_declares_null` had been asking a narrower
  question — did the author *declare* null meaningful, where a schema that
  merely failed to forbid it didn't count — and that reading needed a fresh
  patch for each new spelling while still getting three cases wrong:
  `{"enum": ["a", null]}` (what an optional `Literal["a", None]` emits, with
  no `type` at all) dropped an explicitly supplied `None` and handed the
  handler its default; `{"const": null}` and `{}` did the same; and a nullable
  `anyOf` was read as nullable while a sibling `allOf` forbade null. It
  delegates to `null_validates_against` now, which composes instead of
  laddering. Two tests changed with it — one had pinned the old asymmetry, and
  one had asserted that a generated model must accept a `null` its own
  `anyOf` forbids, which the executor has always rejected.
- **A rejected argument no longer degrades tool health.** An argument the
  handler's own type refuses is caller-supplied input, not a sick tool, but it
  was being recorded as a failure — so after five malformed calls the circuit
  breaker opened and the next *valid* call came back `CIRCUIT_OPEN`, letting a
  caller disable a healthy tool for everyone. The schema validation path has
  always left health alone; this now matches it.
- **`true` and `false` are schemas.** Draft-06 made a bare boolean a valid
  schema — `true` matches every value, `false` none — but combinator branches
  were filtered to dicts, so `{"anyOf": [true, {"type": "integer"}]}`
  (semantically "anything") rejected a string, a `false` branch silently
  stopped counting against `oneOf` exclusivity, and `allOf: [false]` permitted
  everything instead of nothing.
- **The `$ref` expansion budget bounded the wrong thing.** Its guard ran on
  entry to every node rather than at an expansion, so once spent it replaced
  *every* remaining value with `{}` — a `type` string, a `required` list — and
  a model wide enough to exhaust it emitted malformed metadata a provider
  rejects rather than merely unconstrained subschemas.
- **`additionalProperties` typed the wrong keys beside `patternProperties`.**
  JSON Schema applies it to exactly the keys matched by neither `properties`
  nor a pattern. The framework bridge passed its pattern validator only a
  boolean "is it closed" flag, leaving those keys typed by nothing, while
  typing *every* extra from the additional schema — so a pattern-matched
  `{"s_a": "ok"}` was checked against `additionalProperties: {"type":
  "integer"}` and rejected. Verified against the executor across all forty
  `properties` × `additionalProperties` × payload combinations.
- **A pattern-keyed object was published under OpenAI strict mode.** Strict
  mode can only describe an object whose full key set is written out in
  `properties`, and keys typed by regex are by definition not — so such an
  object is open however `additionalProperties` is set, `false` included.
  `unsupported_strict_paths()` read only `additionalProperties` and called it
  strict-safe, so the transform emitted it with the unsupported
  `patternProperties` keyword still attached and the provider rejected the
  tool declaration.
- **A validated pattern-property value was discarded.** A declared integer
  property has always coerced `"1"` to `1`; a pattern-matched key ran the same
  adapter but threw the result away, so the model accepted `{"n_x": "1"}` while
  keeping the string — which LlamaIndex's `model_dump()` then forwarded
  unchanged for the executor to reject against the integer schema. The
  converted value is written back now, into a new mapping rather than the
  caller's.
- **The framework bridge also treats `true`/`false` as schemas.** The executor
  learned this a commit earlier; the bridge still dropped boolean branches, so
  `{"anyOf": [true, {"type": "integer"}]}` was reduced to an integer field
  that rejected the schema-valid strings the engine accepts, and `allOf:
  [false]` was permissive where it should forbid everything. Both sides now
  agree across all twelve boolean-branch cases.
- **A bare collection ABC was advertised as a string.** `def f(m: Mapping)`
  has `get_origin() is None` and matched no concrete-class check, so it fell
  through to the string fallback — the tool asked for a string for a parameter
  the handler needs a mapping for, and the executor then rejected the
  correctly shaped object a caller sent. `Mapping`, `Sequence`, `Set` and
  friends are now classified like their concrete kin (`Mapping` checked first,
  since it is also a `Collection`), and a bare `Set` is rebuilt for the same
  reason `set` is. A bare `set` consequently carries `uniqueItems` now, as
  `set[str]` always has.
- **A parameterized generic took a different branch on Python 3.10.** There a
  parameterized builtin — `dict[str, int]` — *is* an instance of `type`, where
  on 3.11+ it is not, so 3.10 entered the direct-type branch and reached the
  abstract-base checks with a generic alias, where `issubclass` raises
  `TypeError: arg 1 must be a class`. `get_origin` now gates that branch, so
  every version routes a parameterized generic the same way.
- **A boolean schema is honoured wherever a schema may appear.** Draft-06
  booleans were handled only in combinator branches, but they are valid in
  every schema position — so `properties: {"disabled": false}` let an
  `AttributeError` escape `execute()` instead of returning a validation
  failure, `patternProperties: {"^blocked_": false}` accepted the keys it
  exists to forbid (and counted them declared, slipping them past a closed
  object), and `items: false` beside `prefixItems` — the standard spelling of
  a fixed-length tuple — accepted the extra elements. Both the executor's
  validator and the framework bridge now read them at their own single
  subschema funnel, so the two agree by construction.
- **A `null` can itself be a value worth rebuilding.** Reconstruction
  short-circuited on `None`, assuming a null is never worth converting — but
  an `Enum` with a `None`-valued member (`class Mode(Enum): UNSET = None`)
  emits `enum: [null]`, and a call supplying null reached the handler as raw
  `None` rather than `Mode.UNSET`. The adapter answers this correctly without
  a special case: an annotation admitting `None` returns it unchanged, and one
  that doesn't was already a mismatch between the schema and the handler's own
  type.
- **A property forbidden by its schema has no strict-mode representation.**
  Strict mode makes every property *required*, so a property whose schema is
  `false` — satisfiable by no value — was emitted as required and
  unsatisfiable at once: a schema with no valid instance, turning an otherwise
  callable tool into an uncallable one. It is reported as strict-unsupported
  now, so the tool falls back to non-strict where the property is simply
  omitted. Widening it to a null-only placeholder was the alternative and is
  worse: it would *permit* a null the schema forbids.
- **A boolean `items` schema types a plain array too.** The positional path
  routed booleans through the annotation builder, but a plain array with no
  `prefixItems` fell through to a bare `list` in the framework bridge and
  accepted the elements the executor rejects.
- **A `TypedDict`'s members are rebuilt, though the container isn't.** A
  `TypedDict` *is* a dict at runtime, so the container arrives as itself — but
  a member annotated `datetime`, `Enum`, `set` or a dataclass does not, and
  declining reconstruction for the whole thing installed no coercer at all, so
  `payload["at"].year` failed on a schema-valid call. Pydantic recognizes only
  `typing_extensions.TypedDict` before 3.12, so the coercer retries through
  the same rebuild the schema path already used — otherwise the fix would have
  been silently inert on exactly the versions that need it.
- **A property Pydantic cannot make a field from is declined explicitly.**
  `isidentifier()` accepts `_token`, but Pydantic reserves leading underscores
  for private attributes. It raises today, which already produced the
  documented fallback — but it *silently drops* such a name when one arrives
  as a keyword rather than through the field mapping, so the bridge no longer
  depends on which spelling Pydantic sees.
- **An over-quota call is refused before it buys schema validation.**
  Validating arguments ahead of the security policy (so the confirmation
  probe's rate-limit exemption could be decided accurately) meant *every*
  request ran the full recursive validator first — including one already over
  quota. Since the validator runs `re.search` against schema-supplied
  `pattern`/`patternProperties` on caller-controlled input, the limits stopped
  protecting the work they exist to protect. A read-only admission peek now
  runs first: it records no call, consumes no token and prunes no window, so
  the accounting is unchanged and `acquire`/`check_permission` remain the
  authority — it only short-circuits a call that is *certainly* over quota.
  Each limiter's own result shape is preserved, so a `SecurityPolicy` refusal
  still reports `PERMISSION_DENIED` rather than being relabelled.
- **`required` holds in an object declaring no properties.** A `required` name
  needs no matching `properties` entry, so
  `{"type": "object", "properties": {}, "required": ["token"]}` is valid — but
  the no-properties shortcut ran before the required loop and accepted `{}`
  outright.
- **Google ADK placeholders match the annotation they accompany.** Once a
  formatted string became a `datetime` and an enum a `Literal`, the synthetic
  default was still derived from the raw JSON type and stayed `""` —
  recreating the annotation/default mismatch `type_matched_defaults` exists to
  avoid, which ADK rejects during signature processing.
- **A nullable open mapping is still strict-unsupported.** `type` is a *list*
  whenever nullability is spelled into it — which introspection emits for a
  required `dict[str, int] | None` — and the strict-mode check matched only
  the scalar string `"object"`. A nullable open mapping therefore passed as
  strict-safe, and the provider rejected the whole tool request rather than
  that one parameter. A nullable *closed* object is unaffected: it remains
  perfectly representable.
- **A pattern-only object is strict-unsupported without a `type`.** JSON Schema
  applies an object's keywords whenever the instance *is* an object, so a
  property carrying only `patternProperties` — no `type`, no `properties` —
  constrains objects just as much as one spelling the type out. Gating the
  check on type-or-properties let that spelling past as strict-safe while its
  typed twin was flagged.
- **A required property forbidden by its schema is no longer widened to null.**
  A boolean property schema is replaced with `{}` internally so the later
  lookups work, and `{}` admits null under the unified nullability rule — so a
  required `false` property was unioned with `None` and accepted an explicit
  null the executor rejects outright. Nullability is now decided from the
  original schema.
- **LlamaIndex coercion no longer injects defaults into typed map values.**
  `model_dump()` recursively materializes every omitted optional field,
  including inside a typed map's values, where an optional child with no schema
  default becomes `None`. Executor normalization walks named `properties` but
  not schema-valued map entries, so that injected null survived to be rejected
  — turning a call the caller made correctly into an error. The dump now
  carries only what the caller supplied; the coercions that function exists for
  are unaffected.
- **A concrete sequence that isn't a `list` is rebuilt.** The reconstruction
  rule named `set`/`frozenset`/`tuple` outright, so any other concrete
  sequence fell through whenever its member type needed nothing itself:
  `collections.deque[int]` was advertised as a JSON array and reported as
  needing no rebuild, so the handler received a plain `list` and `popleft()`
  raised on a schema-valid call. The rule is now the one the docstring always
  claimed — rebuild whenever a JSON array is not already an instance of the
  declared container — which leaves `list`, `Sequence` and `Iterable`
  untouched as before.
- **`tuple[()]` advertised an unrestricted array.** The empty tuple has no
  type arguments, so the truthy check that gated the arity bounds skipped the
  one tuple that permits *no* items at all. The call then validated and failed
  at reconstruction, giving a dispatch error for a value the schema had said
  was acceptable. Both bounds are now pinned at zero.
- **Formatted values are restored to JSON at every depth.** The dispatch guard
  read only the parent schema's `format`, so it covered a top-level `datetime`
  parameter and nothing below it. `list[datetime]` and `dict[str, UUID]` are
  advertised on the signature that framework adapters introspect, so a
  framework hands back a *container* of real Python objects and every element
  was then rejected against the canonical formatted-string schema. The guard
  now recurses through `items`, `prefixItems`, `properties`,
  `patternProperties`, `additionalProperties` and combinator branches, still
  only where the schema declares one of those formats.
- **Object and array keywords apply without a declared `type`.** JSON Schema
  applies them whenever the *instance* is of that kind, but both the executor
  and the framework bridge gated their checks on `type` alone — so a property
  such as `{"properties": {"token": {"type": "string"}}, "required":
  ["token"]}`, which an MCP or OpenAPI import produces, governed nothing at
  all and `{}` was dispatched despite the missing key. Both now dispatch on
  what the value is, which leaves a schema asserting nothing unconstrained and
  still admits a string under object keywords that cannot apply to one.

### Performance

- **Generated framework models are memoized.** `pydantic_model_from_schema` is
  a pure function of its name and schema, but the live CrewAI and LlamaIndex
  adapters rebuild their tools on *every* retrieval — so the whole recursive
  `create_model` (plus its `TypeAdapter` constructions and `re.compile` calls)
  reran per query, per tool. Bounded and LRU.
- **The handler-coercer cache evicts instead of stopping.** Its hard cap left
  the first 512 handlers pinned for the life of the process and re-ran
  signature inspection on every call for every handler after them. It is an
  LRU now, keeping the targeted invalidation `functools.lru_cache` can't do.

### Known limitation

- A *multi-member* union parameter (`int | str`, as opposed to `T | None`)
  keeps only its first member in the emitted schema. Most provider dialects
  reject union-typed parameters outright, so a narrowed schema is more useful
  than one they refuse; the collapse is logged at debug level and documented
  on `build_parameters_schema`.
- `python_signature` annotates an object parameter with declared
  `properties` as a bare `dict` rather than rebuilding it as a nested model
  (which CrewAI and LlamaIndex do get, via `pydantic_model_from_schema`). The
  frameworks reading that signature — Semantic Kernel, AG2, Google ADK's
  fallback path — therefore still see an untyped object for a nested Pydantic
  or dataclass parameter. Changing it would alter what those frameworks
  introspect in a way this suite can't exercise against their real schema
  derivation, so it is left as a tracked gap rather than an untested change.

### Tests

- **A concurrency benchmark asserts overlap rather than a wall-clock ratio.**
  `test_concurrent_retrieval_throughput` exists to prove the embedder does not
  block the event loop, and used `speedup > 1.2x` as a proxy. The proxy is what
  broke on a contended CI runner: identical code measured 18.9x locally and
  0.97x on a runner taking 7m27s for a suite that runs in 90s, because a ratio
  between two wall-clock spans measures the machine once CPU is scarce. The
  mock embedder now records how many embeds are in flight, and the test asserts
  they overlapped — the property itself, immune to how slow the runner is, and
  still failing on a genuinely blocking embedder.

### CI

- **The weekly drift-check now tests what it claims.** The semantic-kernel
  cell silently resolved back to 1.36.0 (sk ≥1.43.1 needs a pre-release
  `azure-ai-agents`; `uv pip install` refuses pre-releases by default) — it
  now passes `--prerelease=allow`. New cells cover langchain/langgraph/
  llamaindex/autogen-core at latest and classic AG2 (the `register()` API,
  previously never installed in any job). Single-framework cells also run
  `test_real_packages.py` (the only suite that builds *and invokes* the
  native tool object). New real-package guards: `GantryLiveAgnoAgent.build`
  and `GantryLiveSmolAgent.build` (the exact gap class that let the haystack
  3.0 `ToolInvoker` removal slip past the stubbed suites), AG2 registration,
  and `disable_af_instrumentation`.
- `pyproject.toml` audit comments corrected 2026-08-26: the universal lock
  holds agent-framework at 1.5.0 and google-adk at 1.14.1 (pre-release
  `azure-ai-agents` refusal and semantic-kernel 1.36.0's `pydantic<2.12`
  pin, respectively) — the comments previously claimed the resolver picked
  1.13.0 / 2.x. Adapters re-verified standalone against agent-framework
  1.15.0, google-adk 2.7.1, semantic-kernel 1.44.1, crewai 1.15.17,
  langchain 1.3.17, langgraph 1.2.11, llama-index-core 0.14.24, pydantic-ai
  2.35.0, openai-agents 0.22.0, smolagents 1.26.0, haystack-ai 3.1.0, agno
  3.0.1, strands-agents 1.53.0, and dspy 3.3.1.

### Docs

- `examples/agent_frameworks/semantic_kernel_example.py` rewritten to use
  `SemanticKernelAdapter` (it previously hand-rolled a plugin and never used
  the integration it exemplifies); `langchain_example.py` moved off
  `langgraph.prebuilt.create_react_agent` (removed in LangGraph 2.0) to
  `langchain.agents.create_agent`; the examples README gained the four
  missing entries (strands, dspy, generic adapters, AF harness) and the AG2
  install guidance.

## [0.11.0] - 2026-08-20

### Added (tool-use loop)

- **The round-trip layer is now usable end to end.** Every dialect adapter
  could already parse one tool-call payload and format one result, but nothing
  joined those ends — no wrapper, no facade method, and no example used them,
  so callers hand-rolled `json.loads(tc.function.arguments)` and the
  parallel-call case was left to each caller to rediscover. New:
  `extract_tool_calls(response, dialect)` pulls *every* call out of a whole
  response (OpenAI chat and Responses, Anthropic, Gemini; SDK objects or plain
  dicts), and `AgentGantry.execute_tool_calls(response)` runs them
  concurrently through the full protection stack and returns provider-shaped
  results ready to append to the conversation. A failing tool comes back as an
  error-flagged result rather than raising, because that is what a tool-use
  loop needs.
- **Streaming tool calls are reassembled.** Nothing accumulated OpenAI
  `delta.tool_calls` fragments or Anthropic `input_json_delta` — the
  production-normal path got no help at all. `StreamingToolCallAccumulator`
  folds chunks into complete calls (parallel streams stay separate, a
  truncated stream yields empty arguments rather than raising) and its output
  feeds straight into `execute_tool_calls`.
- Both are exported from the top-level package.

### Fixed (frameworks)

- **The two Anthropic `execute_tool_calls` implementations are now one.**
  `AnthropicClient` ran its tools sequentially while `SkillsClient` gathered
  them, and both re-implemented `AnthropicAdapter.format_tool_result` inline.
  Both now delegate to the facade, so they share one concurrency model and one
  formatter.

- **The sync bridge no longer serializes every tool call, and no longer
  deadlocks on a nested one.** `ToolSpec.invoke` hands its coroutine to a
  worker thread when a loop is already running. That pool was `max_workers=1`
  and process-wide, so every sync tool call in the process queued behind every
  other — a multi-agent CrewAI run ran strictly one tool at a time — and a
  handler that itself called `invoke` waited on the single worker it was
  occupying, hanging forever. The pool now sizes like a normal
  `ThreadPoolExecutor`, a re-entrant call gets its own thread instead of a pool
  slot, and lazy construction is locked so a race cannot build two pools.
- **OpenAI Agents tools keep their optional parameters optional.**
  `FunctionTool.strict_json_schema` defaults to `True`, and the SDK's
  `ensure_strict_json_schema` then rewrites `required` to list *every*
  property. The adapter set only a top-level `additionalProperties: False`, so
  that rewrite silently made every optional Gantry parameter mandatory with no
  `null` union, raised `UserError` on nested `additionalProperties: true`, and
  on older SDKs sent a non-strict schema with `strict=true` (a 400). It now
  uses the shared strict transform.
- **Structured tool results reach the model as JSON.** The OpenAI Agents
  adapter and the AutoGen workbench rendered results with `str()`, so a dict
  arrived as Python repr (single quotes, `None`, `True`) for the model to
  guess at. Both now serialize non-strings as JSON, matching the Agent
  Framework bridge.

### Added

- **Token usage is now measured.** `TokenUsageEvent` was defined and never
  constructed, and `record_token_usage` existed on the telemetry protocol and
  both adapters yet was never called by library code — so the flagship
  prompt-reduction claim went unmeasured. `with_semantic_tools` and the two
  Anthropic clients now report each call's provider `usage` block to telemetry,
  best effort: a response without usage is not an error, and a telemetry
  failure never breaks the user's call. Savings are deliberately not inferred,
  because that needs a real baseline and `agent_gantry.metrics.token_usage`
  refuses approximate estimators so the numbers stay auditable — callers who
  run a baseline can still pass both usages to `calculate_token_savings`.
- `AgentGantry.telemetry` exposes the configured adapter, so integration layers
  no longer have to reach into a private attribute.

### Performance

- **Model loading no longer stalls the event loop.** The sentence-transformers
  and Nomic embedders and the cross-encoder reranker construct their model on
  first use. `encode`/`predict` were already offloaded with
  `asyncio.to_thread`, but construction — the expensive part, downloading
  weights on a cold cache — ran inline in the coroutine, freezing every other
  task on the loop for seconds. Construction now runs in a worker thread,
  guarded so concurrent first calls load the model exactly once. The sync
  `dimension` property keeps working.
- **LanceDB reads project only the columns they use.** `search_skills`,
  `list_all`, and `list_all_skills` had no `.select()`, so every returned row
  also materialized its full embedding vector just to discard it — measured at
  ~630x the payload for a 50-row scan. `list_all_skills` matters most: the
  facade's embedder-migration check calls it with a limit of 1,000,000.
- **LanceDB fingerprint reads are columnar.** `get_stored_fingerprints` called
  `to_pylist()` on the whole Arrow table, allocating a dict per row just to
  read two fields out of each. It now converts the two columns it needs and
  zips them, which drops the per-row allocation on large stores. A null
  fingerprint is coerced to `""` so the declared `dict[str, str]` holds.

### Fixed

- **Tool execution now honours the namespace selection resolved.** Selection is
  namespace-aware everywhere — `ToolSpec` carries `_namespace`, pinning
  distinguishes `"other.foo"` from `"foo"`, the Agent Framework bridge caches
  per namespace — but `ToolCall` carried only a bare `tool_name`, and the
  registry's bare-name lookup prefers `default.<name>`. With two MCP servers
  exposing a same-named tool (a supported configuration, since MCP tools are
  registered under per-server namespaces), the selected `other.search` could
  silently execute `default.search`. `ToolCall` gains an optional `namespace`,
  a qualified `tool_name` ("billing.search") is accepted, and every internal
  call site that already knows which tool it selected — `ToolSpec.ainvoke`, the
  Agent Framework bridge, `search_and_execute` — now passes it. A bare-name
  call whose name exists in several namespaces logs a warning instead of
  resolving silently. Bare-name execution still works unchanged, since a
  provider tool-call payload cannot express more than the name the model saw.

- **OpenAI strict mode now emits a schema the API accepts.** `strict=True`
  only set the `strict` flag; it never reshaped the parameter schema. OpenAI
  rejects a strict tool unless every object sets `additionalProperties: false`
  and lists all of its properties in `required`, so any tool with an optional
  parameter — including Agent-Gantry's own introspected tools — produced a 400.
  Both the Chat Completions and Responses adapters now transform the schema,
  widening formerly-optional properties to admit `null` so optionality is
  preserved in meaning. The tool's canonical schema is never mutated.
- **Per-dialect options now reach the adapter.** `retrieve_tools` forwarded
  `**kwargs` into `ToolQuery`, whose `extra="ignore"` dropped anything that was
  not a query field, then called `to_dialect` with no options — so
  `retrieve_tools(..., strict=True)`, `OpenAIAdapter(gantry).tools(q,
  strict=True)` and `with_semantic_tools(...)` all silently returned non-strict
  schemas. Keywords are now split: `ToolQuery` fields configure retrieval, the
  rest go to the adapter. `retrieve_tools` and `with_semantic_tools` also take
  an explicit `dialect_options` dict.
- **Gemini and Vertex AI schemas are sanitized.** The Gemini adapter passed
  `parameters_schema` through verbatim, but the Google SDKs reject unknown
  JSON-Schema keywords rather than ignoring them. `additionalProperties`,
  `default`, `title` and similar are now stripped, `const` is converted to a
  one-value `enum`, and local `$ref`/`$defs` pairs (what Pydantic emits for
  nested models) are inlined since the SDKs will not follow the pointers.
  Structural keywords are deliberately preserved — dropping one would silently
  change which values a schema accepts. Pass `sanitize=False` to opt out.
- **Anthropic cache tokens count towards the prompt.** `ProviderUsage.from_usage`
  read only `input_tokens`, ignoring `cache_creation_input_tokens` and
  `cache_read_input_tokens`. Those tokens were processed, so omitting them made
  a cached run look nearly free against an uncached baseline — a run that truly
  saved 58% reported 98%. They are now included in `prompt_tokens` and also
  surfaced separately as `cached_prompt_tokens`.
- **Emitted schemas no longer alias the registry.** OpenAI, Anthropic-strict
  and Gemini conversions embedded `ToolDefinition.parameters_schema` by
  reference, so a caller mutating a returned schema corrupted every later
  conversion of that tool. The transforming paths now deep-copy.

- **Anthropic convenience clients no longer silently drop every tool.**
  `AnthropicClient.create_message` and `SkillsClient.create_message` built
  their `ToolQuery` without `score_threshold`, inheriting the schema default
  of 0.5. That default is documented as a silent-drop trap for convenience
  layers — long queries dilute absolute similarity, so retrieval could return
  zero tools with no error. Both now pass `0.0`, matching every other
  convenience surface.
- **A configured reranker now actually runs.** `retrieve()` enables reranking
  when `ToolQuery.enable_reranking is None`, but the field was `bool = False`
  and never `None`, so the branch was dead and a configured reranker was
  silently skipped unless the caller passed `enable_reranking=True`. The field
  is now tri-state (`bool | None`, default `None`): `None` defers to the
  reranker config, `True`/`False` force the behaviour.
- **LangChain messages are now understood by the query strategies.**
  `_msg_role` read only `.role`, but LangChain carries the role in `.type`
  (`"human"`/`"ai"`/`"tool"`). Every LangChain message therefore resolved to
  role `""`, so `latest_activity` could let an `AIMessage` drive retrieval and
  never applied the tool-result character cap. Known LangChain type values are
  now mapped to roles; an unrelated `.type` attribute is still ignored.
- **Malformed tool-call arguments are logged instead of silently dropped.**
  The OpenAI and OpenAI-Responses adapters swallowed `json.JSONDecodeError`
  and returned `{}`, so the tool failed later with a misleading "missing
  required parameter". Both now warn with the tool name and the offending
  payload, matching the Agent Framework adapter's existing behaviour.
- **`RateLimiter.acquire` holds one lock across check, strategy, and
  increment.** It previously released the lock between the concurrency check
  and the increment. No live overshoot was reachable (today's strategy checks
  contain no `await` and an uncontended `asyncio.Lock` does not yield), but
  the invariant rested on that staying true; it is now structural, and each
  acquire takes one lock cycle instead of two.
- **Security policies and rate limits key off the resolved tool.** The
  qualified-name calling convention added above meant `call.tool_name` could be
  `"billing.search"`, but `_check_security_policy` fnmatched policies against
  the raw string and `_check_rate_limit` keyed the limiter off it — so one tool
  matched different policy patterns depending on the convention used, and
  produced two independent rate-limit budgets (`"billing.billing.search"` vs
  `"billing.search"`), doubling the allowance for a caller who alternated
  styles. Both now use the resolved tool: the policy sees the bare name from
  either convention, preserving existing policy semantics exactly.
- **A permission denial is reported the same from every path.**
  `_execute_handler_with_retries` mapped `PermissionDeniedError` to
  `ExecutionStatus.FAILURE` while `_check_rate_limit` mapped the same exception
  to `PERMISSION_DENIED`, so whether a permission failure was distinguishable
  depended on which code path raised it. Both now report `PERMISSION_DENIED`.

### Changed

- **`DEFAULT_TOOL_LIMIT` is now honoured everywhere (default 3 -> 5).** The
  shared constant exists so the static and live adapter families cannot drift
  apart, but `_LLMToolAdapter` (the `OpenAIAdapter`/`AnthropicAdapter`/... LLM
  SDK wrappers) and `ToolRefresher` still hardcoded 3. Both now use the
  constant, so they surface 5 tools per call by default. Pass
  `default_limit=3` / `limit=3` to restore the previous behaviour.
- **`auto_sync` on `SemanticToolSelector` / `with_semantic_tools` is
  deprecated.** It was accepted, stored, and never read — `AgentGantry.retrieve()`
  always calls `ensure_synced()`. Passing `auto_sync=False` now raises a
  `DeprecationWarning` and still changes nothing; the parameter will be removed
  in a future release.

### Security

- **Identifier fields reject embedded newlines.** `ToolCall`, `ToolResult`, the
  telemetry event models, `RetrievalResult`, and the MCP/A2A config models
  carried free-form identifiers (`tool_name`, `trace_id`, `span_id`, `name`,
  `namespace`) with no newline validation, so a crafted identifier could inject
  extra lines into logs or headers built from them. The existing
  `reject_newlines` validator is now applied across those models, with
  `validate_assignment=True` so a later assignment cannot bypass it.

### Documentation

- Docs site accessibility: the decorative sidebar logo is now `aria-hidden`,
  the homepage hero CTAs are grouped as a semantic list so screen readers
  announce their count and boundaries, and a global `:focus-visible` outline
  replaces browser defaults for keyboard navigation.

### Internal

- `agent_gantry.integrations.refresh` reuses the canonical `_msg_text` /
  `_msg_role` from `agent_gantry.query.strategies` instead of keeping a second
  copy that had already drifted (the canonical pair also understands
  Responses-API `input_text` parts and Agent Framework `function_result`
  blocks).

## [0.10.0] - 2026-08-07

### Added

- **Semantic skill selection.** The `Skill` schema (procedural memory:
  guidance retrieved by meaning and injected into prompts, never executed)
  now has a facade API — `add_skill`/`add_skills`, `retrieve_skills`,
  `retrieve_skills_as_prompt` (pre-formatted system-prompt block),
  `delete_skill`, `list_skills`, `count_skills` — using the same embedder
  and vector store as tools. The default `InMemoryVectorStore` gained full
  skill support (add/search/get/delete/list/count with namespace/category
  filters and per-dimension matrices), joining LanceDB, which already
  persisted skills; stores without skill support raise a clear
  `NotImplementedError`. `Skill`, `SkillCategory`, and `SkillSearchResult`
  are exported from the top-level package. Example:
  `examples/basics/skills_example.py`.
- **Qdrant quantized vector search.** `QdrantVectorStore(quantization=...)`
  enables int8 scalar quantization (`"scalar"`: ~4x smaller vectors kept in
  RAM, minimal recall loss) or binary quantization (`"binary"`: ~32x
  smaller, best for high-dimensional embeddings) at collection creation.
  Searches oversample and rescore candidates against the original vectors,
  so returned scores stay exact. Existing collections are not migrated —
  recreate the collection to change quantization.

- **mcp 1.x and 2.x are both supported** — the `mcp` dependency range widened
  from the emergency `<2.0.0` cap to `>=1.27.2,<3`. mcp 2.0.0 kept the entire
  v1 client surface (`ClientSession` / `StdioServerParameters` /
  `stdio_client`), so the persistent-session client works verbatim; the two
  real breaks are handled in one code path: `servers/mcp_server.py` registers
  handlers via the 1.x decorators or the 2.x constructor callbacks
  (`on_list_tools` / `on_call_tool`, whose handlers return full
  `ListToolsResult` / `CallToolResult` models and must mark failures with
  `is_error` themselves), and the client reads tool schemas dual-spelled
  (`input_schema` on 2.x, `inputSchema` on 1.x — the 1.x-only read silently
  replaced every v2 tool's schema with an empty default). The full MCP test
  suite passes against both mcp 1.28.1 and 2.0.0, including real stdio
  subprocess round-trips (`tests/test_mcp_execution.py` now spins up a
  version-appropriate server: FastMCP on 1.x, `MCPServer` on 2.x).
  Cross-version protocol interop over stdio was verified in both directions.
  The combined `all` extra still locks mcp 1.x because openai-agents and
  agent-framework pin `mcp<2`; standalone `agent-gantry[mcp]` installs may
  resolve 2.x.
- **haystack-ai 3.0 support.** haystack 3.0 removed `ToolInvoker` (the
  `Agent` component now owns tool execution), which broke
  `GantryLiveHaystackToolInvoker.build()` with a *misleading* "install
  haystack-ai" `ImportError` even when haystack 3 was installed — the stubbed
  test suites never exercised `build()` against the real package. `build()`
  now branches: on haystack 2.x it returns a fresh `ToolInvoker` as before;
  on >=3.0 it builds a per-call `haystack.components.agents.Agent` when the
  builder was given `chat_generator=...`, and otherwise raises a clear
  `RuntimeError` pointing at the alternatives. New real-package guard tests
  (`tests/frameworks/test_haystack_build_live.py`) cover the 2.x invoker
  path, the 3.x Agent path, and the 3.x error path; `haystack_example.py`
  and the adapter docs are version-aware.
- **MCP-discovered tools are now executable through `gantry.execute()`.**
  `add_mcp_server()` and `discover_tools_from_server()` register an execution
  handler per discovered tool that proxies the call to the server via
  `MCPClient.call_tool`, so MCP tools run through the full engine path
  (security policy, retries, timeouts, telemetry) like `@gantry.register`-ed
  tools. Previously discovered MCP tools were retrievable but failed with
  "No handler found" on execution — `MCPClient.call_tool` had no callers.
  In-band MCP tool failures (`isError`/`is_error` on the call result, how the
  protocol reports a tool that raised) are surfaced as exceptions so the
  engine records them as failures — with retries, health, and telemetry —
  instead of passing the error object through as a successful result. The
  persistent session survives such failures (tool error ≠ broken connection).
  Symmetrically, `MCPServer._handle_execute_tool` raises on failed
  executions instead of returning error text, so the served result carries
  `isError` and MCP clients don't record the failure as a success.
  Qualified-name collisions are first-wins for definition AND handler: an
  MCP tool whose `namespace.name` is already registered by a different
  source is skipped with a warning instead of silently hijacking the
  existing tool's handler (validation/authorization and dispatch would
  otherwise disagree about which tool runs). Re-discovery from the same
  server refreshes the stored definition along with the handler, so
  re-adding a reconfigured server can't leave validation running against
  the old schema while calls go to the new subprocess — and tools the
  reconfigured server no longer exposes are removed (registry, handlers,
  vector store), since their handlers would reconnect to the replaced
  command. Discovered definitions enter the registry immediately
  (mirroring `add_tool()`), so MCP tools are executable before the next
  `sync()` even with `auto_sync=False`.
  See the new end-to-end suite `tests/test_mcp_execution.py`, which runs a
  real stdio MCP server subprocess.
- **Persistent MCP sessions.** `MCPClient.call_tool` and `list_tools` share
  one long-lived connection per server (owned by a dedicated background task
  so anyio cancel scopes stay in one task) — discovery seeds the connection
  the first tool call reuses — instead of spawning the server subprocess and
  re-running the initialize handshake on every call — previously hundreds of
  milliseconds to seconds (for `npx`-launched servers) of overhead per tool
  execution. Transport errors invalidate the session so the next call
  reconnects. New lifecycle hooks: `MCPClient.close()`,
  `MCPClientPool.close_all()`, `MCPRegistry.close_all_clients()`, and
  `AgentGantry.close()` closes all MCP clients it created.
- **Incremental sync for Qdrant, Chroma, and PGVector.** All three remote
  stores now persist per-tool fingerprints (Qdrant payload field, Chroma
  metadata field, new PG `fingerprint` column with in-place `ALTER TABLE`
  migration) and implement `get_stored_fingerprints()` plus the sync-metadata
  API (`get_metadata`/`set_metadata`/`update_sync_metadata`, backed by a small
  side collection/table). Previously these backends returned the protocol
  default (empty fingerprints), so **every** `sync()` re-embedded and
  re-upserted the entire registry — on every process restart, and per
  `add_tool()` call with `auto_sync=True`.
  Deployment note for PGVector: the first `initialize()` against a
  pre-existing table performs the one-time `ALTER TABLE ... ADD COLUMN IF
  NOT EXISTS` and `CREATE TABLE IF NOT EXISTS <table>__meta`, so the app's
  DB role needs DDL on its own table (implicit for table owners; grant
  explicitly if your role only has DML).

- **`required` / `always_include` pinned-tool selection, ported to every
  framework adapter.** Previously only the Microsoft Agent Framework provider
  (`GantryContextProvider(required=..., always_include=...)`) could guarantee
  a named tool's presence in the selection or pin a tool onto every round
  regardless of semantic score. `GantryToolset.select` / `.select_or_empty`
  (`integrations/frameworks/base.py`) now accept the same two keywords, and
  `BaseFrameworkAdapter.select` and every adapter's `live(...)` (plus the
  bespoke live methods and constructors it delegates to — `live_wrappers.py`,
  every `*_live.py` module) thread them through, so all 15 framework
  integrations get the same guarantee. `required=[...]` (bare or
  `namespace.name`-qualified names) must resolve against the registry or
  `select` raises the new shared `MissingRequiredToolError`
  (`integrations/frameworks/errors.py`, re-exported from `agent_gantry`,
  `agent_gantry.integrations`, and `agent_gantry.integrations.frameworks` —
  `agent_gantry.integrations.agent_framework_provider.MissingRequiredToolError`
  now imports from this shared module, keeping the historical import path
  working); `always_include=[...]` logs a `WARNING` and skips unresolvable
  names instead of raising. Both are appended after the semantic slice
  (`required` before `always_include`), deduplicated, and never counted
  against `limit` — matching `GantryContextProvider`'s own choice that
  `top_k` bounds only the dynamic/semantic slice. The Microsoft Agent
  Framework provider's own `required`/`always_include` implementation was
  left in place (it is entangled with skills, `static_tools`, and
  `ContextVar`-scoped retrieval history with no equivalent in the plain
  adapter layer) — only the error type is shared, so its 90+ existing tests
  keep passing unmodified. See `integrations/frameworks/README.md`
  ("Guaranteed & pinned tools") and the new `tests/frameworks/test_selection.py`.
- **Reverse-direction framework importers** — `agent_gantry.integrations.importers`
  adds `register_langchain_tools`, `register_crewai_tools`, and
  `register_llamaindex_tools`, the missing other half of every
  `<Framework>Adapter` (which only ever *exported* Gantry tools outward).
  Each coroutine converts existing `langchain_core.tools.BaseTool`,
  `crewai.tools.BaseTool`, or `llama_index.core.tools.FunctionTool` objects
  into `ToolDefinition`s (new `ToolSource.FRAMEWORK`) with an execution
  handler wired through `gantry.add_tool(tool, handler=...)` — a new optional
  `handler` parameter on `AgentGantry.add_tool()` — so imported tools run
  through the normal `gantry.execute()` path (security policy, retries,
  circuit breakers, telemetry) exactly like `@gantry.register`-ed ones, gain
  semantic routing, and are re-exportable to any *other* framework via the
  existing export adapters. Malformed/unrecognized tools are skipped with a
  logged warning rather than aborting the batch; an empty `tools` argument
  raises. See `agent_gantry/integrations/README.md` ("Importing existing
  framework tools") and `examples/frameworks/importers_example.py`.
- **Uniform `live_tier` / `live()` entry point on every framework adapter.**
  All 13 `<Framework>Adapter` classes (`BaseFrameworkAdapter` subclasses) plus
  `AgentFrameworkAdapter` (Microsoft Agent Framework) now expose
  `adapter.live_tier` (`"per-turn"` or `"per-call"` — the deepest dynamic
  re-selection tier that framework supports) and
  `adapter.live(*, limit=None, score_threshold=0.0, namespaces=None,
  **framework_kwargs)`, which returns the framework-appropriate live object
  (hook / toolset / provider / builder) by delegating to that framework's
  existing bespoke live method (`react_agent`, `toolset`, `tool_hook`,
  `agent_builder`, …). No bespoke method was removed or renamed — `live()` is
  a thin, uniform layer over them, so framework-agnostic code no longer needs
  to know each framework's own live-method name. See
  `integrations/frameworks/README.md` for the full per-framework table
  (`live_tier`, delegate, return type, where to plug it in).
  `tests/frameworks/test_conformance.py` locks the new surface: every adapter
  is checked for a valid `live_tier` matching the documented capability table,
  and a stub-based test proves `live()` calls the right bespoke method with
  the right kwargs.
- **`namespaces` now threads through every live/per-turn provider**, not just
  the static `select()` path and `OpenAIAgentsAdapter`'s live methods.
  `LangGraphAdapter.react_agent`/`.areact_agent`/`.select_for_state`,
  `LlamaIndexAdapter.tool_retriever`/`.function_agent`,
  `PydanticAIAdapter.toolset`, `SemanticKernelAdapter.function_provider`/
  `.refresh`, `GoogleADKAdapter.before_model_callback`/`.agent`,
  `AutoGenAdapter.workbench`, `StrandsAdapter.tool_hook`/`.agent`, and the
  per-call builders (`CrewAIAdapter.agent_builder`, `AgnoAdapter.agent_builder`,
  `HaystackAdapter.tool_invoker_builder`, `SmolagentsAdapter.agent_builder`)
  all now accept and forward `namespaces` to every re-selection.
- **`GantryToolset.select_or_empty`** — a new selection primitive that returns
  `[]` immediately for a blank/whitespace-only query instead of running a
  nonsensical selection on an empty embedding. Every `integrations/frameworks/
  *_live.py` module previously re-implemented this exact guard by hand (each
  with its own "consistent with the other live providers" comment); the
  duplicated guard+select code is now centralized here, and
  `ToolRefresher.refresh_specs` uses the same primitive.
- **Native AWS Strands Agents adapter** — `StrandsAdapter`
  (`from agent_gantry.strands import StrandsAdapter`), joining the per-framework
  `<Framework>Adapter` family. `await adapter.select(query, limit=...)` /
  `adapter.convert(spec)` wrap Gantry tools as Strands
  `DecoratedFunctionTool`s (built from `spec.callable_for_signature()`, with
  Gantry's own name/description/JSON-Schema parameters passed straight through
  via `strands.tool()`'s `name`/`description`/`inputSchema` overrides). Strands
  genuinely supports per-turn re-selection — it fires a `BeforeModelCallEvent`
  hook before every model call and only reads the tool registry afterward — so
  `StrandsAdapter(gantry).tool_hook()` / `.agent(...)` re-select tools on
  **every model call**, matching the depth of Google ADK's
  `before_model_callback` rather than the per-top-level-call rebuild used for
  CrewAI/Agno/Haystack/Smolagents.
- **Native DSPy adapter** — `DSPyAdapter`
  (`from agent_gantry.dspy import DSPyAdapter`), joining the per-framework
  `<Framework>Adapter` family. `await adapter.select(query, limit=...)` /
  `adapter.convert(spec)` wrap Gantry tools as `dspy.Tool`s for DSPy's
  agentic module, `dspy.ReAct`, with Gantry's own name/description/JSON-Schema
  parameters passed straight through via
  `dspy.adapters.types.tool.convert_input_schema_to_tool_args` (the same
  schema bridge DSPy's own MCP/LangChain tool converters use). The wrapped
  function is intentionally **synchronous** (`ToolSpec.invoke`'s loop-safe
  bridge), not `callable_for_signature()`'s async wrapper: `dspy.Tool.__call__`
  — the path `ReAct.forward()`/plain `react(...)` uses, DSPy's own documented
  call convention — raises on an async tool unless the caller opts in with
  `dspy.configure(allow_tool_async_sync_conversion=True)` or always calls
  `await react.acall(...)`; a sync wrapper works correctly under both entry
  points with no DSPy configuration. `dspy.ReAct` fixes its tool list at
  construction with no runtime re-selection hook (`dspy.utils.callback`'s
  `on_tool_start`/`on_module_start` fire around an already-selected call, not
  before the model picks the next tool), so `DSPyAdapter(gantry).agent_builder(
  signature, ...)` follows the same per-top-level-call rebuild tier as
  CrewAI/Agno/Smolagents rather than a fabricated per-turn hook.
- **`framework-adapters` CI job now runs on ubuntu × Python 3.10–3.13 plus a
  macOS 3.12 cell** (was a single ubuntu/3.12 cell), matching the OS/Python
  coverage the main `test` matrix already gives the other framework
  integrations. Its isolated-env install now also covers `strands-agents`.
- **New scheduled `latest-frameworks.yml` workflow** validates the latest
  release of each framework that `pyproject.toml` deliberately floors below
  current (google-adk, crewai, semantic-kernel, agent-framework) plus the
  native-adapter set, each in its own isolated env, so drift on unpinned
  latest versions is caught by a weekly run instead of staying invisible
  between manual audits.
- **Documented and tested error-handling policy for all 14 native framework
  adapters + Microsoft Agent Framework.** New "Error-handling policy" section
  in `integrations/frameworks/README.md` states the default contract (a
  failing `gantry.execute()` raises `ToolExecutionError` out of every
  adapter's native tool object, uncaught, letting the framework's own error
  handling take over), the four deliberate "framework absorbs the error"
  exceptions with their rationale (Microsoft Agent Framework's JSON error
  string, AutoGen's live `Workbench.call_tool`, Strands' real `Agent`
  tool-execution loop, and — not a Gantry deviation but documented for
  completeness — DSPy's own `ReAct.forward`/`.aforward`), and a uniform rule
  for the eight per-turn live providers' *selection* failures (`WARNING` log
  + graceful degradation, never raise). `tests/frameworks/test_conformance.py`
  gained an `AdapterCase.error_kind`/`.invoke_failure` field and a
  parametrized `test_adapter_tool_failure_matches_documented_error_kind` that
  forces a failing tool through every adapter's real native call convention
  (proving the contract survives the `_run_coroutine_sync` worker-thread
  bridge for the sync wrappers too), plus one `test_*_live_selection_failure_degrades_gracefully`
  test per per-turn provider and a `test_tool_execution_error_message_format`
  test locking `ToolExecutionError`'s message shape. `tests/frameworks/
  test_dspy_live.py` and `test_strands_live.py` each gained a dedicated test
  proving their respective "framework absorbs it" behavior end-to-end against
  the real installed package.

### Changed

- **BREAKING — `LLMConfig.model` default is now `"gpt-5.4-mini"`** (was
  `"gpt-4o-mini"`, which OpenAI shuts down on 2026-10-23). Leaving the
  retiring model as the default would break every deployment relying on it
  at the shutdown date; callers who need the old model until then must set
  `model="gpt-4o-mini"` explicitly. This only affects LLM-based intent
  classification (`use_llm_for_intent=True`), which is off by default.
  `LLMClient.classify_intent` builds reasoning-model-compatible requests for
  the gpt-5 family and o-series: `max_completion_tokens` (with headroom for
  reasoning tokens) instead of the legacy `max_tokens`, and no `temperature`
  (reasoning models accept only the default) — otherwise every request would
  fail and silently degrade classification to the UNKNOWN fallback.
- **CI native-adapter caps lifted** (added earlier in this cycle as a
  temporary mitigation): pydantic-ai verified against 2.23.0 with **zero
  adapter changes needed** (every construction site was already keyword-only,
  which spans 1.x and 2.x); dspy verified against 3.3.0 (fully backward
  compatible on the adapter surface — the rebuilt ReActV2 is a separate,
  explicitly experimental class); haystack-ai 3.0 supported via the
  version-branched `build()` (see Added). The isolated CI job installs all
  three uncapped again.
- **LanceDB implementation consolidated.** `lancedb_mixins.py` carried full
  duplicate copies of `add_tools`/`search`/the skills API that were shadowed
  by the identical class-body definitions in `LanceDBVectorStore` (MRO:
  class body wins) — ~600 lines of dead code that silently diverged whenever
  only one copy was fixed. The mixins now contain only what is actually
  inherited (tools schema migration and the sync-metadata API), and those
  live methods run their blocking LanceDB calls off the event loop via
  `asyncio.to_thread` like the main store methods.
- **Performance.**
  - `ToolRegistry.get_tool_by_name` is O(1) via a name index instead of a
    linear scan over all registered tools; it runs on every
    `ExecutionEngine.execute()` call (~800× faster at 5k tools in
    microbenchmarks).
  - `SemanticRouter.route` resolves intent concurrently with query embedding
    + vector search; with LLM-based intent classification this removes an
    entire LLM round-trip from the retrieval critical path.
  - Candidate scoring normalizes conversation context (summary/messages
    lowercasing, used/failed-tool sets) once per query instead of once per
    candidate.
  - LanceDB queries now run off the event loop (`asyncio.to_thread`) — disk
    I/O no longer freezes concurrent coroutines — and `search` fetches only
    the columns it needs instead of materializing every row's embedding
    vector; the tag filter no longer parses each row's JSON twice.
  - `OpenAIEmbedder.embed_batch` issues batch requests concurrently (bounded
    at 4 in flight) instead of strictly sequential round-trips — large syncs
    are ~3-4× faster at typical API latency.
  - `add_mcp_server()` / `discover_tools_from_server()` / `add_a2a_agent()`
    register all discovered tools then sync once, instead of triggering a
    full sync (fingerprint scan + size-1 embed batch) per tool under
    `auto_sync=True`.
  - The execution engine caches one `A2AExecutor` (previously constructed per
    call, which also discarded its per-agent client cache), and `A2AClient`
    holds a persistent `httpx.AsyncClient` per event loop so repeated task
    sends reuse connections instead of paying DNS + TCP + TLS each call.
    `ExecutionEngine.close()` / `A2AExecutor.close()` / `A2AClient.close()`
    release the connections; `AgentGantry.close()` calls through.
  - `gantry.sync()` compares synced tools by qualified name instead of
    Pydantic deep-equality list membership (was O(N²) over full schemas).
- **Dependency floors refreshed (2026-08-03 audit)** — see `pyproject.toml`
  comments for full rationale per package:
  - `mcp` was emergency-capped `<2.0.0` at this audit (mcp 2.0.0, released
    2026-07-28, moved every v1 API Gantry used — `Server`, `stdio_server`,
    `ClientSession`, `stdio_client` — so an uncapped standalone
    `agent-gantry[mcp]` install broke at import). **Superseded later in this
    same cycle**: the cap was replaced by full 1.x/2.x dual-version support
    and the range is now `>=1.27.2,<3` — see the mcp entry under Added.
  - `crewai>=1.15.0` — its opentelemetry conflict with `agent-framework` was
    resolved upstream in 1.15.0; the combined `agent-frameworks` extra now
    locks crewai 1.15.10 (was held at 1.6.1).
  - `langchain>=1.3.14`, `langchain-openai>=1.4.1`, `langgraph>=1.2.10`,
    `llama-index-core>=0.14.23`, `llama-index-llms-openai>=0.7.10`,
    `openai>=2.45.0`, `anthropic>=0.120.2`, `cohere>=7.0.8`, `groq>=1.6.0`.
  - `semantic-kernel` stays at `>=1.36.0`: a bump to 1.43.1+ is blocked by a
    *new* conflict (sk 1.43+ pins `azure-ai-projects<1.1`, agent-framework
    needs `>=2.2`) — verified via `uv lock` and documented.
  - The obsolete `azure-search-documents>=11.7.0b2` uv override was removed
    (stable 12.0.0 exists and is what the AF search beta now requires).
  - The `mistralai` PyPI quarantine has been lifted upstream (comment
    updated); Gantry deliberately keeps the OpenAI-compatible path.
  - CI's native-adapter smoke-test job temporarily caps `pydantic-ai-slim<2`,
    `haystack-ai<3`, and `dspy<3.3` — each shipped breaking changes for the
    adapter surface (pydantic-ai 2.0 `ToolDefinition` arg reorder and
    `builtin_tools`→capabilities; haystack 3.0 `ToolInvoker` removal;
    dspy 3.3 ReActV2 trajectory format). Migrating the adapters and lifting
    the caps is tracked follow-up work.
- **`ToolRefresher` (`agent_gantry.integrations.refresh`) is now explicitly
  documented as the standalone, hand-rolled-agent-loop utility**, cross-linked
  with the new `adapter.live()` uniform entry point. It was already
  framework-agnostic by design (no framework adapter called it) — the
  per-framework `*_live.py` modules' query-derivation logic is genuinely
  framework-specific (a LangGraph `state["messages"]`, a Pydantic AI
  `RunContext`, an ADK `callback_context`, …) and was deliberately left
  un-merged with `ToolRefresher`'s generic message-list walker; only the
  shared, framework-independent "guard against an empty query, then select"
  step was extracted, into `GantryToolset.select_or_empty` (used by both
  `ToolRefresher` and every `*_live.py` module).
- **BREAKING — static framework adapters now default `limit` to 5, not 3.**
  `GantryToolset`, `BaseFrameworkAdapter`, and every native per-framework static
  helper (`agent_gantry.langchain`, `.crewai`, `.llamaindex`, `.autogen`,
  `.google_adk`, `.agno`, `.haystack`, `.pydantic_ai`, `.smolagents`,
  `.openai_agents`, `.semantic_kernel`) previously surfaced 3 tools per call by
  default while every live/deep per-turn provider (`live_wrappers.py`,
  `integrations/frameworks/*_live.py`, `AgentFrameworkAdapter`) already
  defaulted to 5. Both families now share a single
  `agent_gantry.integrations.frameworks.base.DEFAULT_TOOL_LIMIT = 5` constant.
  Callers relying on the old default of 3 tools per static selection should
  pass `limit=3` explicitly. (`integrations/frameworks/langgraph_live.py` is
  excluded from this pass — it is mid-migration in a parallel change.)
- **`with_semantic_tools` / `SemanticToolSelector` / `SemanticToolsDecorator`
  now default `score_threshold` to `0.0`, not `0.5`**, matching every
  framework adapter in `agent_gantry.integrations.frameworks` (which already
  documented and used a `0.0` default to avoid silently dropping every tool on
  a non-trivial query). The raw `ToolQuery` schema default remains `0.5` for
  backward compatibility — see the note on
  `agent_gantry.schema.query.ToolQuery.score_threshold`.
- **`BaseFrameworkAdapter.select` gained explicit `score_threshold`,
  `namespaces`, and `tools_already_used` keyword parameters** instead of
  swallowing them in `**select_kwargs`. Previously `namespaces` was a
  discoverable, first-class kwarg only on `OpenAIAgentsAdapter`'s live
  methods; it (and the other two) are now explicit and documented on every
  adapter's `select`. `SemanticKernelAdapter.select` keeps its extra
  `plugin_name` kwarg alongside the same three. This is additive
  (keyword-only) and does not change behavior for existing callers.
- **`fetch_framework_tools`'s `framework` parameter now accepts every native
  adapter name**, not just `langgraph`, `semantic-kernel`, `crew_ai`,
  `google_adk`, `strands`, and `agent_framework` (fixes #101). It now covers
  `langchain`, `llamaindex`, `crewai`, `autogen`, `semantic_kernel`, `agno`,
  `haystack`, `pydantic_ai`, `openai_agents`, and `smolagents` too, matching
  the native per-framework adapter module names. The legacy spellings
  `crew_ai` and `semantic-kernel` are still accepted and normalized
  internally to `crewai` / `semantic_kernel`.
- **LangGraph live tool provider migrated off the deprecated `create_react_agent`.**
  `agent_gantry.integrations.frameworks.langgraph_live` now builds the per-turn
  live agent with `langchain.agents.create_agent` (the documented replacement;
  `langgraph.prebuilt.create_react_agent` is removed outright in LangGraph 2.0).
  Per-turn tool re-selection — the ability for Gantry to rebind a different tool
  subset to the model on every conversation turn — moved from the old
  dynamic-`model` callable to a `wrap_model_call` `AgentMiddleware` hook (the
  same mechanism `langchain.agents.middleware.LLMToolSelectorMiddleware` uses),
  with identical externally-observable behavior. No fallback to the deprecated
  API is kept: this project's floors (`langchain>=1.3.4`, `langgraph>=1.2.4`,
  pinned together in the `agent-frameworks` extra) already guarantee
  `langchain.agents.create_agent` is available. `LangGraphAdapter.react_agent` /
  `areact_agent` / `select_for_state` are unaffected — this is purely an
  internal implementation change, aside from `**agent_kwargs` now being
  forwarded to `create_agent` (e.g. use `system_prompt=` instead of the old
  `create_react_agent`'s `prompt=`).

### Documentation

- **Dedicated runnable examples for the 5 native-tool-adapter frameworks that
  had none.** `examples/agent_frameworks/{agno,haystack,pydantic_ai,
  openai_agents,smolagents}_example.py` each register a small tool set,
  select + convert through the framework's `*Adapter` (no hard-coded tool
  names), and exercise both the static tier (`select` → native tool objects)
  and the deep per-call/per-turn tier (`agent_builder` / `toolset` /
  `run_hooks` + `refresh` / `live_tools` + `tool_invoker_builder`). All five
  degrade gracefully with a clear `pip install` hint when the framework isn't
  installed; the Pydantic AI example runs end-to-end offline via
  `pydantic_ai.models.test.TestModel`, and the others gate their live
  agent/model run behind `OPENAI_API_KEY`. `examples/agent_frameworks/
  README.md` documents all five.

### Fixed

- **Incremental sync on the default in-memory store never worked.**
  `InMemoryVectorStore.add_tools` stored `tool.content_hash` while
  `SyncManager.detect_changes` compares `compute_tool_fingerprint()` output
  (`"v1.0:<hash>"` over a different field set), so the comparison never
  matched and every `sync()` re-embedded the full registry. The store now
  persists the fingerprint format the sync manager actually checks; a repeat
  `sync()` with unchanged tools is a true no-op (regression-tested).
- **Rate-limiter concurrency-slot leak in the execution engine.** After a
  successful `RateLimiter.acquire()`, the early-return paths (argument
  validation failure, A2A dispatch, missing handler, confirmation-required)
  returned without releasing the slot, permanently consuming
  `max_concurrent` capacity — repeated invalid-argument calls (an LLM
  hallucinating parameters) would eventually brick the tool with
  "Concurrent execution limit exceeded". Every path past acquire now
  releases in a `finally`.
- **LanceDB + MMR diversity crashed.** `LanceDBVectorStore.search` ignored
  `include_embeddings=True` (returning 2-tuples with a warning), which the
  router unpacks as 3-tuples whenever `diversity_factor > 0` → `ValueError`.
  The search now returns the stored vector (it was already in the fetched
  row), which also removes the router's re-embed fallback for MMR.
- `MCPManager.add_server` returned a `(count, tools)` tuple while annotated
  `-> int`; the annotation now matches the return value.
- **BREAKING (bugfix) — a broken `gantry.retrieve()` mid-conversation no
  longer crashes six of the eight per-turn live providers.**
  `integrations/frameworks/{autogen_live,langgraph_live,llamaindex_live,
  openai_agents_live,pydantic_ai_live,semantic_kernel_live}.py` previously let
  a selection failure propagate straight out of the framework's own
  turn-driving hook (`Workbench.list_tools`, the `create_agent` tool-selection
  middleware, `ObjectRetriever.aretrieve`, `RunHooks.on_llm_start`,
  `AbstractToolset.get_tools`, `GantryFunctionProvider.refresh`) — a transient
  vector-store hiccup could kill the entire agent run. They now catch the
  failure, log a `WARNING` with `exc_info=True`, and degrade gracefully
  instead: to "no tools this turn" for the four stateless per-turn providers
  (LangGraph, LlamaIndex, and — already-existing behavior — Google ADK) or to
  "leave the previous turn's tools in place" for the stateful ones (AutoGen,
  Pydantic AI, OpenAI Agents SDK, Semantic Kernel), matching the precedent
  already set by Google ADK's `before_model_callback` and Strands'
  `BeforeModelCallEvent` hook (both of which already degraded gracefully and
  are unchanged in behavior, only normalized from `logger.exception`/ERROR to
  `logger.warning(..., exc_info=True)` for consistency with the other six).
  Callers who relied on a selection failure raising out of one of these six
  providers (e.g. to abort a run) must now check the `WARNING` log or wrap
  `gantry.retrieve()`/the vector store itself instead.

## [0.9.0] - 2026-06-16

### Removed

- **BREAKING — `ToolDefinition.to_openai_schema()`, `to_anthropic_schema()`, and
  `to_gemini_schema()` removed.** These were thin deprecated shims over
  `to_dialect()` with no internal or example callers. Use
  `ToolDefinition.to_dialect("openai" | "anthropic" | "gemini", ...)` instead.
  (`Skill` / `SkillRegistry.to_anthropic_schema` is a separate, actively-used
  method and is unchanged.)

### Fixed

- **MCP initialisation no longer masks real failures.** `AgentGantry` now
  separates the *import* guard from the *construction* guard: an expected-absent
  MCP install is logged at DEBUG and degrades silently to no-MCP, while an
  unexpected construction failure (e.g. a broken/partial `mcp` install) is logged
  at WARNING with a traceback instead of being swallowed at DEBUG. Either way
  `AgentGantry()` still constructs successfully.
- **`GantryContextProvider` is a class again.** It is now a thin class whose
  `__new__` delegates to a cached implementation class, so it remains valid in
  type annotations and `isinstance()` checks return `False` rather than raising
  `TypeError`. The `score_threshold` property is now typed `float | str` to match
  the config (relative-threshold strings are valid).

### Changed

- **Internal refactors with no public-API impact.** `AgentGantry.__init__` was
  decomposed into focused builders and the adapter/embedder factories extracted to
  `agent_gantry/core/factories.py`; a shared `BaseFrameworkAdapter` removes the
  per-framework boilerplate across the framework adapters; the Agent Framework
  provider and tool bridge were split into smaller helpers with the implementation
  class cached via `functools.cache`; and the schema layer now shares a single
  newline validator and a common health-metric base.

### Documentation

- Migrated the documentation site from Jekyll to an Astro build, added
  per-framework guide pages, and added an Agent Framework TUI example.

## [0.8.0] - 2026-06-15

### Added

- **Built-in console trace middleware for the Agent Framework provider.**
  `GantryContextProvider.trace()` returns an AF *function* middleware that prints
  a readable per-round line — `>>> round N: tool(args)  [surfaced: name:score, …]`
  then `<<< round N: tool -> <result preview>` — using `last_selection` for the
  surfaced set and `render_result` for the preview. `provider.attach_to(agent,
  trace=True)` wires it (and the per-call retrieval middleware) in one call. This
  replaces hand-rolled `@function_middleware` trace glue.
- **`agent_gantry.render_result(result, *, limit=None, collapse_whitespace=False)`**
  — a framework-agnostic helper that renders any tool result (including Agent
  Framework `Content`-block lists, bytes, dicts, and arbitrary objects) to
  readable text for logs and traces.
- **Per-round retrieval history.** `GantryContextProvider.selections` exposes the
  bounded sequence of `RetrievalDecision`s (oldest first), so callers can
  correlate *what was surfaced* with *what the model called* across the whole run
  rather than only the latest round. `last_selection` remains the single
  most-recent slot.
- **Framework-agnostic tool-call event hook.** `AgentGantry.on_tool_call(callback)`
  registers a listener (sync or async) fired with a `ToolCallEvent(call, result)`
  after every `gantry.execute` — and once per call in `execute_batch`. Because
  `execute` is the single choke point every framework adapter flows through, one
  registration yields logging/metrics across all of them. Callbacks are
  error-isolated (a raising listener never breaks the tool run) and the method
  returns an unsubscribe callable. `ToolCallEvent` is exported from `agent_gantry`.
- **`agent_gantry.enable_console_logging(level=logging.INFO)`** — explicit opt-in
  that attaches a console handler (once) and sets the `agent_gantry` logger level,
  replacing the implicit handler/level side effect that `ConsoleTelemetryAdapter`
  used to perform on construction.

### Changed

- **Logging hygiene (behaviour change).** Importing `agent_gantry` now attaches a
  `logging.NullHandler` to the package logger, and `ConsoleTelemetryAdapter` no
  longer adds a handler or raises the logger level as a side effect of
  construction. A default `AgentGantry()` therefore no longer emits INFO
  "Span started" / "Tool execution" lines or clobbers the root log level —
  telemetry records simply propagate to whatever logging the application
  configured (a `NullHandler` swallows them if none). Opt back into console output
  with `agent_gantry.enable_console_logging()`, or construct
  `ConsoleTelemetryAdapter(attach_handler=True)` for the old direct-construction
  convenience.

## [0.7.0] - 2026-06-15

### Changed

- **BREAKING — framework & LLM integrations are now one class per integration.**
  The free functions (`for_<framework>()`, `spec_to_<framework>()`) and the
  assorted live helpers (`gantry_workbench`, `gantry_toolset`,
  `gantry_tool_retriever`, `gantry_function_agent`, `create_gantry_react_agent`,
  `acreate_gantry_react_agent`, `select_tools_for_state`, `gantry_plugin`,
  `refresh_kernel_tools`, `register_with_autogen`, `gantry_before_model_callback`,
  `gantry_adk_agent`, `gantry_run_hooks`, `run_with_gantry`, `refresh_agent_tools`,
  `select_function_tools`, `gantry_crew_tools`, `gantry_haystack_tools`) were
  **removed** in favour of a single `<Framework>Adapter` class per framework,
  imported from the same clean namespace
  (`from agent_gantry.langchain import LangChainAdapter`). Each adapter exposes
  `await adapter.select(query, limit=...)` (was `for_<fw>`), the
  `adapter.convert(spec)` staticmethod (was `spec_to_<fw>`), and that framework's
  deep per-turn live capability as methods — e.g.
  `GoogleADKAdapter(gantry).agent(...)` / `.before_model_callback(...)`,
  `LlamaIndexAdapter(gantry).function_agent(llm)` / `.tool_retriever()`,
  `AutoGenAdapter(gantry).workbench()` / `.register(...)`,
  `PydanticAIAdapter(gantry).toolset()`,
  `LangGraphAdapter(gantry).react_agent(model)` / `.areact_agent(model)`,
  `OpenAIAgentsAdapter(gantry).run(...)` / `.session(...)` / `.run_hooks(...)`,
  `SemanticKernelAdapter(gantry).plugin(...)` / `.function_provider(kernel)`,
  and `CrewAIAdapter(gantry).agent_builder(...)` / `.live_tools(...)` for the
  fixed-tool frameworks (CrewAI/Agno/Haystack/Smolagents).
- **Microsoft Agent Framework gains a unified `AgentFrameworkAdapter`**
  (`from agent_gantry.agent_framework import AgentFrameworkAdapter`) whose methods
  build the `GantryContextProvider` (`.context_provider(...)`), `GantryToolBridge`
  (`.tool_bridge(...)`), and the approval / observability / tool-choice middleware.
  The underlying classes remain importable as the returned types.

### Added

- **One-class LLM SDK adapters** — `OpenAIAdapter`, `AnthropicAdapter`,
  `GeminiAdapter`, `GroqAdapter`, `VertexAIAdapter`, `MistralAdapter`
  (e.g. `from agent_gantry.openai import OpenAIAdapter`). `await adapter.tools(query,
  limit=...)` returns tool schemas in that provider's dialect (equivalent to
  `gantry.retrieve_tools(query, dialect="...")`); `OpenAIAdapter.responses_tools(...)`
  emits the OpenAI Responses API shape.

### Removed

- The internal `agent_gantry._framework_ns` lazy-namespace helper — folded into the
  adapter classes, whose methods import their third-party framework lazily on use,
  so `import agent_gantry` (and `import agent_gantry.<framework>`) stays
  dependency-free.

## [0.6.0] - 2026-06-15

### Changed

- **Bundled Claude Skill refreshed for the v0.5.0+ API surface** (`agent_gantry/skills/agent-gantry/SKILL.md`):
  rewrote the framework integration guidance around the native `for_<framework>`
  adapters and clean per-framework import namespaces (`from agent_gantry.langchain
  import for_langchain`), added the five frameworks introduced in 0.5.0 (Pydantic AI,
  OpenAI Agents SDK, Smolagents, Haystack, Agno), documented the deep per-turn "live"
  providers and the `ToolRefresher` multi-turn API, and corrected the stale
  `fetch_framework_tools` examples (the previous `framework="langchain"/"autogen"/
  "llamaindex"` names were never valid and now point at the schema-only adapter's real
  name set). Frontmatter still follows the Anthropic Agent Skills format
  (`name`/`description` only). Skill trigger description expanded to the new frameworks.

- **Install/usage instructions now lead with uv** (with pip kept as an explicit
  fallback) across the bundled skill, `README.md`, and the `skill_path()` error hint —
  `uv add "agent-gantry[...]"` for dependencies and `uv run agent-gantry ...` for the CLI.

### Added

- **`agent-gantry install-skill --claude`** installs the bundled skill straight into
  `~/.claude/skills` so Claude Code discovers it with no further wiring (previously the
  command only supported `--target <dir>`).

## [0.5.0] - 2026-06-14

### Added

- **Native tool adapters for 12 agent frameworks** (`agent_gantry.integrations.frameworks`):
  LangChain, LangGraph, LlamaIndex, CrewAI, Pydantic AI, OpenAI Agents SDK,
  Smolagents, Haystack, Agno, AutoGen, Semantic Kernel, Google ADK. Each
  `for_<fw>()` / `spec_to_<fw>()` builds the framework's native tool object and
  routes execution through `gantry.execute`.
- **Deep per-turn "live" providers** that re-select tools every turn via each
  framework's own dynamic-tool hook — matching the Microsoft Agent Framework
  `GantryContextProvider` depth: LlamaIndex `tool_retriever`, Pydantic AI
  `AbstractToolset`, AutoGen `Workbench`, Google ADK `before_model_callback`,
  LangGraph dynamic model, Semantic Kernel plugin refresh, OpenAI Agents
  `RunHooks`. Best-effort per-call live wrappers for CrewAI/Agno/Haystack/Smolagents.
- **`ToolRefresher`** — framework-agnostic multi-turn re-selection (recency-aware;
  serves autonomous tool pipelines and conversational agents).
- **Clean per-framework import namespaces**: `from agent_gantry.<framework> import …`
  (e.g. `from agent_gantry.langchain import for_langchain`).
- **`GantryToolset` / `ToolSpec`** shared adapter base.
- NumPy-vectorized `InMemoryVectorStore.search` (~36–59× faster at 50–1000 tools).
- Selection/multi-turn benchmarks and a real-package adapter CI job.
- Automated release-on-merge-to-main workflow (build → publish to PyPI → tag → GitHub Release).

### Fixed

- `*args`/`**kwargs` tools are no longer emitted as required schema params.
- `retrieve_tools` / `search_and_execute` / `fetch_framework_tools` default
  `score_threshold` to `0.0` (was `0.5`, which silently dropped correct tools).
  **Migration note:** pass `score_threshold=0.5` explicitly to keep the previous
  filtering behaviour — the new default surfaces more candidate tools.
- Dynamic MCP server selection documented accurately (it is functional).

### Changed (2026-06-08 modernisation audit)

- **`pyproject.toml`: bump `anthropic` floor `>=0.105.2 → >=0.107.1`** — three new
  releases since the 2026-06-05 audit.  0.106.0 formally marks `claude-opus-4-1` as
  deprecated in the SDK (retiring 2026-08-05) and fixes Foundry client methods and a
  schema `$ref`/`$defs` transform bug.  0.107.0 adds minor Managed Agents type updates.
  0.107.1 fixes Foundry x-api-key header authentication.  No breaking changes to the
  Messages API or tool-use surfaces used by Gantry across 0.105.2→0.107.1.
  `uv.lock` regenerated: `anthropic 0.105.2 → 0.107.1`.
  *Risk: safe internal — floor bump only.*
  Source: https://pypi.org/pypi/anthropic/json (verified 2026-06-08)
          https://github.com/anthropics/anthropic-sdk-python/releases (verified 2026-06-08)

- **`agent_gantry/integrations/anthropic_features.py`**: replaced vague "earlier Claude
  4 models" in three docstring locations with an explicit model list:
  `claude-opus-4-5`, `claude-sonnet-4-5`, `claude-opus-4-1` (deprecated, retiring
  2026-08-05). Added retirement notice for `claude-sonnet-4` and `claude-opus-4`, which
  were retired on **2026-06-15**. Neither retiring model ID appears in Gantry source or
  examples. Updated the interleaved-thinking beta header comment to remove reference to
  the now-retired "earlier Claude 4 models".
  *Risk: safe internal — documentation only.*
  Source: https://platform.claude.com/docs/en/docs/about-claude/models/overview (verified 2026-06-08)

- **`docs/reference/llm_sdk_compatibility.md`**: updated three install-pin examples:
  `anthropic>=0.101.0 → >=0.107.1`, `openai>=2.40.0 → >=2.41.0` (two occurrences),
  `groq>=1.2.0 → >=1.4.0`. These now match the floors declared in `pyproject.toml`.
  *Risk: safe internal — documentation only.*

### Deprecation notices

- `agent_gantry/schema/config.py` default `"gpt-4o-mini"` must be migrated to
  `"gpt-5.4-mini"` before OpenAI's 2026-10-23 shutdown. This is a **breaking change**
  requiring a major version bump and is tracked in AUDIT.md §10.
- `claude-opus-4-1` (`claude-opus-4-1-20250805`) was marked deprecated in the Anthropic
  SDK (0.106.0, 2026-06-05); retirement date **2026-08-05**. Not referenced in Gantry
  source or examples — no code action required.

### Changed (2026-06-03 modernisation audit)

- **`pyproject.toml`: bump `langchain` floor `>=1.3.2 → >=1.3.4`** — patch release;
  no API changes to `ChatOpenAI`, `ChatAnthropic`, or `BaseTool` surfaces used by
  Gantry's `framework_adapters.py`.
  *Risk: safe internal — floor bump only.*
  Source: https://pypi.org/pypi/langchain/json

- **`pyproject.toml`: bump `langgraph` floor `>=1.2.2 → >=1.2.4`** — patch release;
  graph checkpoint and serialisation fixes; `StateGraph`, `CompiledGraph`, and
  interrupt/resume APIs unchanged. `langgraph-sdk` resolves to `0.4.2` (was `0.3.13`)
  as a transitive consequence — not directly imported by Gantry.
  *Risk: safe internal — floor bump only.*
  Source: https://pypi.org/pypi/langgraph/json

- **`pyproject.toml`: bump `cohere` floor `>=6.0.0 → >=7.0.3`** — cohere 7.0.0 is a
  major release; the only breaking change is raising the minimum Python version from
  `^3.8` to `^3.10`. Gantry already requires `python>=3.10`, so no user-facing change.
  `AsyncClientV2.rerank()` signature and return type are unchanged.
  *Risk: safe with shim (Python constraint already satisfied).*
  Source: https://github.com/cohere-ai/cohere-python/releases;
          https://docs.cohere.com/v2/reference/rerank

- **`docs/reference/llm_sdk_compatibility.md`**: replace discontinued
  `gpt-4o-realtime-preview` with `gpt-realtime-1.5`. OpenAI discontinued
  `gpt-4o-realtime-preview` on 2026-05-07; `gpt-realtime-1.5` is the current
  production realtime model.
  *Risk: safe internal (documentation only).*
  Source: https://developers.openai.com/api/docs/deprecations

- **All example files and documentation**: replace `gpt-4o` → `gpt-5.5` and
  `gpt-4o-mini` → `gpt-5.4-mini` (39 occurrences across 23 files). OpenAI has set a
  shutdown date of **2026-10-23** for both deprecated models; replacements are the
  GPT-5.x generation flagship models.
  Files updated: `README.md`, `agent_gantry/README.md`, `agent_gantry/core/README.md`,
  `agent_gantry/integrations/README.md`, `agent_gantry/schema/config.py` (comment
  only), `agent_gantry/skills/agent-gantry/SKILL.md`,
  `docs/reference/llm_sdk_compatibility.md`, `examples/fast_track_demo.py`,
  `examples/llm_integration/llm_demo.py`, `examples/llm_integration/openai_demo.py`,
  `examples/llm_integration/multi_turn_conversation.py`,
  `examples/llm_integration/token_savings_demo.py`,
  `examples/llm_intent_classification_example.py`,
  `examples/observability/multi_provider_metrics_demo.py`,
  `examples/observability/token_savings_demo.py`, `examples/project_demo/main.py`,
  `examples/project_demo/main_persistent.py`,
  `examples/testing_limits/real_world_30_tools_test.py`,
  `examples/tool_vector_db/main.py`, `examples/tool_vector_db/README.md`,
  `examples/agent_frameworks/autogen_example.py`,
  `examples/agent_frameworks/crewai_example.py`,
  `examples/agent_frameworks/langchain_example.py`,
  `examples/agent_frameworks/langgraph_example.py`,
  `examples/agent_frameworks/llamaindex_example.py`,
  `examples/agent_frameworks/semantic_kernel_example.py`.
  *Risk: safe internal — examples and documentation only.*

### Deprecation notice (action required by 2026-10-23)

- `agent_gantry/schema/config.py` default `"gpt-4o-mini"` must be migrated to
  `"gpt-5.4-mini"` before OpenAI's 2026-10-23 shutdown. This is a **breaking change**
  requiring a major version bump and is tracked in AUDIT.md §10.

### Added

- **`disable_af_instrumentation()` helper** — new top-level function
  (`from agent_gantry import disable_af_instrumentation`) that calls
  `agent_framework.telemetry.disable_instrumentation()` when AF >=1.6.0 is
  installed. Required for concurrent `asyncio.gather()` / `TaskGroup` workflows
  on AF 1.6.0 (which defaults to ContextVar-based instrumentation that crashes
  when tokens are reset across child asyncio contexts). No-op on AF <1.6.0 or
  when AF is not installed. Returns `True` if instrumentation was disabled,
  `False` if it was not applicable.
  *Risk: safe additive — existing callers unaffected.*
  Source: https://pypi.org/pypi/agent-framework/json (1.6.0 release notes)

- **`GantryToolBridge(disable_af_instrumentation=True)`** — new optional
  constructor parameter that applies the shim automatically at bridge
  construction time. Useful when the bridge is constructed near the point where
  concurrent agents are built. Defaults to `False`.
  *Risk: safe additive — new keyword arg with a `False` default.*

### Fixed

- **`AnthropicClient.create_message()` no longer sends `tools=[]`** when no
  tools are retrieved. Previously an empty list was always passed, which causes
  the Anthropic API to inject the tool-use system prompt even with no tools
  (adding ≈346 extra input tokens for Claude 4 models). The `tools` key is now
  omitted entirely when the retrieved list is empty, preserving existing
  behaviour for non-empty lists.
  *Risk: safe fix — only changes the API payload when `tools` would have been
  `[]`; all callers that rely on non-empty tool lists are unaffected.*
  Source: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
          (pricing table — tool-use system-prompt token overhead)

### Fixed (prior)

- **`AnthropicAdapter.to_provider_schema(strict=True)` now auto-injects
  `additionalProperties: false`** into the emitted `input_schema`. Anthropic's
  strict tool-use mode requires this field to activate grammar-constrained
  sampling; previously the requirement was documented but not enforced, so
  users who omitted it from their JSON Schema would silently get non-strict
  behaviour. The original `ToolDefinition.parameters_schema` is never mutated
  (a shallow copy is made). No change for `strict=False` (default).
  *Risk: safe additive — only affects callers who explicitly pass `strict=True`;
  the injected key is additive and may suppress an implicit `true` default on
  very old schemas (check with `jsonschema` lint if concerned).*
  Source: https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use

### Changed

- **`agent-framework` range updated to `>=1.5.0,<2.0.0`** — upper bound relaxed
  from `<1.6.0`. AF 1.6.0 (released 2026-05-22) introduces instrumentation
  enabled by default using `asyncio.ContextVar` tokens, which triggers a hard
  `ValueError` when two `Agent.run()` calls are awaited concurrently via
  `asyncio.gather()` or `TaskGroup`. Sequential workflows (``WorkflowAgent``,
  `SequentialBuilder`, `HandoffBuilder`) are **not** affected. A Gantry
  compatibility shim is now provided:
  ```python
  from agent_gantry import disable_af_instrumentation
  disable_af_instrumentation()   # call once at startup for concurrent workflows
  ```
  Or pass ``GantryToolBridge(gantry, disable_af_instrumentation=True)`` to
  apply it automatically. See the ``disable_af_instrumentation`` entry in
  the Added section above.
  *Risk: safe — upper bound relaxed; the shim is opt-in and a no-op on AF <1.6.0.*
  Source: https://pypi.org/pypi/agent-framework/json
          https://github.com/microsoft/agent-framework/releases (1.6.0 notes)

- **`openai` floor bumped to `>=2.38.0`** (was `>=2.37.0`). Released 2026-05-21;
  adds `service_tier` parameter to `responses compact`, eager pydantic iterator
  validation, and workload-identity auth cleanup. No breaking changes. Floor
  updated in the `openai`, `mistral`, and `llm-providers` extras.
  *Risk: safe internal — floor bump only.*
  Source: https://pypi.org/pypi/openai/json

- **`anthropic` floor bumped to `>=0.104.1`** (was `>=0.103.1`). 0.104.0 added
  thinking-token-count beta in streaming; 0.104.1 (released 2026-05-22) patches a
  bug where `encrypted_content` was not carried through the beta compaction
  accumulator. No breaking changes; no Gantry code changes required.
  *Risk: safe internal — floor bump only.*
  Source: https://pypi.org/pypi/anthropic/json

- **`langgraph` floor bumped to `>=1.2.1`** (was `>=1.2.0`). Patch release;
  no API changes.
  *Risk: safe internal — floor bump only.*
  Source: https://pypi.org/pypi/langgraph/json

- **`pyproject.toml` held-package comments updated** to reflect latest stable
  versions: `google-genai` 2.6.0 (was 2.5.0), `google-adk` 2.1.0 (was 1.34.0,
  major version bump — new Workflow Runtime and Task API; conflicts with
  `langgraph>=1.2.1` prevent upgrade in the combined `agent-frameworks` extra).
  Floor pins are unchanged; notes updated to document the google-adk 2.x conflict
  and the standalone install path.
  *Risk: safe internal — documentation only.*
  Source: https://pypi.org/pypi/google-genai/json, https://pypi.org/pypi/google-adk/json

- **`examples/llm_integration/google_genai_demo.py`**: Extended the function-call
  scenario to show the full tool-result round-trip using the SDK-idiomatic
  `types.Part.from_function_response()` helper with `id` forwarding for parallel
  call correlation. Adds a follow-up `generate_content` call that includes the
  model's function-call turn and the tool result so Gemini can compose a final
  text answer.
  *Risk: safe — example-only change.*
  Source: https://ai.google.dev/gemini-api/docs/function-calling

- **`GantryObservabilityMiddleware` docstring** updated with an AF 1.6.0
  double-instrumentation note explaining the interaction with AF's new
  default-enabled OTel spans and how to suppress them when a single span
  source is preferred.
  *Risk: safe internal — documentation only.*

- **`llama-index-core` floor bumped to `>=0.14.22`** (was `>=0.14.21`). Patch
  release; no API changes.
  *Risk: safe internal — floor bump only.*
  Source: https://pypi.org/pypi/llama-index-core/json

## [0.4.0+2026-05-16] — pre-release patch (CHANGELOG entry retroactively named)

### Added

- **`AnthropicClient.create_message()` now accepts `output_schema`** — an optional
  JSON Schema dict that constrains Claude's response to a specific JSON structure
  via the `output_config.format` parameter introduced in the Anthropic Messages API.
  When supplied the dict is injected as
  `output_config={"format":{"type":"json_schema","schema":{...}}}` unless the caller
  already provides their own `output_config` (caller wins). Default `None` preserves
  all existing behaviour — no changes required for existing callers.
  *Risk: safe additive.*
  Source: https://platform.claude.com/docs/en/build-with-claude/structured-outputs

### Changed

- **`agent-framework` floor bumped to `>=1.4.0,<2.0.0`** (was `>=1.3.0,<2.0.0`).
  Agent Framework 1.4.0 was released 2026-05-15. Breaking changes in 1.4.0 are
  confined to the experimental skills API (file-skill folder discovery aligns with
  agentskills.io spec; skill metadata extracted into `SkillFrontmatter`) — neither
  is used by Gantry. New features include MCP tool-call metadata forwarding,
  `list[str]` support in file skills, and AG-UI tool-result display channel.
  *Risk: safe internal — floor bump only; no Gantry code changes required.*
  Source: https://pypi.org/pypi/agent-framework/json, https://github.com/microsoft/agent-framework/releases

- **`openai` floor bumped to `>=2.37.0`** (was `>=2.36.0`). Current stable release;
  no API surface changes for Gantry's Responses API or Chat Completions call sites.
  Floor updated in the `openai`, `mistral`, and `llm-providers` extras.
  *Risk: safe internal — floor bump only.*
  Source: https://pypi.org/pypi/openai/json

- **`langchain` floor bumped to `>=1.3.1`** (was `>=1.3.0`). Patch release; no API
  changes.
  *Risk: safe internal — floor bump only.*
  Source: https://pypi.org/pypi/langchain/json

## [0.4.0] - 2026-05-15

### Added

- **`RetrievalDecision` introspection** on `GantryContextProvider` and
  `GantryToolBridge.get_tools_with_decision`. Carries the ranked candidate
  list (kept/dropped), the injected tools, and the effective threshold.
  Exposed on the provider as `provider.last_selection`. The decision is
  attached to the new `gantry.bridge_retrieval` telemetry span as
  structured attributes. Pair with `verbose=True` on the provider for
  a one-line INFO summary per round.
- **`provider.dry_run_retrieve(query)`**: officially supported diagnostic
  that uses the *exact same* code path as the live middleware. Use to
  validate "would the LLM see X?" without spinning up an agent.
- **Relative score thresholds**: `score_threshold="relative:0.8"` retains
  any candidate within 80% of the top score. Length-robust where absolute
  cosine cutoffs collapse with long pipeline-style queries.
- **`static_tools=[...]` on `GantryContextProvider`**: pin AF-native tools
  that live *outside* the gantry registry into every round's surface.
- **`provider.attach_to(agent)` helper**: appends the provider and
  (when in `per_call`) the chat middleware in one call.
- **`agent_gantry.query.keyword_focused`** and **`truncated`**: drop-in
  query generators that strip imperative scaffolding and cap query
  length respectively. Mitigate the long-query degradation pattern.
- **`GantryToolChoiceMiddleware`**: AF chat middleware that re-derives
  `tool_choice` per round from a user-supplied callable. Enables the
  "force tool calls for N rounds, then text on summarisation" pattern.
- **Registry linter**: `gantry.analyze_registry()` and
  `gantry.pairwise_similarity()` Python APIs flag tool descriptions that
  cross-reference other tools, pairs whose searchable text is too
  similar, and tags with low discriminative value. Exposed via the CLI
  as `gantry lint` and `gantry sim toolA toolB`.
- **`gantry sync --dry-run` CLI**: reports which tools would be
  (re-)embedded and why, without invoking the embedder.
- **`CachedEmbedder`** (`agent_gantry.adapters.embedders.cached`): wraps
  any embedder with a disk-backed SQLite cache keyed by embedder_id and
  text hash. Eliminates re-embedding spend across cold starts. Default
  cache path `~/.cache/agent_gantry/embeddings.sqlite`. Dedups duplicate
  strings within a batch so the underlying embedder is never called
  twice for the same text. SQLite I/O is offloaded to a thread so it
  doesn't block the event loop.
- **Bundled Claude Skill** at `agent_gantry/skills/agent-gantry/SKILL.md`,
  shipped in the wheel and discoverable via `from agent_gantry.skills
  import skill_path`. Install into a project's skills directory with
  `agent-gantry install-skill --target ./skills`. The skill also
  publishes under the standard `share/claude/skills/agent-gantry/`
  wheel data path so Claude Code can find it automatically.
- **`gantry.embedder`** public property — sibling modules should use
  this instead of reaching into `gantry._embedder`.

### Changed

- **`agent-framework` floor bumped to `>=1.4.0`** (was `>=1.3.0`). Agent Framework
  1.4.0 released 2026-05-15. Changes in 1.4.0: MCP tool-call metadata forwarding,
  path-traversal fix in checkpoint storage, A2A SDK v1.0 alignment. Two breaking
  changes in 1.4.0 that do NOT affect Gantry: (1) SkillFrontmatter extraction in the
  experimental file-based skills API; (2) DevUI CORS tightening. Lock file updated:
  `agent-framework` and `agent-framework-core` both moved from 1.3.0 → 1.4.0.
  *Risk: safe internal — no changes to Agent, WorkflowBuilder, ContextProvider,
  FunctionMiddleware, or any other API surface that Gantry consumes.*
  Source: https://pypi.org/pypi/agent-framework/json

- **`google-adk` floor stays at `>=1.14.1`** — upgrade to 1.33.0 blocked by two
  independent conflicts. (1) **langgraph** (primary blocker): google-adk 1.33.0
  requires `langgraph<0.4.8`, which is mutually exclusive with `langgraph>=1.2.0` in
  the `agent-frameworks` extra; no floor bump resolves this. (2) **pydantic**
  (partially resolved): google-adk 1.33.0 requires `pydantic>=2.12`; semantic-kernel
  1.42.0 has relaxed its upper bound to `<2.14`, but the langgraph conflict still
  blocks co-installation regardless. `pyproject.toml` comment updated to document
  both blockers. To use google-adk 1.33.0, install it in a standalone environment
  without LangChain/LangGraph.
  *Risk: safe internal — comment only, floor unchanged.*
  Source: https://pypi.org/pypi/google-adk/1.33.0/json

- **`semantic-kernel` comment updated** — 1.42.0 is the current stable release.
  1.42.0 relaxes the pydantic upper bound from `<2.12` to `<2.14`, resolving the
  pydantic conflict with google-adk 1.33.0 as far as semantic-kernel is concerned.
  However, (a) the opentelemetry-api conflict with agent-framework on some Python
  versions remains unresolved, keeping the floor at `>=1.36.0`; and (b) google-adk
  1.33.0 has an independent langgraph<0.4.8 conflict that blocks it regardless.
  Floor stays at `>=1.36.0` until opentelemetry conflict is confirmed resolved.
  *Risk: safe internal — comment only, floor unchanged.*
  Source: https://pypi.org/pypi/semantic-kernel/1.42.0/json

- **`google-genai` comment updated** — latest stable is now 2.3.0 (was 2.2.0).
  No change to floor or installation instructions.
  *Risk: safe internal — comment only.*
  Source: https://pypi.org/pypi/google-genai/json

- **`crewai` comment updated** — latest stable is 1.14.4. The co-installation
  conflict with `agent-framework` via `opentelemetry-api` version incompatibility
  is documented. Floor stays at `>=1.6.1`.
  *Risk: safe internal — comment only.*

- **LangChain `tool` import migrated to `langchain_core`** in
  `examples/agent_frameworks/langchain_example.py` (this PR) and
  `examples/agent_frameworks/langgraph_example.py` (prior commit). Both now use
  `from langchain_core.tools import tool` at module level. The `langchain.tools`
  shim may be removed in a future LangChain 1.x minor release.
  *Risk: safe with compatibility shim.*
  Source: https://python.langchain.com/docs/concepts/tools/

- **`examples/agent_frameworks/agent_framework_example.py`** and
  **`agent_framework_orchestration_example.py`**: Version reference in docstring
  updated from 1.3.0 to 1.4.0 to match the bumped `agent-framework` floor.
  *Risk: safe internal — documentation only.*

- **Default `score_threshold` on `GantryContextProvider` and
  `GantryToolBridge` lowered from `0.3` to `0.0`** (no filtering).
  Long queries dilute absolute cosine similarities, so the previous
  default silently filtered relevant tools on multi-step pipelines.
  Filtering is now opt-in. Pair with `score_threshold="relative:<frac>"`
  for length-robust filtering.
- **`per_call` default query generator** is now
  `fallback_chain(last_tool_result, last_user_text)` (was
  `last_user_text`). `last_user_text` returns the same string every
  round, which silently disabled per-round adaptation. `per_run` still
  defaults to `last_user_text`. Explicitly passing `last_user_text`
  with `query_strategy="per_call"` now logs a WARNING.
- **`OpenAIEmbedder` honours `config.api_base` / `OPENAI_BASE_URL`**.
  Previously hard-coded to OpenAI's endpoint, blocking Requesty,
  OpenRouter, Together, vLLM and other OpenAI-compatible providers.
- **`per_call` without `as_chat_middleware()` attached now warns once**
  on the first `before_run`. The previous behavior silently degraded
  to `per_run` semantics.
- **Threshold-filtered-everything WARNING**: when `score_threshold`
  drops every candidate, the bridge logs a WARNING with the threshold
  (and the resolved cutoff for relative mode) plus the top scores so
  users see "it was the threshold, not relevance".

### Fixed

- Relative `score_threshold` over-fetches the candidate pool by 4× to
  compute the cutoff. The over-fetched limit is now clamped to the
  `ToolQuery.limit` upper bound (50) so callers passing `limit >= 13`
  with a relative threshold no longer hit a Pydantic validation error.
- Relative threshold falls back to `0.0` cutoff when the top score is
  non-positive (defensive — `ScoredTool.semantic_score` is Pydantic
  clamped `>= 0`, but the guard prevents the degenerate "filtered the
  top match" path if a custom embedder ever returned negative scores).
- Registry linter pre-compiles regex patterns once instead of inside
  the nested loop — O(N²) → O(N) compile cost.
- `_default_query_generator` reference removed (the import was renamed
  to `last_user_text` in this release).

- **`anthropic` floor bumped to `>=0.102.0`** (was `>=0.101.0`). Anthropic 0.102.0
  released 2026-05-13; no breaking changes for Gantry's Messages API call sites.
  *Risk: safe internal — floor bump only.*
  Source: https://pypi.org/pypi/anthropic/json

- **`langchain` floor bumped to `>=1.3.0`** (was `>=1.2.18`). LangChain 1.3.0 GA
  released 2026-05-14. This lifts the `langgraph<1.2.0` upper bound that 1.2.18
  imposed. The previous hold comment is removed.
  *Risk: safe internal — floor bump only.*
  Source: https://pypi.org/pypi/langchain/json

- **`langgraph` floor bumped to `>=1.2.0`** (was `>=1.1.10`). Unblocked by
  LangChain 1.3.0 GA. LangGraph 1.2.0 is the current stable release.
  *Risk: safe internal — floor bump only.*
  Source: https://pypi.org/pypi/langgraph/json

## [0.3.0] - 2026-05-13

### Changed

- **`mistralai` dependency removed — replaced by OpenAI SDK for Mistral calls.**
  The `mistralai` package was quarantined on PyPI on 2026-05-12 and is no longer
  installable. Mistral's chat endpoint is OpenAI-compatible. The `openai` SDK with
  `base_url="https://api.mistral.ai/v1"` is the canonical replacement. Changes:
  - `pyproject.toml` `mistral` extra now depends on `openai>=2.36.0` instead of
    `mistralai>=2.0.0`. `mistral` is removed from `llm-providers`.
  - `agent_gantry/adapters/llm_client.py` — `provider="mistral"` now initialises
    `AsyncOpenAI(base_url="https://api.mistral.ai/v1")` and uses
    `chat.completions.create()`.
  - `examples/llm_integration/mistral_demo.py` — fully updated to use the OpenAI
    SDK with Mistral's base URL.
  - `docs/reference/llm_sdk_compatibility.md` and `README.md` — Mistral snippets
    updated accordingly.
  - Transitive orphan packages `eval-type-backport` and `jsonpath-python` removed
    from the lock file.
  *Migration*: Replace `from mistralai import Mistral; async with Mistral(...) as c: ...`
  with `from openai import AsyncOpenAI; c = AsyncOpenAI(api_key=..., base_url="https://api.mistral.ai/v1"); await c.chat.completions.create(...)`.
  `LLMConfig(provider="mistral")` continues to work — migration is internal.
  *Risk: safe with shim — public Mistral integration behaviour preserved.*
- **`anthropic` floor bumped to `>=0.101.0`** (was `>=0.100.0`). Anthropic 0.101.0
  released 2026-05-11; no breaking changes for Gantry's Messages API call sites.
  *Risk: safe internal — floor bump only.*
  Source: https://pypi.org/pypi/anthropic/json
- **`pyproject.toml`** — `langgraph 1.2.0` hold documented. `langchain==1.2.18`
  pins `langgraph<1.2.0`; the floor remains `>=1.1.10` until `langchain 1.3.0 GA`.

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

[Unreleased]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.12.0...HEAD
[0.12.0]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.1.0...v0.1.2
[0.1.0]: https://github.com/CodeHalwell/Agent-Gantry/releases/tag/v0.1.0
