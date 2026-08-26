import pytest

from agent_gantry.core.executor import ExecutionEngine
from agent_gantry.schema.tool import ToolDefinition


class MockToolRegistry:
    pass

@pytest.fixture
def engine():
    return ExecutionEngine(registry=MockToolRegistry())

@pytest.fixture
def sample_tool():
    return ToolDefinition(
        name="test_tool",
        description="A test tool",
        parameters_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "weight": {"type": "number"},
                "is_active": {"type": "boolean"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["name"]
        }
    )

@pytest.mark.asyncio
async def test_validate_arguments_valid(engine, sample_tool):
    """Test valid arguments pass validation."""
    is_valid, error = await engine._validate_arguments(sample_tool, {"name": "Alice"})
    assert is_valid is True
    assert error is None

    is_valid, error = await engine._validate_arguments(
        sample_tool,
        {"name": "Bob", "age": 30, "weight": 75.5, "is_active": True, "tags": ["admin", "user"]}
    )
    assert is_valid is True
    assert error is None

@pytest.mark.asyncio
async def test_validate_arguments_missing_required(engine, sample_tool):
    """Test missing required arguments fail validation."""
    is_valid, error = await engine._validate_arguments(sample_tool, {"age": 25})
    assert is_valid is False
    assert error == "Missing required parameter: name"

@pytest.mark.asyncio
async def test_validate_arguments_extra_parameter(engine, sample_tool):
    """Test extra parameters fail validation when not explicitly permitted."""
    is_valid, error = await engine._validate_arguments(sample_tool, {"name": "Alice", "unknown_field": "test"})
    assert is_valid is False
    assert error == "Unknown parameter: unknown_field"

@pytest.mark.asyncio
async def test_validate_arguments_wrong_types(engine, sample_tool):
    """Test incorrect types fail validation with specific error messages."""
    # String instead of int
    is_valid, error = await engine._validate_arguments(sample_tool, {"name": "Alice", "age": "30"})
    assert is_valid is False
    assert error == "Parameter 'age' must be an integer"

    # Float instead of int
    is_valid, error = await engine._validate_arguments(sample_tool, {"name": "Alice", "age": 30.5})
    assert is_valid is False
    assert error == "Parameter 'age' must be an integer"

    # Int instead of bool (python bool subclasses int, test strictness)
    is_valid, error = await engine._validate_arguments(sample_tool, {"name": "Alice", "is_active": 1})
    assert is_valid is False
    assert error == "Parameter 'is_active' must be a boolean"

    # String instead of array
    is_valid, error = await engine._validate_arguments(sample_tool, {"name": "Alice", "tags": "admin"})
    assert is_valid is False
    assert error == "Parameter 'tags' must be an array"



# --------------------------------------------------------------------------- #
# Schema-aware validation (strict-mode null, additionalProperties, enum, dict)
# --------------------------------------------------------------------------- #
@pytest.fixture
def schema_aware_tool():
    return ToolDefinition(
        name="schema_tool",
        description="Exercises schema-aware validation",
        parameters_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "mode": {"type": "string", "enum": ["fast", "slow"]},
                "note": {"type": ["string", "null"]},
                "meta": {"type": "object"},
                "config": {
                    "type": "object",
                    "properties": {"level": {"type": "integer"}},
                    "additionalProperties": True,
                },
            },
            "required": ["name"],
        },
    )


@pytest.mark.asyncio
async def test_validate_enum_membership(engine, schema_aware_tool):
    is_valid, _ = await engine._validate_arguments(
        schema_aware_tool, {"name": "a", "mode": "fast"}
    )
    assert is_valid is True

    is_valid, error = await engine._validate_arguments(
        schema_aware_tool, {"name": "a", "mode": "warp"}
    )
    assert is_valid is False
    assert "must be one of" in error


@pytest.mark.asyncio
async def test_validate_type_list_accepts_null_and_members(engine, schema_aware_tool):
    """Strict-mode widening (``["string", "null"]``) must accept both."""
    for value in ("hello", None):
        is_valid, error = await engine._validate_arguments(
            schema_aware_tool, {"name": "a", "note": value}
        )
        assert is_valid is True, error

    is_valid, _ = await engine._validate_arguments(
        schema_aware_tool, {"name": "a", "note": 42}
    )
    assert is_valid is False


@pytest.mark.asyncio
async def test_validate_object_without_properties_accepts_any_keys(
    engine, schema_aware_tool
):
    """A plain ``dict`` parameter ({"type": "object"}) is free-form."""
    is_valid, error = await engine._validate_arguments(
        schema_aware_tool, {"name": "a", "meta": {"anything": 1, "goes": [2]}}
    )
    assert is_valid is True, error


@pytest.mark.asyncio
async def test_validate_nested_additional_properties(engine, schema_aware_tool):
    """additionalProperties: true admits undeclared nested keys."""
    is_valid, error = await engine._validate_arguments(
        schema_aware_tool, {"name": "a", "config": {"level": 3, "extra": "ok"}}
    )
    assert is_valid is True, error


@pytest.mark.asyncio
async def test_normalize_drops_none_for_declared_optional(engine, schema_aware_tool):
    """Explicit ``None`` for a declared optional param is treated as omitted
    (models legitimately send null under strict-mode widened schemas)."""
    normalized = engine._normalize_arguments(
        schema_aware_tool, {"name": "a", "mode": None}
    )
    assert normalized == {"name": "a"}

    # None for a required param — and for undeclared keys — is preserved so
    # the validation error stays accurate.
    kept = engine._normalize_arguments(schema_aware_tool, {"name": None})
    assert kept == {"name": None}
    kept = engine._normalize_arguments(schema_aware_tool, {"name": "a", "bogus": None})
    assert kept == {"name": "a", "bogus": None}


@pytest.mark.asyncio
async def test_validate_typed_additional_properties_without_declared_properties(engine):
    """A ``dict[str, int]`` schema — object with no ``properties`` but a
    schema-valued ``additionalProperties`` — must still validate every value
    against that subschema (PR #381 review: the free-form early return was
    skipping it)."""
    tool = ToolDefinition(
        name="counts_tool",
        description="Takes a mapping of counts",
        parameters_schema={
            "type": "object",
            "properties": {
                "counts": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                },
            },
            "required": ["counts"],
        },
    )

    is_valid, error = await engine._validate_arguments(tool, {"counts": {"a": 1, "b": 2}})
    assert is_valid is True, error

    is_valid, error = await engine._validate_arguments(
        tool, {"counts": {"a": "not-an-int"}}
    )
    assert is_valid is False
    assert "counts.a" in error


@pytest.mark.asyncio
async def test_normalize_preserves_null_when_schema_explicitly_allows_it(engine):
    """A caller-supplied ``None`` for an optional property whose own schema
    explicitly types ``null`` (e.g. ``{"type": ["string", "null"]}``) is a
    distinct, meaningful value the schema declares — not merely strict-mode's
    "not provided" placeholder — and must survive normalization intact
    (codex review, PR #381)."""
    tool = ToolDefinition(
        name="nullable_tool",
        description="Has an explicitly nullable optional field",
        parameters_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                # Explicitly nullable — None is a real value here.
                "note": {"type": ["string", "null"]},
                # Ordinary optional — None here means "not provided".
                "tag": {"type": "string"},
            },
            "required": ["name"],
        },
    )

    normalized = engine._normalize_arguments(tool, {"name": "a", "note": None, "tag": None})
    assert normalized == {"name": "a", "note": None}


@pytest.mark.asyncio
async def test_validate_closed_empty_object_rejects_any_keys(engine):
    """``{"properties": {}, "additionalProperties": false}`` declares an
    object that permits NO keys at all — a "no-argument object" schema, not
    a free-form one. Must reject any payload with keys, and must not be
    conflated with the (much more common) free-form-dict shape where
    ``additionalProperties`` is absent/true (codex review, PR #381)."""
    tool = ToolDefinition(
        name="closed_object_tool",
        description="Takes a strictly closed empty object",
        parameters_schema={
            "type": "object",
            "properties": {
                "opts": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            "required": ["opts"],
        },
    )

    is_valid, error = await engine._validate_arguments(tool, {"opts": {}})
    assert is_valid is True, error

    is_valid, error = await engine._validate_arguments(tool, {"opts": {"unexpected": 1}})
    assert is_valid is False
    assert "opts" in error


@pytest.mark.asyncio
async def test_validate_empty_schema_additional_properties_permits_extras(engine):
    """``additionalProperties: {}`` is, per JSON Schema, spec-equivalent to
    ``true`` (the empty schema validates every value) — plain Python
    truthiness collapses it with ``False``/absent (both falsy), wrongly
    rejecting valid extra keys. Covers all three call sites: top-level,
    nested-with-declared-properties, and nested-with-no-declared-properties
    (claude[bot] review, PR #381)."""
    top_level_tool = ToolDefinition(
        name="top_level_tool",
        description="Empty-schema additionalProperties at the top level",
        parameters_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": {},
        },
    )
    is_valid, error = await engine._validate_arguments(
        top_level_tool, {"name": "a", "extra": 123}
    )
    assert is_valid is True, error

    nested_with_props_tool = ToolDefinition(
        name="nested_tool",
        description="Empty-schema additionalProperties on a nested object",
        parameters_schema={
            "type": "object",
            "properties": {
                "opts": {
                    "type": "object",
                    "properties": {"level": {"type": "integer"}},
                    "additionalProperties": {},
                },
            },
            "required": ["opts"],
        },
    )
    is_valid, error = await engine._validate_arguments(
        nested_with_props_tool, {"opts": {"level": 1, "extra": "x"}}
    )
    assert is_valid is True, error

    # Sanity: absent additionalProperties (Gantry's own strict default,
    # distinct from the JSON Schema spec default) must still reject extras
    # at both call sites — the fix must not blur this back together.
    strict_tool = ToolDefinition(
        name="strict_tool",
        description="No additionalProperties declared anywhere",
        parameters_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "opts": {
                    "type": "object",
                    "properties": {"level": {"type": "integer"}},
                },
            },
            "required": ["name"],
        },
    )
    is_valid, _ = await engine._validate_arguments(strict_tool, {"name": "a", "extra": 1})
    assert is_valid is False
    is_valid, _ = await engine._validate_arguments(
        strict_tool, {"name": "a", "opts": {"level": 1, "extra": "x"}}
    )
    assert is_valid is False


@pytest.mark.asyncio
async def test_confirmation_probe_does_not_consume_rate_limit_budget():
    """A tool-flag confirmation probe returns PENDING_CONFIRMATION without
    running the handler, so it must not spend rate-limit budget: ``acquire``
    records the call in the limiter's window and ``release`` only frees the
    concurrency counter, so charging both the probe and the approved replay
    would consume two units for one logical call — and at a limit of 1 would
    leave the tool permanently unexecutable (PR #381 review)."""
    from agent_gantry.core.rate_limiter import RateLimiter
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.schema.config import RateLimitConfig
    from agent_gantry.schema.execution import ToolCall

    registry = ToolRegistry()
    tool = ToolDefinition(
        name="delete_thing",
        description="Destructive",
        parameters_schema={"type": "object", "properties": {}},
        requires_confirmation=True,
    )
    registry.register_tool(tool)
    registry.register_handler("default.delete_thing", lambda: "deleted")

    engine = ExecutionEngine(
        registry=registry,
        rate_limiter=RateLimiter(
            RateLimitConfig(
                enabled=True,
                max_calls_per_minute=1,
                max_calls_per_hour=100,
                strategy="sliding_window",
            )
        ),
    )

    probe = await engine.execute(ToolCall(tool_name="delete_thing", arguments={}))
    assert probe.status.value == "pending_confirmation"

    approved = await engine.execute(
        ToolCall(tool_name="delete_thing", arguments={}, require_confirmation=False)
    )
    assert approved.status.value == "success", approved.error
    assert approved.result == "deleted"


@pytest.mark.asyncio
async def test_approved_calls_still_consume_rate_limit_budget():
    """The probe exemption must not become a way around the limiter:
    ``require_confirmation=False`` is caller-supplied, so a call that
    actually executes always counts."""
    from agent_gantry.core.rate_limiter import RateLimiter
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.schema.config import RateLimitConfig
    from agent_gantry.schema.execution import ToolCall

    registry = ToolRegistry()
    tool = ToolDefinition(
        name="delete_thing",
        description="Destructive",
        parameters_schema={"type": "object", "properties": {}},
        requires_confirmation=True,
    )
    registry.register_tool(tool)
    registry.register_handler("default.delete_thing", lambda: "deleted")

    engine = ExecutionEngine(
        registry=registry,
        rate_limiter=RateLimiter(
            RateLimitConfig(
                enabled=True,
                max_calls_per_minute=1,
                max_calls_per_hour=100,
                strategy="sliding_window",
            )
        ),
    )

    first = await engine.execute(
        ToolCall(tool_name="delete_thing", arguments={}, require_confirmation=False)
    )
    assert first.status.value == "success", first.error

    second = await engine.execute(
        ToolCall(tool_name="delete_thing", arguments={}, require_confirmation=False)
    )
    assert second.error_type == "RateLimitExceeded"


@pytest.mark.asyncio
async def test_confirmation_probe_does_not_leak_concurrency_slots():
    """The probe skips ``acquire``, so it must skip ``release`` too — an
    unmatched release would decrement a counter it never incremented and
    hand out extra concurrency."""
    from agent_gantry.core.rate_limiter import RateLimiter
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.schema.config import RateLimitConfig
    from agent_gantry.schema.execution import ToolCall

    registry = ToolRegistry()
    tool = ToolDefinition(
        name="delete_thing",
        description="Destructive",
        parameters_schema={"type": "object", "properties": {}},
        requires_confirmation=True,
    )
    registry.register_tool(tool)
    registry.register_handler("default.delete_thing", lambda: "deleted")

    limiter = RateLimiter(
        RateLimitConfig(enabled=True, max_calls_per_minute=100, strategy="sliding_window")
    )
    engine = ExecutionEngine(registry=registry, rate_limiter=limiter)

    for _ in range(3):
        result = await engine.execute(ToolCall(tool_name="delete_thing", arguments={}))
        assert result.status.value == "pending_confirmation"

    assert all(count == 0 for count in limiter._concurrent.values())


@pytest.mark.asyncio
async def test_a2a_call_is_gated_by_confirmation_before_remote_dispatch():
    """The A2A branch used to return before ``_check_confirmation_required``
    ever ran, so a confirmation-gated remote agent executed unasked — and
    once a pending call also skips the rate limiter, a caller could set
    ``require_confirmation=True`` to run A2A calls past both the per-minute
    and concurrency limits (PR #381 review)."""
    from datetime import datetime, timezone

    from agent_gantry.core.rate_limiter import RateLimiter
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.schema.config import RateLimitConfig
    from agent_gantry.schema.execution import ExecutionStatus, ToolCall, ToolResult
    from agent_gantry.schema.tool import ToolSource

    class _StubA2A:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, tool, call, _):
            self.calls += 1
            return ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.SUCCESS,
                result="remote side effect",
                queued_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                trace_id="t",
                span_id="s",
            )

    registry = ToolRegistry()
    registry.register_tool(
        ToolDefinition(
            name="remote_delete",
            description="Destructive remote agent call",
            parameters_schema={"type": "object", "properties": {}},
            source=ToolSource.A2A_AGENT,
            requires_confirmation=True,
        )
    )
    engine = ExecutionEngine(
        registry=registry,
        rate_limiter=RateLimiter(
            RateLimitConfig(
                enabled=True,
                max_calls_per_minute=1,
                max_calls_per_hour=100,
                strategy="sliding_window",
            )
        ),
    )
    stub = _StubA2A()
    engine._a2a_executor = stub

    for _ in range(3):
        result = await engine.execute(
            ToolCall(tool_name="remote_delete", arguments={}, require_confirmation=True)
        )
        assert result.status.value == "pending_confirmation"

    assert stub.calls == 0, "confirmation-gated A2A tool was dispatched remotely"

    approved = await engine.execute(
        ToolCall(tool_name="remote_delete", arguments={}, require_confirmation=False)
    )
    assert approved.status.value == "success"
    assert stub.calls == 1


def test_normalize_preserves_null_declared_through_anyof():
    """``{"anyOf": [{"type": "string"}, {"type": "null"}]}`` is what Pydantic
    and OpenAPI emit for ``str | None`` — far more common than a ``type``
    list, and it gives ``null`` the same declared meaning."""
    tool = ToolDefinition(
        name="anyof_tool",
        description="Nullable declared via an anyOf branch",
        parameters_schema={
            "type": "object",
            "properties": {"note": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
            "required": [],
        },
    )
    assert ExecutionEngine._normalize_arguments(tool, {"note": None}) == {"note": None}

    plain = ToolDefinition(
        name="plain_tool",
        description="Plain optional string parameter",
        parameters_schema={
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": [],
        },
    )
    assert ExecutionEngine._normalize_arguments(plain, {"note": None}) == {}


@pytest.mark.asyncio
async def test_validate_enforces_combinator_only_nested_schemas(engine):
    """A schema can constrain a value purely through combinators, with no
    ``type`` of its own — ``{"anyOf": [{"type": "integer"}, {"type":
    "null"}]}`` is what Pydantic emits for ``int | None``, including inside
    the nested models introspection now inlines. Reading ``type`` alone saw
    ``None`` and waved the value through (PR #381 review)."""
    tool = ToolDefinition(
        name="anyof_nested",
        description="Nested field typed through an anyOf combinator",
        parameters_schema={
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "properties": {
                        "count": {"anyOf": [{"type": "integer"}, {"type": "null"}]}
                    },
                    "required": ["count"],
                }
            },
            "required": ["payload"],
        },
    )

    is_valid, error = await engine._validate_arguments(tool, {"payload": {"count": "bad"}})
    assert is_valid is False
    assert "payload.count" in error

    for good in ({"count": 3}, {"count": None}):
        is_valid, error = await engine._validate_arguments(tool, {"payload": good})
        assert is_valid is True, error


@pytest.mark.asyncio
async def test_validate_allof_branches_are_all_enforced(engine):
    tool = ToolDefinition(
        name="allof_tool",
        description="Value constrained by an allOf combinator",
        parameters_schema={
            "type": "object",
            "properties": {"mode": {"allOf": [{"type": "string"}, {"enum": ["a", "b"]}]}},
            "required": ["mode"],
        },
    )
    is_valid, _ = await engine._validate_arguments(tool, {"mode": "a"})
    assert is_valid is True
    is_valid, _ = await engine._validate_arguments(tool, {"mode": "zzz"})
    assert is_valid is False


@pytest.mark.asyncio
async def test_validate_nullable_enum_still_enforces_membership(engine):
    """``enum`` is an independent JSON-Schema constraint: a property typed
    ``["string", "null"]`` whose enum lists only "a"/"b" does not admit
    ``null``. The null fast-path used to return before the enum check ever
    ran (PR #381 review)."""
    tool = ToolDefinition(
        name="nullable_enum",
        description="Nullable property carrying an enum constraint",
        parameters_schema={
            "type": "object",
            "properties": {"mode": {"type": ["string", "null"], "enum": ["a", "b"]}},
            "required": ["mode"],
        },
    )
    is_valid, error = await engine._validate_arguments(tool, {"mode": None})
    assert is_valid is False
    assert "must be one of" in error

    is_valid, _ = await engine._validate_arguments(tool, {"mode": "a"})
    assert is_valid is True


@pytest.mark.asyncio
async def test_validate_nullable_enum_admits_null_when_enum_lists_it(engine):
    """…and when the enum *does* list ``None``, null stays valid."""
    tool = ToolDefinition(
        name="nullable_enum_ok",
        description="Nullable enum that explicitly permits null",
        parameters_schema={
            "type": "object",
            "properties": {"mode": {"type": ["string", "null"], "enum": ["a", None]}},
            "required": ["mode"],
        },
    )
    is_valid, error = await engine._validate_arguments(tool, {"mode": None})
    assert is_valid is True, error

    is_valid, _ = await engine._validate_arguments(tool, {"mode": "zzz"})
    assert is_valid is False


@pytest.mark.asyncio
async def test_tool_flag_probe_is_not_charged_to_the_security_policy():
    """A tool gated by ``requires_confirmation=True`` rather than by a
    ``require_confirmation`` *pattern* stops at a gate the SecurityPolicy
    cannot see. Without being told, the policy recorded the probe against
    its window and then denied the approved replay for the rest of the
    minute, making the tool unexecutable (PR #381 review)."""
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.core.security import SecurityPolicy
    from agent_gantry.schema.execution import ToolCall

    registry = ToolRegistry()
    registry.register_tool(
        ToolDefinition(
            name="risky_op",
            description="Gated by the tool flag, not by a policy pattern",
            parameters_schema={"type": "object", "properties": {}},
            requires_confirmation=True,
        )
    )
    registry.register_handler("default.risky_op", lambda: "done")
    engine = ExecutionEngine(
        registry=registry,
        security_policy=SecurityPolicy(require_confirmation=[], max_requests_per_minute=1),
    )

    probe = await engine.execute(ToolCall(tool_name="risky_op", arguments={}))
    assert probe.status.value == "pending_confirmation"

    approved = await engine.execute(
        ToolCall(tool_name="risky_op", arguments={}, require_confirmation=False)
    )
    assert approved.status.value == "success", approved.error
    assert approved.result == "done"


@pytest.mark.asyncio
async def test_executed_calls_still_consume_the_security_policy_budget():
    """The probe exemption must not leak into calls that actually run."""
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.core.security import SecurityPolicy
    from agent_gantry.schema.execution import ToolCall

    registry = ToolRegistry()
    registry.register_tool(
        ToolDefinition(
            name="plain_op",
            description="No confirmation gate of any kind",
            parameters_schema={"type": "object", "properties": {}},
        )
    )
    registry.register_handler("default.plain_op", lambda: "done")
    engine = ExecutionEngine(
        registry=registry,
        security_policy=SecurityPolicy(require_confirmation=[], max_requests_per_minute=1),
    )

    assert (await engine.execute(ToolCall(tool_name="plain_op", arguments={}))).status.value == (
        "success"
    )
    second = await engine.execute(ToolCall(tool_name="plain_op", arguments={}))
    assert second.status.value == "permission_denied"


@pytest.mark.asyncio
async def test_validate_oneof_requires_exactly_one_matching_branch(engine):
    """``oneOf`` means *exactly* one, unlike ``anyOf``: ``1`` matches both
    ``number`` and ``integer``, so it violates the schema (PR #381 review)."""
    one_of = ToolDefinition(
        name="oneof_tool",
        description="Overlapping oneOf branches",
        parameters_schema={
            "type": "object",
            "properties": {"v": {"oneOf": [{"type": "number"}, {"type": "integer"}]}},
            "required": ["v"],
        },
    )
    is_valid, error = await engine._validate_arguments(one_of, {"v": 1})
    assert is_valid is False
    assert "exactly one" in error

    is_valid, _ = await engine._validate_arguments(one_of, {"v": "x"})
    assert is_valid is False

    # anyOf is unaffected — overlapping branches are fine there.
    any_of = ToolDefinition(
        name="anyof_tool",
        description="Overlapping anyOf branches",
        parameters_schema={
            "type": "object",
            "properties": {"v": {"anyOf": [{"type": "number"}, {"type": "integer"}]}},
            "required": ["v"],
        },
    )
    is_valid, error = await engine._validate_arguments(any_of, {"v": 1})
    assert is_valid is True, error


@pytest.mark.asyncio
async def test_validate_enforces_const(engine):
    """Pydantic emits ``const`` (not ``enum``) for a single-value ``Literal``,
    so it appears inside the nested model/TypedDict schemas introspection now
    inlines. Checking only ``enum`` let those through (PR #381 review)."""
    tool = ToolDefinition(
        name="const_tool",
        description="Single-value Literal represented as const",
        parameters_schema={
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "properties": {"kind": {"const": "expected", "type": "string"}},
                    "required": ["kind"],
                }
            },
            "required": ["payload"],
        },
    )
    is_valid, error = await engine._validate_arguments(tool, {"payload": {"kind": "unexpected"}})
    assert is_valid is False
    assert "payload.kind" in error

    is_valid, error = await engine._validate_arguments(tool, {"payload": {"kind": "expected"}})
    assert is_valid is True, error


def test_normalize_allof_is_nullable_only_when_every_branch_is():
    """``allOf`` intersects its branches, so the combined schema admits null
    only when every branch does. ``any()`` would preserve a synthetic null
    the schema actually forbids (PR #381 review)."""
    partial = ToolDefinition(
        name="allof_partial",
        description="Only one allOf branch admits null",
        parameters_schema={
            "type": "object",
            "properties": {
                "v": {"allOf": [{"type": ["string", "null"]}, {"type": "string"}]}
            },
            "required": [],
        },
    )
    assert ExecutionEngine._normalize_arguments(partial, {"v": None}) == {}

    every = ToolDefinition(
        name="allof_every",
        description="Every allOf branch admits null",
        parameters_schema={
            "type": "object",
            "properties": {
                "v": {"allOf": [{"type": ["string", "null"]}, {"type": ["string", "null"]}]}
            },
            "required": [],
        },
    )
    assert ExecutionEngine._normalize_arguments(every, {"v": None}) == {"v": None}


@pytest.mark.asyncio
async def test_validate_enforces_constraint_keywords(engine):
    """Pydantic emits these for any constrained field (``Annotated[int,
    Field(gt=0)]`` becomes ``exclusiveMinimum: 0``), so they arrive inside
    the nested schemas introspection now inlines; ``uniqueItems`` is emitted
    by Gantry itself for a ``set`` parameter (PR #381 review)."""
    tool = ToolDefinition(
        name="bounded",
        description="Carries numeric, string and array constraints",
        parameters_schema={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "exclusiveMinimum": 0},
                "m": {"type": "integer", "minimum": 10, "maximum": 20},
                "s": {"type": "string", "minLength": 3, "pattern": "^a"},
                "arr": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "uniqueItems": True,
                },
            },
            "required": [],
        },
    )
    for args in (
        {"n": -1},
        {"m": 99},
        {"s": "ab"},
        {"s": "bcd"},
        {"arr": [1]},
        {"arr": [1, 1]},
    ):
        is_valid, error = await engine._validate_arguments(tool, args)
        assert is_valid is False, f"{args} should have been rejected"
        assert error

    for args in ({"n": 5}, {"m": 15}, {"s": "abc"}, {"arr": [1, 2]}):
        is_valid, error = await engine._validate_arguments(tool, args)
        assert is_valid is True, error


@pytest.mark.asyncio
async def test_numeric_bounds_do_not_apply_to_booleans(engine):
    """``bool`` is an ``int`` subclass, so a stray bound must not reject a
    perfectly good boolean."""
    tool = ToolDefinition(
        name="flagged",
        description="Boolean alongside a bounded integer",
        parameters_schema={
            "type": "object",
            "properties": {"flag": {"type": "boolean"}, "n": {"type": "integer", "minimum": 1}},
            "required": [],
        },
    )
    is_valid, error = await engine._validate_arguments(tool, {"flag": True})
    assert is_valid is True, error


@pytest.mark.asyncio
async def test_validate_treats_empty_combinator_branch_as_wildcard(engine):
    """An empty schema ``{}`` validates every value, so it is a branch that
    always matches. Excluding it turned ``{"anyOf": [{}, {"type": "integer"}]}``
    — semantically "anything" — into an integer-only constraint
    (PR #381 review)."""
    tool = ToolDefinition(
        name="wildcard",
        description="anyOf carrying an empty wildcard branch",
        parameters_schema={
            "type": "object",
            "properties": {"v": {"anyOf": [{}, {"type": "integer"}]}},
            "required": ["v"],
        },
    )
    for value in ("a string", 5, None, [1, 2]):
        is_valid, error = await engine._validate_arguments(tool, {"v": value})
        assert is_valid is True, f"{value!r}: {error}"
