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


@pytest.mark.asyncio
async def test_fractional_multiple_of_uses_decimal_arithmetic(engine):
    """``0.3 % 0.1`` is ~0.0999 in binary floats, so a schema-valid JSON
    number was rejected. Decimal comparison via ``str`` gives the value as
    written (PR #381 review)."""
    tool = ToolDefinition(
        name="stepped",
        description="Number constrained to a fractional multiple",
        parameters_schema={
            "type": "object",
            "properties": {"v": {"type": "number", "multipleOf": 0.1}},
            "required": ["v"],
        },
    )
    for good in (0.3, 1.0, 0.7):
        is_valid, error = await engine._validate_arguments(tool, {"v": good})
        assert is_valid is True, f"{good}: {error}"

    is_valid, _ = await engine._validate_arguments(tool, {"v": 0.25})
    assert is_valid is False


def test_normalize_drops_synthetic_nulls_below_the_top_level():
    """Strict-mode widening applies to *every* object in the schema, so an
    optional nested property also comes back as an explicit null meaning
    "not provided". Normalizing only the top level left it intact, and
    validation then rejected it against the canonical non-null schema
    (PR #381 review)."""
    tool = ToolDefinition(
        name="nested",
        description="Object parameter carrying an optional nested field",
        parameters_schema={
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "nickname": {"type": "string"},
                    },
                    "required": ["name"],
                }
            },
            "required": ["payload"],
        },
    )
    normalized = ExecutionEngine._normalize_arguments(
        tool, {"payload": {"name": "a", "nickname": None}}
    )
    assert normalized == {"payload": {"name": "a"}}

    # A *required* nested null is still preserved, so the validation error
    # stays accurate rather than becoming "missing parameter".
    assert ExecutionEngine._normalize_arguments(tool, {"payload": {"name": None}}) == {
        "payload": {"name": None}
    }

    # Unchanged input keeps its identity (execute() checks `is not`).
    unchanged = {"payload": {"name": "a"}}
    assert ExecutionEngine._normalize_arguments(tool, unchanged) is unchanged


def test_normalize_recurses_into_array_items():
    tool = ToolDefinition(
        name="rows",
        description="Array of objects with an optional field",
        parameters_schema={
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                        "required": ["x"],
                    },
                }
            },
            "required": ["rows"],
        },
    )
    assert ExecutionEngine._normalize_arguments(tool, {"rows": [{"x": 1, "y": None}]}) == {
        "rows": [{"x": 1}]
    }


@pytest.mark.asyncio
async def test_validate_runs_combinators_alongside_an_explicit_type(engine):
    """JSON Schema applies ``anyOf``/``oneOf``/``allOf`` independently of
    ``type``, so gating them on a missing ``type`` let
    ``{"type": "integer", "allOf": [{"minimum": 1}]}`` accept ``0``
    (PR #381 review)."""
    tool = ToolDefinition(
        name="typed_allof",
        description="Typed property carrying an allOf constraint",
        parameters_schema={
            "type": "object",
            "properties": {"n": {"type": "integer", "allOf": [{"minimum": 1}]}},
            "required": ["n"],
        },
    )
    is_valid, error = await engine._validate_arguments(tool, {"n": 0})
    assert is_valid is False
    assert error

    is_valid, error = await engine._validate_arguments(tool, {"n": 5})
    assert is_valid is True, error


@pytest.mark.asyncio
async def test_malformed_confirmation_probe_consumes_quota():
    """A call whose arguments don't match the schema is terminal even on a
    confirmation-gated tool — it returns a ValidationError, never a pending
    prompt — so it must not take the probe exemption (PR #381 review)."""
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.core.security import SecurityPolicy
    from agent_gantry.schema.execution import ToolCall

    registry = ToolRegistry()
    registry.register_tool(
        ToolDefinition(
            name="risky",
            description="Confirmation-gated tool with a required argument",
            parameters_schema={
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
            },
            requires_confirmation=True,
        )
    )
    registry.register_handler("default.risky", lambda n: n)
    engine = ExecutionEngine(
        registry=registry,
        security_policy=SecurityPolicy(require_confirmation=[], max_requests_per_minute=1),
    )

    first = await engine.execute(ToolCall(tool_name="risky", arguments={"n": "bad"}))
    assert first.error_type == "ValidationError"
    # It consumed the budget, so malformed calls can't repeat forever.
    second = await engine.execute(ToolCall(tool_name="risky", arguments={"n": "bad"}))
    assert second.error_type == "PermissionDeniedError"


@pytest.mark.asyncio
async def test_valid_confirmation_probe_is_still_exempt():
    """The malformed-probe carve-out must not undo the exemption itself."""
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.core.security import SecurityPolicy
    from agent_gantry.schema.execution import ToolCall

    registry = ToolRegistry()
    registry.register_tool(
        ToolDefinition(
            name="risky",
            description="Confirmation-gated tool with a required argument",
            parameters_schema={
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
            },
            requires_confirmation=True,
        )
    )
    registry.register_handler("default.risky", lambda n: n)
    engine = ExecutionEngine(
        registry=registry,
        security_policy=SecurityPolicy(require_confirmation=[], max_requests_per_minute=1),
    )

    probe = await engine.execute(ToolCall(tool_name="risky", arguments={"n": 1}))
    assert probe.status.value == "pending_confirmation"
    approved = await engine.execute(
        ToolCall(tool_name="risky", arguments={"n": 1}, require_confirmation=False)
    )
    assert approved.status.value == "success", approved.error


@pytest.mark.asyncio
async def test_validate_positional_prefix_items(engine):
    """``prefixItems`` types each position independently — what Pydantic
    emits for a heterogeneous ``tuple[int, str]``, so it arrives inside the
    nested models introspection now inlines (PR #381 review)."""
    tool = ToolDefinition(
        name="pairs",
        description="Heterogeneous fixed tuple parameter",
        parameters_schema={
            "type": "object",
            "properties": {
                "pair": {
                    "type": "array",
                    "prefixItems": [{"type": "integer"}, {"type": "string"}],
                }
            },
            "required": ["pair"],
        },
    )
    is_valid, error = await engine._validate_arguments(tool, {"pair": [1, "a"]})
    assert is_valid is True, error

    is_valid, error = await engine._validate_arguments(tool, {"pair": ["bad", 42]})
    assert is_valid is False
    assert "pair[0]" in error

    is_valid, error = await engine._validate_arguments(tool, {"pair": [1, 2]})
    assert is_valid is False
    assert "pair[1]" in error


@pytest.mark.asyncio
async def test_prefix_items_and_items_cover_different_positions(engine):
    """``items`` alongside ``prefixItems`` covers the positions past it."""
    tool = ToolDefinition(
        name="tail",
        description="Prefixed tuple with a homogeneous tail",
        parameters_schema={
            "type": "object",
            "properties": {
                "row": {
                    "type": "array",
                    "prefixItems": [{"type": "string"}],
                    "items": {"type": "integer"},
                }
            },
            "required": ["row"],
        },
    )
    is_valid, error = await engine._validate_arguments(tool, {"row": ["a", 1, 2]})
    assert is_valid is True, error

    is_valid, _ = await engine._validate_arguments(tool, {"row": ["a", 1, "b"]})
    assert is_valid is False


def test_normalize_reaches_objects_behind_a_combinator():
    """An optional nested model is ``{"anyOf": [{object...}, {"type": "null"}]}``
    — the shape Pydantic emits for ``Payload | None``. Reading only the node's
    own ``properties`` left every value under it un-normalized, so its
    strict-mode nulls survived to fail validation (PR #381 review)."""
    tool = ToolDefinition(
        name="wrapped",
        description="Optional nested model expressed through a combinator",
        parameters_schema={
            "type": "object",
            "properties": {
                "payload": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "nickname": {"type": "string"},
                            },
                            "required": ["name"],
                        },
                        {"type": "null"},
                    ]
                }
            },
            "required": ["payload"],
        },
    )
    assert ExecutionEngine._normalize_arguments(
        tool, {"payload": {"name": "a", "nickname": None}}
    ) == {"payload": {"name": "a"}}

    # The branch's own ``required`` still decides: a required nested null is
    # preserved so the validation error stays accurate.
    assert ExecutionEngine._normalize_arguments(tool, {"payload": {"name": None}}) == {
        "payload": {"name": None}
    }


def test_normalize_reaches_array_items_behind_a_combinator():
    tool = ToolDefinition(
        name="wrapped_rows",
        description="Optional array of objects expressed through a combinator",
        parameters_schema={
            "type": "object",
            "properties": {
                "rows": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "integer"},
                                    "y": {"type": "integer"},
                                },
                                "required": ["x"],
                            },
                        },
                        {"type": "null"},
                    ]
                }
            },
            "required": ["rows"],
        },
    )
    assert ExecutionEngine._normalize_arguments(tool, {"rows": [{"x": 1, "y": None}]}) == {
        "rows": [{"x": 1}]
    }


def test_normalize_leaves_ambiguous_combinators_alone():
    """Two branches declaring ``properties`` give no single ``required`` list to
    decide against, so dropping a key one of them requires would be worse than
    leaving the value intact."""
    tool = ToolDefinition(
        name="ambiguous",
        description="Union of two object shapes with different required keys",
        parameters_schema={
            "type": "object",
            "properties": {
                "payload": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {"a": {"type": "string"}},
                            "required": [],
                        },
                        {
                            "type": "object",
                            "properties": {"a": {"type": "string"}},
                            "required": ["a"],
                        },
                    ]
                }
            },
            "required": ["payload"],
        },
    )
    arguments = {"payload": {"a": None}}
    assert ExecutionEngine._normalize_arguments(tool, arguments) is arguments


def test_normalize_recurses_into_prefix_items():
    """``prefixItems`` types each position independently, and strict mode
    widens the optional properties of a positional *object* exactly as it does
    anywhere else — so looking only at ``items`` left the nested null intact
    for a ``tuple[Payload, int]`` parameter (PR #381 review)."""
    tool = ToolDefinition(
        name="tup",
        description="Heterogeneous tuple parameter",
        parameters_schema={
            "type": "object",
            "properties": {
                "row": {
                    "type": "array",
                    "prefixItems": [
                        {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "nickname": {"type": "string"},
                            },
                            "required": ["name"],
                        },
                        {"type": "integer"},
                    ],
                }
            },
            "required": ["row"],
        },
    )
    assert ExecutionEngine._normalize_arguments(
        tool, {"row": [{"name": "a", "nickname": None}, 1]}
    ) == {"row": [{"name": "a"}, 1]}

    unchanged = {"row": [{"name": "a"}, 1]}
    assert ExecutionEngine._normalize_arguments(tool, unchanged) is unchanged


def test_normalize_uses_items_for_positions_past_the_prefix():
    tool = ToolDefinition(
        name="tail",
        description="Labelled row with a homogeneous tail",
        parameters_schema={
            "type": "object",
            "properties": {
                "row": {
                    "type": "array",
                    "prefixItems": [{"type": "string"}],
                    "items": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                        "required": ["x"],
                    },
                }
            },
            "required": ["row"],
        },
    )
    assert ExecutionEngine._normalize_arguments(
        tool, {"row": ["label", {"x": 1, "y": None}]}
    ) == {"row": ["label", {"x": 1}]}


async def test_uncompilable_pattern_fails_open_but_warns(engine, caplog):
    """A pattern Python's ``re`` can't compile is often a valid ECMA-262 one
    (``\\p{L}``), so rejecting every value would break a tool whose schema is
    fine everywhere else. Failing open is right, but silently leaves the
    constraint unenforced — the author needs a signal (PR #381 review)."""
    tool = ToolDefinition(
        name="pat",
        description="Parameter with an ECMA-only pattern",
        parameters_schema={
            "type": "object",
            "properties": {"s": {"type": "string", "pattern": r"\p{L}+"}},
            "required": ["s"],
        },
    )
    with caplog.at_level("WARNING"):
        is_valid, error = await engine._validate_arguments(tool, {"s": "anything"})

    assert is_valid is True
    assert error is None
    assert "cannot compile" in caplog.text
    assert "'s'" in caplog.text


def test_unique_items_separates_booleans_from_numbers():
    """Python's ``True == 1`` made ``[1, true]`` look like a duplicate, but
    JSON Schema compares types before values and calls them distinct
    (PR #381 review)."""
    tool = ToolDefinition(
        name="uniq",
        description="Array parameter requiring unique items",
        parameters_schema={
            "type": "object",
            "properties": {"v": {"type": "array", "uniqueItems": True}},
            "required": ["v"],
        },
    )
    engine = ExecutionEngine(registry=MockToolRegistry())

    async def valid(value):
        ok, _ = await engine._validate_arguments(tool, {"v": value})
        return ok

    import asyncio

    assert asyncio.run(valid([1, True])) is True
    assert asyncio.run(valid([[1], [True]])) is True
    assert asyncio.run(valid([{"a": 1}, {"a": True}])) is True
    # Mathematically equal numbers are still one value, as JSON Schema says.
    assert asyncio.run(valid([1, 1.0])) is False
    assert asyncio.run(valid([True, True])) is False
    assert asyncio.run(valid([{"a": 1}, {"a": 1}])) is False


async def test_malformed_args_outrank_a_pattern_confirmation_gate():
    """A call whose arguments don't match the schema is terminal, so it must
    return a ValidationError rather than a pending prompt. That held for the
    tool's own ``requires_confirmation`` flag but not for ``SecurityPolicy``'s
    ``require_confirmation`` *pattern* gate, whose result was returned before
    the validation result was ever consulted — putting a schema violation in
    front of a human to approve and then failing it anyway (PR #381 review)."""
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.core.security import SecurityPolicy
    from agent_gantry.schema.execution import ToolCall

    registry = ToolRegistry()
    tool = ToolDefinition(
        name="delete_thing",
        description="Destructive tool behind a policy pattern",
        parameters_schema={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        },
    )
    registry.register_tool(tool)
    registry.register_handler("default.delete_thing", lambda count: f"deleted {count}")

    engine = ExecutionEngine(
        registry=registry,
        security_policy=SecurityPolicy(require_confirmation=["delete_*"]),
    )

    malformed = await engine.execute(
        ToolCall(tool_name="delete_thing", arguments={"count": "not-an-int"})
    )
    assert malformed.status.value == "failure", malformed.status
    assert malformed.error_type == "ValidationError", malformed.error

    # A well-formed call still stops at the gate, unchanged.
    gated = await engine.execute(ToolCall(tool_name="delete_thing", arguments={"count": 1}))
    assert gated.status.value == "pending_confirmation", gated.status


async def test_denial_still_outranks_a_validation_error():
    """A denial must not leak "your arguments are malformed" about a tool the
    caller may not invoke at all, so it still wins over validation."""
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.core.security import SecurityPolicy
    from agent_gantry.schema.execution import ToolCall

    registry = ToolRegistry()
    tool = ToolDefinition(
        name="fetch",
        description="Fetches a URL from an allowlisted domain",
        parameters_schema={
            "type": "object",
            "properties": {"url": {"type": "string"}, "count": {"type": "integer"}},
            "required": ["url", "count"],
        },
    )
    registry.register_tool(tool)
    registry.register_handler("default.fetch", lambda url, count: url)

    engine = ExecutionEngine(
        registry=registry,
        security_policy=SecurityPolicy(allowed_domains=["allowed.test"]),
    )
    result = await engine.execute(
        ToolCall(
            tool_name="fetch",
            arguments={"url": "https://blocked.test/x", "count": "not-an-int"},
        )
    )
    assert result.status.value == "permission_denied", result.status


async def test_enum_and_const_compare_by_json_identity(engine):
    """``True == 1`` in Python, so a boolean satisfied a numeric
    ``Literal[1, 1.5]`` — which emits an ``enum`` with no single ``type``,
    leaving membership the only constraint. A tuple-valued member likewise
    never matched the array a provider actually returns (PR #381 review)."""
    def tool_with(prop):
        return ToolDefinition(
            name="t",
            description="Parameter constrained by enum or const",
            parameters_schema={
                "type": "object",
                "properties": {"p": prop},
                "required": ["p"],
            },
        )

    numeric = tool_with({"enum": [1, 1.5]})
    assert (await engine._validate_arguments(numeric, {"p": 1}))[0] is True
    assert (await engine._validate_arguments(numeric, {"p": True}))[0] is False

    const = tool_with({"const": 1})
    assert (await engine._validate_arguments(const, {"p": 1}))[0] is True
    assert (await engine._validate_arguments(const, {"p": True}))[0] is False

    # A boolean enum still accepts booleans.
    boolean = tool_with({"enum": [True, False]})
    assert (await engine._validate_arguments(boolean, {"p": True}))[0] is True
    assert (await engine._validate_arguments(boolean, {"p": 1}))[0] is False

    # A composite member matches the array a provider sends back.
    composite = tool_with({"enum": [[0, 0], [1, 1]]})
    assert (await engine._validate_arguments(composite, {"p": [0, 0]}))[0] is True
    assert (await engine._validate_arguments(composite, {"p": [2, 2]}))[0] is False


async def test_object_property_counts_are_enforced(engine):
    """A Pydantic ``dict`` field constrained with ``Field(min_length=1)`` emits
    ``minProperties``/``maxProperties``, so they arrive inside the inlined
    mapping schemas — and the constraint check had no object branch, letting an
    empty or oversized mapping reach the handler (PR #381 review)."""
    def tool_with(prop):
        return ToolDefinition(
            name="t",
            description="Mapping parameter with a property-count bound",
            parameters_schema={
                "type": "object",
                "properties": {"p": prop},
                "required": ["p"],
            },
        )

    at_least_one = tool_with({"type": "object", "minProperties": 1})
    assert (await engine._validate_arguments(at_least_one, {"p": {}}))[0] is False
    assert (await engine._validate_arguments(at_least_one, {"p": {"a": 1}}))[0] is True

    at_most_two = tool_with({"type": "object", "maxProperties": 2})
    assert (await engine._validate_arguments(at_most_two, {"p": {"a": 1}}))[0] is True
    assert (
        await engine._validate_arguments(at_most_two, {"p": {"a": 1, "b": 2, "c": 3}})
    )[0] is False


async def test_execute_records_exactly_one_telemetry_event():
    """``val_result`` is computed before the security policy runs so the
    rate-limit exemption decision is accurate, but it does not always win — a
    denial outranks it and a pending result can be discarded in favour of it.
    Recording where it was built produced two executions for one call, one of
    them an outcome that was never returned (PR #381 review)."""
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.core.security import SecurityPolicy
    from agent_gantry.schema.execution import ToolCall

    class _Telemetry:
        def __init__(self) -> None:
            self.statuses: list[str] = []

        async def record_execution(self, call, result):
            self.statuses.append(result.status.value)

        async def record_retrieval(self, *args, **kwargs):
            pass

    def _engine(policy, telemetry):
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="delete_thing",
            description="Destructive tool behind a policy pattern",
            parameters_schema={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
        )
        registry.register_tool(tool)
        registry.register_handler("default.delete_thing", lambda count: f"deleted {count}")
        return ExecutionEngine(
            registry=registry, security_policy=policy, telemetry=telemetry
        )

    gated = SecurityPolicy(require_confirmation=["delete_*"])
    for arguments in ({"count": "bad"}, {"count": 1}):
        telemetry = _Telemetry()
        engine = _engine(gated, telemetry)
        result = await engine.execute(
            ToolCall(tool_name="delete_thing", arguments=arguments)
        )
        assert telemetry.statuses == [result.status.value], (
            arguments,
            telemetry.statuses,
        )


async def test_malformed_pattern_gated_calls_consume_policy_quota():
    """Deferring the charge to an approved replay is only right when a replay
    can happen. A malformed call to a pattern-gated tool is terminal, so
    nothing was ever counted and such calls were unlimited — the same exemption
    abuse the tool-flag gate was fixed for, reachable through the pattern gate
    (PR #381 review)."""
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.core.security import SecurityPolicy
    from agent_gantry.schema.execution import ToolCall

    registry = ToolRegistry()
    tool = ToolDefinition(
        name="delete_thing",
        description="Destructive tool behind a policy pattern",
        parameters_schema={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        },
    )
    registry.register_tool(tool)
    registry.register_handler("default.delete_thing", lambda count: f"deleted {count}")
    engine = ExecutionEngine(
        registry=registry,
        security_policy=SecurityPolicy(
            require_confirmation=["delete_*"], max_requests_per_minute=2
        ),
    )

    statuses = [
        (
            await engine.execute(
                ToolCall(tool_name="delete_thing", arguments={"count": "bad"})
            )
        ).status.value
        for _ in range(3)
    ]
    assert statuses == ["failure", "failure", "permission_denied"], statuses


async def test_valid_pattern_gated_probe_keeps_its_exemption():
    """The exemption exists so an approved replay isn't denied for the rest of
    the window — that must survive the fix above."""
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.core.security import SecurityPolicy
    from agent_gantry.schema.execution import ToolCall

    registry = ToolRegistry()
    tool = ToolDefinition(
        name="delete_thing",
        description="Destructive tool behind a policy pattern",
        parameters_schema={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        },
    )
    registry.register_tool(tool)
    registry.register_handler("default.delete_thing", lambda count: f"deleted {count}")
    engine = ExecutionEngine(
        registry=registry,
        security_policy=SecurityPolicy(
            require_confirmation=["delete_*"], max_requests_per_minute=1
        ),
    )

    probe = await engine.execute(ToolCall(tool_name="delete_thing", arguments={"count": 1}))
    assert probe.status.value == "pending_confirmation"

    approved = await engine.execute(
        ToolCall(tool_name="delete_thing", arguments={"count": 1}, require_confirmation=False)
    )
    assert approved.status.value == "success", approved.error
    assert approved.result == "deleted 1"


async def test_required_nullable_parameter_accepts_null_end_to_end(engine):
    """The schema fix is only worth anything if validation agrees with it."""
    from agent_gantry.schema.introspection import build_parameters_schema

    def handler(x: int | None) -> str:
        return "none" if x is None else str(x)

    tool = ToolDefinition(
        name="nullable",
        description="Required parameter whose annotation admits null",
        parameters_schema=build_parameters_schema(handler),
    )
    assert (await engine._validate_arguments(tool, {"x": None}))[0] is True
    assert (await engine._validate_arguments(tool, {"x": 1}))[0] is True
    assert (await engine._validate_arguments(tool, {"x": "bad"}))[0] is False


def test_explicit_null_survives_a_non_none_default():
    """``x: int | None = 5`` — an explicit null is a distinct choice, so
    normalization must not drop it as "not provided" and hand the handler
    ``5`` (PR #381 review)."""
    from agent_gantry.schema.introspection import build_parameters_schema

    def handler(x: int | None = 5) -> str:
        return "none" if x is None else str(x)

    tool = ToolDefinition(
        name="defaulted",
        description="Optional nullable parameter with a non-None default",
        parameters_schema=build_parameters_schema(handler),
    )
    assert ExecutionEngine._normalize_arguments(tool, {"x": None}) == {"x": None}


async def test_pattern_properties_are_validated(engine):
    """``patternProperties`` types keys by regex, which Pydantic emits for a
    mapping with constrained keys. Ignoring it meant a matching key's value was
    never checked — and with ``additionalProperties: false`` that every key was
    rejected, because none counted as declared (PR #381 review)."""
    def tool_with(prop):
        return ToolDefinition(
            name="t",
            description="Mapping parameter with pattern-typed keys",
            parameters_schema={
                "type": "object",
                "properties": {"p": prop},
                "required": ["p"],
            },
        )

    patterns = {"^n_[a-z]+$": {"type": "integer"}}
    open_map = tool_with({"type": "object", "patternProperties": patterns})
    assert (await engine._validate_arguments(open_map, {"p": {"n_abc": 1}}))[0] is True
    assert (await engine._validate_arguments(open_map, {"p": {"n_abc": "x"}}))[0] is False
    # A key no pattern matches is still free-form here — nothing forbids it.
    assert (await engine._validate_arguments(open_map, {"p": {"other": "x"}}))[0] is True

    closed = tool_with(
        {"type": "object", "patternProperties": patterns, "additionalProperties": False}
    )
    # A pattern-matched key *is* declared, so it survives the closure.
    assert (await engine._validate_arguments(closed, {"p": {"n_abc": 1}}))[0] is True
    assert (await engine._validate_arguments(closed, {"p": {"zzz": 1}}))[0] is False


async def test_draft04_boolean_exclusivity_is_honoured(engine):
    """OpenAPI 3.0 emits draft-04's *boolean* ``exclusiveMinimum``, a modifier
    on ``minimum`` rather than a bound of its own — and OpenAPI/MCP import is a
    supported way to register a tool. Reading only the modern numeric form left
    the boolean ignored and the bound applied inclusively (PR #381 review)."""
    def tool_with(prop):
        return ToolDefinition(
            name="t",
            description="Numeric parameter with an exclusive bound",
            parameters_schema={
                "type": "object",
                "properties": {"p": prop},
                "required": ["p"],
            },
        )

    draft04 = tool_with({"type": "integer", "minimum": 5, "exclusiveMinimum": True})
    assert (await engine._validate_arguments(draft04, {"p": 5}))[0] is False
    assert (await engine._validate_arguments(draft04, {"p": 6}))[0] is True
    assert (await engine._validate_arguments(draft04, {"p": 4}))[0] is False

    # ``exclusiveMinimum: false`` leaves ``minimum`` inclusive.
    inclusive = tool_with({"type": "integer", "minimum": 5, "exclusiveMinimum": False})
    assert (await engine._validate_arguments(inclusive, {"p": 5}))[0] is True

    # The modern numeric form is unchanged.
    modern = tool_with({"type": "integer", "exclusiveMinimum": 5})
    assert (await engine._validate_arguments(modern, {"p": 5}))[0] is False
    assert (await engine._validate_arguments(modern, {"p": 6}))[0] is True

    upper = tool_with({"type": "integer", "maximum": 5, "exclusiveMaximum": True})
    assert (await engine._validate_arguments(upper, {"p": 5}))[0] is False
    assert (await engine._validate_arguments(upper, {"p": 4}))[0] is True


def test_declared_null_respects_an_enum_that_forbids_it():
    """An optional ``Literal`` that strict mode pre-widened arrives as
    ``{"type": ["string", "null"], "enum": [...]}`` — nullable by its type list
    but not by its enum. Treating it as nullable preserved a strict-mode
    placeholder the canonical schema then rejected (PR #381 review)."""
    from agent_gantry.schema.base import schema_declares_null

    assert schema_declares_null({"type": ["string", "null"], "enum": ["fast", "slow"]}) is False
    assert schema_declares_null({"type": ["string", "null"], "enum": ["fast", None]}) is True
    assert schema_declares_null({"type": ["string", "null"]}) is True
    assert schema_declares_null({"type": "string", "const": "fixed"}) is False
    assert schema_declares_null({"anyOf": [{"type": "string"}, {"type": "null"}]}) is True


def test_pre_widened_enum_null_is_dropped_not_preserved():
    """End-to-end consequence: the placeholder is dropped so the handler's own
    default applies, rather than surviving to fail validation."""
    tool = ToolDefinition(
        name="speed",
        description="Optional literal choice widened by strict mode",
        parameters_schema={
            "type": "object",
            "properties": {"mode": {"type": ["string", "null"], "enum": ["fast", "slow"]}},
            "required": [],
        },
    )
    assert ExecutionEngine._normalize_arguments(tool, {"mode": None}) == {}


def test_a_sibling_type_governs_combinator_nullability():
    """JSON Schema applies a property's own ``type`` alongside its combinator,
    so a branch cannot admit a value the type forbids. Reading branches in
    isolation called ``{"type": "string", "anyOf": [{"type": "null"}, {}]}``
    nullable and preserved a placeholder validation then rejected
    (PR #381 review)."""
    from agent_gantry.schema.base import schema_declares_null

    assert schema_declares_null({"type": "string", "anyOf": [{"type": "null"}, {}]}) is False
    assert schema_declares_null({"anyOf": [{"type": "null"}, {"type": "string"}]}) is True
    assert (
        schema_declares_null({"type": ["string", "null"], "anyOf": [{"type": "null"}]})
        is True
    )


def test_oneof_null_must_match_exactly_one_branch():
    """``oneOf`` means *exactly* one, so a permissive sibling branch makes
    ``null`` match twice — and a value matching two branches is invalid.
    Sharing the ``anyOf`` any-branch reading called that nullable and
    preserved a placeholder validation then rejected (PR #381 review)."""
    from agent_gantry.schema.base import schema_declares_null

    # ``{}`` matches everything, so ``null`` matches both branches.
    assert schema_declares_null({"oneOf": [{"type": "null"}, {}]}) is False
    # Two null-admitting branches are ambiguous for the same reason.
    assert (
        schema_declares_null({"oneOf": [{"type": "null"}, {"type": ["string", "null"]}]})
        is False
    )
    # Exactly one branch admits null — the ordinary nullable spelling.
    assert schema_declares_null({"oneOf": [{"type": "string"}, {"type": "null"}]}) is True
    # ``anyOf`` is unaffected: any branch admitting null is enough.
    assert schema_declares_null({"anyOf": [{"type": "null"}, {}]}) is True


def test_ambiguous_oneof_null_is_dropped_not_preserved():
    """End-to-end consequence: the placeholder is dropped rather than
    surviving to fail the validation that runs immediately after."""
    tool = ToolDefinition(
        name="ambiguous",
        description="Nullability spelled through overlapping oneOf branches",
        parameters_schema={
            "type": "object",
            "properties": {"note": {"oneOf": [{"type": "null"}, {}]}},
            "required": [],
        },
    )
    assert ExecutionEngine._normalize_arguments(tool, {"note": None}) == {}


@pytest.mark.asyncio
async def test_top_level_pattern_properties_are_validated(engine):
    """``patternProperties`` was handled only inside a nested object, so the
    identical construct at the top level of a tool schema both rejected a
    schema-valid key as unknown and skipped the pattern's own constraint
    (PR #381 review)."""
    closed = ToolDefinition(
        name="closed_patterns",
        description="Top-level keys typed by regex, closed to anything else",
        parameters_schema={
            "type": "object",
            "properties": {},
            "patternProperties": {"^n_[a-z]+$": {"type": "integer"}},
            "additionalProperties": False,
        },
    )
    # A matching key *is* declared, so the closure must not reject it.
    assert await engine._validate_arguments(closed, {"n_abc": 1}) == (True, None)
    # Its value is still checked against the pattern's schema.
    ok, err = await engine._validate_arguments(closed, {"n_abc": "bad"})
    assert ok is False and "n_abc" in err
    # And a key matching no pattern is still refused.
    ok, err = await engine._validate_arguments(closed, {"other": 1})
    assert ok is False and "Unknown parameter: other" in err


@pytest.mark.asyncio
async def test_top_level_pattern_properties_apply_when_the_schema_is_open(engine):
    """The other half: ``additionalProperties: true`` admits the key, but a
    pattern that matches it still constrains its value."""
    tool = ToolDefinition(
        name="open_patterns",
        description="Top-level regex-typed keys alongside free-form ones",
        parameters_schema={
            "type": "object",
            "properties": {},
            "patternProperties": {"^n_": {"type": "integer"}},
            "additionalProperties": True,
        },
    )
    assert await engine._validate_arguments(tool, {"n_x": 1}) == (True, None)
    assert await engine._validate_arguments(tool, {"free": "anything"}) == (True, None)
    ok, err = await engine._validate_arguments(tool, {"n_x": "bad"})
    assert ok is False and "n_x" in err


@pytest.mark.asyncio
async def test_top_level_pattern_properties_coexist_with_declared_ones(engine):
    """A named property keeps its own schema; the patterns type the rest."""
    tool = ToolDefinition(
        name="mixed_patterns",
        description="A declared parameter beside regex-typed extras",
        parameters_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "patternProperties": {"^n_": {"type": "integer"}},
            "additionalProperties": False,
            "required": ["name"],
        },
    )
    assert await engine._validate_arguments(tool, {"name": "x", "n_a": 1}) == (True, None)
    ok, _ = await engine._validate_arguments(tool, {"name": "x", "n_a": "bad"})
    assert ok is False
    ok, _ = await engine._validate_arguments(tool, {"name": 1, "n_a": 1})
    assert ok is False


@pytest.mark.asyncio
async def test_const_is_enforced_on_the_null_member_path(engine):
    """A type list naming ``null`` returns early once the value *is* null —
    and that early return checked ``enum`` but not ``const``, so a property
    pinned to a constant accepted null anyway. Both are independent of
    ``type`` and both must survive the shortcut (PR #381 review)."""
    pinned = ToolDefinition(
        name="pinned",
        description="A widened type list pinned by an independent const",
        parameters_schema={
            "type": "object",
            "properties": {"v": {"type": ["string", "null"], "const": "fixed"}},
            "required": ["v"],
        },
    )
    ok, err = await engine._validate_arguments(pinned, {"v": None})
    assert ok is False and "fixed" in err
    assert await engine._validate_arguments(pinned, {"v": "fixed"}) == (True, None)
    ok, _ = await engine._validate_arguments(pinned, {"v": "other"})
    assert ok is False

    # A const of ``None`` is the one that *does* admit null.
    nullable = ToolDefinition(
        name="null_const",
        description="A widened type list whose constant is null itself",
        parameters_schema={
            "type": "object",
            "properties": {"v": {"type": ["string", "null"], "const": None}},
            "required": ["v"],
        },
    )
    assert await engine._validate_arguments(nullable, {"v": None}) == (True, None)
    ok, _ = await engine._validate_arguments(nullable, {"v": "x"})
    assert ok is False


def test_a_constraint_only_oneof_branch_also_matches_null():
    """``{}`` is not the only branch null matches twice through. A
    constraint-only ``{"minimum": 5}`` declares no type, and numeric keywords
    assert nothing about null — so null matches it as well, making
    ``{"oneOf": [{"type": "null"}, {"minimum": 5}]}`` ambiguous and therefore
    not nullable. Counting *declared* nulls saw one branch (PR #381 review)."""
    from agent_gantry.schema.base import null_validates_against, schema_declares_null

    assert schema_declares_null({"oneOf": [{"type": "null"}, {"minimum": 5}]}) is False
    assert schema_declares_null({"oneOf": [{"type": "null"}, {"maxLength": 3}]}) is False
    # A branch whose type excludes null leaves exactly one match, as before.
    assert (
        schema_declares_null({"oneOf": [{"type": "null"}, {"type": "integer", "minimum": 5}]})
        is True
    )
    assert schema_declares_null({"oneOf": [{"type": "null"}, {"type": "string"}]}) is True

    # The matching predicate itself: a schema that merely fails to forbid null
    # admits it, which is what makes the count above right.
    assert null_validates_against({}) is True
    assert null_validates_against({"minimum": 5}) is True
    assert null_validates_against({"type": "null"}) is True
    assert null_validates_against({"type": ["string", "null"]}) is True
    assert null_validates_against({"const": None}) is True
    assert null_validates_against({"type": "string"}) is False
    assert null_validates_against({"enum": ["a"]}) is False
    assert null_validates_against({"const": "x"}) is False
    assert null_validates_against(False) is False
    assert null_validates_against(True) is True
    # Nested combinators are resolved, not assumed to admit null.
    assert null_validates_against({"allOf": [{"type": "null"}, {"type": "string"}]}) is False
    assert null_validates_against({"anyOf": [{"type": "string"}, {"minimum": 5}]}) is True


def test_one_question_decides_whether_a_null_is_kept():
    """This previously pinned an asymmetry — ``oneOf`` counted what *matches*
    while ``anyOf`` asked what the author *declared*, so a branch that merely
    failed to forbid null didn't count. The distinction cost more than it
    bought: it needed a fresh patch per spelling and still got Gantry's own
    emission wrong (an optional ``Literal["a", None]`` emits
    ``{"enum": ["a", null]}`` with no ``type``, and the declaring reading
    dropped an explicitly supplied ``None``). One rule replaces it — keep a
    null iff the executor would accept it (PR #381 review)."""
    from agent_gantry.schema.base import schema_declares_null

    # The reversed case: null validates against a constraint-only branch, so
    # an explicit one is the caller's value, not a placeholder.
    assert schema_declares_null({"anyOf": [{"type": "string"}, {"minimum": 5}]}) is True
    assert schema_declares_null({"anyOf": [{"type": "string"}, {"type": "null"}]}) is True
    # Everything the narrower reading got right, it still gets right.
    assert schema_declares_null({"anyOf": [{"type": "string"}, {"type": "integer"}]}) is False
    assert schema_declares_null({"type": "string", "anyOf": [{"type": "null"}, {}]}) is False


def test_a_sibling_assertion_can_forbid_the_null_a_branch_admits():
    """``anyOf`` naming null did not stop a sibling ``allOf`` forbidding it —
    the early return fired first. Every sibling assertion is resolved now
    (PR #381 review)."""
    from agent_gantry.schema.base import schema_declares_null

    assert (
        schema_declares_null(
            {"anyOf": [{"type": "string"}, {"type": "null"}], "allOf": [{"type": "string"}]}
        )
        is False
    )
    # The same shape with a sibling that permits null stays nullable.
    assert (
        schema_declares_null(
            {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "allOf": [{"type": ["string", "null"]}],
            }
        )
        is True
    )


def test_an_enum_or_const_naming_null_declares_it():
    """``x: Literal["a", None] = "a"`` is emitted as ``{"enum": ["a", null]}``
    with no ``type`` — its members share no scalar kind. The predicate used
    ``enum`` only to *reject* null, so an explicitly supplied ``None`` was
    dropped and the handler got its default instead of the value its own
    annotation permits (PR #381 review)."""
    from agent_gantry.schema.base import schema_declares_null

    assert schema_declares_null({"enum": ["a", None]}) is True
    assert schema_declares_null({"const": None}) is True
    assert schema_declares_null({}) is True
    # And the rejecting direction is unchanged.
    assert schema_declares_null({"enum": ["a", "b"]}) is False
    assert schema_declares_null({"const": "fixed"}) is False


@pytest.mark.asyncio
async def test_an_explicit_null_for_a_nullable_literal_reaches_the_handler():
    """End-to-end consequence of the enum case above."""
    tool = ToolDefinition(
        name="nullable_literal",
        description="An optional Literal that lists None among its members",
        parameters_schema={
            "type": "object",
            "properties": {"x": {"enum": ["a", None], "default": "a"}},
            "required": [],
        },
    )
    assert ExecutionEngine._normalize_arguments(tool, {"x": None}) == {"x": None}


@pytest.mark.asyncio
async def test_boolean_combinator_branches_are_schemas(engine):
    """``true`` and ``false`` are schemas from draft-06 on — ``true`` matches
    every value, ``false`` none. Filtering branches to dicts dropped them, so
    ``{"anyOf": [true, {"type": "integer"}]}`` (semantically "anything")
    rejected a string (PR #381 review)."""

    async def check(prop, value):
        tool = ToolDefinition(
            name="bools",
            description="Combinator branches spelled as boolean schemas",
            parameters_schema={
                "type": "object",
                "properties": {"v": prop},
                "required": ["v"],
            },
        )
        return (await engine._validate_arguments(tool, {"v": value}))[0]

    # ``true`` always matches.
    assert await check({"anyOf": [True, {"type": "integer"}]}, "x") is True
    assert await check({"allOf": [True, {"type": "integer"}]}, 1) is True
    assert await check({"allOf": [True, {"type": "integer"}]}, "x") is False
    # ``false`` never does — including as an allOf branch, which then forbids
    # every value.
    assert await check({"anyOf": [False, {"type": "integer"}]}, "x") is False
    assert await check({"anyOf": [False, {"type": "integer"}]}, 1) is True
    assert await check({"allOf": [False, {"type": "integer"}]}, 1) is False
    # And it counts for ``oneOf`` exclusivity: ``1`` matches both branches.
    assert await check({"oneOf": [True, {"type": "integer"}]}, 1) is False
    assert await check({"oneOf": [True, {"type": "integer"}]}, "x") is True


@pytest.mark.asyncio
async def test_a_boolean_schema_is_honoured_wherever_a_schema_may_appear(engine):
    """Draft-06 booleans are schemas in *any* schema position, not only in a
    combinator branch. Reading ``.get`` on ``False`` let an ``AttributeError``
    escape ``execute()`` instead of returning a validation failure, and the
    positions that skipped booleans accepted values the schema forbids
    (PR #381 review)."""

    async def check(parameters, arguments):
        tool = ToolDefinition(
            name="booleans",
            description="Boolean subschemas in assorted positions",
            parameters_schema=parameters,
        )
        return await engine._validate_arguments(tool, arguments)

    # A named property. ``false`` forbids the key outright; ``true`` permits
    # any value. Previously this raised rather than returning a verdict.
    named = {"type": "object", "properties": {"disabled": False, "ok": True}, "required": []}
    valid, err = await check(named, {"disabled": 1})
    assert valid is False and "disabled" in err
    assert await check(named, {"ok": "anything"}) == (True, None)
    assert await check(named, {}) == (True, None)

    # A pattern entry. ``false`` must reject every matching key — and not
    # count it as declared, which would also slip it past a closed object.
    patterned = {
        "type": "object",
        "properties": {
            "m": {
                "type": "object",
                "patternProperties": {"^blocked_": False, "^ok_": True},
            }
        },
        "required": ["m"],
    }
    valid, _ = await check(patterned, {"m": {"blocked_x": 1}})
    assert valid is False
    assert await check(patterned, {"m": {"ok_x": "anything"}}) == (True, None)

    # An ``items`` tail. ``items: false`` beside ``prefixItems`` is the
    # standard spelling of a fixed-length tuple.
    fixed = {
        "type": "object",
        "properties": {
            "v": {"type": "array", "prefixItems": [{"type": "integer"}], "items": False}
        },
        "required": ["v"],
    }
    assert await check(fixed, {"v": [1]}) == (True, None)
    valid, _ = await check(fixed, {"v": [1, 2]})
    assert valid is False
    # The prefix itself is still typed.
    valid, _ = await check(fixed, {"v": ["x"]})
    assert valid is False
    # ``items: true`` permits any tail, the mirror of the above.
    open_tail = {
        "type": "object",
        "properties": {
            "v": {"type": "array", "prefixItems": [{"type": "integer"}], "items": True}
        },
        "required": ["v"],
    }
    assert await check(open_tail, {"v": [1, "anything"]}) == (True, None)


@pytest.mark.asyncio
async def test_an_over_quota_call_is_refused_before_it_buys_validation():
    """The recursive validator walks the whole payload and runs ``re.search``
    against schema-supplied patterns on caller-controlled input. Running it
    ahead of every admission check handed an over-quota caller unlimited CPU —
    the limits stopped protecting the work they exist to protect
    (PR #381 review)."""
    import agent_gantry.core.executor as executor_module
    from agent_gantry import AgentGantry
    from agent_gantry.adapters.embedders.simple import SimpleEmbedder
    from agent_gantry.core.security import SecurityPolicy
    from agent_gantry.schema.execution import ToolCall

    evaluations = {"count": 0}
    original = executor_module._check_constraints

    def counting(value, schema, path):
        if isinstance(schema, dict) and "pattern" in schema:
            evaluations["count"] += 1
        return original(value, schema, path)

    executor_module._check_constraints = counting
    try:
        gantry = AgentGantry(
            embedder=SimpleEmbedder(dimension=64),
            security_policy=SecurityPolicy(require_confirmation=[], max_requests_per_minute=2),
        )

        async def handler(s):
            return s

        await gantry.add_tool(
            ToolDefinition(
                name="patterned",
                description="A tool whose schema pattern-matches its argument",
                parameters_schema={
                    "type": "object",
                    "properties": {"s": {"type": "string", "pattern": "^a+$"}},
                    "required": ["s"],
                },
                tags=["demo"],
            ),
            handler=handler,
        )
        await gantry.sync()

        for _ in range(2):
            result = await gantry.execute(
                ToolCall(tool_name="patterned", arguments={"s": "aaaa!"})
            )
            assert result.error_type == "ValidationError"
        spent = evaluations["count"]
        assert spent == 2

        # Over quota now: further calls are refused without the pattern ever
        # being evaluated again.
        for _ in range(4):
            denied = await gantry.execute(
                ToolCall(tool_name="patterned", arguments={"s": "aaaa!"})
            )
            assert denied.status.value == "permission_denied"
        assert evaluations["count"] == spent
    finally:
        executor_module._check_constraints = original


def test_the_admission_peek_spends_nothing():
    """It runs before the recording checks, so it has to leave both limiters
    exactly as it found them — otherwise it would double-charge every call it
    admits, the very bug the ordering it protects was introduced to fix."""
    import asyncio

    from agent_gantry.core.rate_limiter import RateLimiter
    from agent_gantry.core.security import SecurityPolicy
    from agent_gantry.schema.config import RateLimitConfig

    for strategy in ("sliding_window", "token_bucket", "fixed_window"):
        limiter = RateLimiter(
            RateLimitConfig(
                enabled=True,
                max_calls_per_minute=3,
                max_calls_per_hour=100,
                strategy=strategy,
            )
        )

        async def spend(limiter=limiter):
            for _ in range(50):
                limiter.would_exceed("t", "default")
            admitted = 0
            for _ in range(10):
                try:
                    await limiter.acquire("t", "default")
                except Exception:
                    break
                admitted += 1
                await limiter.release("t", "default")
            return admitted

        assert asyncio.run(spend()) == 3, strategy

    policy = SecurityPolicy(require_confirmation=[], max_requests_per_minute=2)
    assert policy.would_exceed_rate_limit() is None
    for _ in range(50):
        policy.would_exceed_rate_limit()
    assert not policy._request_timestamps
    for _ in range(2):
        policy.check_permission("t", {})
    assert policy.would_exceed_rate_limit() is not None
    assert len(policy._request_timestamps) == 2


@pytest.mark.asyncio
async def test_required_keys_hold_in_an_object_declaring_no_properties(engine):
    """``required`` names need no matching ``properties`` entry, so
    ``{"type": "object", "properties": {}, "required": ["token"]}`` is valid —
    and the no-properties shortcut ran before the required loop, accepting
    ``{}`` outright (PR #381 review)."""
    tool = ToolDefinition(
        name="propertyless",
        description="A nested object that requires a key it does not declare",
        parameters_schema={
            "type": "object",
            "properties": {
                "payload": {"type": "object", "properties": {}, "required": ["token"]}
            },
            "required": ["payload"],
        },
    )
    ok, err = await engine._validate_arguments(tool, {"payload": {}})
    assert ok is False and "payload.token" in err
    assert await engine._validate_arguments(tool, {"payload": {"token": "x"}}) == (True, None)

    # A genuinely free-form object is untouched: it requires nothing.
    free_form = ToolDefinition(
        name="free_form",
        description="A plain dict parameter, requiring no particular key",
        parameters_schema={
            "type": "object",
            "properties": {"m": {"type": "object"}},
            "required": ["m"],
        },
    )
    assert await engine._validate_arguments(free_form, {"m": {"anything": 1}}) == (True, None)


async def test_object_and_array_keywords_apply_without_a_declared_type(engine):
    """JSON Schema applies an object's or an array's keywords whenever the
    *instance* is of that kind, whether or not the schema also spells out a
    ``type``. Gating the branches on ``type`` alone meant a property such as
    ``{"properties": {...}, "required": [...]}`` — legal, and what an MCP or
    OpenAPI import produces — governed nothing at all (PR #381 review)."""
    tool = ToolDefinition(
        name="typeless",
        description="Object and array assertions carried without a type",
        parameters_schema={
            "type": "object",
            "properties": {
                "payload": {
                    "properties": {"token": {"type": "string"}},
                    "required": ["token"],
                },
                "vals": {"items": {"type": "integer"}},
            },
        },
    )
    ok, err = await engine._validate_arguments(tool, {"payload": {}})
    assert ok is False and "payload.token" in err
    ok, err = await engine._validate_arguments(tool, {"payload": {"token": 1}})
    assert ok is False and "payload.token" in err
    assert await engine._validate_arguments(tool, {"payload": {"token": "x"}}) == (
        True,
        None,
    )

    ok, err = await engine._validate_arguments(tool, {"vals": ["a"]})
    assert ok is False and "vals[0]" in err
    assert await engine._validate_arguments(tool, {"vals": [1, 2]}) == (True, None)

    # The other half of the same rule, and the reason the gate is the value's
    # kind rather than the presence of the keywords: object keywords do not
    # apply to a *string*, so one is still admitted by a schema that declares
    # them and no type.
    assert await engine._validate_arguments(tool, {"payload": "a string"}) == (True, None)


async def test_a_schema_asserting_nothing_still_admits_everything(engine):
    """The companion guard on the fix above: widening the gate must not start
    rejecting values under a property that simply says nothing about them."""
    tool = ToolDefinition(
        name="unconstrained",
        description="A property carrying no assertions at all",
        parameters_schema={
            "type": "object",
            "properties": {"free": {"description": "anything at all"}},
        },
    )
    for value in ({"anything": [1, {"x": None}]}, [1, "two", None], "scalar", 7, None):
        assert await engine._validate_arguments(tool, {"free": value}) == (True, None), value


async def test_a_gated_probe_is_not_refused_by_the_admission_peek():
    """The read-only admission peek ran before the confirmation gate was
    settled, so it consulted the ``RateLimiter`` for a call that never reaches
    ``acquire`` — the earlier check being *stricter* than the one it stands in
    for. Approved replays of the tool saturate its own key, and the next probe
    came back ``RateLimitExceeded`` instead of ``PENDING_CONFIRMATION``
    (PR #381 review)."""
    from agent_gantry.core.rate_limiter import RateLimiter
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.schema.config import RateLimitConfig
    from agent_gantry.schema.execution import ToolCall

    registry = ToolRegistry()
    gated = ToolDefinition(
        name="delete_thing",
        description="Destructive, and gated behind human confirmation",
        parameters_schema={"type": "object", "properties": {}},
        requires_confirmation=True,
    )
    plain = ToolDefinition(
        name="read_thing",
        description="An ordinary tool with no confirmation gate",
        parameters_schema={"type": "object", "properties": {}},
    )
    for tool in (gated, plain):
        registry.register_tool(tool)
        registry.register_handler(f"default.{tool.name}", lambda: "ok")

    engine = ExecutionEngine(
        registry=registry,
        rate_limiter=RateLimiter(
            RateLimitConfig(
                enabled=True,
                max_calls_per_minute=2,
                max_calls_per_hour=100,
                strategy="sliding_window",
            )
        ),
    )

    # Approved replays *do* record, which is what saturates the gated tool's
    # own key — the limiter is per-tool by default.
    for _ in range(2):
        approved = await engine.execute(
            ToolCall(tool_name="delete_thing", arguments={}, require_confirmation=False)
        )
        assert approved.status.value == "success", approved.error
        assert (
            await engine.execute(ToolCall(tool_name="read_thing", arguments={}))
        ).status.value == "success"

    probe = await engine.execute(ToolCall(tool_name="delete_thing", arguments={}))
    assert probe.status.value == "pending_confirmation", probe.error

    # The exemption is the gate's, not a hole in the limiter: an ungated tool
    # at the same saturation — the loop above spent its two calls too — is
    # still refused.
    refused = await engine.execute(ToolCall(tool_name="read_thing", arguments={}))
    assert refused.status.value == "failure"
    assert refused.error_type == "RateLimitExceeded"


async def test_an_over_quota_gated_call_answers_without_validating(engine):
    """The exemption above must not put the validator back in front of the
    quota. Malformed arguments make ``pending_confirmation`` false, so skipping
    the limiter peek outright for a gated tool let an attacker force the full
    recursive validator — ``re.search`` over caller-controlled payloads — on
    every rejected request while already over quota (PR #381 review).

    The peek runs for every call now; what changes is the *answer*. A gated
    call answers with the gate rather than a denial, because its quota never
    charges it and the window may have room again by the time a human
    approves — and it skips validation either way, which is the work the quota
    exists to protect."""
    from agent_gantry.core.rate_limiter import RateLimiter
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.schema.config import RateLimitConfig
    from agent_gantry.schema.execution import ToolCall

    registry = ToolRegistry()
    tool = ToolDefinition(
        name="delete_thing",
        description="Destructive, and gated behind human confirmation",
        parameters_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
        requires_confirmation=True,
    )
    registry.register_tool(tool)
    registry.register_handler("default.delete_thing", lambda x=0: x)

    saturated = ExecutionEngine(
        registry=registry,
        rate_limiter=RateLimiter(
            RateLimitConfig(
                enabled=True,
                max_calls_per_minute=2,
                max_calls_per_hour=100,
                strategy="sliding_window",
            )
        ),
    )
    for i in range(2):
        approved = await saturated.execute(
            ToolCall(
                tool_name="delete_thing", arguments={"x": i}, require_confirmation=False
            )
        )
        assert approved.status.value == "success", approved.error

    validations = {"count": 0}
    original = saturated._validate_call_arguments

    async def counting(*args, **kwargs):
        validations["count"] += 1
        return await original(*args, **kwargs)

    saturated._validate_call_arguments = counting

    malformed = await saturated.execute(
        ToolCall(tool_name="delete_thing", arguments={"x": "not an integer"})
    )
    assert malformed.status.value == "pending_confirmation", malformed.error
    assert validations["count"] == 0

    # With quota to spare, a malformed call to the same tool is terminal — the
    # property the early exit must not cost, since deferring a call that can
    # never succeed puts a schema violation in front of a human to approve.
    spare = ExecutionEngine(
        registry=registry,
        rate_limiter=RateLimiter(
            RateLimitConfig(
                enabled=True,
                max_calls_per_minute=50,
                max_calls_per_hour=100,
                strategy="sliding_window",
            )
        ),
    )
    terminal = await spare.execute(
        ToolCall(tool_name="delete_thing", arguments={"x": "not an integer"})
    )
    assert terminal.status.value == "failure"
    assert terminal.error_type == "ValidationError"
    assert (
        await spare.execute(ToolCall(tool_name="delete_thing", arguments={"x": 1}))
    ).status.value == "pending_confirmation"


async def test_an_over_quota_gated_call_still_reports_a_policy_denial():
    """The over-quota exemption above answered with the gate straight from the
    admission peek — which consults ``would_exceed_rate_limit`` and nothing
    else. A call that also violated ``allowed_domains`` therefore came back
    ``pending_confirmation``, asking a human to approve something policy would
    refuse on replay, and reporting a different status than the identical call
    with quota to spare (PR #385 review).

    A denial outranks a gate everywhere else in ``execute``; it does here too
    now. What must *not* come back is the validator: that is the expensive,
    caller-controlled work the quota is protecting."""
    from agent_gantry.core.rate_limiter import RateLimiter
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.core.security import SecurityPolicy
    from agent_gantry.schema.config import RateLimitConfig
    from agent_gantry.schema.execution import ToolCall

    def build() -> tuple[ToolRegistry, SecurityPolicy]:
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="fetch_page",
            description="Fetches a page, gated behind human confirmation",
            parameters_schema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            requires_confirmation=True,
        )
        registry.register_tool(tool)
        registry.register_handler("default.fetch_page", lambda url="": url)
        return registry, SecurityPolicy(allowed_domains=["good.example"])

    registry, policy = build()
    saturated = ExecutionEngine(
        registry=registry,
        security_policy=policy,
        rate_limiter=RateLimiter(
            RateLimitConfig(
                enabled=True,
                max_calls_per_minute=2,
                max_calls_per_hour=100,
                strategy="sliding_window",
            )
        ),
    )
    # Approved replays record, saturating the tool's own limiter key while
    # leaving the *policy* window untouched — which is the shape that reaches
    # the exemption: ``_admission_denial`` reports the ``RateLimiter``.
    for _ in range(2):
        approved = await saturated.execute(
            ToolCall(
                tool_name="fetch_page",
                arguments={"url": "https://good.example/x"},
                require_confirmation=False,
            )
        )
        assert approved.status.value == "success", approved.error
    assert saturated._rate_limiter.would_exceed("fetch_page", "default")
    assert policy.would_exceed_rate_limit() is None

    validations = {"count": 0}
    original = saturated._validate_call_arguments

    async def counting(*args, **kwargs):
        validations["count"] += 1
        return await original(*args, **kwargs)

    saturated._validate_call_arguments = counting

    denied = await saturated.execute(
        ToolCall(tool_name="fetch_page", arguments={"url": "https://evil.example/x"})
    )
    assert denied.status.value == "permission_denied", denied.error
    assert denied.error_type == "PermissionDeniedError"
    assert "evil.example" in denied.error

    # The gate is still the answer when policy has no objection...
    gated = await saturated.execute(
        ToolCall(tool_name="fetch_page", arguments={"url": "https://good.example/y"})
    )
    assert gated.status.value == "pending_confirmation", gated.error

    # ...and neither answer bought the caller any validation.
    assert validations["count"] == 0

    # The same two calls with quota to spare agree, which is the invariant that
    # broke: the peek changes *when* an answer is reached, never which one.
    spare_registry, spare_policy = build()
    spare = ExecutionEngine(
        registry=spare_registry,
        security_policy=spare_policy,
        rate_limiter=RateLimiter(
            RateLimitConfig(
                enabled=True,
                max_calls_per_minute=50,
                max_calls_per_hour=100,
                strategy="sliding_window",
            )
        ),
    )
    unsaturated_denial = await spare.execute(
        ToolCall(tool_name="fetch_page", arguments={"url": "https://evil.example/x"})
    )
    assert unsaturated_denial.status.value == denied.status.value
    assert unsaturated_denial.error_type == denied.error_type


async def test_an_over_quota_gated_call_runs_a_custom_policy_check():
    """``would_exceed_rate_limit`` is the only question the admission peek can
    ask, so a replacement policy's own denial reason was invisible to it — the
    finding is not confined to ``allowed_domains``. Every check
    ``check_permission`` makes has to run before a human is asked to approve
    the call (PR #385 review)."""
    from agent_gantry.core.rate_limiter import RateLimiter
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.core.security import PermissionDeniedError, SecurityPolicy
    from agent_gantry.schema.config import RateLimitConfig
    from agent_gantry.schema.execution import ToolCall

    class RefusesFridays(SecurityPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.probes = 0

        def check_permission(self, tool_name, arguments, **kwargs):
            self.probes += 1
            super().check_permission(tool_name, arguments, **kwargs)
            if arguments.get("day") == "friday":
                raise PermissionDeniedError("Execution denied: not on Fridays.")

    registry = ToolRegistry()
    tool = ToolDefinition(
        name="deploy",
        description="Deploys, gated behind human confirmation",
        parameters_schema={
            "type": "object",
            "properties": {"day": {"type": "string"}},
            "required": ["day"],
        },
        requires_confirmation=True,
    )
    registry.register_tool(tool)
    registry.register_handler("default.deploy", lambda day="": day)

    policy = RefusesFridays()
    engine = ExecutionEngine(
        registry=registry,
        security_policy=policy,
        rate_limiter=RateLimiter(
            RateLimitConfig(
                enabled=True,
                max_calls_per_minute=1,
                max_calls_per_hour=100,
                strategy="sliding_window",
            )
        ),
    )
    approved = await engine.execute(
        ToolCall(
            tool_name="deploy", arguments={"day": "monday"}, require_confirmation=False
        )
    )
    assert approved.status.value == "success", approved.error
    assert engine._rate_limiter.would_exceed("deploy", "default")

    refused = await engine.execute(ToolCall(tool_name="deploy", arguments={"day": "friday"}))
    assert refused.status.value == "permission_denied", refused.error
    assert "not on Fridays" in refused.error
    assert policy.probes == 2


@pytest.mark.asyncio
async def test_root_level_assertions_are_enforced(engine):
    """The top-level walk read ``required``, ``properties``,
    ``patternProperties`` and ``additionalProperties`` off the root and
    nothing else, so every other assertion JSON Schema applies to the argument
    object as a whole bound nothing — a root ``allOf`` naming further required
    keys, a root ``anyOf``/``oneOf``, a root ``const``/``enum``, a root
    ``not``, ``minProperties``/``maxProperties``. Merged and imported schemas
    (MCP, OpenAPI) put constraints there routinely (PR #385 review)."""

    def tool_with(schema: dict) -> ToolDefinition:
        return ToolDefinition(
            name="rooted",
            description="A tool whose root schema carries assertions",
            parameters_schema=schema,
        )

    both = {"a": {"type": "integer"}, "b": {"type": "integer"}}

    # ``allOf`` — a branch's own ``required`` now binds.
    allof = tool_with({"type": "object", "properties": both, "allOf": [{"required": ["a"]}]})
    ok, err = await engine._validate_arguments(allof, {"b": 1})
    assert ok is False and "Missing required parameter: a" in err
    assert await engine._validate_arguments(allof, {"a": 1}) == (True, None)

    # ``anyOf`` — at least one branch must hold.
    anyof = tool_with(
        {
            "type": "object",
            "properties": both,
            "anyOf": [{"required": ["a"]}, {"required": ["b"]}],
        }
    )
    ok, err = await engine._validate_arguments(anyof, {})
    assert ok is False and "does not match any permitted schema" in err
    assert await engine._validate_arguments(anyof, {"b": 1}) == (True, None)

    # ``oneOf`` — exactly one.
    oneof = tool_with(
        {
            "type": "object",
            "properties": both,
            "oneOf": [{"required": ["a"]}, {"required": ["b"]}],
        }
    )
    ok, err = await engine._validate_arguments(oneof, {"a": 1, "b": 2})
    assert ok is False and "exactly one must match" in err
    assert await engine._validate_arguments(oneof, {"a": 1}) == (True, None)

    # ``not`` — a shape the schema exists to forbid.
    negated = tool_with({"type": "object", "properties": both, "not": {"required": ["a"]}})
    ok, err = await engine._validate_arguments(negated, {"a": 1})
    assert ok is False and "required not to" in err
    assert await engine._validate_arguments(negated, {"b": 1}) == (True, None)

    # Object-length constraints.
    sized = tool_with({"type": "object", "properties": both, "minProperties": 2})
    ok, err = await engine._validate_arguments(sized, {"a": 1})
    assert ok is False and "at least 2 properties" in err
    assert await engine._validate_arguments(sized, {"a": 1, "b": 2}) == (True, None)

    capped = tool_with({"type": "object", "properties": both, "maxProperties": 1})
    ok, err = await engine._validate_arguments(capped, {"a": 1, "b": 2})
    assert ok is False and "at most 1 properties" in err

    # A root ``const``/``enum`` pins the whole argument object.
    pinned = tool_with({"type": "object", "properties": both, "const": {"a": 1}})
    ok, err = await engine._validate_arguments(pinned, {"a": 2})
    assert ok is False and "must be" in err
    assert await engine._validate_arguments(pinned, {"a": 1}) == (True, None)


@pytest.mark.asyncio
async def test_root_assertions_do_not_reopen_a_closed_tool_schema(engine):
    """The root is handed to ``_validate_value`` as a schema of *just* its
    assertion keywords. Passing the whole thing would re-walk the properties
    the top-level loop already covers at every dispatch, and would resolve the
    root's ``additionalProperties`` under the nested branch's rules — where
    absent means free-form, while a tool schema treats it as closed."""
    closed = ToolDefinition(
        name="closed",
        description="An ordinary tool schema with no additionalProperties key",
        parameters_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "minProperties": 1,
        },
    )
    # The assertion holds and the closure still stands.
    assert await engine._validate_arguments(closed, {"a": 1}) == (True, None)
    ok, err = await engine._validate_arguments(closed, {"a": 1, "z": 2})
    assert ok is False and "Unknown parameter: z" in err

    # A no-argument tool keeps rejecting arguments, which is the top-level
    # walk's own rule rather than the nested branch's.
    empty = ToolDefinition(
        name="empty",
        description="A tool that declares no parameters at all",
        parameters_schema={"type": "object", "properties": {}},
    )
    ok, err = await engine._validate_arguments(empty, {"z": 1})
    assert ok is False and "Unknown parameter: z" in err


@pytest.mark.asyncio
async def test_the_not_keyword_is_evaluated(engine):
    """``not`` was parsed nowhere: the provider transforms walk into it, so it
    survived a round-trip through them and then bound nothing — a schema whose
    whole purpose is to forbid a shape accepted it (PR #385 review)."""
    forbidding = ToolDefinition(
        name="unlucky",
        description="Takes any integer except one",
        parameters_schema={
            "type": "object",
            "properties": {"n": {"type": "integer", "not": {"const": 13}}},
            "required": ["n"],
        },
    )
    ok, err = await engine._validate_arguments(forbidding, {"n": 13})
    assert ok is False and "Parameter 'n'" in err and "required not to" in err
    assert await engine._validate_arguments(forbidding, {"n": 7}) == (True, None)

    # Boolean schemas mean what they mean here too: ``not: false`` forbids
    # nothing, ``not: true`` forbids everything.
    for negation, accepted in ((False, True), (True, False), ({}, False)):
        tool = ToolDefinition(
            name="boolean_not",
            description="A not-branch spelled as a boolean schema",
            parameters_schema={
                "type": "object",
                "properties": {"n": {"type": "integer", "not": negation}},
            },
        )
        ok, _ = await engine._validate_arguments(tool, {"n": 1})
        assert ok is accepted, negation

    # A ``not`` that is not a schema at all is ignored rather than read as an
    # always-matching branch — which would reject every value and turn a
    # malformed schema into an uncallable tool.
    malformed = ToolDefinition(
        name="malformed_not",
        description="A not-branch that is not a schema at all",
        parameters_schema={
            "type": "object",
            "properties": {"n": {"type": "integer", "not": "nonsense"}},
        },
    )
    assert await engine._validate_arguments(malformed, {"n": 1}) == (True, None)


@pytest.mark.asyncio
async def test_an_unevaluable_not_branch_forbids_nothing(engine):
    """`_validate_value` returns *valid* for anything it cannot interpret —
    right everywhere the answer is an assertion about the value. Under ``not``
    that reading inverts: a branch reported as matching forbids the value, so
    a schema the validator merely fails to understand rejected
    **everything**. `not: {"$ref": …}` is legal and reachable, since imported
    (MCP, OpenAPI) schemas are stored as given, and it turned its tool into
    one no call could satisfy (PR #386 review).

    Fail-open is the safe direction here: an unenforced ``not`` is the
    behaviour that shipped before the keyword was implemented at all, whereas
    a wrongly-enforced one is a tool nobody can call."""

    def tool_with(prop: dict) -> ToolDefinition:
        return ToolDefinition(
            name="referenced",
            description="A not-branch reaching for a local definition",
            parameters_schema={
                "type": "object",
                "$defs": {"forbidden": {"type": "string", "const": "nope"}},
                "properties": {"n": prop},
                "required": ["n"],
            },
        )

    # Every keyword the validator skips, in the position that inverts.
    for unevaluable in (
        {"$ref": "#/$defs/forbidden"},
        {"if": {"const": 1}, "then": {"const": 2}},
        {"contains": {"const": 1}},
        # One level down does the same damage: the outer structure evaluates,
        # the nested reference reports valid, and the branch again matches
        # more values than it should.
        {"type": "object", "properties": {"a": {"$ref": "#/$defs/forbidden"}}},
    ):
        tool = tool_with({"type": "integer", "not": unevaluable})
        for value in (1, 13, 42):
            assert await engine._validate_arguments(tool, {"n": value}) == (
                True,
                None,
            ), (unevaluable, value)

    # A branch it *can* evaluate still binds, annotations and inert
    # ``$defs`` included — those carry no constraint, so their presence must
    # not make a schema indeterminate.
    for evaluable in (
        {"const": 13},
        {"const": 13, "title": "unlucky", "$comment": "no thirteens"},
        {"const": 13, "$defs": {"unused": {"$ref": "#/nowhere"}}},
    ):
        tool = tool_with({"type": "integer", "not": evaluable})
        ok, err = await engine._validate_arguments(tool, {"n": 13})
        assert ok is False and "required not to" in err, evaluable
        assert await engine._validate_arguments(tool, {"n": 7}) == (True, None), evaluable


def test_fully_evaluable_certifies_only_what_the_validator_applies():
    """The predicate behind the ``not`` guard, checked directly: boolean
    schemas and the empty schema are evaluable (they are exactly what they
    say), a keyword the validator skips is not, and the check recurses through
    every position a subschema can occupy."""
    from agent_gantry.core.executor import _fully_evaluable

    for evaluable in (
        True,
        False,
        {},
        {"const": 1},
        {"type": "object", "properties": {"a": {"type": "string"}}},
        {"anyOf": [{"type": "string"}, {"type": "null"}]},
        {"items": {"type": "integer"}, "minItems": 1},
        {"additionalProperties": False},
        # Inert: nothing names these without a ``$ref``, which is itself caught.
        {"const": 1, "$defs": {"a": {"$ref": "#/x"}}},
    ):
        assert _fully_evaluable(evaluable) is True, evaluable

    for indeterminate in (
        {"$ref": "#/x"},
        {"unevaluatedProperties": False},
        {"propertyNames": {"pattern": "^a"}},
        {"type": "object", "properties": {"a": {"$ref": "#/x"}}},
        {"anyOf": [{"type": "string"}, {"$ref": "#/x"}]},
        {"items": {"$ref": "#/x"}},
        {"not": {"$ref": "#/x"}},
        {"patternProperties": {"^a": {"$ref": "#/x"}}},
        "not a schema at all",
        None,
    ):
        assert _fully_evaluable(indeterminate) is False, indeterminate

    # A self-referential structure terminates rather than spinning, and the
    # depth cutoff answers "indeterminate", which is the fail-open side.
    deep: dict = {"type": "object"}
    node = deep
    for _ in range(40):
        node["not"] = {"type": "object"}
        node = node["not"]
    assert _fully_evaluable(deep) is False


@pytest.mark.asyncio
async def test_a_combinator_branch_constrains_without_closing(engine):
    """A branch says what it says *about the keys it names* — it is not a
    description of the whole object. Evaluating one through the same object
    path as a tool schema applied Gantry's closed-by-default reading to it, so
    a root ``allOf`` branch declaring ``b`` rejected ``{"a": 1, "b": 2}`` with
    ``Unknown parameter: a``, ``a`` being the *root's* own property
    (PR #386 review).

    The mirror of it: with the branches now evaluated, a key only a branch
    declares was checked against that branch and then reported unknown,
    because only the root's own ``properties`` counted as declared. Validating
    a key and rejecting it as undeclared in one pass is incoherent whichever
    default is right."""

    def tool_with(schema: dict) -> ToolDefinition:
        return ToolDefinition(
            name="composed", description="A composed schema", parameters_schema=schema
        )

    composed = tool_with(
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "allOf": [{"properties": {"b": {"type": "integer"}}}],
        }
    )
    assert await engine._validate_arguments(composed, {"a": 1, "b": 2}) == (True, None)
    assert await engine._validate_arguments(composed, {"a": 1}) == (True, None)

    # Declared by a branch still means *checked* against it.
    ok, err = await engine._validate_arguments(composed, {"a": 1, "b": "x"})
    assert ok is False and "Parameter 'b' must be an integer" in err

    # And a key no branch declares is still refused, so this is not a hole in
    # the tool schema's own closure.
    ok, err = await engine._validate_arguments(composed, {"a": 1, "z": 9})
    assert ok is False and "Unknown parameter: z" in err

    # Sibling branches each contribute their names.
    siblings = tool_with(
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "anyOf": [
                {"properties": {"b": {"type": "integer"}}},
                {"properties": {"c": {"type": "integer"}}},
            ],
        }
    )
    assert await engine._validate_arguments(siblings, {"a": 1, "c": 3}) == (True, None)

    # The same holds one level down, where a nested object carries its own
    # combinator.
    nested = tool_with(
        {
            "type": "object",
            "properties": {
                "p": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "allOf": [{"properties": {"b": {"type": "integer"}}}],
                }
            },
        }
    )
    assert await engine._validate_arguments(nested, {"p": {"a": 1, "b": 2}}) == (True, None)
    ok, err = await engine._validate_arguments(nested, {"p": {"a": 1, "z": 9}})
    assert ok is False and "Unknown parameter: p.z" in err

    # An *explicit* closure inside a branch is the schema saying so rather
    # than a default being inferred, so it is still honoured: this branch
    # permits nothing but ``b``, and ``allOf`` intersects.
    explicit = tool_with(
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "allOf": [
                {"properties": {"b": {"type": "integer"}}, "additionalProperties": False}
            ],
        }
    )
    ok, err = await engine._validate_arguments(explicit, {"a": 1, "b": 2})
    assert ok is False and "Unknown parameter: a" in err

    # ``not`` names the shape a value must *not* take, so it declares nothing.
    negated = tool_with(
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "not": {"properties": {"b": {"type": "integer"}}},
        }
    )
    ok, err = await engine._validate_arguments(negated, {"a": 1, "b": 2})
    assert ok is False and "required not to" in err


@pytest.mark.asyncio
async def test_a_plain_tool_schema_is_still_closed(engine):
    """The guard on the change above: Gantry closes an object whose
    ``additionalProperties`` is absent, stricter than the spec's open default,
    and that must survive for every schema that is not a combinator branch."""
    plain = ToolDefinition(
        name="plain",
        description="An ordinary tool schema",
        parameters_schema={"type": "object", "properties": {"a": {"type": "integer"}}},
    )
    ok, err = await engine._validate_arguments(plain, {"a": 1, "z": 9})
    assert ok is False and "Unknown parameter: z" in err

    nested = ToolDefinition(
        name="nested",
        description="A nested object with no combinator in sight",
        parameters_schema={
            "type": "object",
            "properties": {
                "p": {"type": "object", "properties": {"a": {"type": "integer"}}}
            },
        },
    )
    ok, err = await engine._validate_arguments(nested, {"p": {"a": 1, "z": 9}})
    assert ok is False and "Unknown parameter: p.z" in err

    empty = ToolDefinition(
        name="empty",
        description="A tool that declares no parameters at all",
        parameters_schema={"type": "object", "properties": {}},
    )
    ok, err = await engine._validate_arguments(empty, {"z": 1})
    assert ok is False and "Unknown parameter: z" in err


def test_root_assertions_are_all_keywords_the_validator_implements():
    """`_ROOT_ASSERTIONS` is hand-maintained and has to stay in step with what
    `_validate_value` actually applies — listing a keyword it ignores would
    advertise an assertion as enforced without enforcing it. `_EVALUATED_KEYWORDS`
    is the same claim from the other direction, so pinning one against the
    other catches the drift a future keyword would introduce."""
    from agent_gantry.core.executor import _EVALUATED_KEYWORDS, _ROOT_ASSERTIONS

    assert set(_ROOT_ASSERTIONS) <= _EVALUATED_KEYWORDS
    # The structural keywords stay with the top-level walk, which owns them.
    assert not set(_ROOT_ASSERTIONS) & {
        "properties",
        "required",
        "additionalProperties",
        "patternProperties",
    }
