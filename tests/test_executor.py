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
