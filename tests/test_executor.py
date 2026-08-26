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
