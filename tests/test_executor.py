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

@pytest.mark.asyncio
async def test_validate_arguments_wrong_array_item_type(engine, sample_tool):
    """Test incorrect array item types fail validation."""
    is_valid, error = await engine._validate_arguments(sample_tool, {"name": "Alice", "tags": ["admin", 123]})
    assert is_valid is False
    assert error == "Item at index 1 in 'tags' must be a string"
