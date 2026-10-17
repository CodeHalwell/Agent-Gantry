"""Native tool exports own their schemas and cannot change registry validation."""

from __future__ import annotations

import pytest

from agent_gantry import ToolCall
from agent_gantry.integrations.frameworks.base import GantryToolset
from agent_gantry.schema.execution import ExecutionStatus


@pytest.fixture
def registered(gantry):
    @gantry.register
    def lookup(value: int) -> str:
        """Look up an item using its numeric identifier."""
        return str(value)

    return gantry


async def test_spec_schema_is_a_snapshot_of_the_registry(registered):
    specs = await GantryToolset(registered).select("look up an item", limit=1)
    original = registered.list_tools_sync()[0]
    original.parameters_schema["properties"]["value"]["description"] = "A new description"
    assert "description" not in specs[0].parameters["properties"]["value"]
    specs[0].parameters["properties"]["value"]["type"] = "string"
    assert original.parameters_schema["properties"]["value"]["type"] == "integer"


@pytest.mark.parametrize("framework", ["langchain", "agno", "strands", "dspy", "pydantic_ai"])
async def test_editing_native_schema_cannot_change_the_registry(registered, framework):
    module = {"langchain": "langchain_core", "pydantic_ai": "pydantic_ai"}.get(framework, framework)
    pytest.importorskip(module)
    tool = (await registered.framework(framework).select("look up an item", limit=1))[0]
    if framework == "langchain":
        properties = tool.args_schema["properties"]
    elif framework == "agno":
        properties = tool.parameters["properties"]
    elif framework == "strands":
        properties = tool.tool_spec["inputSchema"]["json"]["properties"]
    elif framework == "dspy":
        properties = tool.args
    else:
        properties = tool.function_schema.json_schema["properties"]
    properties["value"]["type"] = "string"
    assert (
        registered.list_tools_sync()[0].parameters_schema["properties"]["value"]["type"]
        == "integer"
    )
    result = await registered.execute(
        ToolCall(tool_name="lookup", arguments={"value": "not-an-integer"})
    )
    assert result.status == ExecutionStatus.FAILURE
    assert result.error_type == "ValidationError"
    next_spec = (await GantryToolset(registered).select("look up an item", limit=1))[0]
    assert next_spec.parameters["properties"]["value"]["type"] == "integer"
