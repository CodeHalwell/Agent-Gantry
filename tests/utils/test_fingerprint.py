from datetime import datetime, timezone

from agent_gantry.schema.tool import ToolCapability, ToolDefinition
from agent_gantry.utils.fingerprint import compute_tool_fingerprint


def test_compute_tool_fingerprint_valid():
    tool = ToolDefinition(
        name="test_tool",
        description="A test tool",
        parameters_schema={"type": "object", "properties": {}},
        capabilities=[ToolCapability.READ_DATA],
    )
    fp = compute_tool_fingerprint(tool)
    assert fp.startswith("v1.1:")
    assert len(fp.split(":")[1]) == 16


def test_compute_tool_fingerprint_determinism():
    tool1 = ToolDefinition(
        name="test_tool",
        description="A test tool",
        parameters_schema={"type": "object", "properties": {"a": {"type": "string"}}},
        tags=["a", "b", "c"],
        examples=["ex1", "ex2"],
        capabilities=[ToolCapability.READ_DATA, ToolCapability.WRITE_DATA],
    )

    # Tool 2 has the exact same content but the lists are in different order
    tool2 = ToolDefinition(
        name="test_tool",
        description="A test tool",
        parameters_schema={"type": "object", "properties": {"a": {"type": "string"}}},
        tags=["c", "a", "b"],
        examples=["ex2", "ex1"],
        capabilities=[ToolCapability.WRITE_DATA, ToolCapability.READ_DATA],
    )

    assert compute_tool_fingerprint(tool1) == compute_tool_fingerprint(tool2)


def test_compute_tool_fingerprint_sensitivity():
    base_tool = ToolDefinition(
        name="test_tool",
        description="A test tool",
        parameters_schema={"type": "object", "properties": {"a": {"type": "string"}}},
        tags=["a"],
        examples=["ex1"],
        capabilities=[ToolCapability.READ_DATA],
    )
    base_fp = compute_tool_fingerprint(base_tool)

    # Change semantic field: name
    tool_diff_name = base_tool.model_copy(update={"name": "test_tool_diff"})
    assert compute_tool_fingerprint(tool_diff_name) != base_fp

    # Change semantic field: description
    tool_diff_desc = base_tool.model_copy(update={"description": "Different description"})
    assert compute_tool_fingerprint(tool_diff_desc) != base_fp

    # Change semantic field: capabilities
    tool_diff_cap = base_tool.model_copy(update={"capabilities": [ToolCapability.WRITE_DATA]})
    assert compute_tool_fingerprint(tool_diff_cap) != base_fp

    # Change semantic field: requires_confirmation
    tool_diff_req = base_tool.model_copy(update={"requires_confirmation": True})
    assert compute_tool_fingerprint(tool_diff_req) != base_fp

    # Persisted routing/lifecycle fields are covered too — stores serve the
    # stored ToolDefinition back to the router, so definition-only changes
    # must re-sync the stored copy
    tool_diff_source = base_tool.model_copy(update={"source_uri": "http://example.com/tool"})
    assert compute_tool_fingerprint(tool_diff_source) != base_fp

    tool_diff_version = base_tool.model_copy(update={"version": "2.0.0"})
    assert compute_tool_fingerprint(tool_diff_version) != base_fp

    tool_diff_meta = base_tool.model_copy(update={"metadata": {"extra": "data"}})
    assert compute_tool_fingerprint(tool_diff_meta) != base_fp

    # Volatile per-instantiation fields stay excluded — covering created_at
    # or health would defeat incremental sync entirely
    tool_same_created = base_tool.model_copy(update={"created_at": datetime.now(timezone.utc)})
    assert compute_tool_fingerprint(tool_same_created) == base_fp
