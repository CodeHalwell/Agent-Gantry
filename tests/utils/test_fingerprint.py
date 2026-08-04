from datetime import datetime, timezone

import pytest

from agent_gantry.schema.tool import ToolCapability, ToolDefinition
from agent_gantry.utils.fingerprint import compute_tool_fingerprint, parse_fingerprint


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


def test_compute_tool_fingerprint_unsupported_version():
    tool = ToolDefinition(
        name="test_tool",
        description="A test tool",
        parameters_schema={"type": "object", "properties": {}},
    )
    with pytest.raises(ValueError, match="Unsupported fingerprint version: v9.9"):
        compute_tool_fingerprint(tool, version="v9.9")


def test_parse_fingerprint_valid():
    version, hash_val = parse_fingerprint("v1.0:a1b2c3d4e5f67890")
    assert version == "v1.0"
    assert hash_val == "a1b2c3d4e5f67890"


def test_parse_fingerprint_legacy():
    version, hash_val = parse_fingerprint("a1b2c3d4e5f67890")
    assert version == "v1.0"
    assert hash_val == "a1b2c3d4e5f67890"


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

    fp1 = compute_tool_fingerprint(tool1)
    fp2 = compute_tool_fingerprint(tool2)

    assert fp1 == fp2


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

    # v1.1: persisted routing/lifecycle fields are covered too — stores serve
    # the stored ToolDefinition back to the router, so definition-only
    # changes must re-sync the stored copy
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


def test_parse_fingerprint_edge_cases():
    # Test valid formats
    assert parse_fingerprint("v1.0:123456") == ("v1.0", "123456")
    assert parse_fingerprint("legacyhash123") == ("v1.0", "legacyhash123")

    # Empty string should be considered legacy hash
    assert parse_fingerprint("") == ("v1.0", "")

    # Colon at start
    assert parse_fingerprint(":hash123") == ("", "hash123")

    # Colon at end
    assert parse_fingerprint("v1.0:") == ("v1.0", "")

    # Multiple colons (split on first colon only)
    assert parse_fingerprint("v1.0:hash:extra") == ("v1.0", "hash:extra")
