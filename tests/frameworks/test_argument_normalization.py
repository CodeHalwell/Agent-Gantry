"""Normalize framework values without mutating caller-owned argument trees."""

from __future__ import annotations

from datetime import datetime

import pytest

from agent_gantry.integrations.frameworks.base import _json_native


@pytest.mark.parametrize(
    "value,schema",
    [
        ([1, 2, 3], {"type": "array", "items": {"type": "integer"}}),
        ({"a": 1}, {"type": "object", "additionalProperties": {"type": "integer"}}),
        ([{"a": 1}], {"type": "array"}),
        ({"a": [1, 2]}, {"type": "object"}),
    ],
)
def test_native_arguments_keep_their_identity(value, schema):
    assert _json_native(value, schema) is value


def test_copy_only_the_path_with_a_converted_value():
    when = datetime(2026, 10, 16, 12, 30)
    untouched = [1, 2, 3]
    rows = [{"when": "2026-10-15T12:30:00"}, {"when": when}]
    value = {"untouched": untouched, "rows": rows}
    schema = {
        "properties": {
            "rows": {"items": {"properties": {"when": {"format": "date-time"}}}},
        }
    }

    result = _json_native(value, schema)

    assert result == {"untouched": [1, 2, 3], "rows": [rows[0], {"when": when.isoformat()}]}
    assert result is not value
    assert result["rows"] is not rows
    assert result["rows"][0] is rows[0]
    assert result["untouched"] is untouched
    assert value["rows"][1]["when"] is when


def test_tuple_items_are_normalized_alongside_array_conversion():
    when = datetime(2026, 10, 16)
    value = (1, when)
    result = _json_native(value, {"prefixItems": [{}, {"format": "date-time"}]})
    assert result == [1, when.isoformat()]
    assert value[1] is when


def test_combinator_fallback_still_normalizes_nested_values():
    when = datetime(2026, 10, 16)
    assert _json_native(
        [when],
        {
            "anyOf": [
                {"type": "null"},
                {"type": "array", "items": {"format": "date-time"}},
            ]
        },
    ) == [when.isoformat()]


def test_all_matching_property_schemas_apply_before_additional_properties():
    when = datetime(2026, 10, 16)
    schema = {
        "properties": {"a_when": {"type": "string"}},
        "patternProperties": {"^a": {"type": "string"}, "_when$": {"format": "date-time"}},
        "additionalProperties": {"format": "date-time"},
    }
    value = {"a_when": when, "other": when}
    assert _json_native(value, schema) == {"a_when": when.isoformat(), "other": when.isoformat()}
    assert value == {"a_when": when, "other": when}
