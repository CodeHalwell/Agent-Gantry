"""Tests for the JSON-Schema → Pydantic model bridge (``schema_bridge.py``).

``pydantic_model_from_schema`` backs CrewAI's ``args_schema`` and LlamaIndex's
``fn_schema`` — both frameworks want a native ``pydantic.BaseModel`` describing
a tool's parameters rather than a bare JSON schema. These tests exercise the
bridge directly rather than through either framework's adapter, since it's
framework-agnostic and best tested in isolation.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from agent_gantry.integrations.frameworks.schema_bridge import (
    pydantic_model_from_schema,
)


def test_scalar_properties_round_trip():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The name."},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "active": {"type": "boolean"},
        },
        "required": ["name", "count"],
    }
    model = pydantic_model_from_schema("Args", schema)
    assert model is not None
    assert issubclass(model, BaseModel)

    instance = model(name="x", count=1, ratio=0.5, active=True)
    assert instance.name == "x"
    assert instance.count == 1

    with pytest.raises(ValidationError):
        model(count=1)  # missing required "name"

    assert model.model_fields["name"].description == "The name."


def test_optional_field_gets_declared_default():
    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "default": "unset"},
        },
        "required": [],
    }
    model = pydantic_model_from_schema("Args", schema)
    instance = model()
    assert instance.label == "unset"


def test_optional_field_admits_none():
    schema = {
        "type": "object",
        "properties": {"note": {"type": "string"}},
        "required": [],
    }
    model = pydantic_model_from_schema("Args", schema)
    # Frameworks/models routinely send explicit null for "not provided".
    instance = model(note=None)
    assert instance.note is None


def test_enum_becomes_literal():
    schema = {
        "type": "object",
        "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}},
        "required": ["color"],
    }
    model = pydantic_model_from_schema("Args", schema)
    assert model(color="red").color == "red"
    with pytest.raises(ValidationError):
        model(color="purple")


def test_enum_of_exotic_values_falls_back_to_base_type():
    # A JSON-Schema enum could (unusually) contain a null or a nested
    # object; Literal[...] can't express that, so the bridge falls back to
    # the property's declared/base type instead of raising.
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string", "enum": [{"nested": True}, "plain"]}},
        "required": ["value"],
    }
    model = pydantic_model_from_schema("Args", schema)
    assert model is not None
    assert model(value="anything").value == "anything"


def test_typed_array_items():
    schema = {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["tags"],
    }
    model = pydantic_model_from_schema("Args", schema)
    assert model(tags=["a", "b"]).tags == ["a", "b"]
    with pytest.raises(ValidationError):
        model(tags=[1, 2])


def test_nested_object_with_declared_properties_builds_submodel():
    schema = {
        "type": "object",
        "properties": {
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "zip": {"type": "string"},
                },
                "required": ["street"],
            },
        },
        "required": ["address"],
    }
    model = pydantic_model_from_schema("Args", schema)
    instance = model(address={"street": "Main St"})
    assert instance.address.street == "Main St"
    assert issubclass(type(instance.address), BaseModel)
    with pytest.raises(ValidationError):
        model(address={"zip": "12345"})  # missing nested-required "street"


def test_required_nullable_field_admits_none_but_stays_required():
    """``type: ["string", "null"]`` inside ``required`` means the caller must
    supply the key, but the value itself may legitimately be null — distinct
    from an *optional* field, which may be omitted entirely."""
    schema = {
        "type": "object",
        "properties": {"middle_name": {"type": ["string", "null"]}},
        "required": ["middle_name"],
    }
    model = pydantic_model_from_schema("Args", schema)

    assert model(middle_name=None).middle_name is None
    assert model(middle_name="Q").middle_name == "Q"
    with pytest.raises(ValidationError):
        model()  # still required — omitting it entirely must fail


def test_typed_additional_properties_without_declared_properties():
    """A ``dict[str, int]``-shaped schema (no declared ``properties``, a
    schema-valued ``additionalProperties``) should keep the value type
    instead of widening to a bare ``dict`` that accepts any value type."""
    schema = {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "additionalProperties": {"type": "integer"},
            },
        },
        "required": ["scores"],
    }
    model = pydantic_model_from_schema("Args", schema)
    assert model(scores={"a": 1, "b": 2}).scores == {"a": 1, "b": 2}
    with pytest.raises(ValidationError):
        model(scores={"a": "not an int"})


def test_untyped_additional_properties_stays_bare_dict():
    schema = {
        "type": "object",
        "properties": {"metadata": {"type": "object"}},
        "required": ["metadata"],
    }
    model = pydantic_model_from_schema("Args", schema)
    # No declared properties and no schema-valued additionalProperties:
    # free-form dict, any value type is fine.
    assert model(metadata={"a": 1, "b": "two"}).metadata == {"a": 1, "b": "two"}


def test_undeclared_keys_are_rejected_like_the_executor_rejects_them():
    """Pydantic defaults to ``extra="ignore"``, which would silently drop an
    argument the executor would have rejected — a misspelled or hallucinated
    key would vanish inside CrewAI/LlamaIndex instead of surfacing as an
    error. The model mirrors the schema instead (PR #381 review)."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    model = pydantic_model_from_schema("Args", schema)
    with pytest.raises(ValidationError):
        model(name="a", typoed_arg=123)


def test_declared_additional_properties_permits_extras():
    """The ``**kwargs`` shape (``additionalProperties: true``) must stay
    open — mirroring the schema means permissive as well as strict."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": True,
    }
    model = pydantic_model_from_schema("Args", schema)
    assert model(name="a", extra=1).model_dump() == {"name": "a", "extra": 1}


def test_empty_additional_properties_schema_permits_extras():
    """``additionalProperties: {}`` is spec-equivalent to ``true`` — the
    same distinction the executor's ``_permits_additional`` draws."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": {},
    }
    model = pydantic_model_from_schema("Args", schema)
    assert model(name="a", extra=1).model_dump() == {"name": "a", "extra": 1}


def test_typed_additional_properties_constrains_extra_values():
    """A *typed* ``additionalProperties`` constrains the extras' value type.
    Bare ``extra="allow"`` would take every extra as ``Any``, so the
    framework would accept a string where the schema — and the executor,
    which does check the subschema — demand an integer."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": {"type": "integer"},
    }
    model = pydantic_model_from_schema("Args", schema)
    assert model(name="a", hits=3).model_dump() == {"name": "a", "hits": 3}
    with pytest.raises(ValidationError):
        model(name="a", hits="not an int")


def test_untyped_additional_properties_leaves_extras_unconstrained():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": True,
    }
    model = pydantic_model_from_schema("Args", schema)
    assert model(name="a", whatever="ok").model_dump() == {"name": "a", "whatever": "ok"}


def test_closed_empty_nested_object_rejects_all_keys():
    """``{"properties": {}, "additionalProperties": false}`` permits no keys
    at all. A bare ``dict`` annotation would accept anything, so the
    framework would wave through a payload the executor rejects — the same
    closed-empty-object gap already fixed in the executor."""
    schema = {
        "type": "object",
        "properties": {
            "opts": {"type": "object", "properties": {}, "additionalProperties": False}
        },
        "required": ["opts"],
    }
    model = pydantic_model_from_schema("Args", schema)
    assert model(opts={}) is not None
    with pytest.raises(ValidationError):
        model(opts={"anything": 1})


def test_invalid_identifier_property_name_returns_none():
    schema = {
        "type": "object",
        "properties": {"not-an-identifier": {"type": "string"}},
        "required": [],
    }
    assert pydantic_model_from_schema("Args", schema) is None


def test_non_object_schema_returns_none():
    assert pydantic_model_from_schema("Args", {"type": "string"}) is None


def test_self_referential_schema_hits_depth_limit_and_returns_none():
    schema: dict = {"type": "object", "properties": {}, "required": ["child"]}
    schema["properties"]["child"] = schema  # self-reference
    assert pydantic_model_from_schema("Args", schema) is None


def test_base_name_is_sanitized_to_valid_identifier():
    schema = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    }
    model = pydantic_model_from_schema("weird name!!", schema)
    assert model is not None
    assert model(x="ok").x == "ok"
