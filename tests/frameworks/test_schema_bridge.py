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


def test_combinator_only_field_becomes_a_union():
    """``{"anyOf": [{"type": "integer"}, {"type": "null"}]}`` is what Pydantic
    emits for ``int | None``, so it appears in every nested model this bridge
    inlines. Falling through to ``Any`` advertised an unconstrained field that
    accepted values the executor rejects after dispatch (PR #381 review)."""
    schema = {
        "type": "object",
        "properties": {"count": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
        "required": ["count"],
    }
    model = pydantic_model_from_schema("Args", schema)
    assert model(count=3).count == 3
    assert model(count=None).count is None
    with pytest.raises(ValidationError):
        model(count="bad")


def test_null_only_field_rejects_other_types():
    """A property permitting *only* null must not widen to ``Any`` — the
    framework would then accept strings and numbers the canonical schema
    forbids (PR #381 review)."""
    for schema_type in ("null", ["null"]):
        schema = {
            "type": "object",
            "properties": {"nothing": {"type": schema_type}},
            "required": ["nothing"],
        }
        model = pydantic_model_from_schema("Args", schema)
        assert model(nothing=None).nothing is None
        with pytest.raises(ValidationError):
            model(nothing="oops")


def test_const_becomes_a_single_value_literal():
    """``const`` is what Pydantic emits for a single-value ``Literal``, so it
    turns up inside the nested models this bridge inlines. Ignoring it
    advertised an unconstrained scalar the executor then rejected
    (PR #381 review)."""
    schema = {
        "type": "object",
        "properties": {"kind": {"type": "string", "const": "expected"}},
        "required": ["kind"],
    }
    model = pydantic_model_from_schema("Args", schema)
    assert model(kind="expected").kind == "expected"
    with pytest.raises(ValidationError):
        model(kind="anything")


def test_oneof_requires_exactly_one_matching_branch():
    """A Python union is ``anyOf`` semantics: it accepts a value matching
    several branches, which ``oneOf`` forbids. ``1`` satisfies both
    ``number`` and ``integer`` (PR #381 review)."""
    schema = {
        "type": "object",
        "properties": {"v": {"oneOf": [{"type": "number"}, {"type": "integer"}]}},
        "required": ["v"],
    }
    model = pydantic_model_from_schema("Args", schema)
    with pytest.raises(ValidationError):
        model(v=1)  # matches both branches
    assert model(v=1.5).v == 1.5  # matches only "number"
    with pytest.raises(ValidationError):
        model(v="x")  # matches neither


def test_oneof_with_disjoint_branches_accepts_each():
    schema = {
        "type": "object",
        "properties": {"v": {"oneOf": [{"type": "string"}, {"type": "integer"}]}},
        "required": ["v"],
    }
    model = pydantic_model_from_schema("Args", schema)
    assert model(v="a").v == "a"
    assert model(v=7).v == 7


def test_anyof_still_permits_overlapping_branches():
    """The exclusivity check must not leak into ``anyOf``."""
    schema = {
        "type": "object",
        "properties": {"v": {"anyOf": [{"type": "number"}, {"type": "integer"}]}},
        "required": ["v"],
    }
    model = pydantic_model_from_schema("Args", schema)
    assert model(v=1).v == 1
    assert model(v=1.5).v == 1.5


def test_enum_including_null_becomes_a_literal():
    """``{"enum": ["auto", None]}`` is how a nullable choice is expressed —
    ``None`` is a valid ``Literal`` member, and excluding it dropped the whole
    enum to the unconstrained fallback (PR #381 review)."""
    schema = {
        "type": "object",
        "properties": {"mode": {"enum": ["auto", None]}},
        "required": ["mode"],
    }
    model = pydantic_model_from_schema("Args", schema)
    assert model(mode="auto").mode == "auto"
    assert model(mode=None).mode is None
    with pytest.raises(ValidationError):
        model(mode="zzz")


def test_oneof_exclusivity_sees_the_raw_input_for_object_branches():
    """The check runs before the union converts. An ``AfterValidator`` would
    receive the model the union already built and count only that one branch,
    silently passing a payload matching several (PR #381 review)."""
    branch = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    model = pydantic_model_from_schema(
        "Args",
        {"type": "object", "properties": {"v": {"oneOf": [branch, branch]}}, "required": ["v"]},
    )
    with pytest.raises(ValidationError):
        model(v={"a": 1})  # satisfies both identical branches


def test_oneof_with_disjoint_object_branches_still_works():
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {
                "v": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {"a": {"type": "integer"}},
                            "required": ["a"],
                        },
                        {
                            "type": "object",
                            "properties": {"b": {"type": "string"}},
                            "required": ["b"],
                        },
                    ]
                }
            },
            "required": ["v"],
        },
    )
    assert model(v={"a": 1}).v.a == 1
    assert model(v={"b": "x"}).v.b == "x"


def test_closed_object_without_properties_key_rejects_all_keys():
    """``{"type": "object", "additionalProperties": false}`` permits no keys
    whether or not it spells out an empty ``properties`` — the executor
    enforces that either way (PR #381 review)."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"v": {"type": "object", "additionalProperties": False}},
            "required": ["v"],
        },
    )
    assert model(v={}) is not None
    with pytest.raises(ValidationError):
        model(v={"anything": 1})


def test_type_list_with_several_members_becomes_a_union():
    """The executor validates a list-typed property against *any* listed
    member, so collapsing to the first would reject values it accepts."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"v": {"type": ["string", "integer"]}},
            "required": ["v"],
        },
    )
    assert model(v="a").v == "a"
    assert model(v=7).v == 7


def test_nullable_enum_field_matches_executor_semantics():
    """``enum`` is an independent constraint: ``{"type": ["string","null"],
    "enum": ["a","b"]}`` does not admit null, and the executor enforces that.
    Widening here would pass framework validation and fail at dispatch."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"mode": {"type": ["string", "null"], "enum": ["a", "b"]}},
            "required": ["mode"],
        },
    )
    with pytest.raises(ValidationError):
        model(mode=None)
    assert model(mode="a").mode == "a"

    # …but an enum that *does* list null stays nullable.
    permissive = pydantic_model_from_schema(
        "Args2",
        {
            "type": "object",
            "properties": {"mode": {"type": ["string", "null"], "enum": ["a", None]}},
            "required": ["mode"],
        },
    )
    assert permissive(mode=None).mode is None


def test_oneof_rejects_values_matching_no_branch():
    """Rejecting only ``> 1`` left the zero-match case to the union, whose
    coercion accepts a value no branch admits — ``"1"`` matches neither a
    strict ``number`` nor a strict ``integer`` (PR #381 review)."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"v": {"oneOf": [{"type": "number"}, {"type": "integer"}]}},
            "required": ["v"],
        },
    )
    with pytest.raises(ValidationError):
        model(v="1")
    assert model(v=1.5).v == 1.5


def test_constraint_keywords_reach_the_generated_field():
    """The executor enforces these, so omitting them let the framework
    advertise and accept values the engine rejects at dispatch
    (PR #381 review)."""
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "exclusiveMinimum": 0},
            "s": {"type": "string", "minLength": 3, "pattern": "^a"},
            "arr": {"type": "array", "items": {"type": "integer"}, "minItems": 2},
        },
        "required": ["n", "s", "arr"],
    }
    model = pydantic_model_from_schema("Args", schema)
    assert model(n=1, s="abc", arr=[1, 2]) is not None
    for bad in (
        {"n": -1, "s": "abc", "arr": [1, 2]},
        {"n": 1, "s": "ab", "arr": [1, 2]},
        {"n": 1, "s": "bcd", "arr": [1, 2]},
        {"n": 1, "s": "abc", "arr": [1]},
    ):
        with pytest.raises(ValidationError):
            model(**bad)


def test_constraints_on_an_optional_field_still_admit_none():
    """Constraints are folded into the inner annotation, so the field stays
    free to be unioned with ``None`` and carry its own default."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"opt": {"type": "integer", "minimum": 5}},
            "required": [],
        },
    )
    assert model(opt=None).opt is None
    assert model(opt=9).opt == 9
    with pytest.raises(ValidationError):
        model(opt=3)


def test_float_enum_members_are_preserved():
    """A float-valued ``Enum``/``Literal`` parameter is what introspection
    emits; excluding floats dropped the enum to an unconstrained ``float``
    that accepted non-members (PR #381 review)."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"v": {"type": "number", "enum": [0.5, 1.5]}},
            "required": ["v"],
        },
    )
    assert model(v=0.5).v == 0.5
    with pytest.raises(ValidationError):
        model(v=2.5)


def test_constraints_apply_to_array_items():
    """Constraints ride along on the *item* schema too — ``list[Annotated[
    int, Field(gt=0)]]`` puts them there, not on the outer property
    (PR #381 review)."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {
                "nums": {"type": "array", "items": {"type": "integer", "exclusiveMinimum": 0}}
            },
            "required": ["nums"],
        },
    )
    assert model(nums=[1, 2]).nums == [1, 2]
    with pytest.raises(ValidationError):
        model(nums=[-1])


def test_constraints_apply_to_typed_additional_properties():
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "object",
                    "additionalProperties": {"type": "integer", "minimum": 0},
                }
            },
            "required": ["scores"],
        },
    )
    assert model(scores={"a": 1}).scores == {"a": 1}
    with pytest.raises(ValidationError):
        model(scores={"a": -1})


def test_combinators_apply_alongside_an_explicit_type():
    """A schema may carry both — ``{"type": "integer", "anyOf": [...]}`` means
    the value must be an integer *and* satisfy a branch. Gating the combinator
    on a missing ``type`` let this through as a bare ``int``, and a
    constraint-only branch needs the parent's type pushed into it to mean
    anything (PR #381 review)."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "anyOf": [{"minimum": 10}, {"maximum": 0}]}
            },
            "required": ["n"],
        },
    )
    assert model(n=15).n == 15
    assert model(n=-3).n == -3
    for bad in (5, "x"):
        with pytest.raises(ValidationError):
            model(n=bad)


def test_plain_combinator_without_a_type_is_unaffected():
    """The common Pydantic ``int | None`` shape must still resolve normally."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"v": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
            "required": ["v"],
        },
    )
    assert model(v=3).v == 3
    assert model(v=None).v is None
    with pytest.raises(ValidationError):
        model(v="bad")


def test_prefix_items_type_each_position_independently():
    """``prefixItems`` is what Pydantic emits for ``tuple[int, str]``, and the
    executor validates each position against its own entry. A bare ``list``
    advertised an array of anything (PR #381 review)."""
    model = pydantic_model_from_schema(
        "Args",
        {
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
    assert model(pair=[1, "a"]).pair == [1, "a"]
    # The runtime type stays a list — the executor's validator requires a JSON
    # array, so coercing to a tuple here would trade a permissive framework
    # for a rejected dispatch.
    assert isinstance(model(pair=[1, "a"]).pair, list)
    with pytest.raises(ValidationError):
        model(pair=["bad", 42])


def test_items_alongside_prefix_items_covers_the_tail():
    """``items`` types the positions past the prefix, matching the executor."""
    model = pydantic_model_from_schema(
        "Args",
        {
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
    assert model(row=["label", 1, 2]).row == ["label", 1, 2]
    with pytest.raises(ValidationError):
        model(row=["label", 1, "not-an-int"])


def test_float_const_becomes_a_single_value_literal():
    """The adjacent ``enum`` path already admits floats; excluding them here
    dropped ``{"type": "number", "const": 0.5}`` to an unconstrained ``float``
    that accepted values the executor rejects (PR #381 review)."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"ratio": {"type": "number", "const": 0.5}},
            "required": ["ratio"],
        },
    )
    assert model(ratio=0.5).ratio == 0.5
    with pytest.raises(ValidationError):
        model(ratio=1.5)


def test_unique_items_rejects_duplicates():
    """Gantry's own introspection emits ``uniqueItems`` for every ``set``
    parameter and the executor rejects duplicates at dispatch, so a model that
    accepted them waved through a call the engine refuses (PR #381 review)."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
            },
            "required": ["tags"],
        },
    )
    assert model(tags=["a", "b"]).tags == ["a", "b"]
    with pytest.raises(ValidationError):
        model(tags=["a", "a"])


def test_unique_items_handles_unhashable_members():
    """Array members may be objects or arrays, which a ``set`` cannot hold —
    the executor compares by equality and so must the bridge."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"x": {"type": "integer"}}},
                    "uniqueItems": True,
                }
            },
            "required": ["rows"],
        },
    )
    assert len(model(rows=[{"x": 1}, {"x": 2}]).rows) == 2
    with pytest.raises(ValidationError):
        model(rows=[{"x": 1}, {"x": 1}])


def test_empty_anyof_branch_admits_everything():
    """``{}`` is the always-valid JSON Schema, not an absent branch. Dropping
    it made ``anyOf: [{}, {"type": "integer"}]`` — which admits every value —
    reject strings (PR #381 review)."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"v": {"anyOf": [{}, {"type": "integer"}]}},
            "required": ["v"],
        },
    )
    assert model(v=1).v == 1
    assert model(v="str").v == "str"
    assert model(v=[1]).v == [1]


def test_empty_oneof_branch_makes_other_branches_ambiguous():
    """With ``oneOf`` the empty branch matches everything, so a value another
    branch also admits matches twice and must be rejected — only a value no
    other branch accepts is valid."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"v": {"oneOf": [{}, {"type": "integer"}]}},
            "required": ["v"],
        },
    )
    assert model(v="str").v == "str"
    with pytest.raises(ValidationError):
        model(v=1)


def test_nullable_type_list_survives_below_a_field():
    """A top-level field gets ``None`` back from ``_is_nullable`` in
    ``_build_model``, but an array item never passes through there — so
    ``{"type": "array", "items": {"type": ["string", "null"]}}`` became
    ``list[str]`` and rejected a schema-valid ``[null]`` (PR #381 review)."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": ["string", "null"]}}},
            "required": ["tags"],
        },
    )
    assert model(tags=["a", None]).tags == ["a", None]
    with pytest.raises(ValidationError):
        model(tags=[1])


def test_nullable_type_list_survives_in_additional_properties():
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {
                "counts": {
                    "type": "object",
                    "additionalProperties": {"type": ["integer", "null"]},
                }
            },
            "required": ["counts"],
        },
    )
    assert model(counts={"a": None}).counts == {"a": None}
    with pytest.raises(ValidationError):
        model(counts={"a": "x"})


def test_multi_member_nullable_type_list_keeps_null():
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {
                "v": {"type": "array", "items": {"type": ["string", "integer", "null"]}}
            },
            "required": ["v"],
        },
    )
    assert model(v=["a", 1, None]).v == ["a", 1, None]


def test_enum_excluding_null_still_wins_over_the_type_list():
    """``{"type": ["string", "null"], "enum": ["a", "b"]}`` does *not* admit
    null — the enum is an independent constraint, and the executor enforces
    it. The recursive path must respect that as the top level does."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {
                "v": {
                    "type": "array",
                    "items": {"type": ["string", "null"], "enum": ["a", "b"]},
                }
            },
            "required": ["v"],
        },
    )
    assert model(v=["a"]).v == ["a"]
    with pytest.raises(ValidationError):
        model(v=["a", None])


def test_unique_items_separates_booleans_from_numbers():
    """Python says ``True == 1``; JSON Schema compares types before values, so
    ``[1, true]`` is two distinct items (PR #381 review). Shares the identity
    helper with the executor so the two cannot drift."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"v": {"type": "array", "uniqueItems": True}},
            "required": ["v"],
        },
    )
    assert model(v=[1, True]).v == [1, True]
    assert model(v=[[1], [True]]).v == [[1], [True]]
    # Numbers stay equal when mathematically equal, as JSON Schema requires.
    with pytest.raises(ValidationError):
        model(v=[1, 1.0])


def test_anyof_and_oneof_are_both_honoured():
    """They are independent assertions a value must satisfy together, but the
    translation returned on whichever it found first — so a schema carrying
    both silently lost ``oneOf``'s exclusivity and the bridge accepted a value
    the executor rejects (PR #381 review)."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {
                "v": {
                    "anyOf": [{"type": "integer"}, {"type": "number"}],
                    "oneOf": [{"type": "integer"}, {"type": "number"}],
                }
            },
            "required": ["v"],
        },
    )
    # ``1`` satisfies ``anyOf`` but matches *both* ``oneOf`` branches.
    with pytest.raises(ValidationError):
        model(v=1)
    # ``"x"`` matches no branch of either.
    with pytest.raises(ValidationError):
        model(v="x")


def test_anyof_alone_is_unaffected_by_the_oneof_pass():
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"v": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
            "required": ["v"],
        },
    )
    assert model(v=3).v == 3
    assert model(v=None).v is None
    with pytest.raises(ValidationError):
        model(v="x")


def test_allof_branches_are_all_enforced():
    """``allOf`` was silently ignored, so an ``allOf``-only property became a
    bare ``Any`` that accepted values the executor rejects. There's no faithful
    Python annotation for an intersection, so each branch is checked against
    the caller's raw value instead (PR #381 review)."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {
                "n": {"allOf": [{"type": "integer", "minimum": 1}, {"maximum": 3}]}
            },
            "required": ["n"],
        },
    )
    assert model(n=2).n == 2
    for bad in (0, 99, "x"):
        with pytest.raises(ValidationError):
            model(n=bad)


def test_allof_pushes_a_sibling_branch_type_into_typeless_branches():
    """``allOf`` intersects, so a type declared by *any* branch applies to the
    value as a whole. Without that, ``{"maximum": 3}`` stayed a bare ``Any``
    and the upper bound went unenforced. Union combinators can't do this —
    there each branch stands alone."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"n": {"type": "integer", "allOf": [{"minimum": 10}]}},
            "required": ["n"],
        },
    )
    assert model(n=10).n == 10
    with pytest.raises(ValidationError):
        model(n=5)


def test_allof_carries_const_branches():
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"m": {"allOf": [{"type": "string"}, {"const": "fixed"}]}},
            "required": ["m"],
        },
    )
    assert model(m="fixed").m == "fixed"
    with pytest.raises(ValidationError):
        model(m="other")


def test_numeric_literal_rejects_a_boolean():
    """Pydantic matches ``True`` against ``Literal[1, 1.5]`` because Python
    says ``True == 1``, but JSON Schema compares types before values and the
    executor's ``enum`` check agrees with the spec (PR #381 review)."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"v": {"enum": [1, 1.5]}},
            "required": ["v"],
        },
    )
    assert model(v=1).v == 1
    with pytest.raises(ValidationError):
        model(v=True)


def test_boolean_enum_still_accepts_booleans():
    model = pydantic_model_from_schema(
        "Args",
        {"type": "object", "properties": {"v": {"enum": [True, False]}}, "required": ["v"]},
    )
    assert model(v=True).v is True


def test_string_enum_keeps_a_bare_literal():
    """The identity guard is attached only where bool/number confusion is
    possible, so a plain string enum pays nothing for it."""
    model = pydantic_model_from_schema(
        "Args",
        {"type": "object", "properties": {"v": {"enum": ["a", "b"]}}, "required": ["v"]},
    )
    assert model(v="a").v == "a"
    with pytest.raises(ValidationError):
        model(v="c")


def test_object_property_counts_reach_the_generated_field():
    """Mirrors the executor's object branch, so the args model doesn't accept a
    mapping the engine rejects at dispatch (PR #381 review)."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"m": {"type": "object", "minProperties": 1, "maxProperties": 2}},
            "required": ["m"],
        },
    )
    assert model(m={"a": 1}).m == {"a": 1}
    for bad in ({}, {"a": 1, "b": 2, "c": 3}):
        with pytest.raises(ValidationError):
            model(m=bad)


def test_nullable_parent_type_reaches_combinator_branches():
    """A nullable parent (``{"type": ["integer", "null"], "anyOf": [...]}``,
    which imported schemas produce) still names one real member type. Not
    inheriting it left both constraint-only branches as a bare ``Any`` that
    matched everything, so the model accepted a value the executor rejects
    (PR #381 review)."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {
                "n": {
                    "type": ["integer", "null"],
                    "anyOf": [{"minimum": 10}, {"maximum": 0}],
                }
            },
            "required": ["n"],
        },
    )
    assert model(n=15).n == 15
    assert model(n=-3).n == -3
    assert model(n=None).n is None
    for bad in (5, "x"):
        with pytest.raises(ValidationError):
            model(n=bad)


def test_genuine_multi_type_parent_is_not_pushed_down():
    """No single type applies to every branch there, so inheriting one would
    invent a constraint the schema never declared."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {
                "v": {"type": ["integer", "string"], "anyOf": [{"minimum": 10}]}
            },
            "required": ["v"],
        },
    )
    assert model(v="anything").v == "anything"


def test_parent_type_intersects_a_combinator_with_typed_branches():
    """A branch declaring its own type doesn't take the inherited one, so the
    parent's ``type`` assertion was simply lost — the model became
    ``float | str`` while the executor applies both and rejects each
    (PR #381 review)."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "anyOf": [{"type": "number"}, {"type": "string"}]}
            },
            "required": ["n"],
        },
    )
    assert model(n=1).n == 1
    for bad in (1.5, "x"):
        with pytest.raises(ValidationError):
            model(n=bad)


def test_parent_number_still_admits_an_integer():
    """JSON Schema's number/integer relationship must survive the
    intersection: an integer *is* a valid number."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {
                "n": {"type": "number", "anyOf": [{"type": "integer"}, {"type": "number"}]}
            },
            "required": ["n"],
        },
    )
    assert model(n=1).n == 1
    assert model(n=1.5).n == 1.5
    with pytest.raises(ValidationError):
        model(n="x")


def test_nullable_parent_intersection_still_admits_null():
    """The parent's type list declares null, so intersecting on the non-null
    member must not reject the very value the schema allows."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {
                "n": {
                    "type": ["integer", "null"],
                    "anyOf": [{"type": "number"}, {"type": "string"}],
                }
            },
            "required": ["n"],
        },
    )
    assert model(n=1).n == 1
    assert model(n=None).n is None
    for bad in (1.5, "x"):
        with pytest.raises(ValidationError):
            model(n=bad)


def test_combinator_without_a_parent_type_is_unconstrained_by_one():
    """Nothing to intersect with — the union stands alone, as before."""
    model = pydantic_model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {"v": {"anyOf": [{"type": "integer"}, {"type": "string"}]}},
            "required": ["v"],
        },
    )
    assert model(v=1).v == 1
    assert model(v="x").v == "x"
    with pytest.raises(ValidationError):
        model(v=1.5)
