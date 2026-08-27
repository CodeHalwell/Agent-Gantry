"""Tests for the signature annotations in ``base.py``.

``ToolSpec.python_signature`` is what frameworks that rebuild their own LLM
schema from a callable — Semantic Kernel, AG2, Google ADK's fallback path —
actually see. Anything the annotation drops is invisible to the model on those
paths, however faithful ``parameters_schema`` itself is, so these tests pin the
mapping directly rather than through any one framework.
"""

from __future__ import annotations

from typing import Literal

from agent_gantry.integrations.frameworks.base import _annotation_for_prop


def test_enum_becomes_a_literal():
    """Previously a bare ``str``, so Semantic Kernel/AG2/ADK-fallback
    advertised ``Literal["fast", "slow"]`` as unconstrained text even though
    ``parameters_schema`` carried the enum (PR #381 review)."""
    assert _annotation_for_prop({"type": "string", "enum": ["fast", "slow"]}) == Literal[
        "fast", "slow"
    ]


def test_const_becomes_a_single_member_literal():
    assert _annotation_for_prop({"type": "string", "const": "fixed"}) == Literal["fixed"]


def test_float_enum_members_are_kept():
    """PEP 586 disallows a float member statically, but every validator
    downstream enforces it correctly at runtime — the alternative is
    advertising no constraint at all."""
    assert _annotation_for_prop({"type": "number", "enum": [0.5, 1.5]}) == Literal[0.5, 1.5]


def test_enum_including_null_is_kept():
    assert _annotation_for_prop(
        {"type": ["string", "null"], "enum": ["a", None]}
    ) == Literal["a", None]


def test_exotic_enum_members_fall_back_to_the_plain_type():
    """No ``Literal`` can hold a dict, so the declared type is better than a
    wrong annotation."""
    assert _annotation_for_prop({"type": "string", "enum": [{"a": 1}]}) is str


def test_array_items_are_annotated_recursively():
    assert _annotation_for_prop(
        {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}}
    ) == list[Literal["a", "b"]]


def test_array_of_plain_strings_still_works():
    assert _annotation_for_prop({"type": "array", "items": {"type": "string"}}) == list[str]


def test_untyped_array_items_keep_the_bare_container():
    """``str`` is the fallback for an unrecognized schema, not a real
    annotation — asserting it would claim an item type the schema never
    declared."""
    assert _annotation_for_prop({"type": "array", "items": {"description": "x"}}) is list
    assert _annotation_for_prop({"type": "array"}) is list


def test_typed_mapping_keeps_its_value_type():
    assert _annotation_for_prop(
        {"type": "object", "additionalProperties": {"type": "integer"}}
    ) == dict[str, int]


def test_object_with_declared_properties_stays_a_dict():
    """A documented gap: rebuilding it as a nested model would change what
    these frameworks introspect in a way the suite can't exercise against
    their real schema derivation."""
    assert (
        _annotation_for_prop({"type": "object", "properties": {"x": {"type": "integer"}}})
        is dict
    )


def test_plain_scalars_are_unchanged():
    assert _annotation_for_prop({"type": "string"}) is str
    assert _annotation_for_prop({"type": "integer"}) is int
