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


def test_null_only_schema_maps_to_nonetype():
    """``_json_type_to_python`` falls back to ``str`` for anything it doesn't
    recognize, so a parameter annotated ``None`` — which introspection emits
    as ``{"type": "null"}`` — was advertised as a string, and the string the
    model produced was then rejected by the executor (PR #381 review)."""
    assert _annotation_for_prop({"type": "null"}) is type(None)
    assert _annotation_for_prop({"type": ["null"]}) is type(None)


def test_nullable_scalar_keeps_its_real_type():
    """``["string", "null"]`` still annotates as the non-null member — the
    signature's optionality is carried by its default, not the annotation."""
    assert _annotation_for_prop({"type": ["string", "null"]}) is str


async def test_ainvoke_keeps_a_null_the_schema_declares():
    """``ToolSpec.ainvoke`` drops ``None`` for optional parameters because
    frameworks materialize unset fields that way — but ``x: int | None = 5``
    advertises ``["integer", "null"]``, so an explicit null is a distinct
    choice. Dropping it handed the handler ``5`` where the caller asked for
    ``None`` (PR #381 review)."""
    from agent_gantry import AgentGantry
    from agent_gantry.adapters.embedders.simple import SimpleEmbedder
    from agent_gantry.integrations.frameworks.base import GantryToolset

    gantry = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @gantry.register(tags=["demo"])
    def set_value(x: int | None = 5) -> str:
        """Set a configured value to the given number."""
        return "none" if x is None else f"got {x}"

    @gantry.register(tags=["demo"])
    def label_it(note: str = "hi") -> str:
        """Attach a textual label to the current item."""
        return f"label:{note}"

    await gantry.sync()
    specs = await GantryToolset(gantry).select("set a configured value", limit=2)
    by_name = {s.name: s for s in specs}
    assert "set_value" in by_name, list(by_name)

    spec = by_name["set_value"]
    assert spec.parameters["properties"]["x"]["type"] == ["integer", "null"]
    # The explicit null survives; omission still lets the default apply.
    assert await spec.ainvoke(x=None) == "none"
    assert await spec.ainvoke() == "got 5"

    # A parameter whose schema does *not* declare null still has it dropped,
    # which is what stops frameworks' synthetic Nones failing validation.
    if "label_it" in by_name:
        assert await by_name["label_it"].ainvoke(note=None) == "label:hi"


def test_a_composite_enum_is_annotated_as_its_container_not_a_string():
    """A tuple-valued ``Enum`` emits ``{"enum": [[0, 0], [1, 1]]}`` — no
    ``type``, since its members share no scalar kind. With no ``Literal`` to
    build and no ``type`` to read, this fell through to the ``str`` fallback
    and advertised a *string* for an array-valued parameter to Semantic
    Kernel, AG2 and ADK's fallback path, so the string the model produced was
    rejected by the executor (PR #381 review)."""
    assert _annotation_for_prop({"enum": [[0, 0], [1, 1]]}) is list
    assert _annotation_for_prop({"enum": [{"a": 1}, {"a": 2}]}) is dict
    assert _annotation_for_prop({"const": [1, 2]}) is list


def test_a_declared_type_outranks_the_members_it_disagrees_with():
    """An explicit ``type`` is the author's statement, so member inference is
    consulted only when there is none."""
    assert _annotation_for_prop({"type": "string", "enum": [{"a": 1}]}) is str
    # Mixed members name no single container, so the fallback still applies.
    assert _annotation_for_prop({"enum": [[0, 0], "x"]}) is str


def test_reconstructed_string_formats_survive_into_the_signature():
    """Semantic Kernel, AG2 and Google ADK's fallback path rebuild their
    provider schema from this signature, so flattening
    ``{"type": "string", "format": "date-time"}`` to ``str`` dropped the format
    the schema had just gained — and the model could then answer with a string
    the handler can't take (PR #381 review)."""
    import datetime
    import uuid

    assert _annotation_for_prop({"type": "string", "format": "date-time"}) is datetime.datetime
    assert _annotation_for_prop({"type": "string", "format": "date"}) is datetime.date
    assert _annotation_for_prop({"type": "string", "format": "time"}) is datetime.time
    assert _annotation_for_prop({"type": "string", "format": "uuid"}) is uuid.UUID
    # Formats Gantry neither emits nor reconstructs stay plain strings.
    assert _annotation_for_prop({"type": "string", "format": "email"}) is str
    assert _annotation_for_prop({"type": "string"}) is str
