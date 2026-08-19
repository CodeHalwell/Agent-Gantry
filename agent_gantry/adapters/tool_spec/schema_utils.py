"""JSON-Schema transforms shared by the provider adapters.

Providers disagree about which JSON Schema they accept. Two transforms cover
the cases that actually break requests:

- :func:`strict_json_schema` — OpenAI's structured-outputs subset, which
  requires ``additionalProperties: false`` and demands that *every* property
  appear in ``required`` (optionality is expressed by admitting ``null``).
- :func:`sanitize_gemini_schema` — strips the keywords Gemini and Vertex AI
  reject outright rather than ignoring.

Both deep-copy their input. Adapters must never hand out a structure that
aliases ``ToolDefinition.parameters_schema``: the registry holds one canonical
copy per tool, so a caller mutating an emitted schema would corrupt every
later conversion of that tool.
"""

from __future__ import annotations

import copy
from typing import Any

__all__ = ["sanitize_gemini_schema", "strict_json_schema"]

#: Keys that introduce a nested subschema whose value is itself a schema.
_SUBSCHEMA_KEYS = ("items", "additionalItems", "contains", "not")

#: Keys whose value is a list of subschemas.
_SUBSCHEMA_LIST_KEYS = ("anyOf", "oneOf", "allOf", "prefixItems")


def _make_nullable(subschema: dict[str, Any]) -> None:
    """Widen ``subschema`` in place so ``null`` is a valid value.

    OpenAI strict mode has no notion of an omitted property — every property is
    required — so a parameter that was optional has to accept ``null`` instead.
    """
    if "anyOf" in subschema and isinstance(subschema["anyOf"], list):
        branches = subschema["anyOf"]
        if not any(isinstance(b, dict) and b.get("type") == "null" for b in branches):
            branches.append({"type": "null"})
        return

    declared = subschema.get("type")
    if isinstance(declared, str):
        if declared != "null":
            subschema["type"] = [declared, "null"]
    elif isinstance(declared, list):
        if "null" not in declared:
            subschema["type"] = [*declared, "null"]
    else:
        # No usable type to widen (e.g. an enum-only or unconstrained schema).
        # Wrapping it in anyOf keeps the original constraints intact.
        remainder = {k: v for k, v in subschema.items() if k != "description"}
        if remainder:
            description = subschema.get("description")
            subschema.clear()
            subschema["anyOf"] = [remainder, {"type": "null"}]
            if description is not None:
                subschema["description"] = description


def _strict_in_place(node: Any) -> None:
    """Recursively apply OpenAI's strict-mode constraints to ``node``."""
    if isinstance(node, list):
        for item in node:
            _strict_in_place(item)
        return
    if not isinstance(node, dict):
        return

    properties = node.get("properties")
    if isinstance(properties, dict):
        # An object schema: every property must be required, and anything the
        # caller left optional has to admit null instead of being omitted.
        previously_required = node.get("required")
        optional = set(properties) - set(
            previously_required if isinstance(previously_required, list) else []
        )
        for name, subschema in properties.items():
            if isinstance(subschema, dict):
                _strict_in_place(subschema)
                if name in optional:
                    _make_nullable(subschema)
        node["required"] = list(properties)
        node["additionalProperties"] = False

    for key in _SUBSCHEMA_KEYS:
        if key in node:
            _strict_in_place(node[key])
    for key in _SUBSCHEMA_LIST_KEYS:
        if isinstance(node.get(key), list):
            _strict_in_place(node[key])
    if isinstance(node.get("$defs"), dict):
        for subschema in node["$defs"].values():
            _strict_in_place(subschema)
    if isinstance(node.get("definitions"), dict):
        for subschema in node["definitions"].values():
            _strict_in_place(subschema)


def strict_json_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Return a deep copy of ``schema`` conforming to OpenAI strict mode.

    OpenAI rejects a tool marked ``strict: true`` unless the parameter schema
    sets ``additionalProperties: false`` and lists every property in
    ``required``. Setting the flag alone — without reshaping the schema — makes
    the API reject any tool that has an optional parameter.

    Optional properties are preserved semantically by widening their type to
    admit ``null`` rather than by dropping them from ``required``.

    Args:
        schema: The tool's JSON-Schema ``parameters`` object.

    Returns:
        A new schema safe to publish alongside ``strict: true``.
    """
    if not schema:
        return {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    transformed = copy.deepcopy(schema)
    _strict_in_place(transformed)
    return transformed


#: Keywords Gemini and Vertex AI reject whose removal cannot change which
#: values a schema accepts — they are annotations or constraints the Google
#: SDKs do not model. ``additionalProperties`` matters most in practice:
#: Agent-Gantry's own introspection emits it for ``**kwargs`` handlers, so
#: leaving it in breaks tools the library itself produces.
#:
#: Structural keywords (``$ref``, ``allOf``, ``oneOf``, ``not``) are
#: deliberately absent: deleting one silently widens or empties the schema,
#: which is worse than the request error it would have caused. Local ``$ref``
#: pointers are resolved by inlining instead (see :func:`_inline_local_refs`).
_GEMINI_UNSUPPORTED = frozenset(
    {
        "additionalProperties",
        "additionalItems",
        "unevaluatedProperties",
        "patternProperties",
        "$schema",
        "$id",
        "$comment",
        "default",
        "examples",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "readOnly",
        "writeOnly",
        "title",
    }
)

#: Guard against pathological or self-referential ``$defs`` graphs.
_MAX_INLINE_DEPTH = 12


def _resolve_pointer(root: dict[str, Any], ref: str) -> dict[str, Any] | None:
    """Resolve a local JSON pointer like ``#/$defs/Address`` against ``root``."""
    if not ref.startswith("#/"):
        return None
    node: Any = root
    for raw in ref[2:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or token not in node:
            return None
        node = node[token]
    return node if isinstance(node, dict) else None


def _inline_local_refs(node: Any, root: dict[str, Any], depth: int = 0) -> Any:
    """Replace local ``$ref`` pointers with the schema they name.

    Pydantic emits ``$defs`` + ``$ref`` for any nested model, and the Google
    SDKs do not follow those pointers. Inlining keeps the schema's meaning;
    dropping the ``$ref`` would not. A self-referential model cannot be
    expressed this way, so the pointer is left untouched past the depth guard
    and the caller gets a real error rather than a silently wrong schema.
    """
    if isinstance(node, list):
        return [_inline_local_refs(item, root, depth) for item in node]
    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")
    if isinstance(ref, str) and depth < _MAX_INLINE_DEPTH:
        target = _resolve_pointer(root, ref)
        if target is not None:
            merged = {k: v for k, v in node.items() if k != "$ref"}
            resolved = _inline_local_refs(copy.deepcopy(target), root, depth + 1)
            if isinstance(resolved, dict):
                # Keys alongside the $ref (e.g. an overriding description) win.
                return {**resolved, **merged}

    return {key: _inline_local_refs(value, root, depth) for key, value in node.items()}


def _sanitize_gemini_in_place(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _sanitize_gemini_in_place(item)
        return
    if not isinstance(node, dict):
        return

    for key in list(node):
        if key in _GEMINI_UNSUPPORTED:
            del node[key]

    # ``const`` is unsupported but is exactly a one-value ``enum``, which is
    # supported — convert rather than drop, so the constraint survives.
    if "const" in node:
        node["enum"] = [node.pop("const")]

    properties = node.get("properties")
    if isinstance(properties, dict):
        for subschema in properties.values():
            _sanitize_gemini_in_place(subschema)

    for key in _SUBSCHEMA_KEYS:
        if key in node:
            _sanitize_gemini_in_place(node[key])
    for key in _SUBSCHEMA_LIST_KEYS:
        if isinstance(node.get(key), list):
            _sanitize_gemini_in_place(node[key])


def sanitize_gemini_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Return a deep copy of ``schema`` accepted by Gemini and Vertex AI.

    Both reject unknown fields in a ``FunctionDeclaration`` rather than
    ignoring them, so JSON-Schema keywords that are valid everywhere else
    (``additionalProperties``, ``default``, ``title``, …) turn into request
    errors. Local ``$ref``/``$defs`` pairs — what Pydantic emits for any nested
    model — are inlined, since the SDKs will not follow the pointers.

    Args:
        schema: The tool's JSON-Schema ``parameters`` object.

    Returns:
        A new schema containing only keywords the Google SDKs accept.
    """
    if not schema:
        return {"type": "object", "properties": {}}
    inlined = _inline_local_refs(copy.deepcopy(schema), schema)
    transformed = inlined if isinstance(inlined, dict) else copy.deepcopy(schema)
    # ``$defs`` exists only to back the pointers just inlined.
    transformed.pop("$defs", None)
    transformed.pop("definitions", None)
    _sanitize_gemini_in_place(transformed)
    return transformed
