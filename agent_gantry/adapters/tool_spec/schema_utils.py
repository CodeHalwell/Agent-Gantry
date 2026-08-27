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

__all__ = [
    "sanitize_gemini_schema",
    "strict_json_schema",
    "unsupported_strict_paths",
]

#: Keys that introduce a nested subschema whose value is itself a schema.
_SUBSCHEMA_KEYS = ("items", "additionalItems", "contains", "not")

#: Keys whose value is a list of subschemas.
_SUBSCHEMA_LIST_KEYS = ("anyOf", "oneOf", "allOf", "prefixItems")


def _admit_null_in_enum(subschema: dict[str, Any]) -> None:
    """Let ``null`` through an ``enum`` alongside a widened ``type``.

    ``enum`` is an independent constraint, so widening ``type`` to admit
    ``null`` is not enough on its own: an optional ``Literal["fast",
    "slow"] | None`` would advertise ``type: ["string", "null"]`` while its
    enum still listed only the two strings. Strict mode makes every property
    required, so the model could then not express "not provided" at all —
    the constrained grammar would force it to invent ``"fast"`` or
    ``"slow"`` rather than let the handler apply its ``None`` default.
    """
    enum_values = subschema.get("enum")
    if isinstance(enum_values, list) and enum_values and None not in enum_values:
        subschema["enum"] = [*enum_values, None]


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

    if "const" in subschema:
        # ``const`` is an independent constraint that no ``type`` widening can
        # satisfy — a single-value ``Literal`` (what Pydantic emits as
        # ``{"type": "string", "const": "fixed"}``) would still forbid null,
        # and strict mode makes the property required, so the model could not
        # express omission at all. Wrapping keeps the constant intact while
        # adding a null alternative beside it.
        description = subschema.get("description")
        remainder = {k: v for k, v in subschema.items() if k != "description"}
        subschema.clear()
        subschema["anyOf"] = [remainder, {"type": "null"}]
        if description is not None:
            subschema["description"] = description
        return

    declared = subschema.get("type")
    if isinstance(declared, str):
        if declared != "null":
            subschema["type"] = [declared, "null"]
            _admit_null_in_enum(subschema)
    elif isinstance(declared, list):
        if "null" not in declared:
            subschema["type"] = [*declared, "null"]
            _admit_null_in_enum(subschema)
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

    An object with arbitrary keys (a ``dict[str, int]`` parameter, an
    untyped ``dict``) has no strict-mode representation and is left
    untouched rather than being forced to ``additionalProperties: false``,
    which would turn it into an object accepting no keys at all and
    silently discard the parameter's data. The result is then *not* safe to
    publish with ``strict: true`` — call :func:`unsupported_strict_paths`
    first and fall back to a non-strict request when it reports anything.

    Args:
        schema: The tool's JSON-Schema ``parameters`` object.

    Returns:
        A new schema safe to publish alongside ``strict: true``, provided
        :func:`unsupported_strict_paths` reported nothing for it.
    """
    if not schema:
        return {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    transformed = copy.deepcopy(schema)
    _strict_in_place(transformed)
    return transformed


def _is_open_map(node: dict[str, Any]) -> bool:
    """Whether an object schema admits keys it does not enumerate.

    Strict mode can only describe an object whose full key set is written
    out in ``properties``. Anything else — a ``dict[str, int]`` parameter
    (schema-valued ``additionalProperties``, no ``properties``), a bare
    ``{"type": "object"}`` from an untyped ``dict`` parameter — is an open
    map with no strict-mode representation.
    """
    properties = node.get("properties")
    if isinstance(properties, dict) and properties:
        additional = node.get("additionalProperties")
        # ``additionalProperties: true`` alongside real properties (what a
        # ``**kwargs`` handler emits) is a narrowing strict mode handles by
        # forcing it to false — and so is the empty schema ``{}``, which is
        # spec-equivalent to ``true``. A *typed* ``additionalProperties``
        # still describes keys strict mode cannot express, though: forcing
        # it to false there silently drops the typed extras, the same data
        # loss this function exists to catch for a bare map.
        return isinstance(additional, dict) and bool(additional)
    if node.get("additionalProperties") is False:
        return False  # explicitly closed: an object permitting no keys
    if isinstance(properties, dict):
        # ``properties: {}`` with no explicit ``additionalProperties`` is the
        # "tool takes no arguments" shape ``strict_json_schema`` itself emits.
        additional = node.get("additionalProperties")
        return additional is True or isinstance(additional, dict)
    return True


def _collect_open_maps(node: Any, path: str, out: list[str]) -> None:
    if isinstance(node, list):
        for index, item in enumerate(node):
            _collect_open_maps(item, f"{path}[{index}]", out)
        return
    if not isinstance(node, dict):
        return

    properties = node.get("properties")
    if node.get("type") == "object" or isinstance(properties, dict):
        if _is_open_map(node):
            out.append(path or "<root>")

    if isinstance(properties, dict):
        for name, subschema in properties.items():
            _collect_open_maps(subschema, f"{path}.{name}" if path else name, out)
    additional = node.get("additionalProperties")
    if isinstance(additional, dict):
        _collect_open_maps(additional, f"{path}.<values>" if path else "<values>", out)
    for key in _SUBSCHEMA_KEYS:
        if key in node:
            _collect_open_maps(node[key], f"{path}.{key}" if path else key, out)
    for key in _SUBSCHEMA_LIST_KEYS:
        if isinstance(node.get(key), list):
            _collect_open_maps(node[key], f"{path}.{key}" if path else key, out)
    for defs_key in ("$defs", "definitions"):
        if isinstance(node.get(defs_key), dict):
            for name, subschema in node[defs_key].items():
                _collect_open_maps(subschema, f"{defs_key}.{name}", out)


def unsupported_strict_paths(schema: dict[str, Any] | None) -> list[str]:
    """Locations in ``schema`` that OpenAI strict mode cannot express.

    Strict mode requires every object to enumerate its properties and set
    ``additionalProperties: false``; it has no representation for an object
    with arbitrary keys. A ``dict[str, int]`` parameter — which
    ``build_parameters_schema`` emits as an object with a schema-valued
    ``additionalProperties`` and no ``properties`` — therefore cannot be
    published alongside ``strict: true``: OpenAI rejects the whole request
    rather than ignoring the shape, so the tool becomes unusable instead of
    merely unconstrained.

    :func:`strict_json_schema` deliberately leaves such a node alone —
    forcing ``additionalProperties: false`` on it would produce an object
    accepting *no* keys, silently discarding the parameter's data. Callers
    should check this first and fall back to a non-strict request for the
    affected tool.

    Args:
        schema: The tool's JSON-Schema ``parameters`` object.

    Returns:
        Dotted paths (``"counts"``, ``"payload.tags"``) of the offending
        object nodes; empty when the schema is expressible in strict mode.
    """
    if not schema:
        return []
    found: list[str] = []
    _collect_open_maps(schema, "", found)
    return found


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


def _contains_ref(node: Any) -> bool:
    """Whether any ``$ref`` survives anywhere in ``node``."""
    if isinstance(node, list):
        return any(_contains_ref(item) for item in node)
    if not isinstance(node, dict):
        return False
    if "$ref" in node:
        return True
    return any(
        _contains_ref(value) for key, value in node.items() if key not in ("$defs", "definitions")
    )


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
    # ``$defs`` exists only to back the pointers just inlined -- but the depth
    # guard can leave a ``$ref`` unresolved on a deeply nested or recursive
    # model. Dropping the definitions then would turn a schema the SDK rejects
    # into a schema with a pointer to nothing, which is strictly worse. Keep
    # them whenever anything still points at them.
    if not _contains_ref(transformed):
        transformed.pop("$defs", None)
        transformed.pop("definitions", None)
    _sanitize_gemini_in_place(transformed)
    return transformed
