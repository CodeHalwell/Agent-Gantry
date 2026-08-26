"""JSON-Schema → Pydantic model bridge for framework adapters.

Several frameworks accept an *args model* (a ``pydantic.BaseModel`` subclass)
as their native way to describe a tool's parameters — CrewAI's
``args_schema``, LlamaIndex's ``fn_schema``. Handing those frameworks a model
built from the Gantry tool's own JSON schema preserves everything the schema
declares (per-parameter descriptions, enums, typed array items, nested
objects, defaults) where deriving a schema from the wrapper function's
signature would flatten it to bare types.

Best-effort by design: :func:`pydantic_model_from_schema` returns ``None``
for anything it can't express faithfully (non-identifier property names,
pydantic-reserved names, exotic keywords), and callers fall back to their
previous behaviour — the tool still works, sans the richer schema.
"""

from __future__ import annotations

from typing import Any, Literal

#: Hard bound on nested-object recursion (self-referential schemas).
_MAX_DEPTH = 8

_SCALARS: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def pydantic_model_from_schema(name: str, schema: dict[str, Any]) -> Any:
    """Build a ``BaseModel`` subclass mirroring a JSON-Schema object.

    Args:
        name: Base name for the generated model class (sanitized to a valid
            identifier; nested models get a suffixed name).
        schema: A JSON-Schema mapping with ``type: object`` semantics —
            ``properties``, ``required``, per-property ``description`` /
            ``enum`` / ``items`` / ``default``.

    Returns:
        The generated model class, or ``None`` when the schema (or any part
        of it) can't be expressed — callers should fall back to their
        function-signature path in that case.
    """
    try:
        return _build_model(_sanitize_identifier(name) or "ToolArgs", schema, 0)
    except Exception:  # noqa: BLE001 - best-effort; schema stays advisory
        return None


def _sanitize_identifier(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if cleaned and cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _build_model(name: str, schema: dict[str, Any], depth: int) -> Any:
    if depth > _MAX_DEPTH:
        raise ValueError("schema nesting too deep")
    from pydantic import Field, create_model

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("not an object schema")
    required = set(schema.get("required") or [])

    fields: dict[str, Any] = {}
    for prop_name, prop in properties.items():
        if not prop_name.isidentifier():
            raise ValueError(f"property {prop_name!r} is not a valid identifier")
        prop = prop if isinstance(prop, dict) else {}
        annotation = _annotation(f"{name}_{prop_name}", prop, depth)
        description = prop.get("description")
        field_kwargs: dict[str, Any] = {}
        if isinstance(description, str) and description:
            field_kwargs["description"] = description
        if prop_name in required:
            if _is_nullable(prop):
                # Required-but-nullable (``type: ["string", "null"]`` in
                # ``required``): the field must still be supplied, but a
                # schema-valid ``None`` must not be rejected by Pydantic.
                fields[prop_name] = (annotation | None, Field(..., **field_kwargs))
            else:
                fields[prop_name] = (annotation, Field(..., **field_kwargs))
        else:
            # Optional: admit ``None`` (frameworks and models routinely send
            # null for "not provided"; ``ToolSpec.ainvoke`` drops it) and
            # surface the schema's own default when it declares one.
            default = prop.get("default", None)
            fields[prop_name] = (
                annotation | None,
                Field(default=default, **field_kwargs),
            )

    return create_model(name, **fields)


def _is_nullable(prop: dict[str, Any]) -> bool:
    """Whether a property schema's declared type admits ``null``."""
    json_type = prop.get("type")
    return json_type == "null" or (isinstance(json_type, list) and "null" in json_type)


def _annotation(name: str, prop: dict[str, Any], depth: int) -> Any:
    """Python annotation for one property schema (recursive)."""
    enum_values = prop.get("enum")
    if isinstance(enum_values, list) and enum_values:
        if all(isinstance(v, (str, int, bool)) for v in enum_values):
            return Literal[tuple(enum_values)]
        # Enum of exotic values — fall back to the declared/base type.

    json_type = prop.get("type")
    if isinstance(json_type, list):  # e.g. ["string", "null"]
        json_type = next((t for t in json_type if t != "null"), None)

    if json_type in _SCALARS:
        return _SCALARS[json_type]
    if json_type == "array":
        items = prop.get("items")
        if isinstance(items, dict) and items:
            return list[_annotation(f"{name}_item", items, depth + 1)]
        return list
    if json_type == "object":
        properties = prop.get("properties")
        if isinstance(properties, dict) and properties:
            return _build_model(f"{name}_obj", prop, depth + 1)
        additional = prop.get("additionalProperties")
        if isinstance(additional, dict) and additional:
            # No declared properties, but a typed ``additionalProperties``
            # (e.g. ``dict[str, int]``) — preserve the value type instead of
            # widening to a bare ``dict`` that would accept any value type.
            return dict[str, _annotation(f"{name}_value", additional, depth + 1)]
        return dict
    return Any


__all__ = ["pydantic_model_from_schema"]
