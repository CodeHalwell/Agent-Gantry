"""
Tool introspection utilities for Agent-Gantry.

Provides schema building from Python function signatures and type hints.

The schema is what every downstream consumer sees — LLM providers (OpenAI /
Anthropic / Gemini dialects), the ~15 agent-framework adapters, MCP/A2A
servers, and the executor's own argument validation — so fidelity here
directly affects tool-call quality everywhere. ``build_parameters_schema``
therefore preserves as much of the function's declared intent as JSON Schema
can express:

- **Descriptions** from ``Annotated[T, "..."]`` metadata (preferred) or the
  function docstring's parameter section (Google ``Args:``, NumPy
  ``Parameters``, or Sphinx ``:param name:`` styles).
- **Enums** from ``Literal[...]`` and :class:`enum.Enum` subclasses.
- **Defaults** for optional parameters (when JSON-serializable).
- **Container types**: ``list[T]``/``set[T]``/``tuple`` → ``array`` (with
  typed ``items``), ``dict``/``Mapping`` → ``object``.
- **Nested models**: Pydantic models, dataclasses and ``TypedDict``s via
  Pydantic's own schema generation, with local ``$ref``s inlined.
- **Optionality**: ``Optional[T]`` / ``T | None`` map to ``T``'s schema
  (requiredness is carried by the ``required`` list, matching what LLM
  providers expect).
- **String formats** for ``datetime``/``date``/``time``/``UUID``.
"""

from __future__ import annotations

import collections.abc as _abc
import copy
import dataclasses
import datetime
import enum
import inspect
import math
import re
import types
import uuid
from collections.abc import Callable
from typing import Any

#: Generic origins that describe an ordered/unordered collection of items.
#: ``typing.get_origin`` normalizes ``typing.Sequence[int]`` and
#: ``collections.abc.Sequence[int]`` alike to the ``collections.abc`` class,
#: so the ABCs — not the ``typing`` aliases — are what must be matched.
#: Mappings are excluded by being handled first (a Mapping is a Collection).
_SEQUENCE_ORIGINS: tuple[type, ...] = (
    list,
    tuple,
    set,
    frozenset,
    _abc.Sequence,
    _abc.Set,
    _abc.Collection,
    _abc.Iterable,
)

#: Of those, the ones whose items are unordered and distinct.
_UNIQUE_ORIGINS: tuple[type, ...] = (set, frozenset, _abc.Set)

#: Hard bound on ``$ref`` inlining recursion (self-referential models).
_MAX_REF_DEPTH = 16


def build_parameters_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """
    Build JSON Schema for function parameters from Python type hints.

    Handles scalar types, containers, enums/Literals, nested Pydantic models
    and dataclasses, with automatic detection of required vs optional
    parameters based on defaults. Per-parameter descriptions are taken from
    ``Annotated`` metadata or the docstring's ``Args:``/``Parameters``/
    ``:param:`` section; defaults are recorded in the schema when they are
    JSON-serializable.

    Args:
        func: The function to introspect

    Returns:
        JSON Schema dict with type, properties, and required fields

    Example:
        >>> def my_func(x: int, y: str = "default") -> str:
        ...     return f"{x}: {y}"
        >>> schema = build_parameters_schema(my_func)
        >>> schema["required"]
        ['x']
        >>> schema["properties"]["y"]["type"]
        'string'
        >>> schema["properties"]["y"]["default"]
        'default'
    """
    import typing

    sig = inspect.signature(func)

    # Use get_type_hints to resolve string annotations (from __future__ import
    # annotations). include_extras keeps Annotated metadata — the conventional
    # home of per-parameter descriptions.
    try:
        type_hints = typing.get_type_hints(func, include_extras=True)
    except (NameError, TypeError):
        # Fall back to raw annotations if get_type_hints fails
        # NameError: forward references that can't be resolved
        # TypeError: invalid type annotations
        try:
            type_hints = func.__annotations__
        except AttributeError:
            type_hints = {}

    param_docs = _parse_param_docs(inspect.getdoc(func) or "")

    properties: dict[str, Any] = {}
    required: list[str] = []
    allow_additional = False

    for param_name, param in sig.parameters.items():
        # Skip self and cls parameters
        if param_name in ("self", "cls"):
            continue

        # *args / **kwargs are not named JSON-Schema properties. Emitting them
        # as properties (with no default, hence "required") makes the tool
        # impossible to call with a normal argument dict — e.g. a bare
        # ``def f(**kwargs)`` would reject ``execute(arguments={})`` with
        # "Missing required parameter: kwargs". Skip them instead; a
        # ``**kwargs`` simply means extra keys are allowed.
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            allow_additional = True
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue

        param_type = type_hints.get(param_name, str)
        param_type, annotated_desc = _split_annotated(param_type)
        param_schema = _type_to_json_schema(param_type)

        description = annotated_desc or param_docs.get(param_name)
        if description and "description" not in param_schema:
            param_schema["description"] = description

        # Mark as required if no default value; record JSON-safe defaults so
        # the model (and humans reading the schema) can see them. A ``None``
        # default only signals optionality — it is not a meaningful value.
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        elif param.default is not None and "default" not in param_schema:
            default = (
                param.default.value if isinstance(param.default, enum.Enum) else param.default
            )
            if _json_safe(default):
                param_schema["default"] = default

        properties[param_name] = param_schema

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    if allow_additional:
        schema["additionalProperties"] = True
    return schema


def _split_annotated(param_type: Any) -> tuple[Any, str | None]:
    """Unwrap ``Annotated[T, ...]``, returning ``(T, description)``.

    The description is the first ``str`` metadata item, mirroring how
    Pydantic AI / Semantic Kernel / AG2 read parameter descriptions from
    ``Annotated``. Non-``Annotated`` types pass through unchanged.
    """
    import typing

    if typing.get_origin(param_type) is not typing.Annotated:
        return param_type, None
    args = typing.get_args(param_type)
    base = args[0] if args else param_type
    description = next((a for a in args[1:] if isinstance(a, str)), None)
    # Pydantic FieldInfo metadata also carries a description attribute.
    if description is None:
        description = next(
            (
                getattr(a, "description")
                for a in args[1:]
                if isinstance(getattr(a, "description", None), str)
            ),
            None,
        )
    return base, description


def _json_safe(value: Any) -> bool:
    """Whether ``value`` can be embedded in a JSON schema as a default."""
    if isinstance(value, float) and not math.isfinite(value):
        # NaN/±inf are Python floats but not JSON values: ``json.dumps``
        # emits the bare tokens ``NaN``/``Infinity`` unless ``allow_nan`` is
        # off, and a provider parsing strict JSON rejects the request.
        return False
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True
    if isinstance(value, (list, tuple)):
        return all(_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_safe(v) for k, v in value.items())
    return False


# Scalar types with a direct JSON-Schema mapping. ``bool`` must be checked
# before ``int`` at runtime (bool subclasses int), but dict lookup is exact so
# ordering here is only cosmetic.
_SCALAR_MAP: dict[Any, dict[str, Any]] = {
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    str: {"type": "string"},
    bytes: {"type": "string"},
    datetime.datetime: {"type": "string", "format": "date-time"},
    datetime.date: {"type": "string", "format": "date"},
    datetime.time: {"type": "string", "format": "time"},
    uuid.UUID: {"type": "string", "format": "uuid"},
}


def _enum_schema(values: tuple[Any, ...]) -> dict[str, Any]:
    """Build an ``enum`` schema from literal values, inferring a shared type.

    Only JSON-representable values can appear in a schema payload —
    ``Literal`` admits ``bytes`` and Enum members can carry arbitrary
    objects. Anything non-JSON degrades to a plain string schema (no
    ``enum``) rather than emitting a payload providers would reject.
    """
    if not all(_json_safe(v) for v in values):
        return {"type": "string"}
    schema: dict[str, Any] = {}
    kinds = set()
    for v in values:
        if isinstance(v, bool):
            kinds.add("boolean")
        elif isinstance(v, int):
            kinds.add("integer")
        elif isinstance(v, float):
            kinds.add("number")
        elif isinstance(v, str):
            kinds.add("string")
        else:
            kinds.add("other")
    if len(kinds) == 1 and "other" not in kinds:
        schema["type"] = next(iter(kinds))
    schema["enum"] = list(values)
    return schema


def _pydantic_object_schema(param_type: Any) -> dict[str, Any] | None:
    """Schema for a Pydantic model / dataclass / TypedDict via Pydantic.

    Returns ``None`` when Pydantic can't produce a schema (never raises).
    Local ``$ref``s are inlined so downstream consumers that don't resolve
    JSON pointers (the executor's validator, several provider dialects) see
    the full nested structure.
    """
    try:
        if isinstance(param_type, type) and hasattr(param_type, "model_json_schema"):
            raw = param_type.model_json_schema()
        else:
            from pydantic import TypeAdapter

            raw = TypeAdapter(param_type).json_schema()
    except Exception:  # noqa: BLE001 - best-effort; fall back to generic mapping
        import typing

        if not typing.is_typeddict(param_type):
            return None
        raw = _typeddict_schema_via_typing_extensions(param_type)
        if raw is None:
            return None
    return _inline_local_refs(raw)


def _typeddict_schema_via_typing_extensions(param_type: type) -> dict[str, Any] | None:
    """Retry TypedDict introspection via ``typing_extensions.TypedDict``.

    On Python < 3.12, Pydantic's schema generator only recognizes TypedDict
    classes built from ``typing_extensions.TypedDict`` — a class written
    with the standard-library ``typing.TypedDict`` (the import most code
    reaches for) raises ``PydanticUserError`` instead, which would
    otherwise silently flatten the parameter to a bare ``{"type":
    "object"}``. Rebuilding an equivalent class from the same field
    annotations and totality and retrying keeps the nested schema intact
    on those Python versions too.
    """
    try:
        import typing_extensions
        from pydantic import TypeAdapter

        rebuilt = typing_extensions.TypedDict(
            param_type.__name__,
            param_type.__annotations__,
            total=param_type.__total__,
        )
        return TypeAdapter(rebuilt).json_schema()
    except Exception:  # noqa: BLE001 - best-effort; fall back to generic mapping
        return None


def _inline_local_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve local ``#/$defs/...`` pointers by inlining their targets."""
    defs = schema.get("$defs") or schema.get("definitions") or {}

    def _resolve(node: Any, depth: int) -> Any:
        if depth > _MAX_REF_DEPTH:
            return {}
        if isinstance(node, list):
            return [_resolve(item, depth) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.rsplit("/", 1)[0] in ("#/$defs", "#/definitions"):
            target = defs.get(ref.rsplit("/", 1)[1])
            if isinstance(target, dict):
                # Drop $ref itself plus $defs/definitions: a top-level
                # schema is commonly ``{"$defs": {...}, "$ref": "#/..."}``
                # (a model's own schema referencing its $defs sibling), and
                # that raw $defs blob must not survive the merge below —
                # it's the *unresolved* source the ref points into, not an
                # overriding sibling key, and re-attaching it would leak an
                # un-inlined $ref straight back into the output.
                merged = {
                    k: v for k, v in node.items() if k not in ("$ref", "$defs", "definitions")
                }
                resolved = _resolve(copy.deepcopy(target), depth + 1)
                # Keys alongside the $ref (e.g. an overriding description) win.
                return {**resolved, **merged}
        return {
            key: _resolve(value, depth)
            for key, value in node.items()
            if key not in ("$defs", "definitions")
        }

    return _resolve(schema, 0)


def _type_to_json_schema(param_type: Any) -> dict[str, Any]:
    """
    Map Python type to JSON Schema.

    Args:
        param_type: Python type annotation

    Returns:
        Dict describing the type as JSON Schema (always at least a ``type``
        or ``enum`` field; ``{"type": "string"}`` for unknown types).
    """
    import typing

    param_type, _ = _split_annotated(param_type)

    # Direct type match (most reliable)
    if isinstance(param_type, type):
        scalar_schema = _SCALAR_MAP.get(param_type)
        if scalar_schema is not None:
            return dict(scalar_schema)
        # Enum classes → enum of their values.
        if issubclass(param_type, enum.Enum):
            return _enum_schema(tuple(member.value for member in param_type))
        # Pydantic models / dataclasses / TypedDict-like classes: checked
        # before the bare-container fallbacks below, because a TypedDict
        # class *is* a `dict` subclass at runtime — matching `issubclass(
        # param_type, dict)` first would silently flatten it to a bare
        # {"type": "object"} and never reach the richer nested schema here.
        if hasattr(param_type, "model_json_schema") or typing.is_typeddict(param_type):
            nested = _pydantic_object_schema(param_type)
            if nested is not None:
                return nested
        if dataclasses.is_dataclass(param_type):
            nested = _pydantic_object_schema(param_type)
            if nested is not None:
                return nested
        # Bare containers.
        if issubclass(param_type, (list, set, frozenset, tuple)):
            return {"type": "array"}
        if issubclass(param_type, dict):
            return {"type": "object"}

    # Generic types (list[str], dict[str, int], Optional[T], Literal, ...)
    try:
        origin = typing.get_origin(param_type)
        args = typing.get_args(param_type)

        if origin is not None:
            # Literal["a", "b"] → enum
            if origin is typing.Literal:
                return _enum_schema(args)

            # Optional[T] / Union[T, None] / T | None → T's schema. Multi-type
            # unions keep the first non-None member (requiredness is carried
            # by the ``required`` list, and most provider dialects reject
            # union-typed parameters).
            if origin is typing.Union or origin is types.UnionType:
                non_none_args = [a for a in args if a is not type(None)]
                if non_none_args:
                    return _type_to_json_schema(non_none_args[0])
                return {"type": "string"}

            # Mappings → object (value schema recorded via additionalProperties).
            # Checked *before* sequences: a Mapping is also a Collection and an
            # Iterable, so the sequence test below would otherwise classify
            # ``dict[str, int]`` as an array.
            if origin is dict or (
                isinstance(origin, type) and issubclass(origin, _abc.Mapping)
            ):
                schema = {"type": "object"}
                if len(args) == 2 and args[1] is not Any:
                    schema["additionalProperties"] = _type_to_json_schema(args[1])
                return schema

            # Sequences and sets → array (with typed items when parameterized).
            # ``typing.get_origin(Sequence[int])`` returns the
            # ``collections.abc`` class, not the ``typing`` alias and not a
            # ``list`` subclass, so matching only aliases and concrete
            # containers dropped ``Sequence``/``Iterable``/``Set`` parameters
            # through to the first-type-argument fallback below — advertising
            # ``Sequence[int]`` as ``{"type": "integer"}``, which the executor
            # then rejected for every valid list payload.
            if isinstance(origin, type) and issubclass(origin, _SEQUENCE_ORIGINS):
                if issubclass(origin, _UNIQUE_ORIGINS):
                    schema = {"type": "array", "uniqueItems": True}
                else:
                    schema = {"type": "array"}
                item_type = _tuple_item_type(origin, args) if origin is tuple else (
                    args[0] if args else None
                )
                if item_type is not None and item_type is not Any:
                    schema["items"] = _type_to_json_schema(item_type)
                return schema

            # Fallback for other generics: use the first argument if available
            if args:
                return _type_to_json_schema(args[0])
    except (AttributeError, ImportError):
        pass

    # Check string representations as fallback (less reliable)
    type_str = str(param_type)
    # Use word boundaries to avoid false positives
    if type_str in ("int", "<class 'int'>"):
        return {"type": "integer"}
    elif type_str in ("float", "<class 'float'>"):
        return {"type": "number"}
    elif type_str in ("bool", "<class 'bool'>"):
        return {"type": "boolean"}
    elif type_str in ("str", "<class 'str'>"):
        return {"type": "string"}

    # Default to string for unknown types
    return {"type": "string"}


def _tuple_item_type(origin: Any, args: tuple[Any, ...]) -> Any:
    """Item type for a ``tuple[...]`` annotation (homogeneous forms only)."""
    if not args:
        return None
    # tuple[T, ...] — homogeneous variable-length tuple.
    if len(args) == 2 and args[1] is Ellipsis:
        return args[0]
    # tuple[T] or tuple[T, T, T] with one distinct member type.
    distinct = {a for a in args if a is not Ellipsis}
    if len(distinct) == 1:
        return next(iter(distinct))
    return None


# --------------------------------------------------------------------------- #
# Docstring parameter descriptions
# --------------------------------------------------------------------------- #

#: Section headers that introduce a parameter block (Google style; NumPy uses
#: an underlined ``Parameters`` heading handled separately).
_GOOGLE_SECTION = re.compile(
    r"^(Args|Arguments|Parameters|Keyword Args|Keyword Arguments)\s*:\s*$"
)
#: Any section header — used to know where a parameter block ends.
_ANY_SECTION = re.compile(
    r"^(Args|Arguments|Parameters|Keyword Args|Keyword Arguments|Returns?|Yields?|"
    r"Raises?|Attributes|Examples?|Notes?|Warnings?|See Also|References)\s*:?\s*$"
)
#: ``name (type): description`` / ``name: description`` entry line.
_GOOGLE_ENTRY = re.compile(r"^(\*{0,2}[\w]+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$")
#: Sphinx ``:param name: description`` (with optional inline type).
_SPHINX_PARAM = re.compile(r"^:param\s+(?:[\w\[\],\. ]+\s+)?(\w+)\s*:\s*(.*)$")


def _parse_param_docs(doc: str) -> dict[str, str]:
    """Extract ``{param_name: description}`` from a docstring.

    Understands the three common conventions — Google (``Args:`` blocks, the
    project's own style), NumPy (underlined ``Parameters`` heading), and
    Sphinx (``:param name:`` fields). Multi-line descriptions are joined;
    unknown formats simply yield an empty mapping. Best-effort by design:
    never raises.
    """
    if not doc:
        return {}
    try:
        return _parse_param_docs_inner(doc)
    except Exception:  # noqa: BLE001 - docstrings are user input; never raise
        return {}


def _parse_param_docs_inner(doc: str) -> dict[str, str]:
    lines = doc.splitlines()
    out: dict[str, str] = {}

    # Sphinx fields can appear anywhere.
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        match = _SPHINX_PARAM.match(stripped)
        if match:
            current = match.group(1)
            out.setdefault(current, match.group(2).strip())
            continue
        if current is not None:
            if stripped.startswith(":") or not stripped:
                current = None
            else:
                out[current] = f"{out[current]} {stripped}".strip()

    # Google-style block (and NumPy's underlined "Parameters" heading).
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        is_google = bool(_GOOGLE_SECTION.match(stripped))
        is_numpy = (
            stripped in ("Parameters", "Other Parameters")
            and i + 1 < len(lines)
            and set(lines[i + 1].strip()) == {"-"}
        )
        if not (is_google or is_numpy):
            i += 1
            continue
        i += 2 if is_numpy else 1
        # The first non-blank line after the header fixes the block's entry
        # indent; entries sit at that indent, continuations sit deeper, and a
        # dedent below it (or a new section header) ends the block.
        base_indent: int | None = None
        name: str | None = None
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                # A blank line ends the current entry's description but not
                # the block (Google style allows blank lines between entries).
                name = None
                i += 1
                continue
            if _ANY_SECTION.match(stripped):
                break
            indent = len(line) - len(line.lstrip())
            if base_indent is None:
                base_indent = indent
            if indent < base_indent:
                break
            if is_numpy:
                entry = re.match(r"^(\*{0,2}\w+)\s*(?::.*)?$", stripped)
                if indent == base_indent and entry:
                    name = entry.group(1).lstrip("*")
                    out.setdefault(name, "")
                elif name is not None:
                    out[name] = f"{out[name]} {stripped}".strip()
            else:
                entry = _GOOGLE_ENTRY.match(stripped)
                if indent == base_indent and entry:
                    name = entry.group(1).lstrip("*")
                    out[name] = entry.group(3).strip()
                elif name is not None:
                    out[name] = f"{out[name]} {stripped}".strip()
            i += 1
    return {k: v.strip() for k, v in out.items() if v and v.strip()}
