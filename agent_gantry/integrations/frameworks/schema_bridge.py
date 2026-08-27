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

import json
import logging
from collections import OrderedDict
from typing import Any, Literal

from agent_gantry.schema.base import (
    RECONSTRUCTED_STRING_FORMATS,
    check_json_constraints,
    json_identity_key,
    resolve_numeric_bounds,
    schema_declares_null,
)

_logger = logging.getLogger(__name__)

#: Memoized generated models, keyed by ``(name, canonical schema JSON)``.
#: The model is a pure function of those two, and the "live" CrewAI and
#: LlamaIndex adapters rebuild their tools on *every* retrieval — so without
#: this the same recursive ``create_model`` (plus its ``TypeAdapter``
#: constructions and ``re.compile`` calls) reran per query, per tool. Bounded
#: and LRU for the same reason the executor's coercer cache is: a long-running
#: gantry with MCP tool churn would otherwise grow without limit.
_MODEL_CACHE: OrderedDict[tuple[str, str], tuple[Any]] = OrderedDict()
_MODEL_CACHE_MAX = 256

#: Hard bound on nested-object recursion (self-referential schemas).
_MAX_DEPTH = 8

#: The bare Python kind each JSON type asserts, used to intersect a parent
#: ``type`` back onto a combinator whose branches declared their own.
_PARENT_KINDS: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

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
    key: tuple[str, str] | None
    try:
        key = (name, json.dumps(schema, sort_keys=True, default=str))
    except (TypeError, ValueError):  # not serializable — build it uncached
        key = None
    if key is not None:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            _MODEL_CACHE.move_to_end(key)
            return cached[0]

    try:
        model = _build_model(_sanitize_identifier(name) or "ToolArgs", schema, 0)
    except Exception:  # noqa: BLE001 - best-effort; schema stays advisory
        model = None

    if key is not None:
        # Wrapped in a tuple so a ``None`` result (an inexpressible schema) is
        # cached too — that path costs a full recursive build before it fails,
        # and it fails identically every time.
        _MODEL_CACHE[key] = (model,)
        if len(_MODEL_CACHE) > _MODEL_CACHE_MAX:
            _MODEL_CACHE.popitem(last=False)
    return model


def _sanitize_identifier(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if cleaned and cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _permits_additional(schema: dict[str, Any]) -> bool:
    """Whether an object schema admits keys beyond those it declares.

    Mirrors the executor's own rule (``core/executor.py``) so a framework's
    args model enforces what Gantry enforces: ``True`` and a subschema —
    including the empty schema ``{}``, which JSON Schema says validates
    everything — permit extras; ``False`` and an absent key forbid them.
    Absent being strict is Gantry's own deliberate default, stricter than
    the JSON Schema default.
    """
    patterns = schema.get("patternProperties")
    if isinstance(patterns, dict) and patterns:
        # A key a pattern matches *is* declared — the executor treats it that
        # way — so extras have to reach the model rather than being refused by
        # ``extra="forbid"``. The pattern validator attached in
        # ``_build_model`` is what then enforces them, including rejecting a
        # key no pattern matches when the object is closed.
        return True
    return _admits_undeclared(schema)


def _admits_undeclared(schema: dict[str, Any]) -> bool:
    """Whether a key matching no declared property and no pattern is allowed.

    The same rule as ``_permits_additional`` minus its ``patternProperties``
    escape hatch, which exists only so matching keys can reach the model.
    Once they have, *this* is the question the pattern validator has to
    answer, and answering it with a bare ``additionalProperties is False``
    read an absent key as permissive wherever the object also declared
    properties — so the generated model accepted a key the engine rejects at
    dispatch.

    Mirrors the executor's asymmetry rather than simplifying it away: an
    object declaring *no* properties and no ``additionalProperties`` is a
    free-form mapping — Gantry's own shape for a plain ``dict`` parameter —
    while one that declares properties and omits ``additionalProperties`` is
    closed, which is the stricter-than-spec default this module documents.
    """
    additional = schema.get("additionalProperties")
    if additional is True or isinstance(additional, dict):
        return True
    if additional is False:
        return False
    properties = schema.get("properties")
    return not (isinstance(properties, dict) and properties)


def _build_model(name: str, schema: dict[str, Any], depth: int) -> Any:
    if depth > _MAX_DEPTH:
        # Raising aborts the whole model, so the caller falls back to its
        # signature path rather than publishing a half-built one. Logged
        # because that fallback is otherwise invisible: a legitimately deep
        # acyclic schema silently loses its args model.
        _logger.debug(
            "Schema nesting exceeded depth %d while building %r; no args model "
            "will be generated and the caller falls back to its own path.",
            _MAX_DEPTH,
            name,
        )
        raise ValueError("schema nesting too deep")
    from pydantic import ConfigDict, Field, create_model, model_validator

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("not an object schema")
    required = set(schema.get("required") or [])
    # Pydantic defaults to ``extra="ignore"``, which would silently drop an
    # undeclared key the executor would have rejected — a misspelled or
    # hallucinated argument would vanish inside the framework instead of
    # surfacing as an error. Mirror the schema instead.
    model_config = ConfigDict(extra="allow" if _permits_additional(schema) else "forbid")
    additional = schema.get("additionalProperties")
    extra_annotation: Any = None
    has_patterns = isinstance(schema.get("patternProperties"), dict) and schema.get(
        "patternProperties"
    )
    if isinstance(additional, dict) and additional and not has_patterns:
        # A *typed* ``additionalProperties`` constrains the extras' value
        # type. Bare ``extra="allow"`` would take every extra as ``Any``,
        # so the framework would accept a string where the schema (and the
        # executor, which does check the subschema) demand an integer.
        # ``__pydantic_extra__`` carries that type into validation.
        #
        # Skipped when the object also declares ``patternProperties``: JSON
        # Schema applies ``additionalProperties`` only to keys matched by
        # *neither* ``properties`` nor a pattern, but ``__pydantic_extra__``
        # types every extra — so a pattern-matched key was checked against the
        # additional schema instead of its own, and a valid ``{"s_a": "ok"}``
        # beside ``additionalProperties: {"type": "integer"}`` was rejected.
        # The pattern validator below applies both, to the right key sets.
        extra_annotation = _annotation(f"{name}_extra", additional, depth + 1)

    fields: dict[str, Any] = {}
    for prop_name, prop in properties.items():
        if not prop_name.isidentifier() or prop_name.startswith("_"):
            # ``isidentifier()`` accepts ``_token``, but Pydantic reserves
            # leading underscores for private attributes and will not make a
            # normal field from one. It raises today, which the caller turns
            # into the documented fallback — but it *silently drops* such a
            # name when one is passed as a keyword rather than through the
            # field mapping, so relying on the raise is relying on which
            # spelling Pydantic happens to see. Declining explicitly keeps the
            # fallback deliberate: a partially faithful model would omit a
            # required canonical property, and reject an explicitly supplied
            # one as an extra when the object is closed.
            raise ValueError(f"property {prop_name!r} cannot be a model field")
        declared_schema = prop  # the original, which may be a bare boolean
        annotation = _annotation(f"{name}_{prop_name}", prop, depth)
        if isinstance(prop, dict):
            annotation = _with_constraints(annotation, prop)
        else:
            # A boolean property schema carries no constraint keywords, and
            # coercing it to ``{}`` first made ``false`` — which forbids every
            # value — indistinguishable from "unconstrained". ``prop`` is
            # replaced only so the ``.get`` calls below work; nullability is
            # decided from ``declared_schema``, because ``{}`` admits null and
            # ``false`` admits nothing at all.
            prop = {}
        description = prop.get("description")
        field_kwargs: dict[str, Any] = {}
        if isinstance(description, str) and description:
            field_kwargs["description"] = description
        if prop_name in required:
            if schema_declares_null(declared_schema):
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

    if extra_annotation is not None:
        fields["__pydantic_extra__"] = (dict[str, extra_annotation], Field(init=False))

    patterns = schema.get("patternProperties")
    validators: dict[str, Any] = {}
    if isinstance(patterns, dict) and patterns:
        # Declared ``properties`` *and* ``patternProperties`` together: the
        # model types the named fields, and this types the rest. Without it the
        # model merely allowed every extra (so a pattern key's value went
        # unchecked, and a key matching no pattern slipped through a closed
        # object) — both of which the executor rejects.
        #
        # Attached as a *model validator* rather than by wrapping the class in
        # ``Annotated``: this function's contract is to return a
        # ``type[BaseModel]``, and CrewAI's ``args_schema`` field is typed that
        # way, so an ``Annotated`` alias raised a ValidationError at tool
        # construction instead of degrading to the documented fallback.
        check = _pattern_key_validator(
            name,
            patterns,
            not _admits_undeclared(schema),
            depth,
            set(properties),
            schema.get("additionalProperties"),
        )
        validators["_check_pattern_properties"] = model_validator(mode="before")(
            classmethod(lambda cls, value, _check=check: _check(value))
        )

    return create_model(
        name, __config__=model_config, __validators__=validators or None, **fields
    )


def _effective_type(prop: dict[str, Any]) -> Any:
    """The property's single JSON type, or ``None`` when it has no one type."""
    json_type = prop.get("type")
    if isinstance(json_type, list):
        non_null = [t for t in json_type if t != "null"]
        return non_null[0] if len(non_null) == 1 else None
    return json_type


def _reject_duplicates(value: Any) -> Any:
    """Enforce ``uniqueItems`` on a parsed array value.

    Shares :func:`json_identity_key` with the executor's validator so the two
    agree on what counts as a duplicate — otherwise this model would accept a
    payload the engine rejects at dispatch, or vice versa.
    """
    if isinstance(value, (list, tuple)):
        seen: set[Any] = set()
        unhashable: list[Any] = []
        for item in value:
            key = json_identity_key(item)
            try:
                if key in seen:
                    raise ValueError("must not contain duplicate items")
                seen.add(key)
            except TypeError:  # a non-JSON value that isn't hashable
                if key in unhashable:
                    raise ValueError("must not contain duplicate items")
                unhashable.append(key)
    return value


def _literal_annotation(values: list[Any]) -> Any:
    """A ``Literal`` over ``values``, guarded by JSON identity where needed.

    Pydantic matches ``True`` against a ``Literal[1, 1.5]`` member because
    Python says ``True == 1``, but JSON Schema compares types before values
    and the executor's ``enum`` check now agrees with the spec. The guard is
    attached only when the members are numeric or boolean — the only place
    the two can be confused — so a plain string enum keeps a bare ``Literal``.
    """
    from typing import Annotated

    from pydantic import BeforeValidator

    literal = Literal[tuple(values)]
    if not any(isinstance(v, (bool, int, float)) for v in values):
        return literal
    member_keys = [json_identity_key(v) for v in values]

    def _by_identity(value: Any) -> Any:
        if json_identity_key(value) not in member_keys:
            raise ValueError(f"is not one of {values}")
        return value

    return Annotated[literal, BeforeValidator(_by_identity)]


def _with_intersection(annotation: Any, parts: list[Any]) -> Any:
    """Attach an "every branch matches" check for ``allOf``.

    ``allOf`` has no faithful Python annotation — intersecting constraints
    isn't expressible — but silently ignoring it left an ``allOf``-only
    property as a bare ``Any`` that accepted values the executor rejects.
    Checking each branch against the caller's raw value gets the semantics
    exactly, whatever the annotation underneath can express.
    """
    from typing import Annotated

    from pydantic import BaseModel, BeforeValidator, TypeAdapter

    # Strict for scalars, lax for model branches — same reasoning as
    # ``_with_exclusivity``: a JSON object arrives as a mapping, and strict
    # mode would reject every dict.
    checks = [
        (TypeAdapter(part), not (isinstance(part, type) and issubclass(part, BaseModel)))
        for part in parts
    ]

    def _all_match(value: Any) -> Any:
        for index, (adapter, strict) in enumerate(checks):
            try:
                adapter.validate_python(value, strict=strict)
            except Exception as exc:  # noqa: BLE001 - reported as a branch miss
                raise ValueError(f"fails allOf branch {index}: {exc}") from None
        return value

    return Annotated[annotation, BeforeValidator(_all_match)]


def _with_constraints(annotation: Any, prop: dict[str, Any]) -> Any:
    """Fold the schema's constraint keywords into ``annotation``.

    The executor enforces these, so leaving them off the generated model
    means the framework advertises and accepts values the engine rejects at
    dispatch. Applied via ``annotated_types``/``StringConstraints`` on the
    *inner* annotation, so the field stays free to be unioned with ``None``
    and to carry its own default.
    """
    from typing import Annotated

    import annotated_types as at

    json_type = _effective_type(prop)
    marks: list[Any] = []

    def _number(key: str) -> Any:
        candidate = prop.get(key)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return candidate
        return None

    def _count(key: str) -> Any:
        candidate = prop.get(key)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
            return candidate
        return None

    if json_type in ("integer", "number"):
        # Shares the executor's bound resolution so the two dialects of
        # exclusivity — modern numeric, and draft-04's boolean modifier on
        # ``minimum``/``maximum`` (what OpenAPI 3.0 emits) — are read the same
        # way here as at dispatch. Reading only the modern form left an
        # imported draft-04 bound applied inclusively in the model while the
        # executor now applies it exclusively.
        for bound, mark in zip(resolve_numeric_bounds(prop), (at.Ge, at.Le, at.Gt, at.Lt)):
            if bound is not None:
                marks.append(mark(bound))
        multiple = _number("multipleOf")
        if multiple is not None and multiple > 0:
            marks.append(at.MultipleOf(multiple))
    elif json_type == "string":
        if prop.get("format") in RECONSTRUCTED_STRING_FORMATS:
            # Checked, *not* converted. Annotating the field ``datetime`` made
            # the model hand a ``datetime`` object back — CrewAI forwards its
            # validated kwargs and LlamaIndex's ``_coerced`` dumps in Python
            # mode — while the canonical schema still types the property as a
            # JSON string, so the executor rejected every *valid* formatted
            # call. The value stays JSON-native across the dispatch boundary;
            # only its shape is asserted, by the same checker the executor
            # runs.
            from pydantic import AfterValidator

            def _check_format(value: Any, _schema: dict[str, Any] = prop) -> Any:
                error = check_json_constraints(value, _schema, "value")
                if error is not None:
                    raise ValueError(error)
                return value

            marks.append(AfterValidator(_check_format))
        low, high = _count("minLength"), _count("maxLength")
        if low is not None:
            marks.append(at.MinLen(low))
        if high is not None:
            marks.append(at.MaxLen(high))
        pattern = prop.get("pattern")
        if isinstance(pattern, str) and pattern:
            from pydantic import StringConstraints

            marks.append(StringConstraints(pattern=pattern))
    elif json_type == "array":
        low, high = _count("minItems"), _count("maxItems")
        if low is not None:
            marks.append(at.MinLen(low))
        if high is not None:
            marks.append(at.MaxLen(high))
        if prop.get("uniqueItems") is True:
            # Gantry's own introspection emits this for every ``set`` and
            # ``frozenset`` parameter, and the executor rejects duplicates at
            # dispatch — so without it the framework happily accepts a list
            # the engine then refuses. Keyed by JSON identity via the same
            # helper the executor uses, so the two cannot disagree.
            from pydantic import AfterValidator

            marks.append(AfterValidator(_reject_duplicates))
    elif json_type == "object":
        # Mirrors the executor's own object branch, which a Pydantic ``dict``
        # field constrained with ``Field(min_length=1)`` reaches. ``MinLen``
        # and ``MaxLen`` apply to a mapping's key count.
        low, high = _count("minProperties"), _count("maxProperties")
        if low is not None:
            marks.append(at.MinLen(low))
        if high is not None:
            marks.append(at.MaxLen(high))

    if json_type is None:
        # A constraint-only schema with no declared ``type`` — what a
        # ``patternProperties`` branch like ``{"minimum": 5}`` looks like. The
        # keyword-family gates above all miss it, so the constraint was
        # silently dropped. It can't be expressed as ``annotated_types``
        # metadata either: ``Annotated[Any, Ge(5)]`` rejects ``"x"``, which the
        # schema permits, and rejecting a valid call is the worse error. The
        # shared checker applies each family only to its own JSON type, so the
        # model agrees with the executor by construction.
        from pydantic import AfterValidator

        def _check(value: Any, _schema: dict[str, Any] = prop) -> Any:
            error = check_json_constraints(value, _schema, "value")
            if error is not None:
                raise ValueError(error)
            return value

        marks.append(AfterValidator(_check))

    if not marks:
        return annotation
    return Annotated[tuple([annotation, *marks])]


def _never_annotation() -> Any:
    """An annotation nothing validates against — the schema ``false``."""
    from typing import Annotated

    from pydantic import AfterValidator

    def _reject(value: Any) -> Any:
        raise ValueError("schema `false` permits no value")

    return Annotated[Any, AfterValidator(_reject)]


def _combinator_parts(
    name: str, branches: list[Any], depth: int, inherited_type: Any
) -> tuple[list[Any], bool]:
    """Translate one combinator's branches, and whether any was the empty schema."""
    parts: list[Any] = []
    has_empty = False
    for index, branch in enumerate(branches):
        if branch is True:
            # ``true`` is the always-valid schema — the other spelling of
            # ``{}``, and handled identically. The executor learned this a
            # commit ago; without it here the bridge reduced ``{"anyOf":
            # [true, {"type": "integer"}]}`` to an integer field and rejected
            # the schema-valid strings the engine accepts.
            has_empty = True
            parts.append(Any)
            continue
        if branch is False:
            # ``false`` validates nothing. As a union member it simply never
            # matches, which is right for ``anyOf`` and for ``oneOf``'s count;
            # as an ``allOf`` branch the intersection then forbids every value,
            # which is also right.
            parts.append(_never_annotation())
            continue
        if not isinstance(branch, dict):
            continue
        if not branch:
            # ``{}`` is the always-valid schema, not an absent branch.
            # Dropping it made ``anyOf: [{}, {"type": "integer"}]`` — which
            # admits every value — reject strings, and let ``oneOf`` accept
            # an integer that in fact matches both branches.
            has_empty = True
            parts.append(Any)
            continue
        if branch.get("type") == "null":
            parts.append(type(None))
            continue
        if inherited_type is not None and "type" not in branch:
            branch = {**branch, "type": inherited_type}
        parts.append(
            _with_constraints(_annotation(f"{name}_{index}", branch, depth + 1), branch)
        )
    return parts, has_empty


def _union_annotation(
    name: str, prop: dict[str, Any], depth: int, inherited_type: Any = None
) -> Any:
    """Annotation for an ``anyOf``/``oneOf`` schema, or ``None`` if absent.

    ``inherited_type`` is the parent's own ``type``, pushed into any branch
    that doesn't declare one. A schema may carry both — ``{"type":
    "integer", "anyOf": [{"minimum": 10}, {"maximum": 0}]}`` means the value
    must be an integer *and* satisfy one of the branches — and a
    constraint-only branch would otherwise translate to a bare ``Any`` that
    enforces nothing. Merging the type down turns each branch into a real
    constrained annotation, so the union expresses the intersection.

    Each branch is translated recursively and the results unioned, so
    ``{"anyOf": [{"type": "integer"}, {"type": "null"}]}`` becomes
    ``int | None`` rather than a bare ``Any``.

    Both keywords are honoured when both appear. They are independent
    assertions that a value must satisfy together, but this used to return on
    whichever it found first, so a schema carrying both silently lost
    ``oneOf``'s exclusivity — the bridge accepted a value the executor then
    rejected. The union comes from ``anyOf`` (the narrower annotation) and
    ``oneOf`` contributes its exactly-one check on top.

    ``allOf`` has no faithful Python annotation either — intersecting
    constraints isn't expressible — but ignoring it left an ``allOf``-only
    property as a bare ``Any`` that accepted values the executor rejects. Its
    branches are checked individually instead, against the caller's raw value.
    """
    annotation: Any = None
    oneof_parts: list[Any] | None = None
    allof_parts: list[Any] | None = None
    for key in ("anyOf", "oneOf", "allOf"):
        branches = prop.get(key)
        if not isinstance(branches, list) or not branches:
            continue
        branch_type = inherited_type
        if key == "allOf" and branch_type is None:
            # ``allOf`` intersects, so a type declared by *any* branch applies
            # to the value as a whole — and therefore to the branches that
            # declare none. Without this, ``{"allOf": [{"type": "integer",
            # "minimum": 1}, {"maximum": 3}]}`` left the second branch as a
            # bare ``Any`` that matched everything, so the upper bound went
            # unenforced. (Union combinators can't do this: there each branch
            # stands alone.)
            branch_type = next(
                (
                    b["type"]
                    for b in branches
                    if isinstance(b, dict) and isinstance(b.get("type"), str)
                ),
                None,
            )
        parts, has_empty = _combinator_parts(
            f"{name}_{key}" if sum(k in prop for k in ("anyOf", "oneOf", "allOf")) > 1 else name,
            branches,
            depth,
            branch_type,
        )
        if not parts:
            continue
        if key == "oneOf":
            oneof_parts = parts
        if key == "allOf":
            allof_parts = parts
            # An intersection is not a union: the branches must all hold, so
            # they never contribute a union member. The base annotation is the
            # first branch that expresses a real type (the others ride on the
            # check below), and ``Any`` when none does.
            if annotation is None:
                annotation = next((p for p in parts if p is not Any), Any)
            continue
        if has_empty:
            # One branch matches everything, so the union constrains nothing.
            # For ``oneOf`` that is not the end of it: the exclusivity check
            # below still rejects any value another branch also admits.
            member: Any = Any
        else:
            member = parts[0]
            for part in parts[1:]:
                member = member | part
        if annotation is None:
            annotation = member

    if annotation is None:
        return None
    if allof_parts:
        annotation = _with_intersection(annotation, allof_parts)
    if oneof_parts is not None and len(oneof_parts) > 1:
        # A Python union is ``anyOf``: it accepts a value matching several
        # branches, which ``oneOf`` forbids (``1`` satisfies both ``number``
        # and ``integer``). The union still earns its place — it rejects
        # everything outside every branch, where ``Any`` would not — so keep
        # it and add the exclusivity the union can't carry.
        annotation = _with_exclusivity(annotation, oneof_parts)
    return annotation


def _with_exclusivity(annotation: Any, parts: list[Any]) -> Any:
    """Attach an "exactly one branch matches" check to a ``oneOf`` union.

    Runs *before* the union converts, not after: an ``AfterValidator`` would
    receive the instance the union already built (for object branches, a
    model of whichever branch won), and counting matches against that instead
    of the caller's raw mapping would find only the one branch — silently
    passing a payload that matches several.
    """
    from typing import Annotated

    from pydantic import BaseModel, BeforeValidator, TypeAdapter

    # Strict validation is what mirrors JSON Schema for scalars: without it
    # ``"1"`` would coerce into an ``int`` branch and inflate the count. A
    # model branch is the exception — a JSON object arrives as a mapping, and
    # strict mode would reject every dict, making the count meaningless.
    checks = [
        (TypeAdapter(part), not (isinstance(part, type) and issubclass(part, BaseModel)))
        for part in parts
    ]

    def _exactly_one(value: Any) -> Any:
        matched = 0
        for adapter, strict in checks:
            try:
                adapter.validate_python(value, strict=strict)
            except Exception:  # noqa: BLE001 - a branch simply not matching
                continue
            matched += 1
        # Exactly one, in both directions. Rejecting only ``> 1`` left the
        # zero-match case to the union below, whose coercion would then
        # accept a value no branch actually admits — ``"1"`` matches neither
        # a strict ``number`` nor a strict ``integer``, but coerces into one.
        if matched != 1:
            raise ValueError(f"matches {matched} oneOf branches; exactly one must match")
        return value

    return Annotated[annotation, BeforeValidator(_exactly_one)]


def _positional_array(name: str, prefix_items: list[Any], items: Any, depth: int) -> Any:
    """Annotation for an array whose positions are typed independently.

    ``prefixItems`` is what Pydantic emits for a heterogeneous
    ``tuple[int, str]``, so it arrives inside the nested models this bridge
    inlines, and the executor validates each position against its own entry.
    Returning a bare ``list`` here advertised an array of anything.

    The annotation stays ``list`` rather than becoming ``tuple[int, str]``:
    the value flows on to :meth:`ExecutionEngine.execute`, whose validator
    requires a JSON array (a ``list``), so coercing to a tuple to gain
    positional typing would only trade a permissive framework for a rejected
    dispatch. A ``BeforeValidator`` gets the same per-position checking while
    leaving the runtime type alone.
    """
    from typing import Annotated

    from pydantic import BeforeValidator, TypeAdapter

    def _adapter(entry: Any, label: str) -> Any:
        if isinstance(entry, bool):
            # ``items: false`` beside ``prefixItems`` is how a fixed-length
            # tuple forbids positions past the prefix; ``true`` permits any.
            return TypeAdapter(_annotation(label, entry, depth + 1))
        if not isinstance(entry, dict) or not entry:
            return None
        return TypeAdapter(_with_constraints(_annotation(label, entry, depth + 1), entry))

    adapters = [_adapter(entry, f"{name}_{index}") for index, entry in enumerate(prefix_items)]
    # ``items`` alongside ``prefixItems`` types the positions past the prefix,
    # matching how the executor applies the two.
    tail = _adapter(items, f"{name}_item")

    def _by_position(value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            return value
        out = list(value)
        for index, item in enumerate(out):
            adapter = adapters[index] if index < len(adapters) else tail
            if adapter is not None:
                out[index] = adapter.validate_python(item)
        return out

    return Annotated[list, BeforeValidator(_by_position)]


def _enum_membership(values: list[Any]) -> Any:
    """Membership check for an enum whose members can't be ``Literal`` members.

    A tuple-valued ``Enum`` canonicalizes to ``[[0, 0], [1, 1]]``, and a list
    is not a valid ``Literal`` member. Such an enum also has no inferred
    ``type``, so falling through advertised an unconstrained ``Any`` while the
    executor enforces membership — the framework accepted anything the model
    invented. Keyed by JSON identity, the same helper the executor's ``enum``
    check uses, so the two agree on what counts as a member.
    """
    from typing import Annotated

    from pydantic import BeforeValidator

    member_keys = [json_identity_key(v) for v in values]

    def _is_member(value: Any) -> Any:
        if json_identity_key(value) not in member_keys:
            raise ValueError(f"is not one of {values}")
        return value

    return Annotated[Any, BeforeValidator(_is_member)]


def _with_pattern_properties(
    name: str,
    patterns: dict[str, Any],
    closed: bool,
    depth: int,
    additional: Any = None,
) -> Any:
    """Annotation for a mapping whose keys are typed by regex.

    Pydantic emits ``patternProperties`` for a mapping with constrained keys,
    and there is no Python annotation for "keys matching this regex have this
    value type" — but ignoring it was wrong in *both* directions: an open
    mapping accepted values the executor rejects, and one with
    ``additionalProperties: false`` built an empty closed model that rejected
    the valid matching keys the executor accepts. The second is the worse
    error, since it makes a correct call impossible.
    """
    from typing import Annotated

    from pydantic import BeforeValidator

    check = _pattern_key_validator(
        name, patterns, closed, depth, frozenset(), additional
    )
    return Annotated[dict, BeforeValidator(check)]


def _pattern_key_validator(
    name: str,
    patterns: dict[str, Any],
    closed: bool,
    depth: int,
    declared: Any,
    additional: Any = None,
) -> Any:
    """A callable enforcing ``patternProperties`` over a mapping's keys.

    ``declared`` names the keys the surrounding model already types. They are
    exempt from the closed-object "matches no pattern" check — a named property
    is declared by definition — but *not* from pattern validation itself: JSON
    Schema requires a key to satisfy its ``properties`` schema **and** every
    matching ``patternProperties`` schema, so ``n_fixed`` declared as an
    integer beside ``{"^n_": {"minimum": 5}}`` must satisfy both.

    ``additional`` is the object's ``additionalProperties`` schema when it has
    one. JSON Schema applies it to exactly the keys no pattern matched, and the
    executor does; passing only a boolean "is it closed" flag left those keys
    typed by nothing, so a mapping with ``{"^s_": {"type": "string"}}`` beside
    ``additionalProperties: {"type": "integer"}`` accepted ``{"other": "bad"}``
    that the engine rejects.
    """
    import re

    from pydantic import TypeAdapter

    compiled: list[tuple[Any, Any]] = []
    for index, (regex, subschema) in enumerate(patterns.items()):
        if not isinstance(regex, str):
            continue
        try:
            matcher = re.compile(regex)
        except re.error:
            # Fail open, as the executor does for an ECMA-only pattern.
            continue
        adapter = None
        if isinstance(subschema, bool):
            # ``{"^blocked_": false}`` forbids every matching key.
            adapter = TypeAdapter(_annotation(f"{name}_pattern{index}", subschema, depth + 1))
        elif isinstance(subschema, dict) and subschema:
            adapter = TypeAdapter(
                _with_constraints(
                    _annotation(f"{name}_pattern{index}", subschema, depth + 1), subschema
                )
            )
        compiled.append((matcher, adapter))

    additional_adapter = None
    if isinstance(additional, dict) and additional:
        additional_adapter = TypeAdapter(
            _with_constraints(
                _annotation(f"{name}_additional", additional, depth + 1), additional
            )
        )

    def _check(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        # A *new* mapping: the adapters' converted values are written back, so
        # a pattern-keyed value behaves like a declared one. Discarding them
        # left ``{"n_x": "1"}`` a string against ``{"type": "integer"}`` —
        # accepted by the model, then forwarded unchanged by LlamaIndex's
        # ``model_dump()`` and rejected by the executor. Declared properties
        # have always coerced here; only pattern keys did not. Building a new
        # dict rather than mutating the caller's.
        rebuilt: dict[Any, Any] = {}
        for key, item in value.items():
            matched = False
            for matcher, adapter in compiled:
                if not matcher.search(str(key)):
                    continue
                matched = True
                if adapter is not None:
                    item = adapter.validate_python(item)
            if matched:
                rebuilt[key] = item
                continue
            if key in declared:
                rebuilt[key] = item
                continue
            if closed:
                raise ValueError(f"key {key!r} matches no patternProperties entry")
            if additional_adapter is not None:
                # Matched no pattern, so this is exactly the set
                # ``additionalProperties`` governs.
                item = additional_adapter.validate_python(item)
            rebuilt[key] = item
        return rebuilt

    return _check


def _annotation(name: str, prop: Any, depth: int) -> Any:
    """Python annotation for one property schema (recursive).

    A schema may be a bare boolean anywhere a schema is allowed — a named
    property, a ``patternProperties`` entry, an ``items`` tail — not only in a
    combinator branch. Handled here, at the one funnel every subschema passes
    through, so the model agrees with the executor, which reads them at the
    equivalent point in its own validator.
    """
    if prop is True:
        return Any
    if prop is False:
        return _never_annotation()
    if not isinstance(prop, dict):
        return Any
    # ``const`` is a one-value ``enum`` — the shape Pydantic emits for a
    # single-value ``Literal``, so it turns up inside the nested models this
    # bridge inlines. Ignoring it advertised an unconstrained scalar that
    # accepted values the executor then rejected.
    if "const" in prop:
        const_value = prop["const"]
        # Floats included, for the same reason the ``enum`` path below admits
        # them: PEP 586 disallows a float ``Literal`` member statically, but
        # ``typing`` and Pydantic both enforce it correctly at runtime, and
        # ``{"type": "number", "const": 0.5}`` is what a single-value float
        # ``Literal`` emits. Excluding it dropped the constraint entirely and
        # advertised an unconstrained ``float``.
        if isinstance(const_value, (str, int, bool, float)) or const_value is None:
            return _literal_annotation([const_value])
        # A composite constant (``{"type": "array", "const": [1, 2]}``) can't
        # be a ``Literal`` member, and falling through advertised a plain
        # ``list`` that accepted anything while the executor enforced the
        # constant. Checked by JSON identity, exactly as a composite enum is.
        return _enum_membership([const_value])

    enum_values = prop.get("enum")
    if isinstance(enum_values, list) and enum_values:
        # ``None`` is a valid Literal member (``Literal["auto", None]``), and
        # an enum listing it is exactly how a nullable choice is expressed —
        # excluding it dropped the whole enum to the unconstrained fallback.
        # Floats are admitted too: PEP 586 disallows them, but a float-valued
        # ``Enum``/``Literal`` parameter is what introspection emits and both
        # ``typing`` and Pydantic enforce it correctly at runtime, which is
        # what matters here — the alternative is no constraint at all.
        if all(
            isinstance(v, (str, int, bool, float)) or v is None for v in enum_values
        ):
            return _literal_annotation(enum_values)
        # Composite members (a tuple-valued ``Enum``) can't be ``Literal``
        # members, and such an enum has no inferred ``type`` — so falling
        # through advertised an unconstrained ``Any``. Check membership
        # directly instead.
        return _enum_membership(enum_values)

    json_type = prop.get("type")
    # A field can be typed purely through a combinator, with no ``type`` of
    # its own — ``{"anyOf": [{"type": "integer"}, {"type": "null"}]}`` is
    # what Pydantic emits for ``int | None``, so it appears in every nested
    # model this bridge inlines. Falling through to ``Any`` would advertise
    # an unconstrained field that accepts values the executor rejects after
    # dispatch. Checked even when a ``type`` *is* present: JSON Schema
    # applies both, so gating on a missing type let ``{"type": "integer",
    # "anyOf": [...]}`` through as a bare ``int``.
    # ``_effective_type`` rather than a bare ``isinstance(json_type, str)``:
    # a *nullable* parent (``{"type": ["integer", "null"], "anyOf": [...]}``,
    # which imported schemas produce) still names one real member type, and
    # not inheriting it left both constraint-only branches as a bare ``Any``
    # that matched everything. It returns ``None`` for a genuine multi-type
    # list, where no single type applies to every branch.
    parent_type = _effective_type(prop)
    union = _union_annotation(name, prop, depth, inherited_type=parent_type)
    if union is not None:
        # A branch that declares its *own* type doesn't take the inherited
        # one, so the parent's ``type`` assertion would simply be lost —
        # ``{"type": "integer", "anyOf": [{"type": "number"}, {"type":
        # "string"}]}`` became ``float | str`` while the executor applies both
        # and rejects each. Intersecting the union with the parent type
        # restores it, and is a no-op for branches that already inherited it.
        base = _PARENT_KINDS.get(parent_type) if isinstance(parent_type, str) else None
        if base is not None:
            if schema_declares_null(prop):
                # The parent admits null, so the intersection must too, or it
                # would reject the very value the type list declares.
                base = base | None
            union = _with_intersection(union, [base])
        return union

    if json_type == "null":
        # A property permitting *only* null. Falling through to ``Any`` would
        # let the framework accept strings and numbers the canonical schema
        # forbids.
        return type(None)

    if isinstance(json_type, list):  # e.g. ["string", "null"]
        non_null = [t for t in json_type if t != "null"]
        if not non_null:  # e.g. ["null"]
            return type(None)
        # ``null`` in the list has to be carried on the annotation itself. A
        # *top-level* field gets it back from ``schema_declares_null`` in
        # ``_build_model``, but an array item, an ``additionalProperties``
        # value and a combinator branch never pass through there — so
        # ``{"type": "array", "items": {"type": ["string", "null"]}}`` became
        # ``list[str]`` and rejected a schema-valid ``[null]``.
        # ``schema_declares_null`` rather than a bare membership test, so an
        # enum or const that excludes null still wins, exactly as it does
        # at the top level.
        admits_null = schema_declares_null(prop)
        if len(non_null) == 1 and admits_null:
            # Re-entered with a scalar ``type``, so this cannot recurse here.
            return _annotation(f"{name}_0", {**prop, "type": non_null[0]}, depth + 1) | None
        if len(non_null) > 1:
            # Several real member types. Collapsing to the first would reject
            # values the executor accepts (it validates against *any* listed
            # member), so union them. Each member is re-entered with a scalar
            # ``type``, so this cannot re-enter this branch.
            annotation = _annotation(f"{name}_0", {**prop, "type": non_null[0]}, depth + 1)
            for index, member in enumerate(non_null[1:], start=1):
                annotation = annotation | _annotation(
                    f"{name}_{index}", {**prop, "type": member}, depth + 1
                )
            return annotation | None if admits_null else annotation
        json_type = non_null[0]

    if json_type in _SCALARS:
        return _SCALARS[json_type]
    if json_type == "array":
        items = prop.get("items")
        prefix_items = prop.get("prefixItems")
        if isinstance(prefix_items, list) and prefix_items:
            return _positional_array(name, prefix_items, items, depth)
        if isinstance(items, bool):
            # ``{"type": "array", "items": false}`` permits only the empty
            # array; ``true`` permits any. The positional path already routed
            # booleans through ``_annotation``; this branch — a plain array
            # with no ``prefixItems`` — fell through to a bare ``list`` and
            # accepted the elements the executor rejects.
            return list[_annotation(f"{name}_item", items, depth + 1)]
        if isinstance(items, dict) and items:
            # Constraints ride along on the *item* schema too
            # (``list[Annotated[int, Field(gt=0)]]`` puts them there), so
            # apply them here as well as on the outer property.
            return list[
                _with_constraints(_annotation(f"{name}_item", items, depth + 1), items)
            ]
        return list
    if json_type == "object":
        properties = prop.get("properties")
        if isinstance(properties, dict) and properties:
            return _build_model(f"{name}_obj", prop, depth + 1)
        patterns = prop.get("patternProperties")
        if isinstance(patterns, dict) and patterns:
            # Checked before the closed-object branch below: that one would
            # build a model permitting *no* keys, rejecting the very keys the
            # patterns declare.
            return _with_pattern_properties(
                name,
                patterns,
                not _admits_undeclared(prop),
                depth,
                prop.get("additionalProperties"),
            )
        if prop.get("additionalProperties") is False:
            # An object that permits no keys at all, which the executor
            # enforces — whether or not it spells out an empty ``properties``.
            # A bare ``dict`` here would accept anything, so the framework
            # would wave through a payload the engine rejects.
            closed = {**prop, "properties": properties if isinstance(properties, dict) else {}}
            return _build_model(f"{name}_obj", closed, depth + 1)
        additional = prop.get("additionalProperties")
        if isinstance(additional, dict) and additional:
            # No declared properties, but a typed ``additionalProperties``
            # (e.g. ``dict[str, int]``) — preserve the value type instead of
            # widening to a bare ``dict`` that would accept any value type.
            return dict[
                str,
                _with_constraints(
                    _annotation(f"{name}_value", additional, depth + 1), additional
                ),
            ]
        return dict
    return Any


__all__ = ["pydantic_model_from_schema"]
