"""Tests for rebuilding JSON arguments into the types a handler declares.

The executor dispatches ``handler(**arguments)`` with JSON-decoded values.
Once ``build_parameters_schema`` advertises a nested model, a ``set`` or a
``datetime``, a provider sends the JSON form of that type — and forwarding it
unchanged handed the handler a ``dict``/``list``/``str`` where its annotation
promised the real thing. These tests pin both halves: the types that *are*
rebuilt, and the ones deliberately left alone.
"""

from __future__ import annotations

import collections.abc as abc
import dataclasses
import datetime
import enum
import typing
import uuid
from typing import Any, TypedDict

import pytest
from pydantic import BaseModel, field_validator

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.schema.execution import ToolCall
from agent_gantry.schema.introspection import build_argument_coercers
from agent_gantry.schema.tool import ToolDefinition


class Payload(BaseModel):
    x: int


class Positive(BaseModel):
    """A model whose invariant no JSON Schema keyword can carry."""

    x: int

    @field_validator("x")
    @classmethod
    def _must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("x must be positive")
        return value


@dataclasses.dataclass
class Address:
    street: str


class Options(TypedDict):
    k: int


class Mode(enum.Enum):
    FAST = "fast"


def test_only_types_whose_json_form_differs_are_coerced():
    """Scalars, ``list``, ``dict``, ``TypedDict`` and ``Any`` already arrive as
    themselves — coercing them would change what every existing handler
    receives for no benefit."""

    def handler(
        p: Payload,
        a: Address,
        tags: set[str],
        pair: tuple[int, str],
        at: datetime.datetime,
        u: uuid.UUID,
        m: Mode,
        maybe: Payload | None,
        s: str,
        i: int,
        items: list[int],
        mapping: dict[str, int],
        opts: Options,
        anything: Any,
    ) -> None: ...

    assert sorted(build_argument_coercers(handler)) == [
        "a",
        "at",
        "m",
        "maybe",
        "p",
        "pair",
        "tags",
        "u",
    ]


def test_the_typing_optional_spelling_is_handled_too():
    """``Optional[X]`` and ``X | None`` are different origins at runtime
    (``typing.Union`` vs ``types.UnionType``); both must be recognized."""
    from agent_gantry.schema.introspection import _needs_reconstruction

    # Built at runtime rather than written as an annotation: the point is
    # the ``typing.Union`` origin, which the modern spelling doesn't produce.
    optional_payload = typing.Union[Payload, None]  # noqa: UP007
    optional_str = typing.Union[str, None]  # noqa: UP007
    assert _needs_reconstruction(optional_payload) is True
    assert _needs_reconstruction(optional_str) is False


def test_handler_with_no_such_parameters_gets_no_coercers():
    def handler(name: str, count: int = 1) -> None: ...

    assert build_argument_coercers(handler) == {}


def test_container_generics_recurse_into_their_members():
    """``list[Payload]`` has origin ``list`` and isn't a bare class, so it fell
    through every check and the handler got a list of raw dicts — the same
    failure reconstruction exists to fix, one container level up
    (PR #381 review)."""
    from agent_gantry.schema.introspection import _needs_reconstruction

    for needs in (
        list[Payload],
        list[datetime.datetime],
        list[Mode],
        dict[str, Payload],
        list[list[Payload]],
    ):
        assert _needs_reconstruction(needs) is True, needs

    for plain in (list[int], list[str], dict[str, int], dict[str, list[str]]):
        assert _needs_reconstruction(plain) is False, plain


@pytest.fixture
async def gantry() -> AgentGantry:
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["demo"])
    def use_model(p: Payload) -> str:
        """Read a field off a nested payload model."""
        return f"x={p.x}"

    @g.register(tags=["demo"])
    def use_dataclass(a: Address) -> str:
        """Read a field off a dataclass address."""
        return f"street={a.street}"

    @g.register(tags=["demo"])
    def use_set(tags: set[str]) -> str:
        """Report the runtime type of a set parameter."""
        return f"{type(tags).__name__}:{sorted(tags)}"

    @g.register(tags=["demo"])
    def use_scalars(name: str, count: int) -> str:
        """Report the runtime types of plain scalar parameters."""
        return f"{type(name).__name__}/{type(count).__name__}"

    @g.register(tags=["demo"])
    def use_many(
        pair: tuple[int, str],
        at: datetime.datetime,
        ident: uuid.UUID,
        mode: Mode,
        frozen: frozenset[str],
        payloads: list[Payload],
    ) -> str:
        """Report the runtime type of every reconstructed parameter kind."""
        return "|".join(
            [
                type(pair).__name__,
                type(at).__name__,
                type(ident).__name__,
                type(mode).__name__,
                type(frozen).__name__,
                type(payloads[0]).__name__,
            ]
        )

    await g.sync()
    return g


async def test_nested_model_reaches_the_handler_as_its_declared_type(gantry):
    result = await gantry.execute(
        ToolCall(tool_name="use_model", arguments={"p": {"x": 1}})
    )
    assert result.status.value == "success", result.error
    assert result.result == "x=1"


async def test_dataclass_reaches_the_handler_as_its_declared_type(gantry):
    result = await gantry.execute(
        ToolCall(tool_name="use_dataclass", arguments={"a": {"street": "Main"}})
    )
    assert result.status.value == "success", result.error
    assert result.result == "street=Main"


async def test_array_becomes_the_declared_set(gantry):
    result = await gantry.execute(
        ToolCall(tool_name="use_set", arguments={"tags": ["a", "b"]})
    )
    assert result.status.value == "success", result.error
    assert result.result == "set:['a', 'b']"


async def test_scalar_arguments_are_passed_through_untouched(gantry):
    """The conservative half: a handler taking scalars must receive exactly
    what it received before this behaviour existed."""
    result = await gantry.execute(
        ToolCall(tool_name="use_scalars", arguments={"name": "n", "count": 2})
    )
    assert result.status.value == "success", result.error
    assert result.result == "str/int"


async def test_an_unconvertible_value_is_a_validation_error_not_a_fallback():
    """Reconstruction failure is terminal.

    Reaching it needs a schema *looser* than the annotation — which is exactly
    what an imported (MCP/OpenAPI) or hand-written schema can be, and what a
    Pydantic ``field_validator`` whose invariant JSON Schema can't express
    produces even for a schema Gantry emitted itself. Here the schema says
    "any object" while the handler says ``Payload``, so ``{"y": 2}`` passes
    validation and then fails reconstruction.

    This *used* to pass the raw mapping through, on the reasoning that
    validation had already run and a handler happy with a dict shouldn't start
    failing. That reasoning doesn't hold: the coercer is installed precisely
    because the handler declares ``Payload``, so the raw mapping is the one
    value it cannot take — ``p.x`` raises ``AttributeError`` deep inside the
    tool, or the tool misbehaves silently. A clear rejection is the honest
    outcome (PR #381 review).
    """
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    def loose(p: Payload) -> str:
        """Report what the handler actually received."""
        return type(p).__name__

    await g.add_tool(
        ToolDefinition(
            name="loose",
            description="Accepts a free-form object despite a typed handler.",
            parameters_schema={
                "type": "object",
                "properties": {"p": {"type": "object"}},
                "required": ["p"],
            },
            tags=["demo"],
        ),
        handler=loose,
    )
    await g.sync()

    # A well-formed Payload is still rebuilt.
    good = await g.execute(ToolCall(tool_name="loose", arguments={"p": {"x": 1}}))
    assert good.status.value == "success", good.error
    assert good.result == "Payload"

    # One the schema admits but ``Payload`` rejects is refused here rather
    # than dispatched as a ``dict`` to a handler annotated ``Payload``.
    rejected = await g.execute(ToolCall(tool_name="loose", arguments={"p": {"y": 2}}))
    assert rejected.status.value == "failure"
    assert rejected.error_type == "ValidationError"
    assert "p" in rejected.error


async def test_every_reconstructed_kind_arrives_typed_end_to_end(gantry):
    """The unit test above only checks *which* parameters get a coercer. This
    checks dispatch actually produces the declared types — the gap that let a
    broken fallback test go unnoticed (PR #381 review)."""
    result = await gantry.execute(
        ToolCall(
            tool_name="use_many",
            arguments={
                "pair": [1, "a"],
                "at": "2026-08-27T00:00:00",
                "ident": "urn:uuid:12345678-1234-5678-1234-567812345678",
                "mode": "fast",
                "frozen": ["a", "b"],
                "payloads": [{"x": 1}],
            },
        )
    )
    assert result.status.value == "success", result.error
    assert result.result == "tuple|datetime|UUID|Mode|frozenset|Payload"


class Point(enum.Enum):
    """An ``Enum`` whose values are tuples, so its JSON form is an array."""

    ORIGIN = (0, 0)
    UNIT = (1, 1)


async def test_composite_enum_member_is_recovered_from_its_json_array():
    """The canonical schema advertises tuple-valued members as JSON *arrays*,
    so a provider sends ``[0, 0]`` — which Pydantic can't match back to the
    member value ``(0, 0)``. The handler received the raw list instead of the
    member (PR #381 review)."""
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["demo"])
    def use_point(pt: Point) -> str:
        """Use a point enum whose values are tuples."""
        return f"{type(pt).__name__}:{pt.name}"

    await g.sync()
    result = await g.execute(ToolCall(tool_name="use_point", arguments={"pt": [0, 0]}))
    assert result.status.value == "success", result.error
    assert result.result == "Point:ORIGIN"


async def test_composite_enum_recovery_reaches_nested_positions():
    """Applied to the whole annotation, not just a bare ``Enum`` parameter, so
    a container or optional carries it too."""
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["demo"])
    def nested(points: list[Point], maybe: Point | None = None) -> str:
        """Handle composite enums nested inside containers."""
        kinds = [type(p).__name__ for p in points]
        return f"{kinds}|{type(maybe).__name__}"

    await g.sync()
    result = await g.execute(
        ToolCall(tool_name="nested", arguments={"points": [[0, 0], [1, 1]], "maybe": [1, 1]})
    )
    assert result.status.value == "success", result.error
    assert result.result == "['Point', 'Point']|Point"


async def test_a_value_matching_no_member_is_still_rejected():
    """Recovery must not become a way to smuggle a non-member through."""
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["demo"])
    def use_point(pt: Point) -> str:
        """Use a point enum whose values are tuples."""
        return pt.name

    await g.sync()
    result = await g.execute(ToolCall(tool_name="use_point", arguments={"pt": [9, 9]}))
    assert result.status.value == "failure"
    assert "must be one of" in (result.error or "")


def test_plain_scalar_enums_need_no_recovery():
    """A ``str``/``int`` enum already round-trips, so its annotation is left
    exactly as declared rather than being wrapped."""
    from agent_gantry.schema.introspection import _with_enum_recovery

    assert _with_enum_recovery(Mode) is Mode
    assert _with_enum_recovery(list[Mode]) == list[Mode]
    assert _with_enum_recovery(Point) is not Point


async def test_heterogeneous_tuple_is_typed_position_by_position(gantry):
    """``tuple[int, str]`` has no single item type, so the emitted schema used
    to be a bare ``{"type": "array"}``. That accepted ``["bad", 1]``,
    reconstruction then couldn't build the tuple, and the fallback handed the
    handler the raw list — the exact failure reconstruction exists to prevent
    (PR #381 review)."""
    tool = await gantry.get_tool("use_many")
    assert tool is not None
    pair = tool.parameters_schema["properties"]["pair"]
    assert pair["prefixItems"] == [{"type": "integer"}, {"type": "string"}]
    assert pair["minItems"] == pair["maxItems"] == 2

    # Right arity, wrong order: rejected at validation rather than reaching
    # the handler as a ``list``.
    swapped = await gantry.execute(
        ToolCall(
            tool_name="use_many",
            arguments={
                "pair": ["bad", 1],
                "at": "2026-08-27T00:00:00",
                "ident": "urn:uuid:12345678-1234-5678-1234-567812345678",
                "mode": "fast",
                "frozen": ["a"],
                "payloads": [{"x": 1}],
            },
        )
    )
    assert swapped.status.value == "failure"
    assert swapped.error_type == "ValidationError"

    # And the arity itself is pinned.
    short = await gantry.execute(
        ToolCall(
            tool_name="use_many",
            arguments={
                "pair": [1],
                "at": "2026-08-27T00:00:00",
                "ident": "urn:uuid:12345678-1234-5678-1234-567812345678",
                "mode": "fast",
                "frozen": ["a"],
                "payloads": [{"x": 1}],
            },
        )
    )
    assert short.status.value == "failure"
    assert short.error_type == "ValidationError"


def test_a_variadic_tuple_keeps_its_homogeneous_item_type():
    """``tuple[int, ...]`` *does* have a single item type — the positional
    branch must not steal it and pin a length of two."""
    from agent_gantry.schema.introspection import _type_to_json_schema

    schema = _type_to_json_schema(tuple[int, ...])
    assert schema["items"] == {"type": "integer"}
    assert "prefixItems" not in schema
    assert "maxItems" not in schema


def test_optional_members_of_a_container_keep_their_null():
    """A top-level ``int | None = None`` can express "no value" by being
    omitted, which is why the emitted schema stays a bare ``integer``. A
    container member has no such escape hatch, so collapsing the union there
    left the schema forbidding a value the handler's own annotation accepts
    (PR #381 review)."""
    from agent_gantry.schema.introspection import _type_to_json_schema

    assert _type_to_json_schema(list[int | None]) == {
        "type": "array",
        "items": {"type": ["integer", "null"]},
    }
    assert _type_to_json_schema(dict[str, int | None]) == {
        "type": "object",
        "additionalProperties": {"type": ["integer", "null"]},
    }
    assert _type_to_json_schema(tuple[int | None, str])["prefixItems"] == [
        {"type": ["integer", "null"]},
        {"type": "string"},
    ]
    # Nesting is threaded, not applied once at the outermost container.
    assert _type_to_json_schema(list[list[int | None]]) == {
        "type": "array",
        "items": {"type": "array", "items": {"type": ["integer", "null"]}},
    }
    # The top level is deliberately untouched: requiredness carries it there.
    assert _type_to_json_schema(int | None) == {"type": "integer"}
    assert _type_to_json_schema(list[int]) == {
        "type": "array",
        "items": {"type": "integer"},
    }


async def test_a_null_container_member_survives_validation():
    """End-to-end: the handler declares ``list[int | None]``, so ``[1, None, 2]``
    is a call it accepts and validation must not refuse."""
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["demo"])
    def count_present(values: list[int | None]) -> str:
        """Count how many members are not None."""
        return f"{sum(1 for v in values if v is not None)}/{len(values)}"

    await g.sync()
    result = await g.execute(
        ToolCall(tool_name="count_present", arguments={"values": [1, None, 2]})
    )
    assert result.status.value == "success", result.error
    assert result.result == "2/3"

    # The member type is still enforced — widening admits null, not anything.
    bad = await g.execute(
        ToolCall(tool_name="count_present", arguments={"values": [1, "x"]})
    )
    assert bad.status.value == "failure"
    assert bad.error_type == "ValidationError"


def test_the_coercer_cache_evicts_rather_than_stopping():
    """A hard cap with no eviction pins the first N handlers for the life of
    the process and re-inspects every handler after them on *every* call. The
    bound has to evict (PR #381 review)."""
    import agent_gantry.core.executor as executor_module

    original_max = executor_module._COERCER_CACHE_MAX
    original_cache = executor_module._COERCER_CACHE.copy()
    executor_module._COERCER_CACHE.clear()
    executor_module._COERCER_CACHE_MAX = 3
    try:

        def make(i: int):
            namespace: dict[str, Any] = {}
            exec(f"def h{i}(tags: set[str]) -> None: ...", namespace)
            return namespace[f"h{i}"]

        handlers = [make(i) for i in range(5)]
        for handler in handlers:
            executor_module._coercers_for(handler)

        assert len(executor_module._COERCER_CACHE) == 3
        assert handlers[0] not in executor_module._COERCER_CACHE  # evicted
        assert handlers[4] in executor_module._COERCER_CACHE

        # Least-*recently-used*, not merely least-recently-inserted.
        executor_module._coercers_for(handlers[2])
        executor_module._coercers_for(make(9))
        assert handlers[2] in executor_module._COERCER_CACHE

        # Targeted invalidation still works — the reason this isn't lru_cache.
        executor_module.forget_handler(handlers[2])
        assert handlers[2] not in executor_module._COERCER_CACHE
    finally:
        executor_module._COERCER_CACHE_MAX = original_max
        executor_module._COERCER_CACHE.clear()
        executor_module._COERCER_CACHE.update(original_cache)


def test_every_fixed_length_tuple_pins_its_arity():
    """``prefixItems`` was what carried the length bounds, so a *homogeneous*
    fixed tuple — which takes the ``items`` branch instead, having a single
    shared item type — advertised an array of any length. ``[1]`` for
    ``tuple[int, int]`` then validated, reconstruction failed, and the
    fallback handed the handler a raw list (PR #381 review)."""
    from agent_gantry.schema.introspection import _type_to_json_schema

    assert _type_to_json_schema(tuple[int, int]) == {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 2,
    }
    assert _type_to_json_schema(tuple[int]) == {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 1,
        "maxItems": 1,
    }
    # Heterogeneous keeps prefixItems and gains nothing it didn't have.
    heterogeneous = _type_to_json_schema(tuple[int, str])
    assert heterogeneous["minItems"] == heterogeneous["maxItems"] == 2
    assert "prefixItems" in heterogeneous
    # A variadic tuple has no fixed arity to pin.
    variadic = _type_to_json_schema(tuple[int, ...])
    assert "minItems" not in variadic and "maxItems" not in variadic


async def test_a_short_homogeneous_tuple_is_rejected_not_reconstructed():
    """End-to-end consequence of the bounds above, on the branch that lacked
    them: ``tuple[int, int]`` shares one item type, so it never had
    ``prefixItems`` to carry the arity."""
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["demo"])
    def span(bounds: tuple[int, int]) -> str:
        """Report the width between a pair of integer bounds."""
        return f"{bounds[1] - bounds[0]}:{type(bounds).__name__}"

    await g.sync()
    ok = await g.execute(ToolCall(tool_name="span", arguments={"bounds": [2, 5]}))
    assert ok.status.value == "success", ok.error
    assert ok.result == "3:tuple"

    for wrong_arity in ([1], [1, 2, 3]):
        result = await g.execute(
            ToolCall(tool_name="span", arguments={"bounds": wrong_arity})
        )
        assert result.status.value == "failure", wrong_arity
        assert result.error_type == "ValidationError"


def test_a_mapping_keyed_by_anything_but_str_needs_rebuilding():
    """JSON object keys are always strings, so a mapping annotated otherwise
    never arrives as itself — however simple its *values* are, which is all
    this predicate used to look at (PR #381 review)."""
    from agent_gantry.schema.introspection import _needs_reconstruction

    assert _needs_reconstruction(dict[int, str]) is True
    assert _needs_reconstruction(dict[uuid.UUID, str]) is True
    assert _needs_reconstruction(typing.Mapping[int, str]) is True
    # A string-keyed mapping still arrives as itself.
    assert _needs_reconstruction(dict[str, int]) is False
    assert _needs_reconstruction(dict[str, str]) is False
    assert _needs_reconstruction(dict) is False
    # ``Any`` names no conversion to perform.
    assert _needs_reconstruction(dict[Any, str]) is False


async def test_integer_mapping_keys_reach_the_handler_as_integers():
    """``{"1": "a"}`` is the only thing a provider can send for
    ``dict[int, str]``, and the handler must still get ``{1: "a"}``."""
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["demo"])
    def sum_keys(counts: dict[int, str]) -> str:
        """Sum the integer keys of a mapping."""
        return f"{sum(counts)}:{type(next(iter(counts))).__name__}"

    await g.sync()
    result = await g.execute(
        ToolCall(tool_name="sum_keys", arguments={"counts": {"1": "a", "2": "b"}})
    )
    assert result.status.value == "success", result.error
    assert result.result == "3:int"


def test_bare_container_and_bytes_annotations_are_rebuilt_too():
    """``get_origin`` is ``None`` for an unparameterized ``set``/``frozenset``/
    ``tuple``, so those spellings missed the generic branch entirely — while
    introspection still advertised them as JSON arrays, leaving the handler a
    ``list``. Bare ``bytes`` is the same story one type over (PR #381
    review)."""
    from agent_gantry.schema.introspection import _needs_reconstruction

    for bare in (set, frozenset, tuple, bytes):
        assert _needs_reconstruction(bare) is True, bare
    # The types that genuinely arrive as themselves are still excluded —
    # coercing them would change what every existing handler receives.
    for unchanged in (list, dict, str, int, float, bool):
        assert _needs_reconstruction(unchanged) is False, unchanged


async def test_a_bare_set_annotation_reaches_the_handler_as_a_set():
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["demo"])
    def dedupe(tags: set) -> str:
        """Report the runtime type and size of a bare-set parameter."""
        return f"{type(tags).__name__}:{len(tags)}"

    await g.sync()
    result = await g.execute(
        ToolCall(tool_name="dedupe", arguments={"tags": ["a", "b"]})
    )
    assert result.status.value == "success", result.error
    assert result.result == "set:2"

    # A bare ``set`` now carries ``uniqueItems`` exactly as ``set[str]``
    # always has, so duplicates are refused by the same rule for both.
    tool = await g.get_tool("dedupe")
    assert tool.parameters_schema["properties"]["tags"]["uniqueItems"] is True
    duplicated = await g.execute(
        ToolCall(tool_name="dedupe", arguments={"tags": ["a", "b", "a"]})
    )
    assert duplicated.status.value == "failure"
    assert duplicated.error_type == "ValidationError"


async def test_a_malformed_formatted_string_is_rejected_not_passed_through():
    """Validation read only the JSON *type*, so a bad ``date-time`` passed,
    reconstruction then failed, and the fallback handed the raw ``str`` to a
    handler annotated ``datetime`` — reported as a success. The format is
    enforced with the same parser reconstruction uses (PR #381 review)."""
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["demo"])
    def when(at: datetime.datetime) -> str:
        """Report the runtime type of a timestamp argument."""
        return type(at).__name__

    await g.sync()
    good = await g.execute(
        ToolCall(tool_name="when", arguments={"at": "2026-08-27T00:00:00"})
    )
    assert good.status.value == "success", good.error
    assert good.result == "datetime"

    bad = await g.execute(ToolCall(tool_name="when", arguments={"at": "not-a-date"}))
    assert bad.status.value == "failure"
    assert bad.error_type == "ValidationError"


def test_only_the_formats_gantry_reconstructs_are_enforced():
    """``format`` is an annotation by default in JSON Schema. Enforcing every
    one of them would reject calls that work against an imported schema using
    ``email``/``uri`` loosely; these four are different because Gantry emits
    them precisely because the handler's annotation demands them."""
    from agent_gantry.schema.base import check_json_constraints

    for fmt, ok, bad in (
        ("date-time", "2026-08-27T00:00:00", "not-a-date"),
        ("date", "2026-08-27", "nope"),
        ("time", "12:30:00", "nope"),
        ("uuid", "12345678-1234-5678-1234-567812345678", "xx"),
    ):
        schema = {"type": "string", "format": fmt}
        assert check_json_constraints(ok, schema, "v") is None, fmt
        assert check_json_constraints(bad, schema, "v") is not None, fmt

    # Left as an annotation, per the spec's default.
    assert check_json_constraints("nope", {"type": "string", "format": "email"}, "v") is None
    # A non-string value is not a format's business.
    assert check_json_constraints(5, {"type": "string", "format": "date-time"}, "v") is None


async def test_an_invariant_json_schema_cannot_express_is_still_terminal():
    """The case that makes the fallback indefensible: a Pydantic
    ``field_validator`` whose rule no JSON Schema keyword can carry. Schema
    validation *cannot* have ruled the value out, so passing the raw mapping
    through hands a ``dict`` to a handler annotated with the model
    (PR #381 review)."""
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["demo"])
    def use_positive(p: Positive) -> str:
        """Read a field off a model with a custom validator."""
        return f"{type(p).__name__}:{p.x}"

    await g.sync()
    ok = await g.execute(ToolCall(tool_name="use_positive", arguments={"p": {"x": 3}}))
    assert ok.status.value == "success", ok.error
    assert ok.result == "Positive:3"

    # ``{"x": -1}`` satisfies the emitted schema — ``x`` is an integer — and
    # only the model's own validator rejects it.
    bad = await g.execute(ToolCall(tool_name="use_positive", arguments={"p": {"x": -1}}))
    assert bad.status.value == "failure"
    assert bad.error_type == "ValidationError"


async def test_a_formatted_value_survives_the_framework_dispatch_boundary():
    """A framework reading ``python_signature`` sees ``datetime`` and may hand
    a real ``datetime`` back. The canonical schema types that property as a
    JSON string, so it has to be serialized before dispatch or a *valid* call
    is rejected (PR #381 review)."""
    from agent_gantry.integrations.frameworks.base import _json_native

    formatted = {"type": "string", "format": "date-time"}
    assert _json_native(datetime.datetime(2026, 8, 27), formatted) == "2026-08-27T00:00:00"
    assert _json_native("2026-08-27T00:00:00", formatted) == "2026-08-27T00:00:00"
    assert _json_native(uuid.UUID(int=1), {"type": "string", "format": "uuid"}) == str(
        uuid.UUID(int=1)
    )
    # Only where the schema declares one of those formats — a parameter
    # genuinely typed object or array is untouched.
    assert _json_native(datetime.datetime(2026, 8, 27), {"type": "string"}) == (
        datetime.datetime(2026, 8, 27)
    )
    assert _json_native({"a": 1}, {"type": "object"}) == {"a": 1}
    assert _json_native("x", None) == "x"


def test_the_ref_budget_bounds_expansion_not_every_visited_node():
    """The guard ran on entry, so once the budget was spent *every* value was
    replaced with ``{}`` — a ``type`` string, a ``required`` list — and a model
    wide enough to exhaust it emitted malformed metadata a provider rejects
    rather than merely unconstrained subschemas (PR #381 review)."""
    from agent_gantry.schema import introspection

    original = introspection._MAX_REF_NODES
    introspection._MAX_REF_NODES = 3
    try:
        # ``title``/``required`` deliberately sit *after* ``properties``:
        # resolution walks keys in order, so metadata declared first would be
        # visited before the budget ran out and the bug would not show.
        raw = {
            "type": "object",
            "$defs": {"Inner": {"type": "object", "properties": {"x": {"type": "integer"}}}},
            "properties": {f"f{i}": {"$ref": "#/$defs/Inner"} for i in range(5)},
            "title": "Wide",
            "required": [f"f{i}" for i in range(5)],
        }
        out = introspection._inline_local_refs(raw)
    finally:
        introspection._MAX_REF_NODES = original

    # Metadata survives intact.
    assert out["type"] == "object"
    assert out["title"] == "Wide"
    assert out["required"] == [f"f{i}" for i in range(5)]
    # Refs within budget are expanded; those past it degrade to ``{}`` only.
    assert out["properties"]["f0"]["properties"]["x"] == {"type": "integer"}
    assert out["properties"]["f4"] == {}


async def test_a_rejected_argument_does_not_degrade_tool_health():
    """A reconstruction failure is caller-supplied input, not a sick tool.
    Recording it opened the circuit breaker after five malformed calls, so a
    caller could disable a healthy tool for everyone — the valid call that
    followed came back ``CIRCUIT_OPEN``. The schema validation path has always
    left health alone; this now matches it (PR #381 review)."""
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["demo"])
    def use_positive(p: Positive) -> str:
        """Read a field off a model with a custom validator."""
        return f"ok:{p.x}"

    await g.sync()
    for _ in range(10):
        bad = await g.execute(
            ToolCall(tool_name="use_positive", arguments={"p": {"x": -1}})
        )
        assert bad.status.value == "failure"
        assert bad.error_type == "ValidationError"

    healthy = await g.execute(
        ToolCall(tool_name="use_positive", arguments={"p": {"x": 5}})
    )
    assert healthy.status.value == "success", healthy.error
    tool = await g.get_tool("use_positive")
    assert tool.health.consecutive_failures == 0
    assert tool.health.circuit_breaker_open is False


def test_bare_collection_abcs_are_classified_like_their_concrete_kin():
    """``def f(m: Mapping)`` has ``get_origin() is None`` and matches no
    concrete-class check, so it fell through to the ``str`` fallback: the tool
    advertised a *string* for a parameter needing a mapping, and the executor
    then rejected the correctly shaped object a caller sent. The parameterized
    forms were already handled (PR #381 review)."""
    from agent_gantry.schema.introspection import (
        _needs_reconstruction,
        _type_to_json_schema,
    )

    assert _type_to_json_schema(abc.Mapping) == {"type": "object"}
    assert _type_to_json_schema(abc.MutableMapping) == {"type": "object"}
    assert _type_to_json_schema(abc.Sequence) == {"type": "array"}
    assert _type_to_json_schema(abc.Iterable) == {"type": "array"}
    # A Mapping is also a Collection and an Iterable, so ordering matters:
    # the sequence test would otherwise call it an array.
    assert _type_to_json_schema(abc.Set) == {"type": "array", "uniqueItems": True}
    # Scalars are unaffected, which matters because ``str`` *is* a Sequence.
    assert _type_to_json_schema(str) == {"type": "string"}
    assert _type_to_json_schema(bytes) == {"type": "string"}

    # Only the ones whose JSON form differs are rebuilt: a list is already a
    # Sequence and a dict already a Mapping, but neither is a Set. Rebuilding
    # an ``Iterable`` would hand the handler a one-shot iterator instead of a
    # perfectly good list.
    assert _needs_reconstruction(abc.Set) is True
    assert _needs_reconstruction(abc.MutableSet) is True
    assert _needs_reconstruction(abc.Sequence) is False
    assert _needs_reconstruction(abc.Mapping) is False
    assert _needs_reconstruction(abc.Iterable) is False


async def test_a_bare_mapping_annotation_accepts_an_object():
    """End-to-end: the handler needs a mapping, so the schema must ask for one.

    ``abc`` is imported at module scope deliberately: this file uses
    ``from __future__ import annotations``, so a function-local import leaves
    ``get_type_hints`` unable to resolve the name and the parameter silently
    falls back to a string schema — the test would then pass against the very
    bug it exists to catch.
    """
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["demo"])
    def count_keys(m: abc.Mapping) -> str:
        """Report how many keys a mapping parameter carries."""
        return f"{len(m)}"

    await g.sync()
    tool = await g.get_tool("count_keys")
    assert tool.parameters_schema["properties"]["m"] == {"type": "object"}
    result = await g.execute(
        ToolCall(tool_name="count_keys", arguments={"m": {"a": 1, "b": 2}})
    )
    assert result.status.value == "success", result.error
    assert result.result == "2"


def test_a_parameterized_generic_never_takes_the_direct_type_branch():
    """On Python 3.10 a parameterized builtin *is* an instance of ``type``,
    where on 3.11+ it is not — so the two versions took different branches for
    the same annotation, and 3.10 reached the abstract-base checks with a
    generic alias, where ``issubclass`` raises ``TypeError: arg 1 must be a
    class``. It surfaced only once the ABCs were added, the concrete-class
    checks having tolerated it (PR #381 review).

    Asserted through the emitted schema rather than the branch taken, so it
    holds on every version: a parameterized generic must keep its parameters,
    which only the generic branch supplies.
    """
    import collections.abc as module_abc

    from agent_gantry.schema.introspection import _type_to_json_schema

    assert _type_to_json_schema(dict[str, int]) == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }
    assert _type_to_json_schema(module_abc.Mapping[str, int]) == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }
    assert _type_to_json_schema(list[int]) == {
        "type": "array",
        "items": {"type": "integer"},
    }
    assert _type_to_json_schema(set[str]) == {
        "type": "array",
        "uniqueItems": True,
        "items": {"type": "string"},
    }
    # And the bare forms still take the direct branch.
    assert _type_to_json_schema(dict) == {"type": "object"}
    assert _type_to_json_schema(module_abc.Mapping) == {"type": "object"}


class NullableMode(enum.Enum):
    """An ``Enum`` one of whose members *is* ``None``."""

    UNSET = None
    FAST = "fast"


def test_a_null_can_itself_be_a_value_worth_rebuilding():
    """``None`` used to short-circuit reconstruction, on the assumption that a
    null is never worth rebuilding. An ``Enum`` with a ``None``-valued member
    emits ``enum: [null]``, and a call supplying null reached the handler as
    raw ``None`` rather than the member (PR #381 review)."""
    from agent_gantry.core.executor import ArgumentReconstructionError, _reconstructed

    def use_mode(mode: NullableMode) -> None: ...

    assert _reconstructed(use_mode, {"mode": None}) == {"mode": NullableMode.UNSET}
    assert _reconstructed(use_mode, {"mode": "fast"}) == {"mode": NullableMode.FAST}

    # An annotation that admits ``None`` still gets it back unchanged, which
    # is why the shortcut could go rather than needing a carve-out.
    def optional_payload(p: typing.Optional[Payload]) -> None: ...  # noqa: UP045

    assert _reconstructed(optional_payload, {"p": None}) == {"p": None}
    assert _reconstructed(optional_payload, {"p": {"x": 1}}) == {"p": Payload(x=1)}

    # And one that doesn't is a schema/annotation mismatch, reported as such
    # rather than dispatched as a null the handler cannot take.
    def strict_payload(p: Payload) -> None: ...

    with pytest.raises(ArgumentReconstructionError):
        _reconstructed(strict_payload, {"p": None})


async def test_a_null_enum_member_reaches_the_handler_end_to_end():
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["demo"])
    def read_mode(mode: NullableMode) -> str:
        """Report the runtime type and member of a nullable-enum parameter."""
        return f"{type(mode).__name__}.{mode.name}"

    await g.sync()
    result = await g.execute(ToolCall(tool_name="read_mode", arguments={"mode": None}))
    assert result.status.value == "success", result.error
    assert result.result == "NullableMode.UNSET"
