"""Tests for rebuilding JSON arguments into the types a handler declares.

The executor dispatches ``handler(**arguments)`` with JSON-decoded values.
Once ``build_parameters_schema`` advertises a nested model, a ``set`` or a
``datetime``, a provider sends the JSON form of that type — and forwarding it
unchanged handed the handler a ``dict``/``list``/``str`` where its annotation
promised the real thing. These tests pin both halves: the types that *are*
rebuilt, and the ones deliberately left alone.
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
import typing
import uuid
from typing import Any, TypedDict

import pytest
from pydantic import BaseModel

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.schema.execution import ToolCall
from agent_gantry.schema.introspection import build_argument_coercers
from agent_gantry.schema.tool import ToolDefinition


class Payload(BaseModel):
    x: int


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


async def test_unconvertible_value_falls_back_to_the_raw_argument():
    """Validation has already run against the canonical schema, so a handler
    that was happy with the raw mapping must not start failing here.

    Reaching the fallback needs a schema *looser* than the annotation — which
    is exactly what an imported (MCP/OpenAPI) or hand-written schema can be.
    Here the schema says "any object" while the handler says ``Payload``, so
    ``{"y": 2}`` passes validation and then fails reconstruction.
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

    # One the schema admits but ``Payload`` rejects falls back to the raw
    # mapping rather than turning a valid call into an error.
    fallback = await g.execute(ToolCall(tool_name="loose", arguments={"p": {"y": 2}}))
    assert fallback.status.value == "success", fallback.error
    assert fallback.result == "dict"


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
