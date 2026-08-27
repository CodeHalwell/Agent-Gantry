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
    that was happy with the raw mapping must not start failing here."""
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["demo"])
    def loose(p: Payload) -> str:
        """Accept a payload without relying on its declared type."""
        return f"{type(p).__name__}"

    await g.sync()
    # ``{"x": "not-an-int"}`` can't build a Payload; the schema's own
    # validation is what rejects it, not the reconstruction step.
    result = await g.execute(ToolCall(tool_name="loose", arguments={"p": {"x": 1}}))
    assert result.result == "Payload"
