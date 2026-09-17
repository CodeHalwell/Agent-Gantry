"""Regression tests for the tool-spec hardening audit.

Each finding below was reproduced against the real provider SDKs (openai,
anthropic, google-genai) before being fixed, so wherever an SDK ships a
validator for the shape in question — ``google.genai.types.FunctionDeclaration``
rejects unknown keys and mistyped values outright, and the OpenAI and
Anthropic SDKs carry the strict-mode transforms their parse helpers apply —
the tests run the fixed output through it rather than asserting on dict
shape alone. The SDKs are optional extras, so those tests skip without them.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import decimal
import json
import logging
import pathlib
import sys
import uuid
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from agent_gantry import AgentGantry
from agent_gantry.adapters.tool_spec.providers import (
    AgentFrameworkAdapter,
    AnthropicAdapter,
    GeminiAdapter,
    OpenAIAdapter,
    OpenAIResponsesAdapter,
)
from agent_gantry.adapters.tool_spec.round_trip import (
    StreamingToolCallAccumulator,
    extract_tool_calls,
)
from agent_gantry.adapters.tool_spec.schema_utils import (
    sanitize_gemini_schema,
    strict_json_schema,
    unsupported_strict_paths,
)
from agent_gantry.schema.introspection import build_parameters_schema
from agent_gantry.schema.tool import SchemaDialect, ToolDefinition

_PROVIDERS_LOGGER = "agent_gantry.adapters.tool_spec.providers"


def _tool(schema: dict[str, Any]) -> ToolDefinition:
    return ToolDefinition(
        name="probe", description="A tool with a schema under test.", parameters_schema=schema
    )


def _gemini_declaration(parameters: dict[str, Any]) -> Any:
    """Build the real SDK model, which rejects any key or value Gemini won't take."""
    genai = pytest.importorskip("google.genai")
    return genai.types.FunctionDeclaration(name="t", parameters=parameters)


def _openai_sdk_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """What the OpenAI SDK itself publishes for ``schema`` under strict mode."""
    pytest.importorskip("openai")
    from openai.lib._pydantic import _ensure_strict_json_schema

    # The SDK mutates in place and resolves pointers against the same object.
    root = copy.deepcopy(schema)
    return _ensure_strict_json_schema(root, path=(), root=root)


def _anthropic_sdk_transform(schema: dict[str, Any]) -> dict[str, Any]:
    """Run the real SDK's strict transform, around the session-wide mock.

    ``tests/test_anthropic_features.py`` and ``tests/test_anthropic_skills.py``
    assign a ``Mock`` to ``sys.modules["anthropic"]`` at module scope and never
    restore it, so by the time this runs in a full session the name resolves to
    a non-package and ``anthropic.lib`` cannot be imported. Rather than skip —
    which would silently retire this check in CI, where the whole suite always
    runs — the mocked entries are lifted out for the duration of the import and
    put back afterwards, so the guard stays live and the other modules keep the
    stub they installed.
    """
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "anthropic" or name.startswith("anthropic.")
    }
    for name in saved:
        del sys.modules[name]
    try:
        try:
            from anthropic.lib._parse._transform import transform_schema
        except ImportError:
            pytest.skip("anthropic SDK is not installed")
        return transform_schema(copy.deepcopy(schema))
    finally:
        # Drop whatever the real import added, then restore the stub so the
        # modules that installed it are unaffected by this test having run.
        for name in [
            n for n in sys.modules if n == "anthropic" or n.startswith("anthropic.")
        ]:
            del sys.modules[name]
        sys.modules.update(saved)


def _closed_object_paths(node: Any, path: str = "") -> set[str]:
    """Every location in ``node`` carrying ``additionalProperties: false``."""
    found: set[str] = set()
    if isinstance(node, list):
        for index, item in enumerate(node):
            found |= _closed_object_paths(item, f"{path}[{index}]")
    elif isinstance(node, dict):
        if node.get("additionalProperties") is False:
            found.add(path or "<root>")
        for key, value in node.items():
            found |= _closed_object_paths(value, f"{path}.{key}" if path else key)
    return found


# --------------------------------------------------------------------------- #
# 1. Gemini: ``type`` lists
# --------------------------------------------------------------------------- #


class TestGeminiTypeLists:
    def test_required_optional_str_becomes_scalar_type_plus_nullable(self) -> None:
        """A required ``str | None`` publishes ``type: ["string", "null"]``;
        ``Schema.type`` is a single enum, so the SDK rejected the tool."""

        def handler(x: str | None) -> None: ...

        emitted = build_parameters_schema(handler)
        assert emitted["properties"]["x"]["type"] == ["string", "null"]

        out = sanitize_gemini_schema(emitted)
        assert out["properties"]["x"] == {"type": "string", "nullable": True}
        _gemini_declaration(out)

    def test_several_remaining_types_become_anyof_branches(self) -> None:
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {
                    "x": {
                        "type": ["string", "integer", "null"],
                        "minLength": 1,
                        "description": "either",
                    }
                },
            }
        )
        x = out["properties"]["x"]
        assert x["anyOf"] == [{"type": "string"}, {"type": "integer"}]
        assert x["nullable"] is True
        assert "type" not in x
        # The node's other keywords stay on the parent, where the SDK applies them.
        assert x["minLength"] == 1
        assert x["description"] == "either"
        _gemini_declaration(out)

    def test_a_list_without_null_is_not_made_nullable(self) -> None:
        out = sanitize_gemini_schema(
            {"type": "object", "properties": {"x": {"type": ["string", "integer"]}}}
        )
        assert "nullable" not in out["properties"]["x"]
        assert out["properties"]["x"]["anyOf"] == [{"type": "string"}, {"type": "integer"}]
        _gemini_declaration(out)

    def test_null_only_list_uses_the_scalar_null_type(self) -> None:
        out = sanitize_gemini_schema({"type": "object", "properties": {"x": {"type": ["null"]}}})
        assert out["properties"]["x"] == {"type": "null"}
        _gemini_declaration(out)

    def test_nested_lists_are_rewritten_too(self) -> None:
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": ["string", "null"]}},
                    "addr": {
                        "type": "object",
                        "properties": {"zip": {"type": ["integer", "null"]}},
                    },
                },
            }
        )
        assert out["properties"]["tags"]["items"] == {"type": "string", "nullable": True}
        assert out["properties"]["addr"]["properties"]["zip"] == {
            "type": "integer",
            "nullable": True,
        }
        _gemini_declaration(out)


# --------------------------------------------------------------------------- #
# 2. Gemini: non-string ``enum`` members
# --------------------------------------------------------------------------- #


class TestGeminiEnums:
    def test_non_string_members_keep_their_type_and_lose_the_enum(self) -> None:
        """``Literal[1, 2]`` publishes ``{"type": "integer", "enum": [1, 2]}``;
        ``Schema.enum`` is ``list[str]``, so the SDK rejects the members.

        Re-spelling them as strings satisfies the SDK and breaks the round
        trip: the model answers ``"1"``, and the executor validates that
        against the *canonical* integer schema and rejects every call. The
        constraint Gemini cannot carry is dropped instead, the canonical type
        stays so the model returns the right JSON kind, and the permitted
        values move into the description.
        """

        def handler(level: Literal[1, 2]) -> None: ...

        emitted = build_parameters_schema(handler)
        assert emitted["properties"]["level"] == {"type": "integer", "enum": [1, 2]}

        out = sanitize_gemini_schema(emitted)
        level = out["properties"]["level"]
        assert level["type"] == "integer"
        assert "enum" not in level
        assert level["description"] == "Allowed values: 1, 2."
        _gemini_declaration(out)

    @pytest.mark.asyncio
    async def test_the_emitted_shape_round_trips_through_execute(self) -> None:
        """The point of the rewrite: what Gemini can answer must be a value
        the executor accepts, and the enum must still be enforced."""
        gantry = AgentGantry()

        @gantry.register
        def set_level(level: Literal[1, 2]) -> str:
            """Set the level to one of the allowed values."""
            return f"level={level}"

        await gantry.sync()
        declarations = await gantry.retrieve_tools("set the level", limit=1, dialect="gemini")
        parameters = declarations[0]["parameters"]
        assert parameters["properties"]["level"] == {
            "type": "integer",
            "description": "Allowed values: 1, 2.",
        }
        _gemini_declaration(parameters)

        from agent_gantry.schema.execution import ExecutionStatus, ToolCall

        ok = await gantry.execute(ToolCall(tool_name="set_level", arguments={"level": 1}))
        assert ok.status is ExecutionStatus.SUCCESS, ok.error
        assert ok.result == "level=1"

        # The canonical schema still carries the enum, so a value outside it
        # is rejected even though Gemini was not told to constrain itself.
        bad = await gantry.execute(ToolCall(tool_name="set_level", arguments={"level": 3}))
        assert bad.status is ExecutionStatus.FAILURE
        assert "one of [1, 2]" in (bad.error or "")
        await gantry.close()

    def test_const_conversion_drops_a_non_string_enum_too(self) -> None:
        """The sanitizer's own ``const`` → ``enum`` rewrite produced a
        non-string member for a non-string constant."""
        out = sanitize_gemini_schema(
            {"type": "object", "properties": {"x": {"type": "integer", "const": 3}}}
        )
        assert out["properties"]["x"]["type"] == "integer"
        assert "enum" not in out["properties"]["x"]
        assert "const" not in out["properties"]["x"]
        assert out["properties"]["x"]["description"] == "Allowed values: 3."
        _gemini_declaration(out)

    def test_values_are_named_in_their_json_spelling(self) -> None:
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {
                    "flag": {"type": "boolean", "enum": [True]},
                    "ratio": {"type": "number", "enum": [0.5, 1]},
                    "mixed": {"enum": [1, "auto"]},
                },
            }
        )
        properties = out["properties"]
        assert properties["flag"] == {
            "type": "boolean",
            "description": "Allowed values: true.",
        }
        assert properties["ratio"] == {
            "type": "number",
            "description": "Allowed values: 0.5, 1.",
        }
        # A mixed-kind enum has no single canonical type to keep; the SDK
        # accepts a property that declares none.
        assert properties["mixed"] == {"description": 'Allowed values: 1, "auto".'}
        _gemini_declaration(out)

    def test_existing_description_is_kept_ahead_of_the_hint(self) -> None:
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {"x": {"type": "integer", "enum": [1], "description": "Pick one."}},
            }
        )
        assert out["properties"]["x"]["description"] == "Pick one. Allowed values: 1."

    def test_string_enum_is_left_alone(self) -> None:
        out = sanitize_gemini_schema(
            {"type": "object", "properties": {"x": {"type": "string", "enum": ["a", "b"]}}}
        )
        assert out["properties"]["x"] == {"type": "string", "enum": ["a", "b"]}
        _gemini_declaration(out)

    def test_null_member_becomes_nullable_not_the_string_null(self) -> None:
        """``Literal["a", None]`` publishes ``{"enum": ["a", null]}``: null is a
        member the SDK rejects as a non-string, and Gemini spells it
        ``nullable`` rather than as a value the model would emit verbatim."""

        def handler(mode: Literal["a", None]) -> None: ...

        emitted = build_parameters_schema(handler)
        assert emitted["properties"]["mode"] == {"enum": ["a", None]}

        out = sanitize_gemini_schema(emitted)
        assert out["properties"]["mode"] == {"enum": ["a"], "nullable": True}
        _gemini_declaration(out)

    def test_nullable_integer_enum_combines_both_rewrites(self) -> None:
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {"x": {"type": ["integer", "null"], "enum": [1, 2, None]}},
            }
        )
        x = out["properties"]["x"]
        # the type list collapses to the non-null type plus ``nullable``...
        assert x["type"] == "integer"
        assert x["nullable"] is True
        # ...and the non-string members leave the enum for the description
        assert "enum" not in x
        assert x["description"] == "Allowed values: 1, 2."
        _gemini_declaration(out)


# --------------------------------------------------------------------------- #
# 3. Gemini: ``prefixItems``
# --------------------------------------------------------------------------- #


class TestGeminiPrefixItems:
    def test_tuple_parameter_folds_into_an_items_union(self) -> None:
        """``tuple[int, str]`` publishes ``prefixItems``, which the SDK forbids
        as an unknown key; it was neither stripped nor converted."""

        def handler(pair: tuple[int, str]) -> None: ...

        emitted = build_parameters_schema(handler)
        assert emitted["properties"]["pair"]["prefixItems"] == [
            {"type": "integer"},
            {"type": "string"},
        ]

        out = sanitize_gemini_schema(emitted)
        pair = out["properties"]["pair"]
        assert "prefixItems" not in pair
        assert pair["items"] == {"anyOf": [{"type": "integer"}, {"type": "string"}]}
        assert pair["minItems"] == 2
        assert pair["maxItems"] == 2
        _gemini_declaration(out)

    def test_a_single_prefix_schema_needs_no_union(self) -> None:
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {"one": {"type": "array", "prefixItems": [{"type": "integer"}]}},
            }
        )
        assert out["properties"]["one"]["items"] == {"type": "integer"}
        _gemini_declaration(out)

    def test_an_existing_items_schema_joins_the_union(self) -> None:
        """``items`` beside ``prefixItems`` types the *rest* elements; Gemini
        cannot say that, so it becomes one more allowed kind rather than
        silently replacing the prefix kinds."""
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "prefixItems": [{"type": "integer"}],
                        "items": {"type": "boolean"},
                    }
                },
            }
        )
        assert out["properties"]["x"]["items"] == {
            "anyOf": [{"type": "integer"}, {"type": "boolean"}]
        }
        _gemini_declaration(out)

    def test_a_boolean_items_beside_the_prefix_is_dropped(self) -> None:
        """``items: false`` (no elements past the prefix) is not a schema the
        SDK can hold and is redundant beside ``maxItems``."""
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "prefixItems": [{"type": "integer"}, {"type": "string"}],
                        "items": False,
                        "maxItems": 2,
                    }
                },
            }
        )
        assert out["properties"]["x"]["items"] == {
            "anyOf": [{"type": "integer"}, {"type": "string"}]
        }
        _gemini_declaration(out)

    def test_prefix_schemas_are_sanitized_after_folding(self) -> None:
        """The folded branches go through the same pass as anything else under
        ``items``, so a keyword Gemini rejects inside a prefix schema is gone."""
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "prefixItems": [
                            {"type": "integer", "default": 1},
                            {"type": ["string", "null"]},
                        ],
                    }
                },
            }
        )
        assert out["properties"]["x"]["items"] == {
            "anyOf": [{"type": "integer"}, {"type": "string", "nullable": True}]
        }
        _gemini_declaration(out)


def test_a_tool_using_all_three_gemini_shapes_is_accepted_by_the_sdk() -> None:
    """End to end through the adapter: what ``AgentGantry`` actually emits."""

    def handler(
        name: str | None,
        level: Literal[1, 2],
        pair: tuple[int, str],
    ) -> None:
        """Exercise every shape the audit found the SDK rejecting."""

    tool = _tool(build_parameters_schema(handler))
    declaration = GeminiAdapter().to_provider_schema(tool)
    genai = pytest.importorskip("google.genai")
    built = genai.types.FunctionDeclaration(**declaration)
    assert set(built.parameters.properties) == {"name", "level", "pair"}


# --------------------------------------------------------------------------- #
# 4. Tool results that are not JSON-native
# --------------------------------------------------------------------------- #


class _Point(BaseModel):
    x: int
    when: datetime.datetime


@dataclasses.dataclass
class _Reading:
    value: decimal.Decimal
    taken: datetime.date


class _Opaque:
    def __str__(self) -> str:
        return "opaque!"


_TEXT_ADAPTERS = [OpenAIAdapter(), OpenAIResponsesAdapter(), AnthropicAdapter()]
_STAMP = datetime.datetime(2024, 1, 2, 3, 4, 5)
_RESULT_CASES = [
    (_STAMP, "2024-01-02T03:04:05"),
    (datetime.date(2024, 1, 2), "2024-01-02"),
    (datetime.time(3, 4, 5), "03:04:05"),
    (decimal.Decimal("1.50"), "1.50"),
    (uuid.UUID("12345678-1234-5678-1234-567812345678"), "12345678-1234-5678-1234-567812345678"),
    (pathlib.PurePosixPath("/tmp/out.txt"), "/tmp/out.txt"),
    ({"b", "a"}, ["a", "b"]),
    (b"bytes here", "bytes here"),
    (_Reading(decimal.Decimal("2.5"), datetime.date(2024, 1, 2)), {"value": "2.5", "taken": "2024-01-02"}),
    (_Point(x=1, when=_STAMP), {"x": 1, "when": "2024-01-02T03:04:05"}),
    ({"nested": [_STAMP, {"deep": decimal.Decimal(3)}]}, {"nested": ["2024-01-02T03:04:05", {"deep": "3"}]}),
    (_Opaque(), "opaque!"),
]


def _text_of(formatted: dict[str, Any]) -> str:
    return formatted["content"] if "content" in formatted else formatted["output"]


class TestToolResultSerialization:
    @pytest.mark.parametrize("adapter", _TEXT_ADAPTERS, ids=lambda a: a.dialect_name)
    @pytest.mark.parametrize(("result", "expected"), _RESULT_CASES, ids=repr)
    def test_text_adapters_serialize_common_return_types(
        self, adapter: Any, result: Any, expected: Any
    ) -> None:
        """A bare ``json.dumps`` raised ``TypeError`` for every one of these,
        which ``AgentGantry.execute_tool_calls`` surfaced as a crash rather
        than a result the model could read."""
        formatted = adapter.format_tool_result("probe", result, "call_1")
        decoded = json.loads(_text_of(formatted))
        if isinstance(result, set):
            decoded = sorted(decoded)
        assert decoded == expected

    @pytest.mark.parametrize(("result", "expected"), _RESULT_CASES, ids=repr)
    def test_gemini_response_is_json_native(self, result: Any, expected: Any) -> None:
        """Gemini's response is a structured object the SDK serializes itself,
        so the conversion has to happen before wrapping, not at dump time."""
        response = GeminiAdapter().format_tool_result("probe", result, "call_1")["functionResponse"][
            "response"
        ]
        # JSON-native throughout: a plain dump with no fallback must succeed.
        json.dumps(response)
        if isinstance(expected, dict):
            # A model or dataclass result is an object, and is used as the
            # response directly — exactly as a dict result always was.
            assert response == expected
        else:
            value = response["result"]
            assert (sorted(value) if isinstance(result, set) else value) == expected

    def test_plain_strings_and_dicts_are_unchanged(self) -> None:
        assert OpenAIAdapter().format_tool_result("probe", "done", "c")["content"] == "done"
        assert json.loads(AnthropicAdapter().format_tool_result("probe", {"k": 1})["content"]) == {
            "k": 1
        }
        assert GeminiAdapter().format_tool_result("probe", {"k": 1})["functionResponse"][
            "response"
        ] == {"k": 1}

    @pytest.mark.asyncio
    async def test_execute_tool_calls_survives_a_datetime_result(self, gantry: AgentGantry) -> None:
        @gantry.register(tags=["time"])
        def when() -> datetime.datetime:
            """Return a fixed timestamp."""
            return _STAMP

        await gantry.sync()

        openai_response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "call_1", "function": {"name": "when", "arguments": "{}"}}
                        ]
                    }
                }
            ]
        }
        [reply] = await gantry.execute_tool_calls(openai_response, dialect="openai")
        assert json.loads(reply["content"]) == "2024-01-02T03:04:05"

        gemini_response = {
            "candidates": [{"content": {"parts": [{"functionCall": {"name": "when", "args": {}}}]}}]
        }
        [reply] = await gantry.execute_tool_calls(gemini_response, dialect="gemini")
        assert reply["functionResponse"]["response"] == {"result": "2024-01-02T03:04:05"}


# --------------------------------------------------------------------------- #
# 5. OpenAI strict mode: ``$ref`` with sibling keys
# --------------------------------------------------------------------------- #


def _address_schema(**field_extras: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"addr": {"$ref": "#/$defs/Address", **field_extras}},
        "required": ["addr"],
        "$defs": {
            "Address": {
                "type": "object",
                "description": "an address",
                "properties": {"city": {"type": "string"}, "zip": {"type": "string"}},
                "required": ["city", "zip"],
            }
        },
    }


class TestStrictDecoratedRefs:
    def test_a_ref_with_a_description_is_inlined(self) -> None:
        """``{"$ref": ..., "description": ...}`` — what Pydantic emits for a
        documented nested-model field — is rejected by OpenAI under strict
        mode. It used to pass through untouched, with nothing reported."""
        schema = _address_schema(description="shipping")
        assert unsupported_strict_paths(schema) == []

        addr = strict_json_schema(schema)["properties"]["addr"]
        assert "$ref" not in addr
        assert addr["type"] == "object"
        assert addr["additionalProperties"] is False
        assert set(addr["required"]) == {"city", "zip"}
        # The sibling wins over the target's own annotation, as in the SDK.
        assert addr["description"] == "shipping"

    def test_the_result_agrees_with_the_openai_sdk(self) -> None:
        """The OpenAI SDK's ``_ensure_strict_json_schema`` unravels exactly
        these nodes; every nested property here is already required, so the
        two transforms have nothing else to disagree about."""
        schema = _address_schema(description="shipping")
        assert strict_json_schema(schema) == _openai_sdk_strict(schema)

    def test_a_bare_ref_stays_a_pointer(self) -> None:
        """Pointers without siblings are accepted as-is, and inlining them
        would only bloat the schema — the SDK leaves them alone too."""
        schema = _address_schema()
        out = strict_json_schema(schema)
        assert out["properties"]["addr"] == {"$ref": "#/$defs/Address"}
        assert out == _openai_sdk_strict(schema)

    def test_an_optional_decorated_ref_is_inlined_then_widened(self) -> None:
        schema = _address_schema(description="shipping")
        schema["required"] = []
        addr = strict_json_schema(schema)["properties"]["addr"]
        assert "$ref" not in addr
        assert addr["type"] == ["object", "null"]

    def test_inlined_content_gets_the_strict_rules(self) -> None:
        """An optional property *inside* the target is widened after inlining
        just as it would be anywhere else."""
        schema = _address_schema(description="shipping")
        schema["$defs"]["Address"]["required"] = ["city"]
        addr = strict_json_schema(schema)["properties"]["addr"]
        assert addr["properties"]["zip"]["type"] == ["string", "null"]
        assert set(addr["required"]) == {"city", "zip"}

    def test_an_unresolvable_pointer_is_left_alone(self) -> None:
        schema = {
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/Missing", "description": "d"}},
            "required": ["x"],
        }
        out = strict_json_schema(schema)
        assert out["properties"]["x"] == {"$ref": "#/$defs/Missing", "description": "d"}

    def test_a_self_referential_decorated_ref_terminates(self) -> None:
        """A recursive model whose recursive field carries a description has
        no finite inlined spelling; the depth guard must stop it rather than
        recurse without bound."""
        node = {
            "type": "object",
            "properties": {"child": {"$ref": "#/$defs/Node", "description": "kid"}},
            "required": ["child"],
        }
        schema = {**copy.deepcopy(node), "$defs": {"Node": copy.deepcopy(node)}}
        out = strict_json_schema(schema)
        assert out["properties"]["child"]["type"] == "object"

    def test_the_canonical_schema_is_not_mutated(self) -> None:
        schema = _address_schema(description="shipping")
        before = copy.deepcopy(schema)
        strict_json_schema(schema)
        assert schema == before


# --------------------------------------------------------------------------- #
# 6. Anthropic strict mode: ``additionalProperties`` on every object
# --------------------------------------------------------------------------- #

_NESTED_OBJECTS = {
    "type": "object",
    "properties": {
        "addr": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        "tags": {
            "type": "array",
            "items": {"type": "object", "properties": {"k": {"type": "string"}}},
        },
        "either": {
            "anyOf": [
                {"type": "object", "properties": {"a": {"type": "integer"}}},
                {"type": "string"},
            ]
        },
        "linked": {"$ref": "#/$defs/Link"},
    },
    "required": ["addr"],
    "$defs": {"Link": {"type": "object", "properties": {"url": {"type": "string"}}}},
}


class TestAnthropicStrictClosesEveryObject:
    def test_every_object_is_closed(self) -> None:
        """Only the root got ``additionalProperties: false``; Anthropic requires
        it on every object and rejected the tool."""
        out = AnthropicAdapter().to_provider_schema(_tool(_NESTED_OBJECTS), strict=True)
        assert out["strict"] is True
        assert _closed_object_paths(out["input_schema"]) == {
            "<root>",
            "properties.addr",
            "properties.tags.items",
            "properties.either.anyOf[0]",
            "$defs.Link",
        }

    def test_closed_paths_agree_with_the_anthropic_sdk(self) -> None:
        """``transform_schema`` is what the SDK's parse helpers apply; the set
        of objects it closes is the set this adapter must close."""
        out = AnthropicAdapter().to_provider_schema(_tool(_NESTED_OBJECTS), strict=True)
        assert _closed_object_paths(out["input_schema"]) == _closed_object_paths(
            _anthropic_sdk_transform(_NESTED_OBJECTS)
        )

    def test_optional_properties_stay_optional(self) -> None:
        """Anthropic keeps optionality, so OpenAI's transform — every property
        required, the rest widened to null — would be the wrong one here."""
        out = AnthropicAdapter().to_provider_schema(_tool(_NESTED_OBJECTS), strict=True)
        schema = out["input_schema"]
        assert schema["required"] == ["addr"]
        assert "required" not in schema["properties"]["tags"]["items"]
        assert schema["properties"]["tags"]["type"] == "array"
        assert schema["properties"]["linked"] == {"$ref": "#/$defs/Link"}

    def test_non_strict_emission_is_untouched(self) -> None:
        out = AnthropicAdapter().to_provider_schema(_tool(_NESTED_OBJECTS))
        assert _closed_object_paths(out["input_schema"]) == set()
        assert out["input_schema"] == _NESTED_OBJECTS

    def test_the_canonical_schema_is_not_mutated(self) -> None:
        tool = _tool(copy.deepcopy(_NESTED_OBJECTS))
        AnthropicAdapter().to_provider_schema(tool, strict=True)
        assert tool.parameters_schema == _NESTED_OBJECTS


# --------------------------------------------------------------------------- #
# 7. OpenAI-family payloads whose arguments decode to a non-object
# --------------------------------------------------------------------------- #


def _openai_family_payload(adapter: Any, arguments: Any) -> dict[str, Any]:
    if isinstance(adapter, OpenAIResponsesAdapter):
        return {"type": "function_call", "call_id": "c1", "name": "probe", "arguments": arguments}
    if isinstance(adapter, AgentFrameworkAdapter):
        return {"id": "c1", "name": "probe", "arguments": arguments}  # AF's simplified shape
    return {"id": "c1", "type": "function", "function": {"name": "probe", "arguments": arguments}}


_OPENAI_FAMILY = [OpenAIAdapter(), OpenAIResponsesAdapter(), AgentFrameworkAdapter()]


class TestNonObjectArguments:
    @pytest.mark.parametrize("adapter", _OPENAI_FAMILY, ids=lambda a: a.dialect_name)
    @pytest.mark.parametrize("raw", ["null", "[]", "1", '"text"', "[{\"a\": 1}]"])
    def test_valid_json_that_is_not_an_object_becomes_empty_arguments(
        self, adapter: Any, raw: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """These decode without a ``JSONDecodeError`` and then failed
        ``ToolCallPayload(arguments: dict)`` with a ``ValidationError``."""
        with caplog.at_level(logging.WARNING, logger=_PROVIDERS_LOGGER):
            payload = adapter.from_provider_payload(_openai_family_payload(adapter, raw))
        assert payload.arguments == {}
        assert payload.tool_name == "probe"
        assert payload.tool_call_id == "c1"
        assert any("rather than an object" in record.message for record in caplog.records)

    @pytest.mark.parametrize("adapter", _OPENAI_FAMILY, ids=lambda a: a.dialect_name)
    def test_an_already_parsed_non_object_is_dropped_too(self, adapter: Any) -> None:
        """An SDK object can carry ``arguments=None``; the guard sits after the
        decode step, so it covers pre-parsed values as well."""
        payload = adapter.from_provider_payload(_openai_family_payload(adapter, None))
        assert payload.arguments == {}

    @pytest.mark.parametrize("adapter", _OPENAI_FAMILY, ids=lambda a: a.dialect_name)
    def test_object_arguments_still_pass_through(
        self, adapter: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_PROVIDERS_LOGGER):
            payload = adapter.from_provider_payload(
                _openai_family_payload(adapter, '{"city": "Oslo"}')
            )
        assert payload.arguments == {"city": "Oslo"}
        assert not caplog.records

    def test_extract_tool_calls_survives_a_null_arguments_string(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "probe", "arguments": "null"}},
                            {"id": "c2", "function": {"name": "probe", "arguments": '{"x": 1}'}},
                        ]
                    }
                }
            ]
        }
        calls = extract_tool_calls(response, "openai")
        assert [c.arguments for c in calls] == [{}, {"x": 1}]


# --------------------------------------------------------------------------- #
# 8. Typeless properties under strict mode
# --------------------------------------------------------------------------- #


def _wrap(prop: Any) -> dict[str, Any]:
    return {"type": "object", "properties": {"x": prop}, "required": ["x"]}


class TestTypelessProperties:
    @pytest.mark.parametrize(
        "prop",
        [
            {"description": "anything"},
            {},
            {"enum": [1, "auto"]},
            {"default": 3, "title": "X"},
            {"properties": {"a": {"type": "string"}}},
        ],
        ids=["description-only", "empty", "typeless-enum", "annotations-only", "untyped-object"],
    )
    def test_a_property_declaring_no_type_is_reported(self, prop: dict[str, Any]) -> None:
        """Only the typeless *enum* used to be caught, so ``{"description":
        "anything"}`` went out ``strict: true`` with no ``type`` at all."""
        assert unsupported_strict_paths(_wrap(prop)) == ["x"]

    @pytest.mark.parametrize(
        "prop",
        [
            {"type": "string"},
            {"type": ["string", "null"]},
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
            {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            {"allOf": [{"type": "string"}]},
            {"$ref": "#/$defs/Thing"},
        ],
        ids=["type", "type-list", "anyOf", "oneOf", "allOf", "ref"],
    )
    def test_any_way_of_declaring_a_type_is_enough(self, prop: dict[str, Any]) -> None:
        schema = _wrap(prop)
        # The ``$ref`` case needs a target. Give it a declared property: an
        # object with none at all is a free-form map, which strict mode cannot
        # express and the walker reports for its own (unrelated) reason.
        schema["$defs"] = {
            "Thing": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
        }
        assert unsupported_strict_paths(schema) == []

    def test_nested_typeless_properties_are_reported_at_their_path(self) -> None:
        schema = _wrap({"type": "object", "properties": {"inner": {"description": "?"}}})
        assert unsupported_strict_paths(schema) == ["x.inner"]

    def test_a_typeless_enum_under_items_is_still_reported(self) -> None:
        """The node-level enum check still covers shapes not reached through
        ``properties``, such as the items of a ``list[Literal[1, "auto"]]``."""
        schema = _wrap({"type": "array", "items": {"enum": [1, "auto"]}})
        assert unsupported_strict_paths(schema) == ["x.items"]

    @pytest.mark.parametrize(
        "adapter", [OpenAIAdapter(), OpenAIResponsesAdapter()], ids=lambda a: a.dialect_name
    )
    def test_openai_adapters_fall_back_to_non_strict(self, adapter: Any) -> None:
        tool = _tool(_wrap({"description": "anything"}))
        out = adapter.to_provider_schema(tool, strict=True)
        function = out["function"] if "function" in out else out
        assert "strict" not in function
        assert function["parameters"] == tool.parameters_schema

    def test_anthropic_adapter_falls_back_to_non_strict(self) -> None:
        tool = _tool(_wrap({"description": "anything"}))
        out = AnthropicAdapter().to_provider_schema(tool, strict=True)
        assert "strict" not in out
        assert out["input_schema"] == tool.parameters_schema


# --------------------------------------------------------------------------- #
# Minor: ``dialect="auto"``
# --------------------------------------------------------------------------- #


class TestAutoDialect:
    def test_extract_tool_calls_accepts_auto(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "probe", "arguments": '{"a": 1}'}}
                        ]
                    }
                }
            ]
        }
        for dialect in ("auto", SchemaDialect.AUTO):
            [call] = extract_tool_calls(response, dialect=dialect)
            assert call.tool_name == "probe"
            assert call.arguments == {"a": 1}

    def test_streaming_accumulator_accepts_auto(self) -> None:
        acc = StreamingToolCallAccumulator(dialect="auto")
        acc.add_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "c1", "function": {"name": "probe"}},
                            ]
                        }
                    }
                ]
            }
        )
        acc.add_chunk(
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}}]}
        )
        [call] = acc.tool_calls()
        assert call.tool_name == "probe"

    def test_unknown_dialects_still_raise(self) -> None:
        with pytest.raises(ValueError, match="No known response shape"):
            extract_tool_calls({}, dialect="nope")
        with pytest.raises(ValueError, match="not implemented"):
            StreamingToolCallAccumulator(dialect="nope")
