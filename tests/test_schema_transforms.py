"""Tests for the provider schema transforms and per-dialect option threading.

Covers the two transforms in ``agent_gantry.adapters.tool_spec.schema_utils``
(OpenAI strict mode, Gemini/Vertex sanitization) and the path that carries
adapter options from the convenience APIs down to the adapter — which used to
drop them silently, making ``strict=True`` unreachable outside a manual
per-tool ``to_dialect`` call.
"""

from __future__ import annotations

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.tool_spec.providers import GeminiAdapter, OpenAIAdapter
from agent_gantry.adapters.tool_spec.schema_utils import (
    sanitize_gemini_schema,
    strict_json_schema,
)
from agent_gantry.schema.tool import ToolDefinition


class TestStrictJsonSchema:
    def test_all_properties_become_required(self) -> None:
        out = strict_json_schema(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                "required": ["a"],
            }
        )
        assert set(out["required"]) == {"a", "b"}
        assert out["additionalProperties"] is False

    def test_optional_property_admits_null(self) -> None:
        out = strict_json_schema(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                "required": ["a"],
            }
        )
        assert out["properties"]["a"]["type"] == "string"
        assert out["properties"]["b"]["type"] == ["integer", "null"]

    def test_nested_objects_are_transformed(self) -> None:
        out = strict_json_schema(
            {
                "type": "object",
                "properties": {
                    "addr": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}, "zip": {"type": "string"}},
                        "required": ["city"],
                    }
                },
                "required": ["addr"],
            }
        )
        nested = out["properties"]["addr"]
        assert nested["additionalProperties"] is False
        assert set(nested["required"]) == {"city", "zip"}
        assert nested["properties"]["zip"]["type"] == ["string", "null"]

    def test_array_item_schemas_are_transformed(self) -> None:
        out = strict_json_schema(
            {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"id": {"type": "string"}, "note": {"type": "string"}},
                            "required": ["id"],
                        },
                    }
                },
                "required": ["items"],
            }
        )
        item = out["properties"]["items"]["items"]
        assert item["additionalProperties"] is False
        assert set(item["required"]) == {"id", "note"}

    def test_existing_anyof_gains_a_null_branch_once(self) -> None:
        out = strict_json_schema(
            {
                "type": "object",
                "properties": {"v": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
                "required": [],
            }
        )
        branches = out["properties"]["v"]["anyOf"]
        assert {"type": "null"} in branches
        assert sum(b == {"type": "null"} for b in branches) == 1

    def test_already_nullable_type_list_is_left_alone(self) -> None:
        out = strict_json_schema(
            {
                "type": "object",
                "properties": {"v": {"type": ["string", "null"]}},
                "required": [],
            }
        )
        assert out["properties"]["v"]["type"] == ["string", "null"]

    def test_input_is_never_mutated(self) -> None:
        original = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": [],
        }
        snapshot = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": [],
        }
        strict_json_schema(original)
        assert original == snapshot

    def test_empty_schema_is_valid_for_strict(self) -> None:
        out = strict_json_schema({})
        assert out["additionalProperties"] is False
        assert out["required"] == []


class TestSanitizeGeminiSchema:
    def test_strips_keywords_gemini_rejects(self) -> None:
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "title": "Args",
                "additionalProperties": True,
                "properties": {"n": {"type": "integer", "default": 1, "exclusiveMinimum": 0}},
            }
        )
        assert "additionalProperties" not in out
        assert "title" not in out
        assert "default" not in out["properties"]["n"]
        assert "exclusiveMinimum" not in out["properties"]["n"]
        assert out["properties"]["n"]["type"] == "integer"

    def test_kwargs_tool_schema_survives(self) -> None:
        """Introspecting a ``**kwargs`` handler emits ``additionalProperties``.

        That alone broke the Gemini path for tools the library itself produces.
        """
        out = sanitize_gemini_schema(
            {"type": "object", "properties": {}, "additionalProperties": True}
        )
        assert "additionalProperties" not in out

    def test_local_refs_are_inlined(self) -> None:
        """Pydantic emits ``$defs``/``$ref`` for nested models; Gemini can't follow them."""
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {"addr": {"$ref": "#/$defs/Address"}},
                "required": ["addr"],
                "$defs": {
                    "Address": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    }
                },
            }
        )
        assert "$defs" not in out
        assert "$ref" not in out["properties"]["addr"]
        assert out["properties"]["addr"]["properties"]["city"]["type"] == "string"

    def test_keys_beside_a_ref_win_over_the_target(self) -> None:
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {
                    "addr": {"$ref": "#/$defs/Address", "description": "shipping address"}
                },
                "$defs": {"Address": {"type": "object", "description": "an address"}},
            }
        )
        assert out["properties"]["addr"]["description"] == "shipping address"

    def test_self_referential_schema_terminates(self) -> None:
        """A cyclic model must not hang or recurse without bound."""
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}},
                "$defs": {
                    "Node": {
                        "type": "object",
                        "properties": {"child": {"$ref": "#/$defs/Node"}},
                    }
                },
            }
        )
        assert out["type"] == "object"

    def test_const_becomes_a_single_value_enum(self) -> None:
        """``const`` is unsupported but is exactly a one-value ``enum``."""
        out = sanitize_gemini_schema(
            {"type": "object", "properties": {"mode": {"const": "fast"}}}
        )
        assert out["properties"]["mode"]["enum"] == ["fast"]
        assert "const" not in out["properties"]["mode"]

    def test_structural_keywords_are_preserved(self) -> None:
        """Dropping ``anyOf``/``enum`` would silently change what is accepted."""
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {
                    "v": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["v"],
            }
        )
        assert out["properties"]["v"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
        assert out["properties"]["unit"]["enum"] == ["c", "f"]
        assert out["required"] == ["v"]

    def test_input_is_never_mutated(self) -> None:
        original = {"type": "object", "additionalProperties": True, "properties": {}}
        sanitize_gemini_schema(original)
        assert original["additionalProperties"] is True

    def test_definitions_survive_when_a_ref_could_not_be_inlined(self) -> None:
        """Never leave a pointer to definitions that were just deleted.

        The depth guard deliberately stops inlining on a deeply nested or
        recursive model. Popping ``$defs`` regardless turned a schema the SDK
        rejects into one with a dangling ``$ref`` — strictly worse. Reported on
        PR #367.
        """
        # A self-referential model: inlining terminates with a $ref still in place.
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}},
                "$defs": {
                    "Node": {
                        "type": "object",
                        "properties": {"child": {"$ref": "#/$defs/Node"}},
                    }
                },
            }
        )

        import json

        rendered = json.dumps(out)
        if "$ref" in rendered:
            assert "$defs" in out, "dangling $ref: definitions were dropped underneath it"

    def test_definitions_are_dropped_once_fully_inlined(self) -> None:
        """The common case still sheds the now-unreferenced definitions."""
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {"addr": {"$ref": "#/$defs/Address"}},
                "$defs": {"Address": {"type": "object", "properties": {}}},
            }
        )

        assert "$defs" not in out
        assert "$ref" not in str(out)


class TestAdapterSchemasDoNotAliasTheRegistry:
    """Emitted schemas must not share structure with the canonical definition.

    The registry holds one ``ToolDefinition`` per tool, so a caller mutating an
    emitted schema would corrupt every later conversion of that tool.
    """

    @pytest.fixture
    def tool(self) -> ToolDefinition:
        return ToolDefinition(
            name="sample_tool",
            description="A sample tool used for aliasing checks.",
            parameters_schema={
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            },
        )

    def test_openai_strict_output_is_independent(self, tool: ToolDefinition) -> None:
        emitted = OpenAIAdapter().to_provider_schema(tool, strict=True)
        emitted["function"]["parameters"]["properties"]["a"]["type"] = "MUTATED"
        assert tool.parameters_schema["properties"]["a"]["type"] == "string"

    def test_gemini_output_is_independent(self, tool: ToolDefinition) -> None:
        emitted = GeminiAdapter().to_provider_schema(tool)
        emitted["parameters"]["properties"]["a"]["type"] = "MUTATED"
        assert tool.parameters_schema["properties"]["a"]["type"] == "string"


class TestDialectOptionThreading:
    """``strict`` must survive the trip from the convenience API to the adapter.

    ``retrieve_tools`` forwarded ``**kwargs`` into ``ToolQuery``, whose default
    ``extra="ignore"`` dropped anything that was not a query field, and then
    called ``to_dialect`` with no options at all — so ``strict=True`` vanished
    without an error.
    """

    @pytest.fixture
    async def gantry(self) -> AgentGantry:
        g = AgentGantry()

        @g.register(tags=["weather"])
        async def get_weather(city: str, unit: str = "celsius") -> str:
            """Get the weather for a city."""
            return "sunny"

        return g

    async def test_strict_reaches_the_openai_adapter(self, gantry: AgentGantry) -> None:
        tools = await gantry.retrieve_tools("weather", dialect="openai", strict=True)
        assert tools
        fn = tools[0]["function"]
        assert fn["strict"] is True
        assert fn["parameters"]["additionalProperties"] is False
        assert set(fn["parameters"]["required"]) == {"city", "unit"}

    async def test_explicit_dialect_options_are_forwarded(self, gantry: AgentGantry) -> None:
        tools = await gantry.retrieve_tools(
            "weather", dialect="openai", dialect_options={"strict": True}
        )
        assert tools[0]["function"]["strict"] is True

    async def test_omitting_strict_keeps_the_plain_schema(self, gantry: AgentGantry) -> None:
        tools = await gantry.retrieve_tools("weather", dialect="openai")
        assert "strict" not in tools[0]["function"]

    async def test_query_fields_still_configure_retrieval(self, gantry: AgentGantry) -> None:
        """A ToolQuery field must go to the query, not to the adapter."""
        tools = await gantry.retrieve_tools("weather", dialect="openai", namespaces=["nope"])
        assert tools == []

    async def test_llm_adapter_passes_strict_through(self, gantry: AgentGantry) -> None:
        from agent_gantry.openai import OpenAIAdapter as OpenAIRetrievalAdapter

        tools = await OpenAIRetrievalAdapter(gantry).tools("weather", strict=True)
        assert tools[0]["function"]["strict"] is True
