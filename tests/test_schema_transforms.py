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
from agent_gantry.adapters.tool_spec.providers import (
    AnthropicAdapter,
    GeminiAdapter,
    OpenAIAdapter,
    OpenAIResponsesAdapter,
)
from agent_gantry.adapters.tool_spec.schema_utils import (
    sanitize_gemini_schema,
    strict_json_schema,
    unsupported_strict_paths,
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


class TestStrictNullableConst:
    """``const`` is an independent constraint no ``type`` widening satisfies
    (PR #381 review)."""

    def test_optional_const_gains_a_null_branch(self) -> None:
        out = strict_json_schema(
            {
                "type": "object",
                "properties": {"mode": {"type": "string", "const": "fixed"}},
                "required": [],
            }
        )
        mode = out["properties"]["mode"]
        # Widening ``type`` alone would leave ``const`` forbidding null, and
        # strict mode makes the property required — so the model could not
        # express omission at all.
        assert mode["anyOf"] == [{"type": "string", "const": "fixed"}, {"type": "null"}]

    def test_required_const_is_left_alone(self) -> None:
        out = strict_json_schema(
            {
                "type": "object",
                "properties": {"mode": {"type": "string", "const": "fixed"}},
                "required": ["mode"],
            }
        )
        assert out["properties"]["mode"] == {"type": "string", "const": "fixed"}


class TestUnsupportedStrictPaths:
    """Objects with arbitrary keys have no strict-mode representation, so
    they must be detected rather than silently mangled (PR #381 review)."""

    def test_typed_mapping_is_unsupported(self) -> None:
        # dict[str, int] as build_parameters_schema emits it.
        assert unsupported_strict_paths(
            {
                "type": "object",
                "properties": {
                    "counts": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    }
                },
                "required": ["counts"],
            }
        ) == ["counts"]

    def test_untyped_dict_is_unsupported(self) -> None:
        # dict[str, Any] → a bare {"type": "object"}; strict mode needs the
        # key set enumerated, which this does not provide.
        assert unsupported_strict_paths(
            {
                "type": "object",
                "properties": {"meta": {"type": "object"}},
                "required": ["meta"],
            }
        ) == ["meta"]

    def test_fully_declared_schema_is_supported(self) -> None:
        assert (
            unsupported_strict_paths(
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "addr": {
                            "type": "object",
                            "properties": {"street": {"type": "string"}},
                            "required": ["street"],
                        },
                    },
                    "required": ["name", "addr"],
                }
            )
            == []
        )

    def test_no_argument_tool_is_supported(self) -> None:
        # properties: {} is the "takes no arguments" shape strict_json_schema
        # itself emits — not an open map.
        assert unsupported_strict_paths({"type": "object", "properties": {}}) == []
        assert unsupported_strict_paths({}) == []

    def test_kwargs_style_additional_properties_is_supported(self) -> None:
        # Real properties plus additionalProperties: true (a **kwargs
        # handler). Strict mode narrows it by forcing false, which is a
        # documented narrowing rather than an inexpressible shape.
        assert (
            unsupported_strict_paths(
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"],
                    "additionalProperties": True,
                }
            )
            == []
        )

    def test_typed_additional_properties_alongside_declared_properties(self) -> None:
        """Declared properties *and* a typed ``additionalProperties`` still
        describes keys strict mode cannot express — forcing it to false there
        silently drops the typed extras (PR #381 review)."""
        assert unsupported_strict_paths(
            {
                "type": "object",
                "properties": {
                    "outer": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "additionalProperties": {"type": "integer"},
                    }
                },
                "required": ["outer"],
            }
        ) == ["outer"]

    def test_empty_additional_properties_alongside_properties_is_supported(self) -> None:
        # ``{}`` is spec-equivalent to ``true``, so it is the same **kwargs
        # narrowing strict mode legitimately applies.
        assert (
            unsupported_strict_paths(
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"],
                    "additionalProperties": {},
                }
            )
            == []
        )

    def test_nested_and_array_item_mappings_are_found(self) -> None:
        found = unsupported_strict_paths(
            {
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "object",
                        "properties": {
                            "tags": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": {"type": "string"},
                                },
                            }
                        },
                        "required": ["tags"],
                    }
                },
                "required": ["payload"],
            }
        )
        assert found == ["payload.tags.items"]

    def test_strict_transform_leaves_open_maps_alone(self) -> None:
        # Forcing additionalProperties: false here would produce an object
        # accepting no keys at all, silently discarding the parameter.
        out = strict_json_schema(
            {
                "type": "object",
                "properties": {
                    "counts": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    }
                },
                "required": ["counts"],
            }
        )
        assert out["properties"]["counts"]["additionalProperties"] == {"type": "integer"}

    def test_pre_widened_nullable_enum_still_admits_null(self) -> None:
        """A schema that already lists ``null`` in its ``type`` is not
        necessarily nullable: ``enum`` is an independent constraint, and some
        emitters produce an optional ``Literal`` pre-widened. Returning early
        there left strict mode making the property required with no value that
        satisfies both constraints (PR #381 review)."""
        out = strict_json_schema(
            {
                "type": "object",
                "properties": {
                    "speed": {"type": ["string", "null"], "enum": ["fast", "slow"]}
                },
                "required": [],
            }
        )
        speed = out["properties"]["speed"]
        assert speed["type"] == ["string", "null"]
        assert None in speed["enum"]
        # Strict mode requires every property, so the only way to express
        # "not provided" is a value both constraints accept.
        assert out["required"] == ["speed"]

    def test_anyof_with_a_sibling_assertion_wraps_the_whole_schema(self) -> None:
        """Appending a null branch only works when the combinator is the whole
        schema. A sibling ``type`` applies independently, so a null added
        *inside* the ``anyOf`` still fails it — and strict mode makes the
        property required, leaving no satisfiable value (PR #381 review)."""
        out = strict_json_schema(
            {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "anyOf": [{"minimum": 10}, {"maximum": 0}]}
                },
                "required": [],
            }
        )
        n = out["properties"]["n"]
        assert n == {
            "anyOf": [
                {"type": "integer", "anyOf": [{"minimum": 10}, {"maximum": 0}]},
                {"type": "null"},
            ]
        }
        assert out["required"] == ["n"]

    def test_bare_anyof_still_gains_a_flat_null_branch(self) -> None:
        """No sibling assertion means the combinator *is* the schema, so
        appending is equivalent and keeps the emitted schema flat. A
        ``description`` alongside it annotates rather than constrains."""
        out = strict_json_schema(
            {
                "type": "object",
                "properties": {
                    "n": {"description": "d", "anyOf": [{"type": "integer"}]}
                },
                "required": [],
            }
        )
        assert out["properties"]["n"] == {
            "description": "d",
            "anyOf": [{"type": "integer"}, {"type": "null"}],
        }

    def test_oneof_is_always_wrapped_rather_than_extended(self) -> None:
        """``oneOf`` demands *exactly* one match, and null passes most
        constraint-only branches vacuously — an appended null branch would
        make null match several and fail."""
        out = strict_json_schema(
            {
                "type": "object",
                "properties": {"n": {"oneOf": [{"type": "integer"}, {"type": "string"}]}},
                "required": [],
            }
        )
        assert out["properties"]["n"] == {
            "anyOf": [
                {"oneOf": [{"type": "integer"}, {"type": "string"}]},
                {"type": "null"},
            ]
        }

    def test_allof_is_wrapped_rather_than_type_widened(self) -> None:
        """``allOf`` intersects, so *every* branch must admit null. Widening
        only the outer ``type`` left a const branch that still rejects it, and
        strict mode makes the property required (PR #381 review)."""
        out = strict_json_schema(
            {
                "type": "object",
                "properties": {"m": {"type": "string", "allOf": [{"const": "fixed"}]}},
                "required": [],
            }
        )
        assert out["properties"]["m"] == {
            "anyOf": [
                {"type": "string", "allOf": [{"const": "fixed"}]},
                {"type": "null"},
            ]
        }

    def test_single_type_enum_widening_is_unchanged(self) -> None:
        out = strict_json_schema(
            {
                "type": "object",
                "properties": {"speed": {"type": "string", "enum": ["fast", "slow"]}},
                "required": [],
            }
        )
        assert out["properties"]["speed"]["type"] == ["string", "null"]
        assert out["properties"]["speed"]["enum"] == ["fast", "slow", None]


class TestStrictFallbackInAdapters:
    """A tool strict mode cannot express is emitted without the flag rather
    than with a schema OpenAI rejects outright."""

    @staticmethod
    def _mapping_tool() -> ToolDefinition:
        return ToolDefinition(
            name="tally",
            description="Count things",
            parameters_schema={
                "type": "object",
                "properties": {
                    "counts": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    }
                },
                "required": ["counts"],
            },
        )

    def test_openai_drops_strict_for_open_maps(self, caplog: pytest.LogCaptureFixture) -> None:
        tool = self._mapping_tool()
        with caplog.at_level("WARNING"):
            out = OpenAIAdapter().to_provider_schema(tool, strict=True)

        assert "strict" not in out["function"]
        # The schema is passed through untouched, so the tool stays callable.
        assert out["function"]["parameters"] == tool.parameters_schema
        assert "tally" in caplog.text
        assert "counts" in caplog.text

    def test_openai_responses_drops_strict_for_open_maps(self) -> None:
        out = OpenAIResponsesAdapter().to_provider_schema(self._mapping_tool(), strict=True)
        assert "strict" not in out

    def test_anthropic_drops_strict_for_open_maps(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Anthropic strict mode also requires ``additionalProperties: false``,
        so setting it on a schema that needs arbitrary keys replaced a typed
        mapping with an object accepting none — the same data loss the OpenAI
        adapters gate against (PR #381 review)."""
        tool = self._mapping_tool()
        with caplog.at_level("WARNING"):
            out = AnthropicAdapter().to_provider_schema(tool, strict=True)

        assert "strict" not in out
        assert out["input_schema"]["properties"]["counts"]["additionalProperties"] == {
            "type": "integer"
        }
        assert "tally" in caplog.text

    def test_anthropic_keeps_kwargs_handlers_callable(self) -> None:
        """A ``**kwargs`` handler advertises an open root object. Forcing it
        closed under strict mode left Claude unable to pass any argument at
        all — strictly worse than emitting the tool unconstrained."""
        tool = ToolDefinition(
            name="passthrough",
            description="Handler accepting arbitrary keyword arguments",
            parameters_schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": True,
            },
        )
        out = AnthropicAdapter().to_provider_schema(tool, strict=True)
        assert "strict" not in out
        assert out["input_schema"]["additionalProperties"] is True

    def test_anthropic_still_applies_strict_to_closed_schemas(self) -> None:
        tool = ToolDefinition(
            name="greet",
            description="Greet someone by name",
            parameters_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        out = AnthropicAdapter().to_provider_schema(tool, strict=True)
        assert out["strict"] is True
        assert out["input_schema"]["additionalProperties"] is False

    def test_strict_still_applies_to_fully_declared_tools(self) -> None:
        tool = ToolDefinition(
            name="greet",
            description="Greet someone",
            parameters_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": [],
            },
        )
        out = OpenAIAdapter().to_provider_schema(tool, strict=True)
        assert out["function"]["strict"] is True
        assert out["function"]["parameters"]["additionalProperties"] is False


class TestEmittedSchemasAreCallerOwned:
    """The registry holds one canonical ``parameters_schema`` per tool, so no
    adapter may hand out a payload that aliases it — a caller augmenting the
    payload would otherwise corrupt every later conversion of that tool and
    the executor's own validation (PR #381 review)."""

    @staticmethod
    def _tool() -> ToolDefinition:
        return ToolDefinition(
            name="tally",
            description="Count things carefully",
            parameters_schema={
                "type": "object",
                "properties": {
                    "counts": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    }
                },
                "required": ["counts"],
            },
        )

    @pytest.mark.parametrize(
        ("adapter_factory", "options", "extract"),
        [
            # strict=True on an open map takes the new fallback path.
            (OpenAIAdapter, {"strict": True}, lambda o: o["function"]["parameters"]),
            (OpenAIAdapter, {}, lambda o: o["function"]["parameters"]),
            (OpenAIResponsesAdapter, {"strict": True}, lambda o: o["parameters"]),
            (OpenAIResponsesAdapter, {}, lambda o: o["parameters"]),
            (AnthropicAdapter, {"strict": True}, lambda o: o["input_schema"]),
            (AnthropicAdapter, {}, lambda o: o["input_schema"]),
            (GeminiAdapter, {"sanitize": False}, lambda o: o["parameters"]),
            (GeminiAdapter, {}, lambda o: o["parameters"]),
        ],
    )
    def test_mutating_the_payload_leaves_the_tool_untouched(
        self, adapter_factory, options, extract
    ) -> None:
        tool = self._tool()
        emitted = extract(adapter_factory().to_provider_schema(tool, **options))

        emitted.setdefault("properties", {})["INJECTED"] = {"type": "string"}
        # Nested too: a ``{**schema}`` spread would leave this dict shared.
        if "counts" in tool.parameters_schema["properties"]:
            nested = emitted.get("properties", {}).get("counts")
            if isinstance(nested, dict):
                nested["MUTATED"] = True

        assert "INJECTED" not in tool.parameters_schema["properties"]
        assert "MUTATED" not in tool.parameters_schema["properties"]["counts"]


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


class TestStrictNullableEnums:
    """Widening ``type`` to admit ``null`` is not enough on its own — ``enum``
    is an independent constraint (PR #381 review)."""

    def test_optional_enum_admits_null_after_widening(self) -> None:
        out = strict_json_schema(
            {
                "type": "object",
                "properties": {"mode": {"type": "string", "enum": ["fast", "slow"]}},
                "required": [],
            }
        )
        mode = out["properties"]["mode"]
        assert mode["type"] == ["string", "null"]
        # Without this the model cannot express "not provided": strict mode
        # makes every property required, and the enum would forbid null.
        assert mode["enum"] == ["fast", "slow", None]

    def test_required_enum_is_left_alone(self) -> None:
        out = strict_json_schema(
            {
                "type": "object",
                "properties": {"mode": {"type": "string", "enum": ["fast", "slow"]}},
                "required": ["mode"],
            }
        )
        mode = out["properties"]["mode"]
        assert mode["type"] == "string"
        assert mode["enum"] == ["fast", "slow"]


def test_pattern_keyed_objects_have_no_strict_mode_representation():
    """Strict mode can only describe an object whose full key set is written
    out in ``properties``. Keys typed by regex are by definition not — so the
    object is open however ``additionalProperties`` is set, ``false``
    included. Reading only ``additionalProperties`` called such a schema
    strict-safe, and the transform then published it with the unsupported
    ``patternProperties`` keyword still attached, which the provider rejects
    (PR #381 review)."""
    from agent_gantry.adapters.tool_spec.schema_utils import unsupported_strict_paths

    closed = {
        "type": "object",
        "properties": {
            "m": {
                "type": "object",
                "patternProperties": {"^n_": {"type": "integer"}},
                "additionalProperties": False,
            }
        },
        "required": ["m"],
    }
    assert unsupported_strict_paths(closed) == ["m"]

    at_root = {
        "type": "object",
        "properties": {},
        "patternProperties": {"^n_": {"type": "integer"}},
        "additionalProperties": False,
    }
    assert unsupported_strict_paths(at_root) == ["<root>"]

    # Schemas strict mode *can* express are unaffected.
    plain = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    assert unsupported_strict_paths(plain) == []
    assert unsupported_strict_paths({"type": "object", "properties": {}, "required": []}) == []


def test_a_property_forbidden_by_its_schema_has_no_strict_representation():
    """Strict mode makes every property *required*, so a property whose schema
    is ``false`` — satisfiable by no value — would be emitted as required and
    unsatisfiable at once, a schema with no valid instance that turns an
    otherwise callable tool into an uncallable one (PR #381 review)."""
    from agent_gantry.adapters.tool_spec.schema_utils import unsupported_strict_paths

    optional = {
        "type": "object",
        "properties": {"disabled": False, "name": {"type": "string"}},
        "required": ["name"],
    }
    assert unsupported_strict_paths(optional) == ["disabled"]
    # Required or optional, the property is equally unrepresentable.
    assert unsupported_strict_paths(
        {"type": "object", "properties": {"disabled": False}, "required": ["disabled"]}
    ) == ["disabled"]
    # Nested, and reported at its path.
    assert unsupported_strict_paths(
        {
            "type": "object",
            "properties": {"m": {"type": "object", "properties": {"d": False}}},
            "required": ["m"],
        }
    ) == ["m.d"]

    # ``true`` is satisfiable by everything, so it needs no fallback — and
    # neither do the schemas strict mode already handles. Widening the
    # unsupported set costs every one of them their strict guarantees.
    assert unsupported_strict_paths(
        {
            "type": "object",
            "properties": {"ok": True, "name": {"type": "string"}},
            "required": ["name"],
        }
    ) == []
    assert unsupported_strict_paths(
        {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": False,
        }
    ) == []


def test_a_nullable_open_mapping_is_still_strict_unsupported():
    """``type`` is a *list* whenever nullability is spelled into it — which
    introspection emits for a required ``dict[str, int] | None`` — so matching
    only the scalar string let a nullable open mapping past the check and into
    strict mode, which cannot represent it. The provider then rejects the
    whole tool request rather than that one parameter (PR #381 review)."""
    from agent_gantry.adapters.tool_spec.schema_utils import unsupported_strict_paths
    from agent_gantry.schema.introspection import build_parameters_schema

    def nullable_map(m: dict[str, int] | None) -> None: ...

    def plain_map(m: dict[str, int]) -> None: ...

    emitted = build_parameters_schema(nullable_map)
    assert emitted["properties"]["m"]["type"] == ["object", "null"]
    assert unsupported_strict_paths(emitted) == ["m"]
    # The non-nullable spelling was already correct and must stay so.
    assert unsupported_strict_paths(build_parameters_schema(plain_map)) == ["m"]

    # A nullable *closed* object is perfectly representable — widening the
    # check must not sweep it up.
    closed = {
        "type": "object",
        "properties": {
            "p": {
                "type": ["object", "null"],
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": False,
            }
        },
        "required": ["p"],
    }
    assert unsupported_strict_paths(closed) == []


def test_a_pattern_only_object_is_strict_unsupported_without_a_type():
    """JSON Schema applies an object's keywords whenever the instance *is* an
    object, so a property carrying only ``patternProperties`` — no ``type``, no
    ``properties`` — constrains objects just as much as one spelling the type
    out. Gating the check on type-or-properties let that spelling past as
    strict-safe while its typed twin was flagged (PR #381 review)."""
    from agent_gantry.adapters.tool_spec.schema_utils import unsupported_strict_paths

    untyped = {
        "type": "object",
        "properties": {
            "m": {
                "patternProperties": {"^n_": {"type": "integer"}},
                "additionalProperties": False,
            }
        },
        "required": ["m"],
    }
    typed = {
        "type": "object",
        "properties": {
            "m": {
                "type": "object",
                "patternProperties": {"^n_": {"type": "integer"}},
                "additionalProperties": False,
            }
        },
        "required": ["m"],
    }
    assert unsupported_strict_paths(untyped) == ["m"]
    assert unsupported_strict_paths(typed) == ["m"]

    # Schemas strict mode genuinely handles must stay handled — this is the
    # gate deciding whether a tool gets strict guarantees at all.
    for supported in (
        {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": False,
        },
        {"type": "object", "properties": {}, "required": []},
        {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
    ):
        assert unsupported_strict_paths(supported) == []


def test_a_typeless_additional_properties_map_is_strict_unsupported():
    """The sibling of the pattern-only fix, and reached only its own keyword:
    ``{"additionalProperties": {"type": "integer"}}`` — an imported
    ``dict[str, int]`` with no ``type`` — was reported strict-safe while its
    typed twin was flagged, so the strict transform published the
    schema-valued keyword strict mode cannot express and the provider rejected
    the whole tool request (PR #381 review).

    The invariant is that the two spellings agree, since JSON Schema applies
    an object's keywords whenever the instance *is* an object and writing the
    type out is optional rather than load-bearing."""

    def wrap(prop: dict) -> dict:
        return {"type": "object", "properties": {"m": prop}, "required": ["m"]}

    for keyword, expected in (
        ({"additionalProperties": {"type": "integer"}}, ["m"]),
        ({"additionalProperties": True}, ["m"]),
        ({"additionalProperties": {}}, ["m"]),
        # Explicitly closed is the one that is genuinely representable, and it
        # still passes through ``_is_open_map`` as safe rather than being
        # excluded by the gate.
        ({"additionalProperties": False}, []),
    ):
        untyped = unsupported_strict_paths(wrap(dict(keyword)))
        typed = unsupported_strict_paths(wrap({"type": "object", **keyword}))
        assert untyped == typed == expected, (keyword, untyped, typed)

    # Only where the node declares no type: unlike properties and
    # patternProperties, which nothing but an object schema carries, a stray
    # additionalProperties beside a scalar type asserts nothing — and flagging
    # it would cost that tool strict mode for no reason.
    assert unsupported_strict_paths(
        wrap({"type": "string", "additionalProperties": {"type": "integer"}})
    ) == []

    # Schemas strict mode genuinely handles must stay handled.
    for supported in (
        {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": False,
        },
        {"type": "object", "properties": {}},
        {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
    ):
        assert unsupported_strict_paths(supported) == []


def test_a_nested_empty_properties_object_is_strict_unsupported():
    """``properties: {}`` with no ``additionalProperties`` is the "tool takes
    no arguments" shape strict mode itself emits — but only at the *root*,
    where the executor agrees and rejects an unknown argument outright.
    Nested, absent ``additionalProperties`` is a free-form mapping the
    executor accepts any keys for, so calling it strict-safe let the transform
    rewrite it closed and made a valid ``{"payload": {"key": 1}}``
    ungeneratable (PR #381 review)."""

    def wrap(prop: dict) -> dict:
        return {
            "type": "object",
            "properties": {"payload": prop},
            "required": ["payload"],
        }

    # The same object written two ways must get the same verdict; the bare
    # spelling was flagged all along.
    assert unsupported_strict_paths(wrap({"type": "object", "properties": {}})) == [
        "payload"
    ]
    assert unsupported_strict_paths(wrap({"type": "object"})) == ["payload"]

    # A nested object that really is closed stays representable.
    assert (
        unsupported_strict_paths(
            wrap({"type": "object", "properties": {}, "additionalProperties": False})
        )
        == []
    )
    assert (
        unsupported_strict_paths(
            wrap(
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "additionalProperties": False,
                }
            )
        )
        == []
    )

    # And the root exemption is untouched, which is the half this must not
    # break: a no-argument tool is genuinely closed, and the executor says so.
    assert unsupported_strict_paths({"type": "object", "properties": {}}) == []
    assert unsupported_strict_paths({"type": "object", "properties": {}, "required": []}) == []


def test_a_typeless_enum_is_strict_unsupported():
    """``_enum_schema`` declares a ``type`` only when the members share one
    scalar JSON kind, so ``Literal[1, "auto"]`` — and a tuple-valued ``Enum``
    — publishes a *typeless* property. Strict mode requires every property to
    name its type, so the provider rejects the whole tool request rather than
    that one parameter (PR #381 review).

    Reported rather than repaired: widening to a type *list* is not something
    strict mode accepts either, so the honest answer is that the schema has no
    strict spelling and the tool should go out non-strict."""

    def wrap(prop: dict) -> dict:
        return {"type": "object", "properties": {"mode": prop}, "required": ["mode"]}

    # What introspection actually emits for each of these, asserted here so
    # the two halves cannot drift apart.
    import typing

    from agent_gantry.schema.introspection import _type_to_json_schema

    assert _type_to_json_schema(typing.Literal[1, "auto"]) == {"enum": [1, "auto"]}
    for typeless in (
        {"enum": [1, "auto"]},
        # ``integer`` and ``number`` are distinct JSON kinds.
        {"enum": [1, 2.5]},
        # A tuple-valued ``Enum``: composite members no scalar type names.
        {"enum": [[0, 0], [1, 1]]},
    ):
        assert unsupported_strict_paths(wrap(typeless)) == ["mode"], typeless

    # Anything that does carry a type is untouched — this gate decides whether
    # a tool gets strict guarantees at all.
    for supported in (
        {"type": "string", "enum": ["a", "b"]},
        {"type": "integer", "enum": [1, 2]},
        {"type": ["string", "null"], "enum": ["a", None]},
    ):
        assert unsupported_strict_paths(wrap(supported)) == [], supported


def test_a_nullable_typeless_enum_is_strict_unsupported():
    """``None`` is a member like any other, and ``_enum_schema`` puts it in the
    catch-all branch, so a ``Literal`` carrying one publishes typeless too
    (PR #385 review).

    The guard used to skip ``None`` before deciding, on the grounds that a
    nullable enum still names one real kind and carries its nullability in the
    ``type`` list. But a node that *has* a ``type`` list already left at the
    first line, so the skip only ever ran where its own justification did not
    hold: ``Literal["a", None]`` reduced to a lone ``string`` kind and went out
    ``strict: true`` with no ``type`` for the provider to read."""

    def wrap(prop: dict) -> dict:
        return {"type": "object", "properties": {"mode": prop}, "required": ["mode"]}

    import typing

    from agent_gantry.schema.introspection import _type_to_json_schema

    # Nothing repairs these on the way out: ``strict_json_schema`` widens and
    # wraps the types it finds and never invents one, so what the guard sees
    # is what the provider gets.
    assert _type_to_json_schema(typing.Literal["a", None]) == {"enum": ["a", None]}
    assert _type_to_json_schema(typing.Literal[None]) == {"enum": [None]}
    for typeless in ({"enum": ["a", None]}, {"enum": [None]}):
        assert unsupported_strict_paths(wrap(typeless)) == ["mode"], typeless
        assert "type" not in strict_json_schema(wrap(typeless))["properties"]["mode"]

    # An ``Optional[Literal[...]]`` that keeps a real type is still fine: the
    # union collapses to one scalar kind and introspection names it.
    assert _type_to_json_schema(typing.Literal["a"] | None) == {
        "type": "string",
        "enum": ["a"],
    }
    assert unsupported_strict_paths(wrap({"type": "string", "enum": ["a"]})) == []


def test_a_typeless_enum_from_an_external_schema_is_unsupported():
    """The guard tests for a ``type``, not for what one *could* be derived
    from the members, so a hand-written or MCP-supplied ``{"enum": [...]}``
    reports too. Deriving ``string`` here would name a type the published
    schema does not carry, which is the whole failure mode."""
    wrapped = {
        "type": "object",
        "properties": {"mode": {"enum": ["a", "b"]}},
        "required": ["mode"],
    }
    assert unsupported_strict_paths(wrapped) == ["mode"]

    # An empty ``enum`` is not a schema a provider accepts at all, and
    # ``_enum_schema`` degrades it to a plain string rather than emitting one,
    # so there is no typeless property here to report.
    empty = {"type": "object", "properties": {"mode": {"enum": []}}, "required": ["mode"]}
    assert unsupported_strict_paths(empty) == []
