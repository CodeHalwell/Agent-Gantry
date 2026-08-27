"""
Tests for new convenience methods and improvements added in the refactoring PR.

Tests:
- AgentGantry.quick_start()
- AgentGantry.search_and_execute()
- set_default_gantry() and decorator improvements
- build_parameters_schema() improvements
- ToolDefinition.to_searchable_text()
"""

import json

import pytest

from agent_gantry import AgentGantry, set_default_gantry, with_semantic_tools
from agent_gantry.schema.introspection import build_parameters_schema
from agent_gantry.schema.tool import ToolDefinition


class TestQuickStart:
    """Tests for AgentGantry.quick_start() convenience method."""

    @pytest.mark.asyncio
    async def test_quick_start_auto_embedder(self):
        """Test quick_start with auto embedder selection."""
        gantry = AgentGantry.quick_start(embedder="auto")
        assert gantry is not None
        assert gantry._embedder is not None

    @pytest.mark.asyncio
    async def test_quick_start_simple_embedder(self):
        """Test quick_start with explicit simple embedder."""
        gantry = AgentGantry.quick_start(embedder="simple")
        assert gantry is not None
        from agent_gantry.adapters.embedders.simple import SimpleEmbedder

        assert isinstance(gantry._embedder, SimpleEmbedder)

    @pytest.mark.asyncio
    async def test_quick_start_openai_without_key(self):
        """Test quick_start raises error for OpenAI without API key."""
        with pytest.raises(ValueError, match="OpenAI embedder requires a valid API key"):
            AgentGantry.quick_start(embedder="openai")

    @pytest.mark.asyncio
    async def test_quick_start_with_tool_registration(self):
        """Test quick_start works with tool registration and sync."""
        gantry = AgentGantry.quick_start()

        @gantry.register
        def test_tool(x: int) -> int:
            """Test tool."""
            return x * 2

        await gantry.sync()
        tools = await gantry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"


class TestSearchAndExecute:
    """Tests for AgentGantry.search_and_execute() convenience method."""

    @pytest.mark.asyncio
    async def test_search_and_execute_basic(self):
        """Test basic search and execute functionality."""
        gantry = AgentGantry.quick_start()

        @gantry.register
        def calculate_tax(amount: float) -> float:
            """Calculate 8% sales tax."""
            return amount * 0.08

        await gantry.sync()

        result = await gantry.search_and_execute(
            "calculate tax", arguments={"amount": 100.0}, score_threshold=0.0
        )

        assert result.result == 8.0
        assert result.status.value == "success"

    @pytest.mark.asyncio
    async def test_search_and_execute_no_tools_found(self):
        """Test search_and_execute raises error when no tools match."""
        gantry = AgentGantry.quick_start()

        @gantry.register
        def unrelated_tool(x: int) -> int:
            """Some unrelated tool."""
            return x

        await gantry.sync()

        with pytest.raises(ValueError, match="No tools found matching query"):
            await gantry.search_and_execute(
                "calculate quantum mechanics", arguments={}, score_threshold=0.9
            )

    @pytest.mark.asyncio
    async def test_search_and_execute_with_namespace(self):
        """Test search_and_execute respects tool namespace."""
        gantry = AgentGantry.quick_start()

        @gantry.register(namespace="default")
        def multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        await gantry.sync()

        result = await gantry.search_and_execute(
            "multiply numbers", arguments={"x": 5, "y": 3}, score_threshold=0.0
        )

        assert result.result == 15


class TestDefaultGantryDecorator:
    """Tests for set_default_gantry() and improved decorator."""

    @pytest.mark.asyncio
    async def test_set_default_gantry(self):
        """Test setting and using default gantry."""
        gantry = AgentGantry.quick_start()

        @gantry.register
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Sunny in {city}"

        await gantry.sync()

        # Set as default
        set_default_gantry(gantry)

        # Use decorator without explicit gantry
        @with_semantic_tools(limit=1, score_threshold=0.0)
        async def mock_llm_call(prompt: str, tools=None):
            return {"tools_count": len(tools) if tools else 0}

        result = await mock_llm_call("What's the weather?")
        assert result["tools_count"] == 1

    @pytest.mark.asyncio
    async def test_decorator_without_default_gantry_raises(self):
        """Test decorator without default gantry raises helpful error."""
        # Reset default
        from agent_gantry.integrations import semantic_tools

        semantic_tools._DEFAULT_GANTRY = None

        with pytest.raises(ValueError, match="No gantry provided and no default set"):

            @with_semantic_tools(limit=1)
            async def generate(prompt: str, tools=None):
                pass

    @pytest.mark.asyncio
    async def test_decorator_with_explicit_gantry_still_works(self):
        """Test decorator still works with explicit gantry parameter."""
        gantry = AgentGantry.quick_start()

        @gantry.register
        def test_tool() -> str:
            """Test tool."""
            return "result"

        await gantry.sync()

        @with_semantic_tools(gantry, limit=1, score_threshold=0.0)
        async def generate(prompt: str, tools=None):
            return {"tools_count": len(tools) if tools else 0}

        result = await generate("test")
        assert result["tools_count"] == 1


class TestBuildParametersSchema:
    """Tests for build_parameters_schema improvements."""

    def test_basic_type_mapping(self):
        """Test basic type mapping for int, float, bool, str."""

        def func(a: int, b: float, c: bool, d: str) -> None:
            pass

        schema = build_parameters_schema(func)
        assert schema["properties"]["a"]["type"] == "integer"
        assert schema["properties"]["b"]["type"] == "number"
        assert schema["properties"]["c"]["type"] == "boolean"
        assert schema["properties"]["d"]["type"] == "string"

    def test_required_vs_optional(self):
        """Test detection of required vs optional parameters."""

        def func(required: int, optional: str = "default") -> None:
            pass

        schema = build_parameters_schema(func)
        assert "required" in schema["required"]
        assert "optional" not in schema["required"]

    def test_optional_type_handling(self):
        """Test handling of Optional[T] type hints."""

        def func(x: int | None = None, y: str | None = None) -> None:
            pass

        schema = build_parameters_schema(func)
        assert schema["properties"]["x"]["type"] == "integer"
        assert schema["properties"]["y"]["type"] == "string"

    def test_skips_self_and_cls(self):
        """Test that self and cls parameters are skipped."""

        class TestClass:
            def method(self, x: int) -> None:
                pass

            @classmethod
            def classmethod(cls, x: int) -> None:
                pass

        schema1 = build_parameters_schema(TestClass.method)
        assert "self" not in schema1["properties"]
        assert "x" in schema1["properties"]

        schema2 = build_parameters_schema(TestClass.classmethod)
        assert "cls" not in schema2["properties"]
        assert "x" in schema2["properties"]

    def test_no_type_hints(self):
        """Test function without type hints defaults to string."""

        def func(x, y):
            pass

        schema = build_parameters_schema(func)
        assert schema["properties"]["x"]["type"] == "string"
        assert schema["properties"]["y"]["type"] == "string"


class TestToolSearchableText:
    """Tests for ToolDefinition.to_searchable_text()."""

    def test_includes_all_metadata(self):
        """Test that searchable text includes all relevant metadata."""
        tool = ToolDefinition(
            name="calculate_tax",
            namespace="finance",
            description="Calculate sales tax for a given amount",
            tags=["math", "money", "finance"],
            examples=["calculate tax on $100", "what is 8% tax on 50"],
            parameters_schema={"type": "object", "properties": {}, "required": []},
        )

        text = tool.to_searchable_text()

        # Verify all components are present
        assert "calculate_tax" in text
        assert "finance" in text
        assert "Calculate sales tax" in text
        assert "math" in text
        assert "money" in text
        assert "calculate tax on $100" in text

    def test_empty_tags_and_examples(self):
        """Test searchable text with empty tags and examples."""
        tool = ToolDefinition(
            name="simple_tool",
            namespace="default",
            description="A simple tool",
            parameters_schema={"type": "object", "properties": {}, "required": []},
        )

        text = tool.to_searchable_text()

        assert "simple_tool" in text
        assert "default" in text
        assert "A simple tool" in text

    def test_consistency_with_router(self):
        """Test that searchable text is consistent across uses."""
        tool = ToolDefinition(
            name="test_tool",
            namespace="test",
            description="Test description",
            tags=["tag1", "tag2"],
            examples=["example1"],
            parameters_schema={"type": "object", "properties": {}, "required": []},
        )

        # Call multiple times
        text1 = tool.to_searchable_text()
        text2 = tool.to_searchable_text()

        # Should be identical
        assert text1 == text2


class TestSchemaFidelity:
    """build_parameters_schema preserves declared intent (descriptions, enums,
    containers, defaults) — what every provider dialect and framework adapter
    ultimately advertises to the LLM."""

    def test_docstring_descriptions_google_style(self):
        def func(city: str, days: int = 3) -> str:
            """Get a weather forecast.

            Args:
                city: Name of the city.
                days: Forecast horizon in days.
            """
            return ""

        schema = build_parameters_schema(func)
        assert schema["properties"]["city"]["description"] == "Name of the city."
        assert schema["properties"]["days"]["description"] == "Forecast horizon in days."
        assert schema["properties"]["days"]["default"] == 3

    def test_annotated_description_wins_over_docstring(self):
        from typing import Annotated

        def func(city: Annotated[str, "City name override"]) -> str:
            """Get weather.

            Args:
                city: Ignored.
            """
            return ""

        schema = build_parameters_schema(func)
        assert schema["properties"]["city"]["description"] == "City name override"

    def test_literal_and_enum_become_enum(self):
        import enum
        from typing import Literal

        # Deliberately *not* a ``str`` mixin. A ``class Color(str, Enum)``
        # compares and serializes equal to its value either way, so it would
        # pass whether or not the default is actually unwrapped — the
        # assertion below would not catch a regression (PR #381 review).
        class Color(enum.Enum):
            RED = "red"
            BLUE = "blue"

        def func(mode: Literal["fast", "slow"], color: Color = Color.RED) -> None:
            pass

        schema = build_parameters_schema(func)
        assert schema["properties"]["mode"] == {
            "type": "string",
            "enum": ["fast", "slow"],
        }
        assert schema["properties"]["color"]["enum"] == ["red", "blue"]
        default = schema["properties"]["color"]["default"]
        assert default == "red"
        # The unwrapping is the point: a bare ``Enum`` member is not JSON
        # serializable, so leaving it in place emits a schema no provider
        # can parse.
        assert type(default) is str
        json.dumps(schema)

    def test_dict_maps_to_object(self):
        from typing import Any

        def func(meta: dict[str, Any], counts: dict[str, int] | None = None) -> None:
            pass

        schema = build_parameters_schema(func)
        assert schema["properties"]["meta"] == {"type": "object"}
        assert schema["properties"]["counts"]["type"] == "object"
        assert schema["properties"]["counts"]["additionalProperties"] == {
            "type": "integer"
        }

    def test_pep604_union_none_first(self):
        def func(x: None | int = None) -> None:
            pass

        schema = build_parameters_schema(func)
        assert schema["properties"]["x"]["type"] == "integer"

    def test_multi_member_union_keeps_only_the_first_member(self, caplog):
        """A genuine ``int | str`` loses every member but the first — a real
        fidelity loss, but a deliberate one: most provider dialects reject
        union-typed parameters outright. Pinned so the behaviour is a decision
        rather than an accident, and logged so a schema author has a signal
        (PR #381 review)."""
        def func(x: int | str) -> None:
            pass

        with caplog.at_level("DEBUG", logger="agent_gantry.schema.introspection"):
            schema = build_parameters_schema(func)

        assert schema["properties"]["x"] == {"type": "integer"}
        assert "multi-member union" in caplog.text

    def test_optional_union_is_not_logged_as_a_loss(self):
        """``T | None`` loses nothing — requiredness carries the optionality —
        so it must not warn."""
        def func(x: int | None = None) -> None:
            pass

        import logging as _logging

        records: list[_logging.LogRecord] = []

        class _Capture(_logging.Handler):
            def emit(self, record: _logging.LogRecord) -> None:
                records.append(record)

        log = _logging.getLogger("agent_gantry.schema.introspection")
        handler = _Capture()
        log.addHandler(handler)
        previous = log.level
        log.setLevel(_logging.DEBUG)
        try:
            schema = build_parameters_schema(func)
        finally:
            log.removeHandler(handler)
            log.setLevel(previous)

        assert schema["properties"]["x"] == {"type": "integer"}
        assert not [r for r in records if "multi-member union" in r.getMessage()]

    def test_non_finite_float_defaults_are_dropped(self):
        """NaN/±inf are Python floats but not JSON values: ``json.dumps``
        emits bare ``NaN``/``Infinity`` tokens, which a provider parsing
        strict JSON rejects (PR #381 review)."""
        import json

        def func(
            ratio: float = float("nan"),
            cap: float = float("inf"),
            ok: float = 1.5,
        ) -> None:
            pass

        schema = build_parameters_schema(func)
        assert "default" not in schema["properties"]["ratio"]
        assert "default" not in schema["properties"]["cap"]
        assert schema["properties"]["ok"]["default"] == 1.5
        # The emitted schema must survive strict JSON serialization.
        json.dumps(schema, allow_nan=False)

    def test_nested_non_finite_defaults_are_dropped(self):
        def func(bounds: list[float] = [1.0, float("inf")]) -> None:
            pass

        schema = build_parameters_schema(func)
        assert "default" not in schema["properties"]["bounds"]

    def test_collections_abc_container_origins(self):
        """``typing.get_origin(Sequence[int])`` is ``collections.abc.Sequence``
        — neither the ``typing`` alias nor a ``list`` subclass — so matching
        only aliases and concrete containers advertised ``Sequence[int]`` as
        ``{"type": "integer"}`` (PR #381 review)."""
        # ``typing.Sequence[int]`` and ``collections.abc.Sequence[int]``
        # normalize to the same origin, so these cover both spellings.
        from collections.abc import Iterable, Mapping, Sequence, Set

        def func(
            a: Sequence[int],
            b: Iterable[str],
            c: Set[int],
            d: Mapping[str, int],
        ) -> None:
            pass

        schema = build_parameters_schema(func)
        props = schema["properties"]
        assert props["a"] == {"type": "array", "items": {"type": "integer"}}
        assert props["b"] == {"type": "array", "items": {"type": "string"}}
        assert props["c"]["type"] == "array"
        assert props["c"]["uniqueItems"] is True
        # A Mapping is also a Collection/Iterable, so it must not be caught by
        # the sequence branch.
        assert props["d"] == {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        }

    def test_typed_containers(self):
        def func(tags: set[str], pair: tuple[int, ...] = ()) -> None:
            pass

        schema = build_parameters_schema(func)
        assert schema["properties"]["tags"]["type"] == "array"
        assert schema["properties"]["tags"]["uniqueItems"] is True
        assert schema["properties"]["tags"]["items"] == {"type": "string"}
        assert schema["properties"]["pair"]["items"] == {"type": "integer"}

    def test_dataclass_param_inlines_nested_schema(self):
        import dataclasses

        @dataclasses.dataclass
        class Address:
            street: str
            city: str = "London"

        def func(addr: Address) -> None:
            pass

        schema = build_parameters_schema(func)
        addr = schema["properties"]["addr"]
        assert addr["type"] == "object"
        assert addr["properties"]["street"]["type"] == "string"
        assert "$ref" not in str(addr)

    def test_sphinx_param_docs(self):
        def func(x: int) -> None:
            """Do a thing.

            :param x: The x value.
            """

        schema = build_parameters_schema(func)
        assert schema["properties"]["x"]["description"] == "The x value."

    def test_numpy_style_docstring_descriptions(self):
        def func(city: str, days: int = 3) -> str:
            """Get a weather forecast.

            Parameters
            ----------
            city : str
                Name of the city.
            days : int
                Forecast horizon in days.
            """
            return ""

        schema = build_parameters_schema(func)
        assert schema["properties"]["city"]["description"] == "Name of the city."
        assert schema["properties"]["days"]["description"] == "Forecast horizon in days."

    def test_pydantic_basemodel_param_inlines_nested_schema(self):
        from pydantic import BaseModel

        class Address(BaseModel):
            street: str
            city: str = "London"

        def func(addr: Address) -> None:
            pass

        schema = build_parameters_schema(func)
        addr = schema["properties"]["addr"]
        assert addr["type"] == "object"
        assert addr["properties"]["street"]["type"] == "string"
        assert addr["properties"]["city"]["default"] == "London"
        assert "$ref" not in str(addr)

    def test_typeddict_param_inlines_nested_schema(self):
        from typing import TypedDict

        class Address(TypedDict):
            street: str
            city: str

        def func(addr: Address) -> None:
            pass

        schema = build_parameters_schema(func)
        addr = schema["properties"]["addr"]
        assert addr["type"] == "object"
        assert addr["properties"]["street"]["type"] == "string"
        assert addr["properties"]["city"]["type"] == "string"
        assert set(addr.get("required", [])) == {"street", "city"}

    def test_none_annotated_parameter_maps_to_null(self):
        """``def f(x: None)`` resolves to ``NoneType``, which isn't in the
        scalar map and fell through to the string fallback — advertising a
        string for a parameter that admits only null (PR #381 review)."""

        def func(x: None) -> None:
            pass

        assert build_parameters_schema(func)["properties"]["x"] == {"type": "null"}

    def test_typeddict_inheritance_keeps_required_keys(self):
        """``class Child(Base, total=False)`` still requires ``Base``'s keys.
        Replaying one ``total=`` flag over the merged annotations made every
        one of them optional (PR #381 review)."""
        from typing import TypedDict

        class Base(TypedDict):
            a: str

        class Child(Base, total=False):
            b: int

        def func(x: Child) -> None:
            pass

        schema = build_parameters_schema(func)["properties"]["x"]
        assert set(schema["properties"]) == {"a", "b"}
        assert schema["required"] == ["a"]

    def test_self_referential_model_hits_ref_depth_limit(self):
        from pydantic import BaseModel

        class Node(BaseModel):
            name: str
            children: list["Node"] = []

        Node.model_rebuild()

        def func(root: Node) -> None:
            pass

        schema = build_parameters_schema(func)
        root = schema["properties"]["root"]
        assert root["type"] == "object"
        assert root["properties"]["name"]["type"] == "string"
        # Self-reference is capped (_MAX_REF_DEPTH) rather than inlined
        # forever or left as an unresolved $ref for consumers that don't
        # follow JSON pointers (the executor's validator, several provider
        # dialects).
        assert "$ref" not in str(root)


def test_non_json_literal_values_degrade_to_string_schema():
    """Literal admits bytes (and Enum members can carry arbitrary objects) —
    non-JSON values must not leak into an ``enum`` payload (PR #381 review)."""
    from typing import Literal

    def func(marker: Literal[b"a", b"b"]) -> None:
        pass

    schema = build_parameters_schema(func)
    assert schema["properties"]["marker"] == {"type": "string"}
    assert "enum" not in schema["properties"]["marker"]


class TestCompositeEnumValues:
    """A Python ``tuple`` is JSON-safe but serializes to an *array*, so
    storing one verbatim left the canonical schema holding a value no provider
    would ever send back (PR #381 review)."""

    def test_composite_enum_members_are_stored_as_arrays(self):
        import enum

        class Point(enum.Enum):
            ORIGIN = (0, 0)
            UNIT = (1, 1)

        def func(pt: Point = Point.ORIGIN) -> None:
            pass

        schema = build_parameters_schema(func)
        prop = schema["properties"]["pt"]
        assert prop["enum"] == [[0, 0], [1, 1]]
        assert prop["default"] == [0, 0]
        # The stored schema is the same document the provider sees, so the
        # executor compares like with like.
        assert json.loads(json.dumps(schema)) == schema

    def test_tuple_defaults_are_stored_as_arrays(self):
        def func(box: tuple[int, int] = (1, 2)) -> None:
            pass

        schema = build_parameters_schema(func)
        assert schema["properties"]["box"]["default"] == [1, 2]
        assert json.loads(json.dumps(schema)) == schema


class TestStringFormats:
    """``build_parameters_schema`` emits a JSON-Schema ``format`` for the
    stdlib scalar types providers understand, so a model gets the shape of the
    string it must produce rather than "any string" (PR #381 review noted this
    was claimed in the changelog but untested)."""

    def test_datetime_family_and_uuid_carry_a_format(self):
        import uuid
        from datetime import date, datetime, time

        def func(
            at: datetime,
            on: date,
            clock: time,
            ident: uuid.UUID,
        ) -> None:
            pass

        props = build_parameters_schema(func)["properties"]
        assert props["at"] == {"type": "string", "format": "date-time"}
        assert props["on"] == {"type": "string", "format": "date"}
        assert props["clock"] == {"type": "string", "format": "time"}
        assert props["ident"] == {"type": "string", "format": "uuid"}

    def test_plain_strings_carry_no_format(self):
        def func(name: str) -> None:
            pass

        assert build_parameters_schema(func)["properties"]["name"] == {"type": "string"}
