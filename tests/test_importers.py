"""Tests for ``agent_gantry.integrations.importers`` (framework -> Gantry).

Stub-based tests build a minimal fake module for ``langchain_core.tools`` /
``crewai.tools`` / ``llama_index.core.tools`` via ``sys.modules`` (mirroring
``tests/frameworks/test_langchain.py``'s approach) so they exercise the
importers' own conversion/wrapping logic without requiring the real
framework to be installed. The ``TestRealPackages`` section re-runs the same
shape of assertions against the actual installed packages (available here
via the ``agent-frameworks`` extra) and is skipped cleanly via
``pytest.importorskip`` when a framework isn't installed, so this file also
passes in a minimal environment.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from typing import Any

import pytest
from pydantic import BaseModel, Field

# CrewAI ships opt-out telemetry that can block for ~30s on a firewalled
# network; disable it before crewai is ever imported for real (mirrors
# tests/frameworks/test_real_packages.py).
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.integrations.importers import (
    _normalize_tool_name,
    register_crewai_tools,
    register_langchain_tools,
    register_llamaindex_tools,
)
from agent_gantry.observability.console import NoopTelemetryAdapter
from agent_gantry.schema.execution import ExecutionStatus, ToolCall
from agent_gantry.schema.tool import ToolSource

_IMPORTERS_LOGGER = "agent_gantry.integrations.importers"


class _RecordingTelemetry(NoopTelemetryAdapter):
    """Records every ``record_execution`` call so tests can assert telemetry fired."""

    def __init__(self) -> None:
        self.executions: list[tuple[Any, Any]] = []

    async def record_execution(self, call: Any, result: Any) -> None:
        self.executions.append((call, result))


@pytest.fixture
def gantry() -> AgentGantry:
    """A fresh AgentGantry with a deterministic offline embedder + telemetry recorder."""
    return AgentGantry(embedder=SimpleEmbedder(dimension=64), telemetry=_RecordingTelemetry())


# --------------------------------------------------------------------------- #
# Shared helper unit tests
# --------------------------------------------------------------------------- #


class TestNormalizeToolName:
    """Unit tests for the name-sanitization shared by all three importers."""

    def test_lowercases_and_replaces_spaces(self) -> None:
        assert _normalize_tool_name("Get Weather") == "get_weather"

    def test_strips_non_alnum_punctuation(self) -> None:
        assert _normalize_tool_name("send-email!!") == "send_email"

    def test_leading_digit_gets_letter_prefix(self) -> None:
        assert _normalize_tool_name("123tool") == "t_123tool"

    def test_empty_or_none_falls_back_to_tool(self) -> None:
        assert _normalize_tool_name("") == "tool"
        assert _normalize_tool_name(None) == "tool"

    def test_collapses_repeated_separators(self) -> None:
        assert _normalize_tool_name("get   the   weather") == "get_the_weather"


# --------------------------------------------------------------------------- #
# LangChain — stub-based
# --------------------------------------------------------------------------- #


class _FakeLCBaseTool:
    """Stand-in for ``langchain_core.tools.BaseTool``."""

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: Any = None,
        return_direct: bool = False,
        fn: Any = None,
        afn: Any = None,
    ) -> None:
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.return_direct = return_direct
        self._fn = fn
        self._afn = afn

    async def ainvoke(self, input: dict[str, Any]) -> Any:  # noqa: A002 - matches LangChain's signature
        if self._afn is not None:
            return await self._afn(**input)
        if self._fn is not None:
            return self._fn(**input)
        raise NotImplementedError("no implementation wired for this fake tool")


class _CityArgs(BaseModel):
    city: str = Field(description="City name")


class _ABArgs(BaseModel):
    a: int
    b: int


class _ValueArg(BaseModel):
    value: str


@pytest.fixture
def fake_langchain(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    core = types.ModuleType("langchain_core")
    tools_mod = types.ModuleType("langchain_core.tools")
    tools_mod.BaseTool = _FakeLCBaseTool  # type: ignore[attr-defined]
    core.tools = tools_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_core", core)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", tools_mod)
    return tools_mod


class TestLangChainImporterStub:
    async def test_schema_and_metadata_fidelity(self, fake_langchain, gantry: AgentGantry) -> None:
        tool = _FakeLCBaseTool(
            name="Get Weather",
            description="Get the current weather for a city.",
            args_schema=_CityArgs,
            return_direct=True,
            fn=lambda city: f"sunny in {city}",
        )
        count = await register_langchain_tools(gantry, [tool], tags=["weather"])
        assert count == 1
        await gantry.sync()

        gtool = await gantry.get_tool("get_weather", "langchain")
        assert gtool is not None
        assert gtool.description == "Get the current weather for a city."
        assert gtool.parameters_schema["properties"]["city"]["type"] == "string"
        assert gtool.parameters_schema["required"] == ["city"]
        assert gtool.source == ToolSource.FRAMEWORK
        assert gtool.source_uri == "langchain://Get Weather"
        assert gtool.tags == ["weather"]
        assert gtool.metadata["native_name"] == "Get Weather"
        assert gtool.metadata["return_direct"] is True
        assert gtool.metadata["framework"] == "langchain"

    async def test_tool_without_args_schema_falls_back_to_empty_object(
        self, fake_langchain, gantry: AgentGantry
    ) -> None:
        tool = _FakeLCBaseTool(
            name="ping", description="Ping with no arguments at all.", fn=lambda: "pong"
        )
        count = await register_langchain_tools(gantry, [tool])
        assert count == 1
        await gantry.sync()

        gtool = await gantry.get_tool("ping", "langchain")
        assert gtool is not None
        assert gtool.parameters_schema == {"type": "object", "properties": {}}

    async def test_sync_and_async_native_tools_both_execute(
        self, fake_langchain, gantry: AgentGantry
    ) -> None:
        sync_tool = _FakeLCBaseTool(
            name="sync_add",
            description="Add two numbers synchronously.",
            args_schema=_ABArgs,
            fn=lambda a, b: a + b,
        )

        async def _amul(a: int, b: int) -> int:
            return a * b

        async_tool = _FakeLCBaseTool(
            name="async_mul",
            description="Multiply two numbers asynchronously.",
            args_schema=_ABArgs,
            afn=_amul,
        )

        count = await register_langchain_tools(gantry, [sync_tool, async_tool])
        assert count == 2
        await gantry.sync()

        r1 = await gantry.execute(ToolCall(tool_name="sync_add", arguments={"a": 2, "b": 3}))
        assert r1.status == ExecutionStatus.SUCCESS
        assert r1.result == 5

        r2 = await gantry.execute(ToolCall(tool_name="async_mul", arguments={"a": 2, "b": 3}))
        assert r2.status == ExecutionStatus.SUCCESS
        assert r2.result == 6

    async def test_execution_routes_through_gantry_execute_and_telemetry(
        self, fake_langchain, gantry: AgentGantry
    ) -> None:
        tool = _FakeLCBaseTool(
            name="echo",
            description="Echo the input value back.",
            args_schema=_ValueArg,
            fn=lambda value: value,
        )
        await register_langchain_tools(gantry, [tool])
        await gantry.sync()

        result = await gantry.execute(ToolCall(tool_name="echo", arguments={"value": "hi"}))
        assert result.status == ExecutionStatus.SUCCESS
        assert result.result == "hi"

        telemetry: _RecordingTelemetry = gantry._telemetry  # type: ignore[assignment]
        assert len(telemetry.executions) == 1
        recorded_call, recorded_result = telemetry.executions[0]
        assert recorded_call.tool_name == "echo"
        assert recorded_result.status == ExecutionStatus.SUCCESS

    async def test_skips_non_base_tool_objects_with_warning(
        self, fake_langchain, gantry: AgentGantry, caplog: pytest.LogCaptureFixture
    ) -> None:
        valid = _FakeLCBaseTool(
            name="valid_tool", description="A valid tool for testing.", fn=lambda: "ok"
        )
        with caplog.at_level(logging.WARNING, logger=_IMPORTERS_LOGGER):
            count = await register_langchain_tools(gantry, [valid, object(), "not-a-tool"])
        assert count == 1
        assert sum("is not a langchain_core BaseTool" in m for m in caplog.messages) == 2

    async def test_skips_tool_that_fails_conversion_with_warning(
        self, fake_langchain, gantry: AgentGantry, caplog: pytest.LogCaptureFixture
    ) -> None:
        # "register" collides with one of Gantry's reserved tool names, so
        # ToolDefinition construction raises -- this must be caught and
        # skipped, not propagated out of register_langchain_tools().
        bad = _FakeLCBaseTool(
            name="register", description="Uses a reserved Gantry tool name.", fn=lambda: "x"
        )
        good = _FakeLCBaseTool(
            name="fine_tool", description="A perfectly fine tool.", fn=lambda: "ok"
        )
        with caplog.at_level(logging.WARNING, logger=_IMPORTERS_LOGGER):
            count = await register_langchain_tools(gantry, [bad, good])
        assert count == 1
        assert any("conversion failed" in m for m in caplog.messages)

    async def test_duplicate_normalized_names_collide_and_second_is_skipped(
        self, fake_langchain, gantry: AgentGantry, caplog: pytest.LogCaptureFixture
    ) -> None:
        t1 = _FakeLCBaseTool(
            name="Get Weather", description="First get weather tool.", fn=lambda: "a"
        )
        t2 = _FakeLCBaseTool(
            name="get-weather",
            description="Second get weather tool, colliding name.",
            fn=lambda: "b",
        )
        with caplog.at_level(logging.WARNING, logger=_IMPORTERS_LOGGER):
            count = await register_langchain_tools(gantry, [t1, t2])
        assert count == 1
        assert any("collides with another" in m for m in caplog.messages)

    async def test_empty_tools_raises(self, fake_langchain, gantry: AgentGantry) -> None:
        with pytest.raises(ValueError, match="requires at least one tool"):
            await register_langchain_tools(gantry, [])

    async def test_missing_langchain_raises_import_error(
        self, monkeypatch: pytest.MonkeyPatch, gantry: AgentGantry
    ) -> None:
        monkeypatch.setitem(sys.modules, "langchain_core", None)
        monkeypatch.setitem(sys.modules, "langchain_core.tools", None)
        with pytest.raises(ImportError, match="langchain-core"):
            await register_langchain_tools(gantry, [object()])


# --------------------------------------------------------------------------- #
# CrewAI — stub-based
# --------------------------------------------------------------------------- #


class _FakeCrewBaseTool:
    """Stand-in for ``crewai.tools.BaseTool`` with a synchronous ``_run``."""

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: Any = None,
        impl: Any = None,
        result_as_answer: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.result_as_answer = result_as_answer
        self._impl = impl or (lambda **kwargs: None)

    def _run(self, **kwargs: Any) -> Any:
        return self._impl(**kwargs)


class _FakeAsyncCrewBaseTool(_FakeCrewBaseTool):
    """Stand-in for a crewai ``BaseTool`` whose author defined an async ``_run``."""

    async def _run(self, **kwargs: Any) -> Any:  # type: ignore[override]
        return await self._impl(**kwargs)


class _MulArgs(BaseModel):
    x: int = Field(description="left operand")
    y: int = Field(description="right operand")


class _XArg(BaseModel):
    x: int


@pytest.fixture
def fake_crewai(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    pkg = types.ModuleType("crewai")
    tools_mod = types.ModuleType("crewai.tools")
    tools_mod.BaseTool = _FakeCrewBaseTool  # type: ignore[attr-defined]
    pkg.tools = tools_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "crewai", pkg)
    monkeypatch.setitem(sys.modules, "crewai.tools", tools_mod)
    return tools_mod


class TestCrewAIImporterStub:
    async def test_composite_description_is_recovered(
        self, fake_crewai, gantry: AgentGantry
    ) -> None:
        # Mirrors the composite string crewai.tools.BaseTool._generate_description
        # actually produces at construction time.
        composite = (
            "Tool Name: Multiply Numbers\n"
            "Tool Arguments: {'x': {'description': 'left', 'type': 'int'}}\n"
            "Tool Description: Multiply two integers together for testing."
        )
        tool = _FakeCrewBaseTool(
            name="Multiply Numbers",
            description=composite,
            args_schema=_MulArgs,
            impl=lambda x, y: x * y,
        )
        count = await register_crewai_tools(gantry, [tool])
        assert count == 1
        await gantry.sync()

        gtool = await gantry.get_tool("multiply_numbers", "crewai")
        assert gtool is not None
        assert gtool.description == "Multiply two integers together for testing."
        assert gtool.parameters_schema["properties"]["x"]["type"] == "integer"
        assert gtool.parameters_schema["properties"]["y"]["type"] == "integer"
        assert gtool.source == ToolSource.FRAMEWORK
        assert gtool.metadata["framework"] == "crewai"

    async def test_description_without_marker_falls_back_to_raw_string(
        self, fake_crewai, gantry: AgentGantry
    ) -> None:
        tool = _FakeCrewBaseTool(
            name="plain", description="A perfectly plain description.", impl=lambda: "x"
        )
        count = await register_crewai_tools(gantry, [tool])
        assert count == 1
        await gantry.sync()

        gtool = await gantry.get_tool("plain", "crewai")
        assert gtool is not None
        assert gtool.description == "A perfectly plain description."

    async def test_sync_run_executes(self, fake_crewai, gantry: AgentGantry) -> None:
        tool = _FakeCrewBaseTool(
            name="sync_tool",
            description="A synchronous tool for testing.",
            args_schema=_XArg,
            impl=lambda x: x * 2,
        )
        await register_crewai_tools(gantry, [tool])
        await gantry.sync()

        result = await gantry.execute(ToolCall(tool_name="sync_tool", arguments={"x": 5}))
        assert result.status == ExecutionStatus.SUCCESS
        assert result.result == 10

    async def test_async_run_is_awaited_directly(self, fake_crewai, gantry: AgentGantry) -> None:
        async def _double(x: int) -> int:
            return x * 2

        tool = _FakeAsyncCrewBaseTool(
            name="async_tool",
            description="An asynchronous tool for testing.",
            args_schema=_XArg,
            impl=_double,
        )
        await register_crewai_tools(gantry, [tool])
        await gantry.sync()

        result = await gantry.execute(ToolCall(tool_name="async_tool", arguments={"x": 5}))
        assert result.status == ExecutionStatus.SUCCESS
        assert result.result == 10

    async def test_skips_non_base_tool_with_warning(
        self, fake_crewai, gantry: AgentGantry, caplog: pytest.LogCaptureFixture
    ) -> None:
        valid = _FakeCrewBaseTool(
            name="valid", description="A valid crewai tool for testing.", impl=lambda: "ok"
        )
        with caplog.at_level(logging.WARNING, logger=_IMPORTERS_LOGGER):
            count = await register_crewai_tools(gantry, [valid, 123])
        assert count == 1
        assert any("is not a crewai.tools.BaseTool" in m for m in caplog.messages)

    async def test_empty_tools_raises(self, fake_crewai, gantry: AgentGantry) -> None:
        with pytest.raises(ValueError, match="requires at least one tool"):
            await register_crewai_tools(gantry, [])


# --------------------------------------------------------------------------- #
# LlamaIndex — stub-based
# --------------------------------------------------------------------------- #


class _FakeToolMetadata:
    """Stand-in for ``llama_index.core.tools.types.ToolMetadata``."""

    def __init__(
        self,
        name: str,
        description: str,
        fn_schema: type[BaseModel] | None = None,
        return_direct: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.fn_schema = fn_schema
        self.return_direct = return_direct

    def get_parameters_dict(self) -> dict[str, Any]:
        if self.fn_schema is not None:
            schema = self.fn_schema.model_json_schema()
            schema.pop("title", None)
            return schema
        # Mirrors LlamaIndex's own fallback for fn_schema=None: a single
        # generic string input.
        return {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        }


class _FakeToolOutput:
    """Stand-in for ``llama_index.core.tools.types.ToolOutput``."""

    def __init__(self, raw_output: Any) -> None:
        self.raw_output = raw_output


class _FakeFunctionTool:
    """Stand-in for ``llama_index.core.tools.FunctionTool``."""

    def __init__(self, metadata: _FakeToolMetadata, fn: Any = None, afn: Any = None) -> None:
        self.metadata = metadata
        self._fn = fn
        self._afn = afn

    async def acall(self, *args: Any, **kwargs: Any) -> _FakeToolOutput:
        if self._afn is not None:
            return _FakeToolOutput(await self._afn(**kwargs))
        return _FakeToolOutput(self._fn(**kwargs))

    def call(self, *args: Any, **kwargs: Any) -> _FakeToolOutput:
        return _FakeToolOutput(self._fn(**kwargs))


class _AddArgs(BaseModel):
    a: int
    b: int


@pytest.fixture
def fake_llamaindex(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    li_pkg = types.ModuleType("llama_index")
    li_core = types.ModuleType("llama_index.core")
    li_tools = types.ModuleType("llama_index.core.tools")
    li_tools.FunctionTool = _FakeFunctionTool  # type: ignore[attr-defined]
    li_core.tools = li_tools  # type: ignore[attr-defined]
    li_pkg.core = li_core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_index", li_pkg)
    monkeypatch.setitem(sys.modules, "llama_index.core", li_core)
    monkeypatch.setitem(sys.modules, "llama_index.core.tools", li_tools)
    return li_tools


class TestLlamaIndexImporterStub:
    async def test_schema_and_raw_output_unwrapping(
        self, fake_llamaindex, gantry: AgentGantry
    ) -> None:
        metadata = _FakeToolMetadata(
            name="li_add", description="Add two numbers together for testing.", fn_schema=_AddArgs
        )
        tool = _FakeFunctionTool(metadata, fn=lambda a, b: a + b)
        count = await register_llamaindex_tools(gantry, [tool])
        assert count == 1
        await gantry.sync()

        gtool = await gantry.get_tool("li_add", "llamaindex")
        assert gtool is not None
        assert gtool.parameters_schema["properties"]["a"]["type"] == "integer"
        assert gtool.source == ToolSource.FRAMEWORK

        result = await gantry.execute(ToolCall(tool_name="li_add", arguments={"a": 2, "b": 3}))
        assert result.status == ExecutionStatus.SUCCESS
        # The raw Python int, unwrapped from ToolOutput.raw_output -- not a
        # stringified ToolOutput.content.
        assert result.result == 5
        assert isinstance(result.result, int)

    async def test_no_fn_schema_falls_back_to_default_input_schema(
        self, fake_llamaindex, gantry: AgentGantry
    ) -> None:
        metadata = _FakeToolMetadata(
            name="generic", description="A tool without a typed schema at all.", fn_schema=None
        )
        tool = _FakeFunctionTool(metadata, fn=lambda input: input.upper())
        count = await register_llamaindex_tools(gantry, [tool])
        assert count == 1
        await gantry.sync()

        gtool = await gantry.get_tool("generic", "llamaindex")
        assert gtool is not None
        assert gtool.parameters_schema["properties"]["input"]["type"] == "string"

    async def test_async_fn_executes(self, fake_llamaindex, gantry: AgentGantry) -> None:
        async def _amul(a: int, b: int) -> int:
            return a * b

        metadata = _FakeToolMetadata(
            name="li_mul",
            description="Multiply two numbers asynchronously for testing.",
            fn_schema=_AddArgs,
        )
        tool = _FakeFunctionTool(metadata, afn=_amul)
        await register_llamaindex_tools(gantry, [tool])
        await gantry.sync()

        result = await gantry.execute(ToolCall(tool_name="li_mul", arguments={"a": 3, "b": 4}))
        assert result.status == ExecutionStatus.SUCCESS
        assert result.result == 12

    async def test_skips_non_function_tool_with_warning(
        self, fake_llamaindex, gantry: AgentGantry, caplog: pytest.LogCaptureFixture
    ) -> None:
        metadata = _FakeToolMetadata(
            name="valid", description="A valid llamaindex tool for testing."
        )
        valid = _FakeFunctionTool(metadata, fn=lambda: "ok")
        with caplog.at_level(logging.WARNING, logger=_IMPORTERS_LOGGER):
            count = await register_llamaindex_tools(gantry, [valid, "nope"])
        assert count == 1
        assert any("is not a" in m and "FunctionTool" in m for m in caplog.messages)

    async def test_empty_tools_raises(self, fake_llamaindex, gantry: AgentGantry) -> None:
        with pytest.raises(ValueError, match="requires at least one tool"):
            await register_llamaindex_tools(gantry, [])


# --------------------------------------------------------------------------- #
# Real packages (skipped cleanly if not installed)
# --------------------------------------------------------------------------- #


class TestRealPackages:
    """Exercises the importers against the actual installed frameworks.

    Skipped cleanly (not failed) per-test when a framework isn't installed,
    via ``pytest.importorskip``, so this module still collects and passes in
    a minimal environment without the ``agent-frameworks`` extra.
    """

    async def test_langchain_real_tool_round_trip(self, gantry: AgentGantry) -> None:
        pytest.importorskip("langchain_core", reason="langchain-core not installed")
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def get_weather(city: str) -> str:
            """Get the current weather for a city."""
            return f"sunny in {city}"

        count = await register_langchain_tools(gantry, [get_weather], tags=["weather"])
        assert count == 1
        await gantry.sync()

        # Executable through gantry.execute -- security/retries/telemetry --
        # exactly like a @gantry.register-ed tool.
        result = await gantry.execute(
            ToolCall(tool_name="get_weather", arguments={"city": "Paris"})
        )
        assert result.status == ExecutionStatus.SUCCESS
        assert result.result == "sunny in Paris"

        telemetry: _RecordingTelemetry = gantry._telemetry  # type: ignore[assignment]
        assert any(c.tool_name == "get_weather" for c, _ in telemetry.executions)

        # Retrievable via semantic search after sync(), and re-exportable as
        # a provider dialect schema -- the full "register once, use
        # anywhere" round trip.
        found = await gantry.retrieve_tools("what's the weather like", limit=5)
        matching = [t for t in found if t["function"]["name"] == "get_weather"]
        assert matching, "imported LangChain tool was not retrievable after sync()"
        assert matching[0]["function"]["parameters"]["properties"]["city"]["type"] == "string"

    async def test_langchain_import_then_reexport_to_llamaindex(self, gantry: AgentGantry) -> None:
        """Full loop: import a LangChain tool, then re-export it as a native
        LlamaIndex FunctionTool via the existing export adapter, and execute
        *that* -- proving an imported tool is a first-class registry citizen,
        not a special case that only works for its originating framework."""
        pytest.importorskip("langchain_core", reason="langchain-core not installed")
        pytest.importorskip("llama_index.core", reason="llama-index-core not installed")
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def send_email(to: str, subject: str = "") -> str:
            """Send an email to a recipient with an optional subject."""
            return f"sent to {to}: {subject}"

        count = await register_langchain_tools(gantry, [send_email])
        assert count == 1
        await gantry.sync()

        from agent_gantry.integrations.frameworks.llamaindex import LlamaIndexAdapter

        native_tools = await LlamaIndexAdapter(gantry).select("send an email", limit=1)
        assert native_tools, "re-export produced no LlamaIndex tools"

        output = await native_tools[0].acall(to="a@b.com", subject="hi")
        assert output.raw_output == "sent to a@b.com: hi"

    async def test_crewai_real_tool(self, gantry: AgentGantry) -> None:
        pytest.importorskip("crewai", reason="crewai not installed")
        from crewai.tools import BaseTool as CrewBaseTool

        class MultiplyTool(CrewBaseTool):
            name: str = "Multiply Numbers"
            description: str = "Multiply two integers together."
            args_schema: type[BaseModel] = _MulArgs

            def _run(self, x: int, y: int) -> int:
                return x * y

        count = await register_crewai_tools(gantry, [MultiplyTool()])
        assert count == 1
        await gantry.sync()

        gtool = await gantry.get_tool("multiply_numbers", "crewai")
        assert gtool is not None
        assert gtool.description == "Multiply two integers together."
        assert gtool.parameters_schema["properties"]["x"]["type"] == "integer"

        result = await gantry.execute(
            ToolCall(tool_name="multiply_numbers", arguments={"x": 6, "y": 7})
        )
        assert result.status == ExecutionStatus.SUCCESS
        assert result.result == 42

    async def test_llamaindex_real_tool(self, gantry: AgentGantry) -> None:
        pytest.importorskip("llama_index.core", reason="llama-index-core not installed")
        from llama_index.core.tools import FunctionTool

        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        native = FunctionTool.from_defaults(
            fn=add, name="li_add", description="Add two numbers together."
        )
        count = await register_llamaindex_tools(gantry, [native])
        assert count == 1
        await gantry.sync()

        result = await gantry.execute(ToolCall(tool_name="li_add", arguments={"a": 10, "b": 15}))
        assert result.status == ExecutionStatus.SUCCESS
        assert result.result == 25
