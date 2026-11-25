"""The module-level API: ``import dotfetch as dot`` with one shared instance."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Iterator

import pytest

import dotfetch as dot
from dotfetch import DotFetch, set_default_fetcher, with_semantic_tools
from dotfetch.adapters.embedders.simple import SimpleEmbedder
from dotfetch.core import default as default_module


@pytest.fixture(autouse=True)
def _fresh_default() -> Iterator[None]:
    """Each test starts with no shared instance and leaves none behind."""
    default_module._reset()
    yield
    default_module._reset()


def _offline() -> DotFetch:
    return dot.configure(embedder=SimpleEmbedder())


def _register_tools() -> None:
    @dot.register(tags=["weather"])
    def get_weather(city: str) -> str:
        """Get the current weather forecast for a city."""
        return f"Sunny in {city}"

    @dot.register
    def calculate_tax(amount: float) -> float:
        """Calculate sales tax for a purchase amount."""
        return amount * 0.2


def _names(tools: list[dict]) -> list[str]:
    return [tool["function"]["name"] for tool in tools]


class _OfflineDotFetch(DotFetch):
    """``DotFetch()`` with no arguments, minus the sentence-transformers import."""

    def __init__(self) -> None:
        super().__init__(embedder=SimpleEmbedder())


class TestSharedInstance:
    def test_created_once_on_first_use(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("dotfetch.core.fetcher.DotFetch", _OfflineDotFetch)
        assert default_module.current_default() is None
        first = dot.default_fetcher()
        assert isinstance(first, _OfflineDotFetch)
        assert dot.default_fetcher() is first

    def test_register_bare_and_with_options(self) -> None:
        fetcher = _offline()
        _register_tools()
        names = {tool.name for tool in fetcher.list_tools_sync()}
        assert names == {"get_weather", "calculate_tax"}
        weather = next(t for t in fetcher.list_tools_sync() if t.name == "get_weather")
        assert weather.tags == ["weather"]

    def test_register_returns_the_function(self) -> None:
        _offline()

        @dot.register
        def double(x: int) -> int:
            """Double a number."""
            return x * 2

        assert double(3) == 6

    def test_context_default_takes_precedence(self) -> None:
        shared = _offline()
        explicit = DotFetch(embedder=SimpleEmbedder())

        def in_context() -> DotFetch:
            set_default_fetcher(explicit)
            return dot.default_fetcher()

        assert contextvars.copy_context().run(in_context) is explicit
        assert dot.default_fetcher() is shared


class TestConfigure:
    def test_builds_from_arguments(self) -> None:
        embedder = SimpleEmbedder()
        fetcher = dot.configure(embedder=embedder)
        assert fetcher.embedder is embedder
        assert dot.default_fetcher() is fetcher

    def test_accepts_an_instance(self) -> None:
        mine = DotFetch(embedder=SimpleEmbedder())
        assert dot.configure(mine) is mine
        assert dot.default_fetcher() is mine

    def test_rejects_instance_and_arguments_together(self) -> None:
        with pytest.raises(TypeError, match="not both"):
            dot.configure(DotFetch(embedder=SimpleEmbedder()), embedder=SimpleEmbedder())

    def test_refuses_to_drop_registered_tools(self) -> None:
        _offline()
        _register_tools()
        with pytest.raises(RuntimeError, match="drop the 2 tool"):
            dot.configure(embedder=SimpleEmbedder())

    def test_same_instance_again_is_allowed(self) -> None:
        fetcher = _offline()
        _register_tools()
        assert dot.configure(fetcher) is fetcher


class TestFetch:
    async def test_module_fetch_syncs_and_ranks(self) -> None:
        _offline()
        _register_tools()
        tools = await dot.fetch("what is the weather forecast in Leeds", limit=1)
        assert _names(tools) == ["get_weather"]

    async def test_matches_retrieve_tools(self) -> None:
        fetcher = _offline()
        _register_tools()
        query = "sales tax on a purchase"
        assert await fetcher.fetch(query, limit=2) == await fetcher.retrieve_tools(query, limit=2)

    async def test_passes_dialect_through(self) -> None:
        _offline()
        _register_tools()
        tools = await dot.fetch("weather forecast", limit=1, dialect="anthropic")
        assert tools[0]["name"] == "get_weather"
        assert "input_schema" in tools[0]

    def test_shared_instance_survives_separate_event_loops(self) -> None:
        # Scripts and notebooks often call asyncio.run more than once. Two
        # explicit loops stand in for that without disturbing pytest-asyncio's.
        _offline()
        _register_tools()

        def run_in_new_loop(query: str) -> list[dict]:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(dot.fetch(query, limit=1))
            finally:
                loop.close()

        first = run_in_new_loop("weather forecast")
        second = run_in_new_loop("sales tax")
        assert _names(first) == ["get_weather"]
        assert _names(second) == ["calculate_tax"]


class TestExecuteToolCalls:
    async def test_runs_calls_from_a_response(self) -> None:
        _offline()
        _register_tools()
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Leeds"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        results = await dot.execute_tool_calls(response)
        assert len(results) == 1
        assert results[0]["tool_call_id"] == "call_1"
        assert "Sunny in Leeds" in results[0]["content"]


class TestWithSemanticTools:
    async def test_uses_the_shared_instance(self) -> None:
        _offline()
        _register_tools()

        @with_semantic_tools(limit=1)
        async def generate(prompt: str, *, tools: list | None = None) -> list | None:
            return tools

        tools = await generate("what is the weather forecast in Leeds")
        assert tools is not None
        assert _names(tools) == ["get_weather"]

    def test_still_raises_with_no_default_at_all(self) -> None:
        with pytest.raises(ValueError, match="No fetcher provided and no default set"):
            with_semantic_tools(limit=1)
