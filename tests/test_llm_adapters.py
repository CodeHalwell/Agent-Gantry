"""Tests for the one-class LLM SDK adapters (dialect + limit fallback).

Each ``<Provider>Adapter`` wraps :meth:`AgentGantry.retrieve_tools` with a
provider *dialect* baked in. These tests use a recording stand-in for the gantry
(no embedder needed) to verify each adapter calls ``retrieve_tools`` with the
correct dialect and honours the adapter's ``default_limit`` — so the dialect
strings (including ``"groq"`` / ``"mistral"`` / ``"gemini"``) and the limit
fallback are exercised, not assumed.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_gantry.anthropic import AnthropicAdapter
from agent_gantry.gemini import GeminiAdapter
from agent_gantry.groq import GroqAdapter
from agent_gantry.mistral import MistralAdapter
from agent_gantry.openai import OpenAIAdapter
from agent_gantry.vertexai import VertexAIAdapter


class _RecordingGantry:
    """Minimal stand-in that records the kwargs passed to ``retrieve_tools``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def retrieve_tools(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append({"query": query, **kwargs})
        return [{"ok": True}]


# (adapter class, expected dialect)
_ADAPTERS = [
    (OpenAIAdapter, "openai"),
    (AnthropicAdapter, "anthropic"),
    (GeminiAdapter, "gemini"),
    (GroqAdapter, "groq"),
    (VertexAIAdapter, "gemini"),
    (MistralAdapter, "mistral"),
]


@pytest.mark.parametrize("adapter_cls, dialect", _ADAPTERS)
async def test_tools_uses_provider_dialect(adapter_cls: type, dialect: str) -> None:
    gantry = _RecordingGantry()
    result = await adapter_cls(gantry).tools("what's the weather?")
    assert result == [{"ok": True}]
    assert len(gantry.calls) == 1
    call = gantry.calls[0]
    assert call["dialect"] == dialect
    assert call["query"] == "what's the weather?"


@pytest.mark.parametrize("adapter_cls, dialect", _ADAPTERS)
async def test_tools_honours_default_limit(adapter_cls: type, dialect: str) -> None:
    # default_limit defaults to 3
    g1 = _RecordingGantry()
    await adapter_cls(g1).tools("q")
    assert g1.calls[-1]["limit"] == 3
    # a custom default_limit is respected
    g2 = _RecordingGantry()
    await adapter_cls(g2, default_limit=7).tools("q")
    assert g2.calls[-1]["limit"] == 7
    # an explicit limit overrides the default
    g3 = _RecordingGantry()
    await adapter_cls(g3, default_limit=7).tools("q", limit=2)
    assert g3.calls[-1]["limit"] == 2


async def test_openai_responses_dialect() -> None:
    gantry = _RecordingGantry()
    await OpenAIAdapter(gantry).responses_tools("q", limit=4)
    assert gantry.calls[-1]["dialect"] == "openai_responses"
    assert gantry.calls[-1]["limit"] == 4


async def test_tools_forwards_score_threshold_and_kwargs() -> None:
    gantry = _RecordingGantry()
    await AnthropicAdapter(gantry).tools("q", score_threshold=0.5, namespaces=["x"])
    call = gantry.calls[-1]
    assert call["score_threshold"] == 0.5
    assert call["namespaces"] == ["x"]


async def test_dialects_resolve_against_real_gantry() -> None:
    """Each adapter's dialect resolves end-to-end against a real gantry (not mocked).

    Confirms the dialect strings — including ``groq`` / ``mistral`` / ``gemini`` —
    are registered in the schema-dialect registry and produce real provider
    schemas (i.e. ``retrieve_tools(dialect=...)`` doesn't raise or mis-route).
    """
    from agent_gantry import AgentGantry

    gantry = AgentGantry()

    @gantry.register(tags=["weather"])
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return "sunny"

    await gantry.sync()

    for adapter_cls, _dialect in _ADAPTERS:
        tools = await adapter_cls(gantry).tools("what's the weather?", limit=1)
        assert isinstance(tools, list) and tools, f"{adapter_cls.__name__}: no tools"
        assert isinstance(tools[0], dict), f"{adapter_cls.__name__}: not a schema dict"
