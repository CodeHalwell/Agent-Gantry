"""Tests for the one-class LLM SDK adapters (dialect + limit fallback).

Each ``<Provider>Adapter`` wraps :meth:`DotFetch.retrieve_tools` with a
provider *dialect* baked in. These tests use a recording stand-in for the fetcher
(no embedder needed) to verify each adapter calls ``retrieve_tools`` with the
correct dialect and honours the adapter's ``default_limit`` — so the dialect
strings (including ``"groq"`` / ``"mistral"`` / ``"gemini"``) and the limit
fallback are exercised, not assumed.
"""

from __future__ import annotations

from typing import Any

import pytest

from dotfetch.anthropic import AnthropicAdapter
from dotfetch.gemini import GeminiAdapter
from dotfetch.groq import GroqAdapter
from dotfetch.integrations.frameworks.base import DEFAULT_TOOL_LIMIT
from dotfetch.mistral import MistralAdapter
from dotfetch.openai import OpenAIAdapter
from dotfetch.vertexai import VertexAIAdapter


class _RecordingDotFetch:
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
    fetcher = _RecordingDotFetch()
    result = await adapter_cls(fetcher).tools("what's the weather?")
    assert result == [{"ok": True}]
    assert len(fetcher.calls) == 1
    call = fetcher.calls[0]
    assert call["dialect"] == dialect
    assert call["query"] == "what's the weather?"


@pytest.mark.parametrize("adapter_cls, dialect", _ADAPTERS)
async def test_tools_honours_default_limit(adapter_cls: type, dialect: str) -> None:
    # default_limit falls back to the shared constant (not a hardcoded number,
    # so the static and live adapter families cannot drift apart again)
    g1 = _RecordingDotFetch()
    await adapter_cls(g1).tools("q")
    assert g1.calls[-1]["limit"] == DEFAULT_TOOL_LIMIT
    # a custom default_limit is respected
    g2 = _RecordingDotFetch()
    await adapter_cls(g2, default_limit=7).tools("q")
    assert g2.calls[-1]["limit"] == 7
    # an explicit limit overrides the default
    g3 = _RecordingDotFetch()
    await adapter_cls(g3, default_limit=7).tools("q", limit=2)
    assert g3.calls[-1]["limit"] == 2


async def test_openai_responses_dialect() -> None:
    fetcher = _RecordingDotFetch()
    await OpenAIAdapter(fetcher).responses_tools("q", limit=4)
    assert fetcher.calls[-1]["dialect"] == "openai_responses"
    assert fetcher.calls[-1]["limit"] == 4


async def test_tools_forwards_score_threshold_and_kwargs() -> None:
    fetcher = _RecordingDotFetch()
    await AnthropicAdapter(fetcher).tools("q", score_threshold=0.5, namespaces=["x"])
    call = fetcher.calls[-1]
    assert call["score_threshold"] == 0.5
    assert call["namespaces"] == ["x"]
