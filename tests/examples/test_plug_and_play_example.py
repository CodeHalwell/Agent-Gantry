from typing import Any

import pytest

from dotfetch import DotFetch, with_semantic_tools


@pytest.mark.asyncio
async def test_toolpack_can_be_loaded_and_filtered() -> None:
    fetcher = await DotFetch.from_modules(["examples.basics.toolpack"])

    tools = await fetcher.retrieve_tools(
        "convert 10 kilometers to miles",
        limit=3,
        score_threshold=0.1,
    )

    tool_names = {tool["function"]["name"] for tool in tools}
    assert "convert_km_to_miles" in tool_names


@pytest.mark.asyncio
async def test_decorator_injects_relevant_tools() -> None:
    fetcher = await DotFetch.from_modules(["examples.basics.toolpack"])

    captured: dict[str, list[str]] = {}

    @with_semantic_tools(fetcher, limit=2, score_threshold=0.1)
    async def chat(prompt: str, *, tools: list[dict[str, Any]] | None = None):
        captured["tools"] = [t["function"]["name"] for t in tools or []]
        return "ok"

    await chat("What is the current UTC time?")
    assert "current_utc_time" in captured["tools"]
