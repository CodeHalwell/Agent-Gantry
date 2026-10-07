from typing import Any

import pytest

from agent_gantry import AgentGantry, with_semantic_tools


@pytest.mark.asyncio
async def test_toolpack_can_be_loaded_and_filtered() -> None:
    gantry = await AgentGantry.from_modules(["examples.basics.toolpack"])

    # The toolpack has only four tools, so a wide limit would prove little:
    # ask for the single best match. The cutoff stays at the 0.0 default, as in the
    # example, because the hash embedder's scores sit far below any useful threshold.
    tools = await gantry.retrieve_tools(
        "convert 10 kilometers to miles",
        limit=1,
        score_threshold=0.0,
    )

    assert [tool["function"]["name"] for tool in tools] == ["convert_km_to_miles"]


@pytest.mark.asyncio
async def test_decorator_injects_relevant_tools() -> None:
    gantry = await AgentGantry.from_modules(["examples.basics.toolpack"])

    captured: dict[str, list[str]] = {}

    @with_semantic_tools(gantry, limit=2, score_threshold=0.0)
    async def chat(prompt: str, *, tools: list[dict[str, Any]] | None = None):
        captured["tools"] = [t["function"]["name"] for t in tools or []]
        return "ok"

    await chat("What is the current UTC time?")
    assert "current_utc_time" in captured["tools"]
