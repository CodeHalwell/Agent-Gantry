import pytest

from agent_gantry import AgentGantry, with_semantic_tools


@pytest.mark.asyncio
async def test_toolpack_can_be_loaded_and_filtered() -> None:
    gantry = await AgentGantry.from_modules(["examples.basics.toolpack"])

    tools = await gantry.retrieve_tools(
        "convert 10 kilometers to miles",
        limit=3,
        score_threshold=0.0,
    )

    tool_names = {tool["function"]["name"] for tool in tools}
    assert "convert_km_to_miles" in tool_names


@pytest.mark.asyncio
async def test_decorator_injects_relevant_tools() -> None:
    gantry = await AgentGantry.from_modules(["examples.basics.toolpack"])

    captured: dict[str, list[str]] = {}

    @with_semantic_tools(gantry, limit=2, score_threshold=0.0)
    async def chat(prompt: str, *, tools=None):
        captured["tools"] = [t["function"]["name"] for t in tools or []]
        return "ok"

    await chat("What is the current UTC time?")

    assert "current_utc_time" in captured["tools"]
    assert captured["tools"][0] == "current_utc_time"
