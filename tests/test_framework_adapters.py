import pytest

from agent_gantry import AgentGantry
from agent_gantry.integrations.framework_adapters import fetch_framework_tools


@pytest.mark.asyncio
async def test_fetch_framework_tools_returns_schema_openai_shape():
    gantry = AgentGantry()

    @gantry.register
    def ping(name: str) -> str:
        return f"hi {name}"

    await gantry.sync()

    tools = await fetch_framework_tools(
        gantry,
        "ping the user",
        framework="langgraph",
        limit=1,
        score_threshold=0.0,
    )

    assert len(tools) == 1
    fn = tools[0]["function"]
    assert fn["name"] == "ping"
    assert "parameters" in fn


@pytest.mark.asyncio
async def test_fetch_framework_tools_invalid_framework_raises():
    gantry = AgentGantry()

    with pytest.raises(ValueError):
        await fetch_framework_tools(gantry, "q", framework="unknown")
