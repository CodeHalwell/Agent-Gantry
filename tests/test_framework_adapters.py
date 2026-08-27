import pytest

from agent_gantry import AgentGantry
from agent_gantry.integrations.framework_adapters import (
    _SUPPORTED_FRAMEWORKS,
    fetch_framework_tools,
)


@pytest.fixture
async def gantry_with_tool():
    gantry = AgentGantry()

    @gantry.register
    def ping(name: str) -> str:
        return f"hi {name}"

    await gantry.sync()
    return gantry


@pytest.mark.asyncio
async def test_fetch_framework_tools_returns_schema_openai_shape(gantry_with_tool):
    tools = await fetch_framework_tools(
        gantry_with_tool,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("framework", sorted(_SUPPORTED_FRAMEWORKS))
async def test_fetch_framework_tools_covers_every_native_framework_name(
    gantry_with_tool, framework
):
    """Every canonical framework name (matching the native adapter modules)
    must be accepted, not just the handful the legacy Literal used to cover.
    """
    tools = await fetch_framework_tools(
        gantry_with_tool,
        "ping the user",
        framework=framework,
        limit=1,
        score_threshold=0.0,
    )
    assert len(tools) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "legacy,canonical",
    [("crew_ai", "crewai")],
)
async def test_fetch_framework_tools_normalizes_legacy_names(gantry_with_tool, legacy, canonical):
    """Legacy spellings are accepted and behave identically to the canonical name."""
    legacy_tools = await fetch_framework_tools(
        gantry_with_tool, "ping the user", framework=legacy, limit=1, score_threshold=0.0
    )
    canonical_tools = await fetch_framework_tools(
        gantry_with_tool, "ping the user", framework=canonical, limit=1, score_threshold=0.0
    )
    assert legacy_tools == canonical_tools
