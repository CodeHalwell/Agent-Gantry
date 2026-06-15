"""Per-turn (live) OpenAI Agents SDK provider tests against the real package.

These exercise the DEEP integration: the agent's ``tools`` list is re-selected by
Gantry as the conversation progresses, so the set of functions the model can
call changes turn by turn. Validated against an installed ``agents``
(``openai-agents``) without ever calling a real LLM — the per-turn mechanism
(the ``RunHooks.on_llm_start`` hook and the ``GantryAgentSession`` refresh) is
driven directly.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("agents")

from agents import Agent, FunctionTool  # noqa: E402

from agent_gantry import AgentGantry  # noqa: E402
from agent_gantry.adapters.embedders.simple import SimpleEmbedder  # noqa: E402
from agent_gantry.integrations.frameworks.openai_agents import (  # noqa: E402
    OpenAIAgentsAdapter,
)
from agent_gantry.integrations.frameworks.openai_agents_live import (  # noqa: E402
    GantryAgentSession,
)


@pytest.fixture
async def gantry():
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["weather"])
    def get_weather(city: str) -> str:
        "Get the current weather forecast for a city."
        return f"sunny in {city}"

    @g.register(tags=["email"])
    def send_email(to: str, body: str = "") -> str:
        "Send an email message to a recipient."
        return f"sent:{to}"

    @g.register(tags=["math"])
    def add(a: int, b: int) -> int:
        "Add two integers together and return the sum."
        return a + b

    await g.sync()
    return g


def _tool_names(agent: Agent) -> set[str]:
    return {t.name for t in agent.tools if isinstance(t, FunctionTool)}


async def test_hook_reselects_agent_tools_per_turn(gantry):
    """on_llm_start re-selects and rewrites agent.tools as the input changes."""
    agent = Agent(name="assistant", tools=[])
    hooks = OpenAIAgentsAdapter(gantry).run_hooks(agent, limit=1)

    # Turn 1: a weather conversation -> the weather tool is now in agent.tools.
    weather_input = [{"role": "user", "content": "what is the weather in Paris today"}]
    await hooks.on_llm_start(None, agent, None, weather_input)
    assert "get_weather" in _tool_names(agent)
    assert "send_email" not in _tool_names(agent)

    # Turn 2: the conversation pivots to email -> selection follows the pivot.
    email_input = [{"role": "user", "content": "please send an email to my boss"}]
    await hooks.on_llm_start(None, agent, None, email_input)
    assert "send_email" in _tool_names(agent)
    assert "get_weather" not in _tool_names(agent)


async def test_session_refresh_reselects_per_turn(gantry):
    """GantryAgentSession.refresh re-selects agent.tools before each run."""
    agent = Agent(name="assistant", tools=[])
    session = GantryAgentSession(agent, gantry, limit=1)

    await session.refresh("what is the weather forecast in London")
    assert _tool_names(agent) == {"get_weather"}

    await session.refresh("send an email message to the team")
    assert _tool_names(agent) == {"send_email"}


async def test_refresh_accepts_message_history(gantry):
    """A conversation history derives the query via latest_activity."""
    agent = Agent(name="assistant", tools=[])
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi, how can I help?"},
        {"role": "user", "content": "please send an email to the team"},
    ]
    await OpenAIAgentsAdapter(gantry).refresh(agent, history, limit=1)
    assert "send_email" in _tool_names(agent)


async def test_selected_function_tool_routes_through_gantry(gantry):
    """A selected FunctionTool's on_invoke_tool runs the real tool via gantry."""
    tools = await OpenAIAgentsAdapter(gantry).select_function_tools(
        "weather forecast for a city", limit=1
    )
    weather = next(t for t in tools if t.name == "get_weather")

    result = await weather.on_invoke_tool(None, json.dumps({"city": "Berlin"}))
    assert result == "sunny in Berlin"


async def test_tools_mutated_in_place(gantry):
    """Re-selection mutates the existing list object (SDK reads it each turn)."""
    agent = Agent(name="assistant", tools=[])
    original_list = agent.tools

    await OpenAIAgentsAdapter(gantry).refresh(agent, "add two numbers", limit=1)
    assert agent.tools is original_list  # same list object, updated contents
    assert _tool_names(agent) == {"add"}
