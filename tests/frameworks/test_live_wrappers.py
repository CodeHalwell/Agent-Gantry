"""Tests for the best-effort "live" per-call wrappers (``live_wrappers``).

CrewAI, Agno, Haystack and Smolagents all freeze an agent's tool list at
construction time, so the wrappers re-run Gantry selection for the query of
*each* top-level call and (re)build the agent / tool set for that call. These
tests exercise that per-call re-selection against the **real** installed
frameworks (each is ``pytest.importorskip``-ed): for every wrapper, selecting
for a weather query must surface the weather tool, and selecting for an email
query must surface the email tool.

Where building a full agent would need an LLM/model, only the *tools* are built
and inspected (via the wrapper's ``select_tools`` / the ``gantry_*_tools``
helper) — never a live agent run.
"""

from __future__ import annotations

import os

# Disable CrewAI / OpenTelemetry network calls before any framework import.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
# CrewAI's Agent constructor requires a model key to exist (no call is made —
# these tests only inspect the agent's tools). Provide a dummy so construction
# succeeds in CI where no real key is set.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used")

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.integrations.frameworks.live_wrappers import (
    GantryLiveAgnoAgent,
    GantryLiveCrewAgent,
    GantryLiveSmolAgent,
    gantry_crew_tools,
    gantry_haystack_tools,
)

WEATHER_QUERY = "what is the weather forecast for the city today"
EMAIL_QUERY = "send an email message to a recipient"


@pytest.fixture
async def gantry():
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["weather", "forecast"])
    def get_weather(city: str) -> str:
        "Get the current weather forecast for a city."
        return f"weather:{city}:sunny"

    @g.register(tags=["email", "communication"])
    def send_email(to: str, body: str = "") -> str:
        "Send an email message to a recipient."
        return f"sent:{to}"

    @g.register(tags=["currency", "exchange"])
    def convert_currency(amount: float, to_currency: str) -> str:
        "Convert a money amount into another currency."
        return f"{amount}:{to_currency}"

    await g.sync()
    return g


def _names(tools) -> set[str]:
    return {t.name for t in tools}


# --------------------------------------------------------------------------- #
# CrewAI
# --------------------------------------------------------------------------- #
async def test_crewai_reselects_tools_per_call(gantry):
    pytest.importorskip("crewai")

    agent_builder = GantryLiveCrewAgent(gantry, limit=1)

    weather_tools = await agent_builder.select_tools(WEATHER_QUERY)
    assert "get_weather" in _names(weather_tools)

    email_tools = await agent_builder.select_tools(EMAIL_QUERY)
    assert "send_email" in _names(email_tools)


async def test_crewai_helper_alias_reselects(gantry):
    pytest.importorskip("crewai")

    weather_tools = await gantry_crew_tools(gantry, WEATHER_QUERY, limit=1)
    assert "get_weather" in _names(weather_tools)

    email_tools = await gantry_crew_tools(gantry, EMAIL_QUERY, limit=1)
    assert "send_email" in _names(email_tools)


async def test_crewai_build_constructs_fresh_agent(gantry):
    crewai = pytest.importorskip("crewai")

    agent_builder = GantryLiveCrewAgent(gantry, limit=1)
    agent = await agent_builder.build(WEATHER_QUERY)

    assert isinstance(agent, crewai.Agent)
    assert "get_weather" in _names(agent.tools)


# --------------------------------------------------------------------------- #
# Agno
# --------------------------------------------------------------------------- #
async def test_agno_reselects_tools_per_call(gantry):
    pytest.importorskip("agno")

    agent_builder = GantryLiveAgnoAgent(gantry, limit=1)

    weather_tools = await agent_builder.select_tools(WEATHER_QUERY)
    assert "get_weather" in _names(weather_tools)

    email_tools = await agent_builder.select_tools(EMAIL_QUERY)
    assert "send_email" in _names(email_tools)


# --------------------------------------------------------------------------- #
# Haystack
# --------------------------------------------------------------------------- #
async def test_haystack_reselects_tools_per_call(gantry):
    pytest.importorskip("haystack")

    weather_tools = await gantry_haystack_tools(gantry, WEATHER_QUERY, limit=1)
    assert "get_weather" in _names(weather_tools)

    email_tools = await gantry_haystack_tools(gantry, EMAIL_QUERY, limit=1)
    assert "send_email" in _names(email_tools)


# --------------------------------------------------------------------------- #
# Smolagents
# --------------------------------------------------------------------------- #
async def test_smolagents_reselects_tools_per_call(gantry):
    pytest.importorskip("smolagents")

    agent_builder = GantryLiveSmolAgent(gantry, limit=1)

    weather_tools = await agent_builder.select_tools(WEATHER_QUERY)
    assert "get_weather" in _names(weather_tools)

    email_tools = await agent_builder.select_tools(EMAIL_QUERY)
    assert "send_email" in _names(email_tools)
