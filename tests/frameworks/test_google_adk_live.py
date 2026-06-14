"""Tests for the deep, per-turn Google ADK dynamic-tool provider.

Exercises :func:`gantry_before_model_callback` against the *real* installed
``google.adk`` contract. The callback is what ADK invokes via
``Agent(before_model_callback=...)`` before every model request: it must derive
the turn's query from the ``callback_context``, re-select tools from Gantry, and
inject their ``FunctionDeclaration``s into the ``LlmRequest`` the model sees this
turn (via :meth:`LlmRequest.append_tools`). Because selection is re-run per call,
the injected tool surface must change turn-to-turn as the user's intent changes.

We fabricate a minimal ``LlmRequest`` (it constructs empty) and a stand-in
callback context exposing ``user_content`` — the attribute the real ADK
``CallbackContext`` carries and our query-derivation reads — then assert the
weather tool is injected on a weather turn and the email tool on an email turn.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("google.adk")

from google.adk.models import LlmRequest
from google.genai import types

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.integrations.frameworks.google_adk_live import (
    gantry_adk_agent,
    gantry_before_model_callback,
)


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

    @g.register(tags=["math", "arithmetic"])
    def add(a: int, b: int) -> int:
        "Add two integers together."
        return a + b

    await g.sync()
    return g


def _context_with_text(text: str) -> SimpleNamespace:
    """A minimal stand-in for ADK's ``CallbackContext``.

    The real ``CallbackContext`` exposes ``user_content`` (a
    ``google.genai.types.Content``) and ``session``; our query derivation reads
    those. We fabricate just ``user_content`` here — the reliable fallback ADK
    always populates — and leave ``session`` absent.
    """
    content = types.Content(role="user", parts=[types.Part(text=text)])
    return SimpleNamespace(user_content=content, session=None)


def _injected_declaration_names(llm_request: LlmRequest) -> set[str]:
    """Names of the ``FunctionDeclaration``s injected into ``llm_request``."""
    names: set[str] = set()
    for tool in llm_request.config.tools or []:
        for decl in tool.function_declarations or []:
            names.add(decl.name)
    return names


async def test_callback_injects_weather_tool_for_weather_turn(gantry):
    callback = gantry_before_model_callback(gantry, limit=2)
    llm_request = LlmRequest()

    result = await callback(
        _context_with_text("what is the weather forecast in Paris today"),
        llm_request,
    )

    # Returning None lets ADK proceed with the mutated request.
    assert result is None

    injected = _injected_declaration_names(llm_request)
    assert "get_weather" in injected
    # Declarations are also registered for execution this turn.
    assert "get_weather" in llm_request.tools_dict


async def test_callback_reselects_email_tool_for_email_turn(gantry):
    callback = gantry_before_model_callback(gantry, limit=2)
    llm_request = LlmRequest()

    await callback(
        _context_with_text("send an email message to a recipient"),
        llm_request,
    )

    injected = _injected_declaration_names(llm_request)
    assert "send_email" in injected
    assert "send_email" in llm_request.tools_dict


async def test_per_turn_reselection_changes_tool_surface(gantry):
    """Each model request gets a freshly selected slice of tools."""
    callback = gantry_before_model_callback(gantry, limit=1)

    weather_req = LlmRequest()
    await callback(_context_with_text("weather forecast for the city"), weather_req)
    weather_injected = _injected_declaration_names(weather_req)

    email_req = LlmRequest()
    await callback(_context_with_text("send an email to a recipient"), email_req)
    email_injected = _injected_declaration_names(email_req)

    assert weather_injected == {"get_weather"}
    assert email_injected == {"send_email"}


async def test_empty_query_injects_nothing(gantry):
    callback = gantry_before_model_callback(gantry, limit=3)
    llm_request = LlmRequest()

    # No user_content and no session -> empty query -> no injection.
    await callback(SimpleNamespace(user_content=None, session=None), llm_request)

    assert _injected_declaration_names(llm_request) == set()
    assert llm_request.tools_dict == {}


async def test_gantry_adk_agent_builds_agent_with_no_static_tools(gantry):
    from google.adk.agents import Agent

    agent = gantry_adk_agent(
        gantry,
        model="gemini-2.0-flash",
        name="assistant",
        instruction="You are helpful.",
        limit=2,
    )

    assert isinstance(agent, Agent)
    assert agent.name == "assistant"
    # The agent ships with no statically registered tools; the callback injects
    # the relevant slice before each model request.
    assert list(agent.tools) == []
    assert agent.before_model_callback is not None
