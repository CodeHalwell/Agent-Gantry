"""Tests for the deep, per-turn LangGraph dynamic-tool provider.

Exercises :meth:`LangGraphAdapter.react_agent` against the *real* installed
``langchain.agents.create_agent`` API surface (the ``langgraph.prebuilt.
create_react_agent`` replacement — the latter is removed in LangGraph 2.0): the
agent must re-select tools from Gantry on every model turn via a
``wrap_model_call`` middleware hook and bind exactly those tools to the chat
model (so the tools advertised to the LLM change turn-to-turn as the
conversation pivots).

No real LLM is called: a tiny stub ``BaseChatModel`` records which tools were
bound to it via ``.bind_tools`` and returns a fixed (tool-free) reply so the
agent loop terminates immediately.
"""

from __future__ import annotations

import warnings

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain")
pytest.importorskip("langchain_core")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.warnings import LangGraphDeprecatedSinceV10

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.langgraph import LangGraphAdapter


class RecordingChatModel(BaseChatModel):
    """Fake chat model that records the tools bound to it via ``bind_tools``.

    Returns a plain, tool-call-free ``AIMessage`` so the ReAct agent finishes
    after a single model turn (no real LLM is invoked).
    """

    # ``BaseChatModel`` is a pydantic model; declare the field so assignment is
    # allowed, and share one list across the original + every ``bind`` clone.
    bound_tool_names: list = []

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        self.bound_tool_names.append(
            [getattr(t, "name", getattr(t, "__name__", str(t))) for t in tools]
        )
        # ``.bind`` returns a Runnable that delegates ``_generate`` back to self,
        # which is all the ReAct loop needs to terminate.
        return self.bind(tools=tools)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="done"))]
        )

    @property
    def _llm_type(self) -> str:
        return "recording-fake"


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
        "Convert a monetary amount into another currency."
        return f"converted:{amount}:{to_currency}"

    await g.sync()
    return g


# -- per-turn selection function (real API surface, fake model) ------------- #


async def test_select_tools_for_weather_state(gantry):
    """Weather-oriented state selects the weather StructuredTool."""
    state = {"messages": [HumanMessage(content="what's the weather in Paris?")]}
    tools = await LangGraphAdapter(gantry).select_for_state(state, limit=1)
    assert tools, "expected at least one tool to be selected"
    assert tools[0].name == "get_weather"


async def test_select_tools_for_email_state(gantry):
    """Email-oriented state selects the email StructuredTool (per-turn pivot)."""
    state = {"messages": [HumanMessage(content="send an email to my boss")]}
    tools = await LangGraphAdapter(gantry).select_for_state(state, limit=1)
    assert tools, "expected at least one tool to be selected"
    assert tools[0].name == "send_email"


async def test_selected_tool_routes_through_gantry(gantry):
    """The selected StructuredTool executes via Gantry."""
    state = {"messages": [HumanMessage(content="weather forecast for Tokyo")]}
    tools = await LangGraphAdapter(gantry).select_for_state(state, limit=1)
    assert await tools[0].coroutine(city="Tokyo") == "weather:Tokyo:sunny"


# -- full compiled graph with the create_agent wrap_model_call middleware --- #


async def test_react_agent_binds_weather_tool_per_turn(gantry):
    """Driving the compiled agent binds the weather tool for a weather state."""
    model = RecordingChatModel()
    model.bound_tool_names.clear()
    agent = LangGraphAdapter(gantry).react_agent(model, limit=1)

    await agent.ainvoke(
        {"messages": [HumanMessage(content="what's the weather in Paris?")]}
    )

    assert model.bound_tool_names, "wrap_model_call middleware was never invoked"
    last_bound = model.bound_tool_names[-1]
    assert last_bound == ["get_weather"]


async def test_react_agent_binds_email_tool_per_turn(gantry):
    """Re-selection: an email state binds the email tool, not weather."""
    model = RecordingChatModel()
    model.bound_tool_names.clear()
    agent = LangGraphAdapter(gantry).react_agent(model, limit=1)

    await agent.ainvoke(
        {"messages": [HumanMessage(content="send an email to my boss")]}
    )

    assert model.bound_tool_names, "wrap_model_call middleware was never invoked"
    last_bound = model.bound_tool_names[-1]
    assert last_bound == ["send_email"]


async def test_react_agent_superset_covers_all_tools(gantry):
    """The static ToolNode superset includes every registered tool."""
    from agent_gantry.integrations.frameworks.langgraph_live import _all_tools

    superset = await _all_tools(gantry)
    names = {t.name for t in superset}
    assert {"get_weather", "send_email", "convert_currency"} <= names


# -- regression guard: the deprecated create_react_agent must stay gone ----- #


async def test_react_agent_build_does_not_use_deprecated_create_react_agent(gantry):
    """Building and driving the live agent must never touch ``create_react_agent``.

    ``langgraph.prebuilt.create_react_agent`` is deprecated in LangGraph 1.x
    (raising ``LangGraphDeprecatedSinceV10`` on use) and is REMOVED outright in
    LangGraph 2.0. This module migrated to ``langchain.agents.create_agent`` +
    a ``wrap_model_call`` middleware for per-turn tool re-selection; escalate
    the specific deprecation warning class to an error here so that if the
    deprecated call path ever sneaks back in, this test fails loudly instead of
    the warning getting lost in normal pytest output.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=LangGraphDeprecatedSinceV10)

        model = RecordingChatModel()
        agent = LangGraphAdapter(gantry).react_agent(model, limit=1)
        await agent.ainvoke(
            {"messages": [HumanMessage(content="what's the weather in Paris?")]}
        )
