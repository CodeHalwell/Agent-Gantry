"""Per-turn (live) OpenAI Agents SDK provider tests against the real package.

These exercise the DEEP integration: the agent's ``tools`` list is re-selected by
Gantry as the conversation progresses, so the set of functions the model can
call changes turn by turn. Validated against an installed ``agents``
(``openai-agents``) without ever calling a real LLM — the per-turn mechanism
(the ``RunHooks`` hooks and the ``GantryAgentSession`` refresh) is driven
directly.
"""

from __future__ import annotations

import json
import types
import warnings

import pytest

pytest.importorskip("agents")

from agents import Agent, FunctionTool  # noqa: E402

from agent_gantry import AgentGantry  # noqa: E402
from agent_gantry.adapters.embedders.simple import SimpleEmbedder  # noqa: E402
from agent_gantry.integrations.frameworks import (  # noqa: E402
    openai_agents as adapter_module,
)
from agent_gantry.integrations.frameworks import (  # noqa: E402
    openai_agents_live as live_module,
)
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


# --------------------------------------------------------------------------- #
# The first model request of a run
# --------------------------------------------------------------------------- #
# The SDK lists a turn's tools once, before the model call. From 0.23.1 it
# records the turn's input and fires ``on_agent_start`` first, so a hook (or an
# ``is_enabled`` gate) can select for the very first request; earlier releases
# list the tools before either. These drive the hooks directly so they hold on
# whichever SDK is installed; ``test_openai_agents_native.py`` runs the same
# paths through a real ``Runner.run``.


def _start(text: str) -> types.SimpleNamespace:
    """The context ``on_agent_start`` receives: the turn's input is on it."""
    return types.SimpleNamespace(turn_input=[{"role": "user", "content": text}])


def _count_retrievals(gantry) -> list[str]:
    queries: list[str] = []
    original = gantry.retrieve

    async def spy(query):
        queries.append(query.context.query)
        return await original(query)

    gantry.retrieve = spy
    return queries


async def test_hook_selects_when_the_agent_starts(gantry, monkeypatch):
    """``on_agent_start`` selects from the run input, ahead of the first request."""
    monkeypatch.setattr(live_module, "_first_request_sees_input", lambda: True)
    agent = Agent(name="assistant", tools=[])
    hooks = OpenAIAgentsAdapter(gantry).run_hooks(agent, limit=1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await hooks.on_agent_start(_start("what is the weather in Paris today"), agent)
    assert _tool_names(agent) == {"get_weather"}


async def test_hook_does_not_repeat_an_unchanged_selection(gantry, monkeypatch):
    """Agent start and the first model call see the same input: one retrieval."""
    monkeypatch.setattr(live_module, "_first_request_sees_input", lambda: True)
    agent = Agent(name="assistant", tools=[])
    hooks = OpenAIAgentsAdapter(gantry).run_hooks(agent, limit=1)
    queries = _count_retrievals(gantry)
    start = _start("what is the weather in Paris today")

    await hooks.on_agent_start(start, agent)
    await hooks.on_llm_start(None, agent, None, start.turn_input)
    assert len(queries) == 1

    pivot = [*start.turn_input, {"role": "user", "content": "please send an email to my boss"}]
    await hooks.on_llm_start(None, agent, None, pivot)
    assert len(queries) == 2
    assert _tool_names(agent) == {"send_email"}


async def test_hook_reselects_when_the_tools_were_rewritten_elsewhere(gantry, monkeypatch):
    """An unchanged query is only skipped while the tools are still that selection.

    A hooks object reused across runs, or a ``refresh`` for another query in
    between, leaves ``agent.tools`` holding something else under the same
    query text.
    """
    monkeypatch.setattr(live_module, "_first_request_sees_input", lambda: True)
    agent = Agent(name="assistant", tools=[])
    adapter = OpenAIAgentsAdapter(gantry)
    hooks = adapter.run_hooks(agent, limit=1)
    start = _start("what is the weather in Paris today")

    await hooks.on_agent_start(start, agent)
    assert _tool_names(agent) == {"get_weather"}

    await adapter.refresh(agent, "send an email message to the team", limit=1)
    assert _tool_names(agent) == {"send_email"}

    await hooks.on_agent_start(start, agent)  # same query as before, stale tools
    assert _tool_names(agent) == {"get_weather"}


async def test_hook_retries_after_a_failed_selection(gantry, monkeypatch):
    """A retrieval failure is not remembered as done: the next hook tries again."""
    monkeypatch.setattr(live_module, "_first_request_sees_input", lambda: True)
    agent = Agent(name="assistant", tools=[])
    hooks = OpenAIAgentsAdapter(gantry).run_hooks(agent, limit=1)
    original = gantry.retrieve
    start = _start("what is the weather in Paris today")

    async def down(query):
        raise RuntimeError("vector store unavailable")

    gantry.retrieve = down
    await hooks.on_agent_start(start, agent)
    assert agent.tools == []

    gantry.retrieve = original
    await hooks.on_llm_start(None, agent, None, start.turn_input)
    assert _tool_names(agent) == {"get_weather"}


@pytest.mark.parametrize("context", [None, types.SimpleNamespace(turn_input=[])])
async def test_hook_agent_start_without_input_keeps_the_tools(gantry, context):
    """No input on the context (older SDKs, a resumed run) must not clear the tools."""
    agent = Agent(name="assistant", tools=[])
    adapter = OpenAIAgentsAdapter(gantry)
    await adapter.refresh(agent, "add two numbers", limit=1)
    hooks = adapter.run_hooks(agent, limit=1)

    await hooks.on_agent_start(context, agent)
    assert _tool_names(agent) == {"add"}


async def test_hook_agent_start_leaves_other_agents_alone(gantry, monkeypatch):
    monkeypatch.setattr(live_module, "_first_request_sees_input", lambda: True)
    agent = Agent(name="assistant", tools=[])
    other = Agent(name="other", tools=[])
    hooks = OpenAIAgentsAdapter(gantry).run_hooks(agent, limit=1)

    await hooks.on_agent_start(_start("what is the weather in Paris today"), other)
    assert agent.tools == [] and other.tools == []


async def test_hook_warns_when_an_old_sdk_already_built_the_first_request(gantry, monkeypatch):
    """Before 0.23.1 the tools are listed before any hook: say so, do not hide it."""
    monkeypatch.setattr(live_module, "_first_request_sees_input", lambda: False)
    agent = Agent(name="assistant", tools=[])
    hooks = OpenAIAgentsAdapter(gantry).run_hooks(agent, limit=1)

    with pytest.warns(RuntimeWarning, match=r"adapter\.refresh\(agent, input\)"):
        await hooks.on_agent_start(_start("what is the weather in Paris today"), agent)
    assert _tool_names(agent) == {"get_weather"}  # still in force from the next turn


async def test_hook_is_quiet_on_an_old_sdk_when_the_caller_selected_first(gantry, monkeypatch):
    """``refresh`` then ``run_hooks`` is the supported pattern there: no warning."""
    monkeypatch.setattr(live_module, "_first_request_sees_input", lambda: False)
    agent = Agent(name="assistant", tools=[])
    adapter = OpenAIAgentsAdapter(gantry)
    prompt = "what is the weather in Paris today"
    await adapter.refresh(agent, prompt, limit=1)
    hooks = adapter.run_hooks(agent, limit=1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await hooks.on_agent_start(_start(prompt), agent)
    assert _tool_names(agent) == {"get_weather"}


@pytest.mark.parametrize(
    ("installed", "sees_input"),
    [
        ((0, 4, 2), False),
        ((0, 20, 0), False),
        ((0, 22, 9), False),
        ((0, 23, 0), False),
        ((0, 23, 1), True),
        ((0, 25), True),
        ((1, 0, 0), True),
        (None, True),  # unreadable version: never refuse on a guess
    ],
)
def test_first_request_support_follows_the_sdk_version(monkeypatch, installed, sees_input):
    monkeypatch.setattr(adapter_module, "_sdk_version", lambda: installed)
    assert adapter_module._first_request_sees_input() is sees_input


async def test_dynamic_tools_refuses_an_sdk_that_hides_the_first_turn(gantry, monkeypatch):
    """On such an SDK every gate would resolve False for request one: fail loudly."""
    monkeypatch.setattr(adapter_module, "_sdk_version", lambda: (0, 20, 0))
    with pytest.raises(ImportError, match=r"openai-agents>=0\.23\.1; 0\.20\.0 is installed"):
        await OpenAIAgentsAdapter(gantry).dynamic_tools(limit=1)


@pytest.mark.parametrize(
    ("reported", "parsed"),
    [
        ("0.23.1", (0, 23, 1)),
        ("0.23.1rc1", (0, 23, 1)),  # a pre-release of the floor has the new run loop
        ("0.24.0.dev3", (0, 24, 0)),
        ("0.22.0.post1", (0, 22, 0)),
        ("1.0", (1, 0)),
        ("unknown", None),
    ],
)
def test_sdk_version_reads_release_numbers(monkeypatch, reported, parsed):
    from importlib import metadata

    monkeypatch.setattr(metadata, "version", lambda name: reported)
    assert adapter_module._sdk_version() == parsed


def test_sdk_version_is_unknown_when_the_distribution_is_missing(monkeypatch):
    from importlib import metadata

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    assert adapter_module._sdk_version() is None
    assert adapter_module._first_request_sees_input() is True
