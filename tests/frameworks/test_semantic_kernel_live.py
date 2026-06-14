"""Per-turn (live) Semantic Kernel provider tests against the real package.

These exercise the DEEP integration: the kernel's ``gantry`` plugin is
re-selected by Gantry on every ``refresh`` call, so the set of functions the
model can call changes turn by turn. Validated against an installed
``semantic_kernel``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("semantic_kernel")

from semantic_kernel import Kernel  # noqa: E402

from agent_gantry import AgentGantry  # noqa: E402
from agent_gantry.adapters.embedders.simple import SimpleEmbedder  # noqa: E402
from agent_gantry.integrations.frameworks.semantic_kernel_live import (  # noqa: E402
    GantryFunctionProvider,
    refresh_kernel_tools,
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


def _plugin_function_names(kernel: Kernel, plugin_name: str = "gantry") -> set[str]:
    plugin = kernel.plugins.get(plugin_name)
    if plugin is None:
        return set()
    return set(plugin.functions.keys())


async def test_refresh_reselects_functions_per_turn(gantry):
    """Each refresh rebuilds the kernel's gantry plugin from a fresh selection."""
    kernel = Kernel()
    provider = GantryFunctionProvider(gantry, kernel, limit=1)

    # Turn 1: weather query -> weather function advertised, not email.
    await provider.refresh("what is the weather forecast in Paris today")
    names = _plugin_function_names(kernel)
    assert "get_weather" in names
    assert "send_email" not in names

    # Turn 2: email query -> selection pivots; email now present, weather gone.
    await provider.refresh("send an email message to my boss")
    names = _plugin_function_names(kernel)
    assert "send_email" in names
    assert "get_weather" not in names


async def test_refresh_accepts_message_history(gantry):
    """A conversation history derives the query via latest_activity."""
    kernel = Kernel()
    provider = GantryFunctionProvider(gantry, kernel, limit=1)

    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi, how can I help?"},
        {"role": "user", "content": "please send an email to the team"},
    ]
    await provider.refresh(history)
    assert "send_email" in _plugin_function_names(kernel)


async def test_selected_kernel_function_executes_through_gantry(gantry):
    """A selected KernelFunction runs the real tool via gantry on invoke."""
    kernel = Kernel()
    provider = GantryFunctionProvider(gantry, kernel, limit=1)
    await provider.refresh("weather forecast for a city")

    kf = kernel.plugins["gantry"].functions["get_weather"]
    result = await kf.invoke(kernel, city="London")
    assert result.value == "sunny in London"


async def test_refresh_kernel_tools_convenience(gantry):
    """The free function performs a single per-turn refresh equivalently."""
    kernel = Kernel()

    selected = await refresh_kernel_tools(gantry, kernel, "add two numbers", limit=1)
    assert "add" in selected
    assert _plugin_function_names(kernel) == {"add"}

    # A second call re-selects and replaces the plugin contents.
    await refresh_kernel_tools(gantry, kernel, "send an email", limit=1)
    assert _plugin_function_names(kernel) == {"send_email"}
