"""Pydantic AI + Agent-Gantry integration example.

Demonstrates both tiers `PydanticAIAdapter` exposes:

1. **Static tier** — ``PydanticAIAdapter.select(query, limit=...)`` runs semantic
   retrieval once and wraps the result as native ``pydantic_ai.tools.Tool``
   objects, ready to hand to ``Agent(model, tools=[...])``.
2. **Dynamic tier** — ``PydanticAIAdapter.toolset(...)`` returns a live
   ``pydantic_ai.toolsets.AbstractToolset``. Handed to ``Agent(model,
   toolsets=[...])``, it re-runs Gantry selection on *every* run/step (derived
   from the current ``RunContext``), so the tool surface tracks the
   conversation's focus without any manual re-selection code.

Both tiers run **fully offline**: this example uses ``pydantic_ai.models.test.
TestModel``, which simulates tool calls without hitting a real LLM, so no API
key is required anywhere in this file.

Pydantic AI is intentionally NOT part of any Agent-Gantry project extra (see
the comment block in ``pyproject.toml`` around lines 152-160 — it can't
co-resolve with the combined ``agent-frameworks`` extra). Install it
standalone:

    pip install agent-gantry pydantic-ai-slim

Run:

    python examples/agent_frameworks/pydantic_ai_example.py
"""

from __future__ import annotations

import asyncio

from agent_gantry import AgentGantry


def build_gantry() -> AgentGantry:
    gantry = AgentGantry()

    @gantry.register(tags=["weather"])
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"It is 21C and sunny in {city}."

    @gantry.register(tags=["email"])
    def send_email(to: str, body: str = "") -> str:
        """Compose and send an email message to a recipient."""
        return f"Email sent to {to}."

    @gantry.register(tags=["finance"])
    def convert_currency(amount: float, frm: str, to: str) -> str:
        """Convert an amount of money from one currency to another."""
        return f"{amount} {frm} = {amount * 1.1:.2f} {to}"

    return gantry


async def main() -> None:
    gantry = build_gantry()
    await gantry.sync()

    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel
    except ImportError as exc:
        print(
            f"Pydantic AI is not installed ({exc}).\n"
            "Install it with `pip install pydantic-ai-slim` to run this example."
        )
        return

    from agent_gantry.pydantic_ai import PydanticAIAdapter

    adapter = PydanticAIAdapter(gantry)

    # --- 1. Static tier: select once, get native pydantic_ai.tools.Tool ----- #
    query = "what's the weather in Paris?"
    # Lowering threshold for SimpleEmbedder compatibility in this example.
    static_tools = await adapter.select(query, limit=2, score_threshold=0.1)
    print(f"[static] selected {len(static_tools)} tool(s) for {query!r}:")
    for tool in static_tools:
        print(f"  - {tool.name}: {tool.description}")

    static_agent = Agent(TestModel(), tools=static_tools)
    static_result = await static_agent.run(query)
    print(f"[static] TestModel run output: {static_result.output}\n")

    # --- 2. Dynamic tier: a live AbstractToolset re-selects every run ------- #
    # One toolset, reused across two runs with very different intents — the
    # tool surface Pydantic AI sees changes each time, driven purely by the
    # run's prompt (no manual re-selection code on our side).
    toolset = adapter.toolset(limit=1, score_threshold=0.1)
    dynamic_agent = Agent(TestModel(), toolsets=[toolset])

    weather_run = await dynamic_agent.run("what's the weather in Tokyo?")
    print(f"[dynamic] weather run output: {weather_run.output}")

    email_run = await dynamic_agent.run("send an email to my manager")
    print(f"[dynamic] email run output:   {email_run.output}")

    print(
        "\nEvery call above (static Tool and dynamic toolset alike) routes "
        "through gantry.execute, so retries, timeouts, circuit breakers, and "
        "the security policy all still apply."
    )


if __name__ == "__main__":
    asyncio.run(main())
