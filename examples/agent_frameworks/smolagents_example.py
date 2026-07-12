"""Smolagents + Agent-Gantry integration example.

Demonstrates both tiers `SmolagentsAdapter` exposes:

1. **Static tier** — ``SmolagentsAdapter.select(query, limit=...)`` runs
   semantic retrieval once and wraps the result as native
   ``smolagents.Tool`` objects, ready to hand to
   ``smolagents.ToolCallingAgent(tools=[...])``.
2. **Dynamic tier** — ``SmolagentsAdapter.agent_builder(...)`` returns a
   ``GantryLiveSmolAgent``. Smolagents fixes an agent's tools at construction
   time (there is no native per-turn hook), so this builder rebuilds a
   *fresh* agent for every call via ``await builder.build(query)``,
   re-selecting tools for that call's query each time — as dynamic as
   smolagents permits.

Tool selection and native-tool wrapping run with no API key. Building a
smolagents model and actually running the agent (``agent.run(...)``) needs a
real LLM, so that step is gated behind ``OPENAI_API_KEY``.

Smolagents is intentionally NOT part of any Agent-Gantry project extra (see
the comment block in ``pyproject.toml`` around lines 152-160 — it can't
co-resolve with the combined ``agent-frameworks`` extra). Install it
standalone:

    pip install agent-gantry smolagents

Run:

    python examples/agent_frameworks/smolagents_example.py
"""

from __future__ import annotations

import asyncio
import os

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
        import smolagents  # noqa: F401
    except ImportError as exc:
        print(
            f"Smolagents is not installed ({exc}).\n"
            "Install it with `pip install smolagents` to run this example."
        )
        return

    from agent_gantry.smolagents import SmolagentsAdapter

    adapter = SmolagentsAdapter(gantry)

    # --- 1. Static tier: select once, get native smolagents.Tool objects ---- #
    query = "what's the weather in Paris?"
    # Lowering threshold for SimpleEmbedder compatibility in this example.
    static_tools = await adapter.select(query, limit=2, score_threshold=0.1)
    print(f"[static] selected {len(static_tools)} tool(s) for {query!r}:")
    for tool in static_tools:
        print(f"  - {tool.name}: {tool.description}")

    # Tools are directly callable via forward() — no agent/model needed.
    print(f"[static] direct forward() call: {static_tools[0].forward(city='Paris')}\n")

    # --- 2. Dynamic tier: per-call agent builder re-selects tools ----------- #
    # Smolagents freezes an agent's tools at construction, so the builder
    # rebuilds a fresh agent per call. Here we just inspect the re-selected
    # tool set (select_tools) without building a full model-backed agent.
    builder = adapter.agent_builder(limit=1, score_threshold=0.1)

    weather_tools = await builder.select_tools("what's the weather in Tokyo?")
    print(f"[dynamic] weather query -> {[t.name for t in weather_tools]}")

    email_tools = await builder.select_tools("send an email to my manager")
    print(f"[dynamic] email query   -> {[t.name for t in email_tools]}\n")

    # --- 3. Optional: build + run a real smolagents agent (needs an LLM) ---- #
    if os.environ.get("OPENAI_API_KEY"):
        from smolagents import OpenAIServerModel

        live_builder = adapter.agent_builder(
            model=OpenAIServerModel(model_id="gpt-5.5"), limit=2, score_threshold=0.1
        )
        live_agent = await live_builder.build(query)
        print("[live] running a smolagents ToolCallingAgent with Gantry-selected tools...")
        # MultiStepAgent.run is synchronous by design (smolagents' own loop).
        answer = live_agent.run(query)
        print(f"[live] answer: {answer}")
    else:
        print("(Set OPENAI_API_KEY to also run a live smolagents agent turn.)")


if __name__ == "__main__":
    asyncio.run(main())
