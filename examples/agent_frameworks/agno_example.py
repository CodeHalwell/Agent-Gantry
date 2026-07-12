"""Agno (formerly Phidata) + Agent-Gantry integration example.

Demonstrates both tiers `AgnoAdapter` exposes:

1. **Static tier** — ``AgnoAdapter.select(query, limit=...)`` runs semantic
   retrieval once and wraps the result as native ``agno.tools.function.
   Function`` objects, ready to hand to ``agno.agent.Agent(tools=[...])``.
2. **Dynamic tier** — ``AgnoAdapter.agent_builder(...)`` returns a
   ``GantryLiveAgnoAgent``. Agno fixes an agent's tools at construction time
   (there is no native per-turn hook), so this builder rebuilds a *fresh*
   ``agno.agent.Agent`` for every call via ``await builder.build(query)``,
   re-selecting tools for that call's query each time — as dynamic as Agno
   permits.

Tool selection and native-tool wrapping run with no API key. Building an Agno
model and actually running the agent (``agent.arun(...)``) needs a real LLM,
so that step is gated behind ``OPENAI_API_KEY``.

Agno is intentionally NOT part of any Agent-Gantry project extra (see the
comment block in ``pyproject.toml`` around lines 152-160 — it can't
co-resolve with the combined ``agent-frameworks`` extra). Install it
standalone:

    pip install agent-gantry agno

Run:

    python examples/agent_frameworks/agno_example.py
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
        import agno  # noqa: F401
    except ImportError as exc:
        print(
            f"Agno is not installed ({exc}).\n"
            "Install it with `pip install agno` to run this example."
        )
        return

    from agent_gantry.agno import AgnoAdapter

    adapter = AgnoAdapter(gantry)

    # --- 1. Static tier: select once, get native agno.tools.function.Function #
    query = "what's the weather in Paris?"
    # Lowering threshold for SimpleEmbedder compatibility in this example.
    static_tools = await adapter.select(query, limit=2, score_threshold=0.1)
    print(f"[static] selected {len(static_tools)} tool(s) for {query!r}:")
    for tool in static_tools:
        print(f"  - {tool.name}: {tool.description}")

    # Functions are directly callable — no agent/model needed.
    print(f"[static] direct entrypoint call: {static_tools[0].entrypoint(city='Paris')}\n")

    # --- 2. Dynamic tier: per-call agent builder re-selects tools ----------- #
    # Agno freezes an agent's tools at construction, so the builder rebuilds a
    # fresh Agent per call. Here we just inspect the re-selected tool set
    # (select_tools) without building a full model-backed Agent.
    builder = adapter.agent_builder(limit=1, score_threshold=0.1)

    weather_tools = await builder.select_tools("what's the weather in Tokyo?")
    print(f"[dynamic] weather query -> {[t.name for t in weather_tools]}")

    email_tools = await builder.select_tools("send an email to my manager")
    print(f"[dynamic] email query   -> {[t.name for t in email_tools]}\n")

    # --- 3. Optional: build + run a real Agno agent (needs an LLM) ---------- #
    if os.environ.get("OPENAI_API_KEY"):
        from agno.models.openai import OpenAIChat

        live_builder = adapter.agent_builder(
            model=OpenAIChat(id="gpt-5.5"), limit=2, score_threshold=0.1
        )
        live_agent = await live_builder.build(query)
        print("[live] running an Agno agent with Gantry-selected tools...")
        response = await live_agent.arun(query)
        print(f"[live] response.content: {response.content}")
    else:
        print("(Set OPENAI_API_KEY to also run a live Agno agent turn.)")


if __name__ == "__main__":
    asyncio.run(main())
