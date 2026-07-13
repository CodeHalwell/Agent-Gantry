"""OpenAI Agents SDK + Agent-Gantry integration example.

Demonstrates the tiers `OpenAIAgentsAdapter` exposes:

1. **Static tier** — ``OpenAIAgentsAdapter.select(query, limit=...)`` runs
   semantic retrieval once and wraps the result as native
   ``agents.FunctionTool`` objects, ready to hand to ``Agent(tools=[...])``.
2. **Dynamic tier** — the SDK reads ``agent.tools`` fresh on every turn, so
   Gantry can re-select mid-conversation via two complementary primitives:

   - ``OpenAIAgentsAdapter.refresh(agent, query_or_input)`` re-selects tools
     for the given text/messages and rewrites ``agent.tools`` in place —
     usable standalone, no LLM call required.
   - ``OpenAIAgentsAdapter.run_hooks(agent)`` builds ``agents.RunHooks`` that
     perform that same refresh automatically before every model call
     (``on_llm_start``), giving intra-run, per-turn dynamism with zero
     application code between turns.

Building tools, refreshing ``agent.tools``, and constructing hooks all run
**without any API key** — only the very last step (actually calling
``agents.Runner.run`` against a real model) needs one, so that step is gated
behind ``OPENAI_API_KEY``.

The OpenAI Agents SDK is intentionally NOT part of any Agent-Gantry project
extra (see the comment block in ``pyproject.toml`` around lines 152-160 — it
can't co-resolve with the combined ``agent-frameworks`` extra). Install it
standalone:

    pip install agent-gantry openai-agents

Run:

    python examples/agent_frameworks/openai_agents_example.py
"""

from __future__ import annotations

import asyncio
import json
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
        from agents import Agent
    except ImportError as exc:
        print(
            f"The OpenAI Agents SDK is not installed ({exc}).\n"
            "Install it with `pip install openai-agents` to run this example."
        )
        return

    from agent_gantry.openai_agents import OpenAIAgentsAdapter

    adapter = OpenAIAgentsAdapter(gantry)

    # --- 1. Static tier: select once, get native agents.FunctionTool -------- #
    query = "what's the weather in Paris?"
    # Lowering threshold for SimpleEmbedder compatibility in this example.
    static_tools = await adapter.select(query, limit=2, score_threshold=0.1)
    print(f"[static] selected {len(static_tools)} tool(s) for {query!r}:")
    for tool in static_tools:
        print(f"  - {tool.name}: {tool.description}")

    result = await static_tools[0].on_invoke_tool(None, json.dumps({"city": "Paris"}))
    print(f"[static] direct on_invoke_tool call: {result}\n")

    # --- 2. Dynamic tier: refresh + hooks re-select as the turn changes ----- #
    agent = Agent(name="assistant", tools=[])

    weather_tools = await adapter.refresh(
        agent, "what's the weather in Tokyo?", limit=1, score_threshold=0.1
    )
    print(f"[dynamic] weather turn -> agent.tools = {[t.name for t in weather_tools]}")

    email_tools = await adapter.refresh(
        agent, "send an email to my manager", limit=1, score_threshold=0.1
    )
    print(f"[dynamic] email turn   -> agent.tools = {[t.name for t in email_tools]}")

    # RunHooks apply that same refresh automatically before every model call —
    # build it here to show it's ready to hand to Runner.run(..., hooks=...).
    hooks = adapter.run_hooks(agent, limit=1, score_threshold=0.1)
    print(f"[dynamic] built {type(hooks).__name__} for intra-run per-turn re-selection\n")

    # --- 3. Optional: an actual model run (needs a real LLM) ---------------- #
    if os.environ.get("OPENAI_API_KEY"):
        run_agent = Agent(name="assistant", tools=[])
        # GantryAgentSession re-selects agent.tools before each run() call and
        # installs the run_hooks above for intra-run dynamism.
        session = adapter.session(run_agent, limit=2, score_threshold=0.1)
        print("[live] running the OpenAI Agents SDK with Gantry-selected tools...")
        run_result = await session.run(query)
        print(f"[live] final_output: {run_result.final_output}")
    else:
        print("(Set OPENAI_API_KEY to also run a live agents.Runner.run(...) turn.)")


if __name__ == "__main__":
    asyncio.run(main())
