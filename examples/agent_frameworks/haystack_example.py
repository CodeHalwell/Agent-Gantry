"""Haystack 2.x + Agent-Gantry integration example.

Demonstrates the tiers `HaystackAdapter` exposes:

1. **Static tier** — ``HaystackAdapter.select(query, limit=...)`` runs
   semantic retrieval once and wraps the result as native
   ``haystack.tools.Tool`` objects, ready to hand to a
   ``haystack.components.tools.ToolInvoker`` or an ``OpenAIChatGenerator``.
2. **Dynamic tier** — Haystack fixes a ``ToolInvoker``'s tools at
   construction time (there is no native per-turn hook), so Gantry offers two
   per-call primitives:

   - ``HaystackAdapter.live_tools(query, limit=...)`` re-selects the raw
     ``Tool`` list for a given call — usable standalone.
   - ``HaystackAdapter.tool_invoker_builder(...)`` returns a
     ``GantryLiveHaystackToolInvoker`` that rebuilds a *fresh* ``ToolInvoker``
     per call via ``await builder.build(query)``.

Tool selection and native-tool wrapping run with no API key. Actually running
an ``OpenAIChatGenerator`` (the LLM turn that decides which tool to call)
needs a real key, so that step is gated behind ``OPENAI_API_KEY``.

Haystack is intentionally NOT part of any Agent-Gantry project extra (see the
comment block in ``pyproject.toml`` around lines 152-160 — it can't
co-resolve with the combined ``agent-frameworks`` extra). Install it
standalone:

    pip install agent-gantry haystack-ai

Run:

    python examples/agent_frameworks/haystack_example.py
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
        import haystack  # noqa: F401
    except ImportError as exc:
        print(
            f"Haystack is not installed ({exc}).\n"
            "Install it with `pip install haystack-ai` to run this example."
        )
        return

    from agent_gantry.haystack import HaystackAdapter

    adapter = HaystackAdapter(gantry)

    # --- 1. Static tier: select once, get native haystack.tools.Tool -------- #
    query = "what's the weather in Paris?"
    # Lowering threshold for SimpleEmbedder compatibility in this example.
    static_tools = await adapter.select(query, limit=2, score_threshold=0.1)
    print(f"[static] selected {len(static_tools)} tool(s) for {query!r}:")
    for tool in static_tools:
        print(f"  - {tool.name}: {tool.description}")

    # Tools are directly callable — no invoker/model needed.
    print(f"[static] direct function() call: {static_tools[0].function(city='Paris')}\n")

    # --- 2. Dynamic tier: per-call live_tools + ToolInvoker builder --------- #
    weather_tools = await adapter.live_tools(
        "what's the weather in Tokyo?", limit=1, score_threshold=0.1
    )
    print(f"[dynamic] weather query -> {[t.name for t in weather_tools]}")

    email_tools = await adapter.live_tools(
        "send an email to my manager", limit=1, score_threshold=0.1
    )
    print(f"[dynamic] email query   -> {[t.name for t in email_tools]}")

    # haystack 3.0 removed ToolInvoker (the Agent component owns tool
    # execution), so the per-call builder behaves differently by version:
    # 2.x -> a fresh ToolInvoker; 3.x -> a fresh Agent (needs chat_generator).
    try:
        from haystack.components.tools import ToolInvoker  # noqa: F401

        haystack_has_tool_invoker = True
    except ImportError:
        haystack_has_tool_invoker = False

    if haystack_has_tool_invoker:
        invoker_builder = adapter.tool_invoker_builder(limit=2, score_threshold=0.1)
        invoker = await invoker_builder.build(query)
        print(f"[dynamic] built a fresh ToolInvoker with tools: {[t.name for t in invoker.tools]}\n")
    else:
        print(
            "[dynamic] haystack >= 3.0 removed ToolInvoker; pass "
            "chat_generator=... to tool_invoker_builder() to build a "
            "per-call Agent instead (see the live section below).\n"
        )

    # --- 3. Optional: a real Haystack chat + tool-invocation round ---------- #
    if os.environ.get("OPENAI_API_KEY"):
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.dataclasses import ChatMessage

        generator = OpenAIChatGenerator(model="gpt-5.5")
        if haystack_has_tool_invoker:
            print("[live] asking the model, offering Gantry-selected tools...")
            reply = generator.run(messages=[ChatMessage.from_user(query)], tools=static_tools)
            replies = reply["replies"]
            if replies and replies[0].tool_calls:
                invocation = invoker.run(messages=replies)
                print(f"[live] tool invocation results: {invocation['tool_messages']}")
            else:
                print(f"[live] model replied directly: {replies[0].text}")
        else:
            # haystack >= 3.0: the builder constructs a per-call Agent that
            # both selects and executes the Gantry tools.
            agent_builder = adapter.tool_invoker_builder(
                limit=2, score_threshold=0.1, chat_generator=generator
            )
            agent = await agent_builder.build(query)
            print("[live] running a per-call haystack Agent with Gantry-selected tools...")
            result = agent.run(messages=[ChatMessage.from_user(query)])
            print(f"[live] agent reply: {result['last_message'].text}")
    else:
        print("(Set OPENAI_API_KEY to also run a live chat + tool-invocation round.)")


if __name__ == "__main__":
    asyncio.run(main())
