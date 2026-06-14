"""Multi-turn tool selection within a single agent run, for any framework.

``ToolRefresher`` re-ranks the *whole* registry on every turn, so an agent can
pick a tool, read its result, then **change direction** and pick a completely
different tool driven by the semantics of the new sub-task — not by a fixed,
pre-bound tool list. This generalizes the Microsoft Agent Framework provider's
per-call retrieval to any framework (or a hand-rolled loop).

This example runs offline (no LLM). A simulated model always picks the
top-ranked tool; the point is that the *selection* pivots turn to turn.

Run::

    python examples/frameworks/multi_turn_refresher_example.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.integrations import ToolRefresher


def build_gantry() -> AgentGantry:
    gantry = AgentGantry(embedder=SimpleEmbedder(dimension=128))
    tools = [
        ("get_weather", "Get the current weather and temperature for a city.", ["weather"]),
        ("search_flights", "Search for available flights between two airports.", ["travel"]),
        ("book_hotel", "Reserve a hotel room for a set of dates.", ["travel"]),
        ("send_email", "Compose and send an email message to a recipient.", ["email"]),
        ("convert_currency", "Convert money from one currency to another.", ["finance"]),
        ("create_todo", "Add a task item to the user's to-do list.", ["productivity"]),
    ]
    for name, desc, tags in tools:
        def make(n: str):
            def fn(arg: str = "") -> str:
                return f"{n} done"
            fn.__name__ = n
            fn.__doc__ = desc
            return fn
        gantry.register(make(name), tags=tags)
    return gantry


async def main() -> None:
    gantry = build_gantry()
    await gantry.sync()

    # Default query generator = fallback_chain(last_user_text, last_tool_result),
    # which is the right choice for user-driven pivots like this conversation.
    refresher = ToolRefresher(gantry, limit=3, dialect="openai")

    conversation = [
        "What's the weather in San Francisco today?",
        "Now find me flights from London to San Francisco.",
        "Book a hotel there for two nights.",
        "Email the trip details to my travel buddy.",
        "Convert the 450 dollar hotel rate into euros.",
        "Add 'pack passport' to my todo list.",
    ]

    messages: list[dict[str, Any]] = []
    picks: list[str] = []
    for turn, utterance in enumerate(conversation, start=1):
        messages.append({"role": "user", "content": utterance})

        # Re-select over the WHOLE registry for this turn.
        schemas = await refresher.refresh(messages)
        names = [s["function"]["name"] for s in schemas]
        pick = names[0] if names else None
        picks.append(pick or "—")

        print(f"turn {turn}: {utterance!r}")
        print(f"         top3={names}  -> picked={pick}")

        # Simulate executing the picked tool and feeding the result back.
        if pick:
            messages.append({"role": "assistant", "content": f"calling {pick}"})
            messages.append({"role": "tool", "name": pick, "content": f"{pick} done"})

    print(f"\nPicks across the run: {picks}")
    print(f"Distinct tools chosen: {len(set(picks))} / {len(conversation)} turns")
    print(f"Tools used (accumulated): {refresher.tools_used}")


if __name__ == "__main__":
    asyncio.run(main())
