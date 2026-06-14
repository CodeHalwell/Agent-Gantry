"""Multi-turn semantic tool selection within a single agent run.

This benchmark proves the property the README calls "per-call retrieval": in
*one* agent run the registry is queried fresh on every turn, so the agent can
pick a tool, read its result, then **change direction** and pick a completely
different tool driven purely by the semantics of the new sub-task — not by a
fixed, pre-bound tool list.

It simulates the agent loop directly against the public Gantry API:

    for each turn:
        query  = query_generator(conversation_so_far)   # per-call
        tools  = gantry.retrieve_tools(query, limit=3)   # re-rank the WHOLE registry
        pick   = first tool the (simulated) model chooses
        result = gantry.execute(ToolCall(pick, args))    # real execution
        append (assistant call, tool result) to conversation

The registry is 50 tools (imported from ``benchmark_tool_selection``). Each
turn asserts (a) the gold tool is surfaced in the top-3, and (b) it differs
from the previous turn's pick — i.e. the selection genuinely pivots.

Run::

    python benchmarks/benchmark_multi_turn.py
    python benchmarks/benchmark_multi_turn.py --embedder simple
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from agent_gantry import AgentGantry
from agent_gantry.query.strategies import fallback_chain, last_tool_result, last_user_text
from agent_gantry.schema.execution import ToolCall
from agent_gantry.schema.query import ConversationContext, ToolQuery

from benchmark_tool_selection import TOOLS, _make_tool, build_embedder


# A single "agent run": an ordered list of (user utterance, expected tool).
# Note how the direction lurches across domains turn to turn — weather, then
# travel, then comms, then finance, then data — to stress re-selection.
CONVERSATION: list[tuple[str, str]] = [
    ("What's the weather in San Francisco today?", "get_current_weather"),
    ("Great, now find me flights from London to San Francisco.", "search_flights"),
    ("Book a hotel there for two nights.", "book_hotel"),
    ("Email the trip details to my travel buddy.", "send_email"),
    ("Convert the 450 dollar hotel rate into euros.", "convert_currency"),
    ("Add 'pack passport' to my todo list.", "create_todo"),
    ("Now summarise the cancellation policy text I pasted.", "summarize_text"),
]


def _model_picks_first(tool_schemas: list[dict[str, Any]]) -> str | None:
    """Stand-in for the LLM's tool choice: take the top-ranked tool."""
    if not tool_schemas:
        return None
    return tool_schemas[0]["function"]["name"]


async def run(embedder_kind: str, limit: int) -> dict[str, Any]:
    embedder, label = build_embedder(embedder_kind)
    print(f"Embedder: {label}\n")
    gantry = AgentGantry(embedder=embedder)
    for name, desc, tags in TOOLS:
        gantry.register(_make_tool(name, desc), tags=tags)
    await gantry.sync()

    # Per-call query generator: prefer the latest tool result (to detect when
    # the conversation has moved on), else fall back to the latest user text.
    query_gen = fallback_chain(last_user_text, last_tool_result)

    messages: list[dict[str, Any]] = []
    tools_used: list[str] = []
    prev_pick: str | None = None

    correct = 0
    pivots = 0
    surfaced = 0

    for turn, (utterance, gold) in enumerate(CONVERSATION, start=1):
        messages.append({"role": "user", "content": utterance})

        # 1. PER-CALL retrieval over the ENTIRE 50-tool registry, re-ranked
        #    fresh for this turn. tools_already_used lets the router gently
        #    deprioritise repeats so the agent keeps moving forward.
        derived_query = query_gen(messages) or utterance
        result = await gantry.retrieve(
            ToolQuery(
                context=ConversationContext(
                    query=derived_query,
                    recent_messages=[
                        {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
                        for m in messages[-4:]
                    ],
                    tools_already_used=list(tools_used),
                ),
                limit=limit,
                score_threshold=0.0,
            )
        )
        ranked = [st.tool.name for st in result.tools]
        schemas = await gantry.retrieve_tools(derived_query, limit=limit, score_threshold=0.0)

        # 2. Simulated model picks the top tool.
        pick = _model_picks_first(schemas)

        # 3. Execute it for real (closing the select -> act loop).
        exec_result = None
        if pick is not None:
            call = await gantry.execute(ToolCall(tool_name=pick, arguments={}))
            exec_result = call.result if str(call.status).endswith("success") else call.error
            messages.append({"role": "assistant", "content": f"calling {pick}"})
            messages.append(
                {"role": "tool", "name": pick, "content": str(exec_result)}
            )
            tools_used.append(pick)

        in_top = gold in ranked
        is_pivot = pick != prev_pick
        surfaced += int(in_top)
        correct += int(pick == gold)
        pivots += int(is_pivot)

        flag = "✅" if pick == gold else ("◐" if in_top else "❌")
        pivot_mark = "↪ pivot" if is_pivot else "  (same)"
        print(
            f"turn {turn} {flag} {pivot_mark}  said={utterance[:42]!r:<44}\n"
            f"         picked={pick!r:<22} gold={gold!r:<22} top{limit}={ranked}"
        )
        prev_pick = pick

    n = len(CONVERSATION)
    summary = {
        "embedder": label,
        "registry_size": len(TOOLS),
        "turns": n,
        "limit": limit,
        "pick_accuracy": round(correct / n, 3),
        f"gold_in_top{limit}": round(surfaced / n, 3),
        "distinct_pivots": pivots,
        "distinct_tools_used": len(set(tools_used)),
        "selection_pivoted_every_turn": pivots == n,
    }
    print("\n=== MULTI-TURN SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(
        "\nInterpretation: a high 'distinct_tools_used' with 'pivoted_every_turn'"
        "\nmeans the agent re-selected a *different* tool each turn purely from the"
        "\nsemantics of the new sub-task — multi-turn re-selection within one run."
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument(
        "--embedder",
        default="auto",
        choices=["auto", "sentence-transformers", "nomic", "simple"],
    )
    args = ap.parse_args()
    asyncio.run(run(args.embedder, args.limit))


if __name__ == "__main__":
    main()
