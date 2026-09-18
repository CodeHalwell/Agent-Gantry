"""
Tuning a selector's threshold from its own scores.

``threshold`` is the one knob on ``JevSelector`` that really matters, and you
should not have to guess it. Every pass returns a probability for *every*
candidate — not just the ones that won — so you can see exactly what a given
threshold would keep and what it would cut, on your own catalogue, before you
commit to a number.

This demo sweeps a range of thresholds over a labelled query set and prints the
trade so you can pick from data rather than from a blog post.

Run with::

    pip install agent-gantry[jev]
    export TYPESAFE_API_KEY=...
    python examples/routing/jev_threshold_tuning_demo.py
"""

from __future__ import annotations

import asyncio
import os

from agent_gantry import AgentGantry, JevSelector, SelectionCandidate

# Tools carry examples because that is what the retrieval path embeds and what
# a selector reads. Skimping here costs more accuracy than any model choice.
TOOLS = [
    ("numbers", "add_numbers", "Add two numbers together and return the sum.",
     ["add 3 and 4", "what is 10 plus 5"]),
    ("numbers", "divide_numbers", "Divide the first number by the second.",
     ["divide 10 by 2", "what is 12 over 4"]),
    ("comms", "send_email", "Send an email message to a named recipient.",
     ["email Bob about the meeting", "send a note to the team"]),
    ("comms", "list_inbox", "List the most recent messages sitting in the inbox.",
     ["show my messages", "any new mail"]),
    ("comms", "delete_message", "Permanently delete a message from the mailbox.",
     ["delete that email", "clear out old messages"]),
    ("db", "run_query", "Run a read-only SQL query against the analytics database.",
     ["how many users signed up", "count active accounts last month"]),
    ("web", "search_web", "Search the public web and return matching pages.",
     ["look this up online", "find recent news"]),
]

# The right answer for each query. An empty set means nothing should fire.
LABELLED = [
    ("what's 12 divided by 4?", {"numbers.divide_numbers"}),
    ("email Priya that the report is ready", {"comms.send_email"}),
    ("show me my messages, but do not send anything", {"comms.list_inbox"}),
    ("count active accounts last month", {"db.run_query"}),
    ("book me a flight to Berlin", set()),
]


def candidates() -> list[SelectionCandidate]:
    """Build the catalogue a selector sees."""
    return [
        SelectionCandidate(
            id=f"{ns}.{name}", name=f"{ns}.{name}", description=desc, group=ns, examples=ex
        )
        for ns, name, desc, ex in TOOLS
    ]


async def main() -> None:
    if not os.getenv("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set — this demo needs the live model to score anything.")
        return

    selector = JevSelector()
    catalogue = candidates()

    # One pass per query, keeping every score. Threshold 0.0 so nothing is cut
    # yet: we want the raw distribution to choose from.
    print("Scoring...\n")
    passes = []
    for query, expected in LABELLED:
        result = await selector.select(query, catalogue, limit=len(catalogue))
        if result.fallback:
            print(f"  selection failed ({result.reason}); cannot tune without scores")
            await selector.aclose()
            return
        passes.append((query, expected, result.scores))

    # What the model actually thought, sorted. This is the view worth keeping:
    # it shows the gap between the right answer and the runner-up.
    for query, expected, scores in passes:
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        print(f"  {query}")
        for name, score in ranked[:4]:
            mark = "<-- wanted" if name in expected else ""
            print(f"      {score:.2f}  {name} {mark}")
        if not expected:
            print("      (nothing here should have fired)")
        print()

    # Sweep. `kept` counts tools handed to the model; `missed` counts right
    # answers cut. You are choosing where to sit between the two.
    print(f"  {'threshold':<11}{'right kept':<13}{'wrong kept':<13}{'right cut':<11}")
    for step in range(0, 10):
        threshold = step / 10
        right = wrong = missed = 0
        for _query, expected, scores in passes:
            kept = {name for name, score in scores.items() if score >= threshold}
            right += len(kept & expected)
            wrong += len(kept - expected)
            missed += len(expected - kept)
        print(f"  {threshold:<11.1f}{right:<13}{wrong:<13}{missed:<11}")

    print(
        "\n  Pick the largest threshold that still cuts nothing you need. Every"
        "\n  'wrong kept' is a full tool schema in the prompt; every 'right cut'"
        "\n  is a task the agent can no longer do."
    )
    await selector.aclose()

    # The same numbers, applied. A gantry configured this way returns only what
    # cleared the bar — including nothing at all, which a top-k retriever
    # cannot express.
    gantry = AgentGantry(selector=JevSelector(threshold=0.5))
    try:
        for ns, name, desc, ex in TOOLS:

            async def handler(**kwargs: object) -> None:
                return None

            from agent_gantry.schema.tool import ToolDefinition

            await gantry.add_tool(
                ToolDefinition(
                    name=name, namespace=ns, description=desc, examples=ex,
                    parameters_schema={"type": "object", "properties": {}},
                ),
                handler,
            )
        print("\n  With threshold=0.5 through the normal retrieval API:")
        for query, _expected in LABELLED:
            tools = await gantry.retrieve_tools(query, limit=3)
            names = [tool["function"]["name"] for tool in tools] or ["(nothing relevant)"]
            print(f"      {query[:46]:<46} -> {names}")
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
