"""Multi-turn tool selection within one agent run — autonomous AND conversational.

``ToolRefresher`` re-ranks the *whole* registry on every turn, so an agent can
pick a tool, read its result, then move to a completely different tool driven by
the semantics of the new sub-task — not by a fixed, pre-bound tool list. Its
default query generator (:func:`agent_gantry.query.latest_activity`) is
recency-aware, so the **same** refresher works for both:

- **Autonomous agents / tool pipelines** — no new user input; each tool's
  *result* selects the next tool.
- **Conversational agents** — each new user message selects the next tool.

This example runs offline (no LLM). A simulated model always picks the
top-ranked tool; the point is that the *selection* advances correctly in both
modes. It uses the strongest embedder available (SentenceTransformers if
installed) and falls back to the offline ``SimpleEmbedder`` — note the toy
embedder is too coarse for the autonomous result-driven routing.

Run::

    python examples/frameworks/multi_turn_refresher_example.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_gantry import AgentGantry
from agent_gantry.integrations import ToolRefresher


def _embedder() -> tuple[Any, str]:
    try:
        import sentence_transformers  # noqa: F401  (validate the install actually imports)

        from agent_gantry.adapters.embedders.sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        return SentenceTransformersEmbedder("all-MiniLM-L6-v2"), "all-MiniLM-L6-v2"
    except Exception:  # noqa: BLE001
        from agent_gantry.adapters.embedders.simple import SimpleEmbedder

        return SimpleEmbedder(dimension=256), "SimpleEmbedder (toy)"


def _register(gantry: AgentGantry, tools: list[tuple[str, str, list[str], list[str]]]) -> None:
    for name, desc, tags, examples in tools:
        def make(n: str, d: str):
            def fn(arg: str = "") -> str:
                return f"{n} done"
            fn.__name__ = n
            fn.__doc__ = d
            return fn
        gantry.register(make(name, desc), tags=tags, examples=examples)


async def autonomous_pipeline_demo(embedder) -> None:
    """No new user input — each tool result drives the next selection."""
    print("\n=== AUTONOMOUS PIPELINE (result-driven) ===")
    gantry = AgentGantry(embedder=embedder)
    try:
        _register(gantry, [
            ("fetch_raw_data", "Fetch raw unprocessed data from the source system.", ["data"],
             ["pull the raw data from the source system", "fetch the unprocessed records"]),
            ("clean_dataset", "Clean and normalize a raw dataset, removing nulls and duplicates.", ["data"],
             ["clean up the nulls and duplicates", "normalize this raw dataset"]),
            ("train_model", "Train a machine learning model on a cleaned dataset.", ["ml"],
             ["train a model on the prepared data", "fit the model to the training set"]),
            ("evaluate_model", "Evaluate a trained machine learning model's accuracy metrics.", ["ml"],
             ["how accurate is the trained model", "evaluate the model's performance metrics"]),
            ("generate_report", "Generate a written report summarizing evaluation results.", ["report"],
             ["write up a report of the findings", "summarize the evaluation results"]),
            ("send_email", "Compose and send an email message to a recipient.", ["email"],
             ["email the results to the team", "send Sam a message"]),
        ])
        await gantry.sync()

        refresher = ToolRefresher(gantry, limit=3)  # default = latest_activity
        # Each result points forward at the next stage (how a real step hands off).
        forward = {
            "fetch_raw_data": "records have missing nulls and duplicate rows that must be cleaned and normalized",
            "clean_dataset": "the prepared training set is ready to fit and train a machine learning model",
            "train_model": "the fitted model now needs its accuracy and performance metrics evaluated",
            "evaluate_model": "please write a summary report describing the evaluation findings",
            "generate_report": "report complete",
        }

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Build and evaluate a model from the raw source data."}
        ]
        for step in range(5):
            schemas = await refresher.refresh(messages)
            names = [s["function"]["name"] for s in schemas]
            pick = names[0] if names else None
            print(f"  step {step + 1}: picked={pick}   (top3={names})")
            if not pick:
                break
            # Autonomously advance — NO new user message, only the tool result.
            messages.append({"role": "assistant", "content": f"calling {pick}"})
            messages.append({"role": "tool", "name": pick, "content": forward.get(pick, "done")})
    finally:
        await gantry.close()


async def conversational_demo(embedder) -> None:
    """New user message each turn — the user's request drives selection."""
    print("\n=== CONVERSATIONAL (user-driven) ===")
    gantry = AgentGantry(embedder=embedder)
    try:
        _register(gantry, [
            ("get_weather", "Get the current weather and temperature for a city.", ["weather"],
             ["what's the weather in San Francisco", "how hot is it in Madrid today"]),
            ("search_flights", "Search for available flights between two airports.", ["travel"],
             ["find me flights from London to San Francisco", "any flights to JFK tomorrow"]),
            ("book_hotel", "Reserve a hotel room for a set of dates.", ["travel"],
             ["book a hotel for two nights", "reserve me a room in Berlin"]),
            ("send_email", "Compose and send an email message to a recipient.", ["email"],
             ["email the trip details to my friend", "send Priya a message"]),
            ("convert_currency", "Convert money from one currency to another.", ["finance"],
             ["convert 450 dollars into euros", "how much is 100 GBP in USD"]),
            ("create_todo", "Add a task item to the user's to-do list.", ["productivity"],
             ["add 'pack umbrella' to my todo list", "remind me to call the bank"]),
        ])
        await gantry.sync()

        refresher = ToolRefresher(gantry, limit=3)
        conversation = [
            "What's the weather in San Francisco today?",
            "Now find me flights from London to San Francisco.",
            "Book a hotel there for two nights.",
            "Email the trip details to my travel buddy.",
            "Convert the 450 dollar hotel rate into euros.",
        ]
        messages: list[dict[str, Any]] = []
        for turn, utterance in enumerate(conversation, start=1):
            messages.append({"role": "user", "content": utterance})
            schemas = await refresher.refresh(messages)
            names = [s["function"]["name"] for s in schemas]
            pick = names[0] if names else None
            print(f"  turn {turn}: {utterance!r}\n           picked={pick}   (top3={names})")
            if pick:
                messages.append({"role": "assistant", "content": f"calling {pick}"})
                messages.append({"role": "tool", "name": pick, "content": f"{pick} done"})
    finally:
        await gantry.close()


async def main() -> None:
    embedder, label = _embedder()
    print(f"Embedder: {label}")
    await autonomous_pipeline_demo(embedder)
    await conversational_demo(embedder)


if __name__ == "__main__":
    asyncio.run(main())
