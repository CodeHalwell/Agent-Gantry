"""Semantic tool-selection benchmark: pick the best top-3 tools out of 50.

This benchmark measures how well Agent-Gantry's semantic router surfaces the
*right* tools when the registry is large (50 tools) and only a small slice
(top-3) is injected into the prompt. It reports retrieval-quality metrics and
latency, plus the context-window saving versus dumping all 50 schemas.

Metrics
-------
- ``top1``      : fraction of queries whose single best tool ranked #1.
- ``hit@3``     : fraction of queries whose best tool appeared in the top-3.
- ``MRR``       : mean reciprocal rank of the gold tool (1.0 == always #1).
- ``p50/p95``   : retrieval latency percentiles (ms), embedding excluded/
                  included depending on cache warmth — see notes printed.
- ``token_savings`` : prompt-token reduction of top-3 vs. all-50 schemas.

The benchmark is embedder-agnostic. It uses the strongest embedder available
in the environment (SentenceTransformers / Nomic), falling back to the
hash-based ``SimpleEmbedder`` (a toy — accuracy will be low, the harness still
runs). Pass ``--embedder`` to force one.

Run::

    python benchmarks/benchmark_tool_selection.py
    python benchmarks/benchmark_tool_selection.py --embedder sentence-transformers --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

from agent_gantry import AgentGantry


# --------------------------------------------------------------------------- #
# 50-tool registry across 10 domains. Each entry: (name, description, tags).
# --------------------------------------------------------------------------- #
TOOLS: list[tuple[str, str, list[str]]] = [
    # --- weather ---
    ("get_current_weather", "Get the current temperature and conditions for a city.", ["weather"]),
    ("get_weather_forecast", "Get a multi-day weather forecast for a location.", ["weather"]),
    ("get_air_quality", "Get the air quality index and pollutant levels for a city.", ["weather", "environment"]),
    ("get_uv_index", "Get the current ultraviolet radiation index for a location.", ["weather"]),
    ("get_marine_forecast", "Get sea state, wave height and tide times for a coastal area.", ["weather", "marine"]),
    # --- finance ---
    ("get_stock_price", "Look up the latest share price for a stock ticker symbol.", ["finance"]),
    ("convert_currency", "Convert an amount of money from one currency to another.", ["finance"]),
    ("calculate_sales_tax", "Calculate sales tax owed on a purchase amount.", ["finance", "tax"]),
    ("get_crypto_price", "Get the current market price of a cryptocurrency.", ["finance", "crypto"]),
    ("compute_loan_payment", "Compute the monthly payment for a loan given rate and term.", ["finance"]),
    # --- email / messaging ---
    ("send_email", "Compose and send an email message to a recipient.", ["email", "communication"]),
    ("search_inbox", "Search the user's email inbox for matching messages.", ["email"]),
    ("send_slack_message", "Post a message to a Slack channel or user.", ["chat", "communication"]),
    ("schedule_meeting", "Create a calendar invite and schedule a meeting.", ["calendar", "communication"]),
    ("send_sms", "Send a text message to a phone number.", ["sms", "communication"]),
    # --- files / storage ---
    ("read_file", "Read the contents of a file from local disk.", ["files", "io"]),
    ("write_file", "Write text content to a file on local disk.", ["files", "io"]),
    ("list_directory", "List the files and folders inside a directory.", ["files", "io"]),
    ("upload_to_s3", "Upload a file to an Amazon S3 bucket.", ["files", "cloud"]),
    ("compress_folder", "Create a zip archive from a folder of files.", ["files"]),
    # --- database ---
    ("run_sql_query", "Execute a read-only SQL query against the database.", ["database", "sql"]),
    ("insert_record", "Insert a new row into a database table.", ["database", "sql"]),
    ("delete_record", "Delete a row from a database table by id.", ["database", "sql"]),
    ("backup_database", "Create a full backup snapshot of the database.", ["database"]),
    ("get_table_schema", "Describe the columns and types of a database table.", ["database", "sql"]),
    # --- web / search ---
    ("web_search", "Search the public web for information on a topic.", ["web", "search"]),
    ("fetch_url", "Download the raw HTML or text content of a web page.", ["web"]),
    ("translate_text", "Translate text from one natural language to another.", ["nlp", "language"]),
    ("summarize_text", "Produce a short summary of a long passage of text.", ["nlp"]),
    ("extract_entities", "Extract named entities (people, places, orgs) from text.", ["nlp"]),
    # --- maps / travel ---
    ("get_directions", "Get driving directions between two addresses.", ["maps", "travel"]),
    ("find_nearby_restaurants", "Find restaurants near a given location.", ["maps", "food"]),
    ("geocode_address", "Convert a street address into latitude/longitude.", ["maps"]),
    ("search_flights", "Search for available flights between two airports.", ["travel"]),
    ("book_hotel", "Reserve a hotel room for a set of dates.", ["travel"]),
    # --- math / data ---
    ("calculate_statistics", "Compute mean, median and standard deviation of numbers.", ["math", "data"]),
    ("solve_equation", "Solve an algebraic equation for its variable.", ["math"]),
    ("plot_chart", "Render a chart image from a series of data points.", ["data", "viz"]),
    ("factorial", "Compute the factorial of a non-negative integer.", ["math"]),
    ("linear_regression", "Fit a linear regression model to x/y data points.", ["math", "data", "ml"]),
    # --- devops ---
    ("deploy_service", "Deploy an application service to the cluster.", ["devops"]),
    ("get_pod_logs", "Fetch recent log lines from a Kubernetes pod.", ["devops", "logs"]),
    ("restart_container", "Restart a running Docker container by name.", ["devops"]),
    ("scale_deployment", "Change the replica count of a deployment.", ["devops"]),
    ("get_cpu_metrics", "Get current CPU utilisation metrics for a host.", ["devops", "monitoring"]),
    # --- calendar / productivity ---
    ("create_todo", "Add a task to the user's to-do list.", ["productivity"]),
    ("set_reminder", "Set a timed reminder for the user.", ["productivity"]),
    ("get_calendar_events", "List upcoming events on the user's calendar.", ["calendar"]),
    ("create_note", "Save a free-text note for later retrieval.", ["productivity"]),
    ("start_timer", "Start a countdown timer for a number of minutes.", ["productivity"]),
]

# Labeled queries: (natural-language query, gold tool name).
QUERIES: list[tuple[str, str]] = [
    ("What's the temperature in Paris right now?", "get_current_weather"),
    ("Will it rain in Tokyo over the next five days?", "get_weather_forecast"),
    ("Is the air safe to breathe in Delhi today?", "get_air_quality"),
    ("How much is one bitcoin worth?", "get_crypto_price"),
    ("Convert 100 dollars into euros", "convert_currency"),
    ("What is Apple's share price?", "get_stock_price"),
    ("Work out the monthly repayment on a 20000 loan", "compute_loan_payment"),
    ("Email the report to my manager", "send_email"),
    ("Find the message from accounting in my mailbox", "search_inbox"),
    ("Drop a note in the engineering channel on Slack", "send_slack_message"),
    ("Text John that I'm running late", "send_sms"),
    ("Set up a 30 minute call with the design team tomorrow", "schedule_meeting"),
    ("Save this text to a file on disk", "write_file"),
    ("Show me everything in the downloads folder", "list_directory"),
    ("Push this document up to our S3 bucket", "upload_to_s3"),
    ("Zip up the project folder", "compress_folder"),
    ("Run a query to count active users in the database", "run_sql_query"),
    ("Add a new customer row to the table", "insert_record"),
    ("Take a full backup of the database", "backup_database"),
    ("Search the internet for news about AI regulation", "web_search"),
    ("Translate this paragraph into Spanish", "translate_text"),
    ("Give me a short summary of this article", "summarize_text"),
    ("How do I drive from the airport to downtown?", "get_directions"),
    ("Find me somewhere to eat near the hotel", "find_nearby_restaurants"),
    ("Look for flights from London to New York", "search_flights"),
    ("Book a room for two nights in Berlin", "book_hotel"),
    ("Compute the average and standard deviation of these numbers", "calculate_statistics"),
    ("Draw a line chart of monthly sales", "plot_chart"),
    ("Fit a regression line to this data", "linear_regression"),
    ("Deploy the payments service to production", "deploy_service"),
    ("Show me the recent logs from the api pod", "get_pod_logs"),
    ("Restart the redis container", "restart_container"),
    ("How busy is the CPU on web-01?", "get_cpu_metrics"),
    ("Remind me to call the dentist at 3pm", "set_reminder"),
    ("Add 'buy milk' to my todo list", "create_todo"),
    ("What's on my calendar this afternoon?", "get_calendar_events"),
]


def _make_tool(name: str, description: str) -> Callable[..., str]:
    # A simple optional parameter keeps the benchmark tools trivially callable.
    # (Bare ``**kwargs`` tools are also fully supported — the schema builder now
    # skips variadic params and sets ``additionalProperties: true`` — but an
    # explicit arg keeps these example tools self-documenting.)
    def fn(arg: str = "") -> str:
        return f"called {name}({arg})"

    fn.__name__ = name
    fn.__doc__ = description
    return fn


def build_embedder(kind: str):
    """Return (embedder, label). ``kind`` == 'auto' tries best-available."""
    order = (
        [kind]
        if kind != "auto"
        else ["sentence-transformers", "nomic", "simple"]
    )
    for k in order:
        try:
            if k == "sentence-transformers":
                from agent_gantry.adapters.embedders.sentence_transformers import (
                    SentenceTransformersEmbedder,
                )

                return SentenceTransformersEmbedder("all-MiniLM-L6-v2"), "sentence-transformers/all-MiniLM-L6-v2"
            if k == "nomic":
                from agent_gantry.adapters.embedders.nomic import NomicEmbedder

                return NomicEmbedder(), "nomic-embed-text-v1.5"
            if k == "simple":
                from agent_gantry.adapters.embedders.simple import SimpleEmbedder

                return SimpleEmbedder(dimension=256), "SimpleEmbedder(hash, toy)"
        except Exception as exc:  # noqa: BLE001
            print(f"  embedder {k!r} unavailable: {exc}")
            continue
    raise RuntimeError("no embedder available")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1))))
    return s[idx]


async def run(limit: int, embedder_kind: str) -> dict[str, Any]:
    embedder, label = build_embedder(embedder_kind)
    print(f"Embedder: {label}")
    gantry = AgentGantry(embedder=embedder)

    for name, desc, tags in TOOLS:
        gantry.register(_make_tool(name, desc), tags=tags)

    t0 = time.perf_counter()
    count = await gantry.sync()
    sync_s = time.perf_counter() - t0
    print(f"Registered + embedded {count} tools in {sync_s:.2f}s\n")

    top1 = 0
    hit_at_k = 0
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []

    for query, gold in QUERIES:
        t = time.perf_counter()
        # score_threshold=0.0 measures pure ranking quality. NOTE: the library
        # default is 0.5, which silently drops correct tools with any embedder
        # whose cosine scores sit below 0.5 (e.g. MiniLM) — see review finding.
        tools = await gantry.retrieve_tools(query, limit=limit, score_threshold=0.0)
        latencies_ms.append((time.perf_counter() - t) * 1000.0)

        names = [t["function"]["name"] for t in tools]
        rank = names.index(gold) + 1 if gold in names else 0
        if rank == 1:
            top1 += 1
        if rank != 0:
            hit_at_k += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        flag = "✅" if gold in names else "❌"
        print(f"{flag} q={query[:46]!r:<48} gold={gold:<24} top{limit}={names}")

    n = len(QUERIES)
    # Token-savings estimate: full 50-tool schema dump vs. top-3.
    all_schemas = await gantry.retrieve_tools(QUERIES[0][0], limit=len(TOOLS), score_threshold=0.0)
    full_tokens = len(json.dumps(all_schemas)) // 4  # ~4 chars/token heuristic
    topk_schemas = await gantry.retrieve_tools(QUERIES[0][0], limit=limit, score_threshold=0.0)
    topk_tokens = len(json.dumps(topk_schemas)) // 4
    savings = 1.0 - (topk_tokens / full_tokens) if full_tokens else 0.0

    results = {
        "embedder": label,
        "tools": len(TOOLS),
        "queries": n,
        "limit": limit,
        "top1_accuracy": round(top1 / n, 3),
        f"hit@{limit}": round(hit_at_k / n, 3),
        "mrr": round(sum(reciprocal_ranks) / n, 3),
        "latency_ms_p50": round(_percentile(latencies_ms, 50), 2),
        "latency_ms_p95": round(_percentile(latencies_ms, 95), 2),
        "latency_ms_mean": round(sum(latencies_ms) / len(latencies_ms), 2),
        "token_savings_topk_vs_all": round(savings, 3),
    }
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3, help="top-K tools to retrieve")
    ap.add_argument(
        "--embedder",
        default="auto",
        choices=["auto", "sentence-transformers", "nomic", "simple"],
    )
    args = ap.parse_args()
    asyncio.run(run(args.limit, args.embedder))


if __name__ == "__main__":
    main()
