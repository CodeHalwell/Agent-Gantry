"""End-to-end verification harness for Agent-Gantry's framework integrations.

Run this to confirm — offline, with no LLM and no third-party agent framework
required — that every piece added in the framework-integration work actually
functions:

1. **P0 fixes**: a ``**kwargs`` tool is executable; the convenience retrieval
   default surfaces tools instead of silently filtering them.
2. **Universal core**: ``GantryToolset.select`` returns ranked ``ToolSpec``s and
   both ``ainvoke`` (async) and ``invoke`` (sync, even inside a running loop)
   route through ``gantry.execute``.
3. **Native adapters**: each ``for_<framework>`` helper builds native tool
   objects when the framework is installed, or reports SKIP with a clean
   ``ImportError`` when it is not — proving the lazy-import contract.
4. **Multi-turn**: ``ToolRefresher`` re-selects a *different* tool as the task
   pivots across turns within one run, and accumulates used tools.

Exit code is ``0`` only when every core check passes (framework SKIPs do not
fail the run). Usage::

    python examples/frameworks/verify_all.py
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from agent_gantry import AgentGantry
from agent_gantry.integrations import ToolRefresher
from agent_gantry.integrations.frameworks import GantryToolset
from agent_gantry.schema.execution import ExecutionStatus, ToolCall

# ---- registry ------------------------------------------------------------- #
# Ten tools across distinct domains so even the hash-based SimpleEmbedder can
# route unambiguously when no real embedder is installed.
TOOLS: list[tuple[str, str, list[str]]] = [
    ("get_current_weather", "Get the current weather and temperature for a city.", ["weather"]),
    ("send_email", "Compose and send an email message to a recipient.", ["email"]),
    ("convert_currency", "Convert an amount of money from one currency to another.", ["finance"]),
    ("run_sql_query", "Execute a read-only SQL query against the database.", ["database"]),
    ("create_todo", "Add a task item to the user's to-do list.", ["productivity"]),
    ("search_flights", "Search for available flights between two airports.", ["travel"]),
    ("translate_text", "Translate a piece of text from one language to another.", ["language"]),
    ("get_stock_price", "Look up the latest share price for a stock ticker.", ["finance"]),
    ("deploy_service", "Deploy an application service to the production cluster.", ["devops"]),
    ("summarize_text", "Produce a short summary of a long passage of text.", ["nlp"]),
]


def _best_embedder() -> tuple[Any, str, bool]:
    """Prefer a real embedder; fall back to the offline hash toy.

    Returns ``(embedder, label, is_real)``. ``is_real`` is ``False`` for the
    hash-based ``SimpleEmbedder``, whose similarity scores are too coarse for
    nuanced result-driven routing — checks that need real semantics skip when
    only it is available.
    """
    try:
        import sentence_transformers  # noqa: F401  (validate the install actually imports)

        from agent_gantry.adapters.embedders.sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        return (
            SentenceTransformersEmbedder("all-MiniLM-L6-v2"),
            "sentence-transformers/all-MiniLM-L6-v2",
            True,
        )
    except Exception:  # noqa: BLE001
        from agent_gantry.adapters.embedders.simple import SimpleEmbedder

        return SimpleEmbedder(dimension=256), "SimpleEmbedder(hash, offline)", False


def _make_tool(name: str, description: str):
    def fn(arg: str = "") -> str:
        return f"{name}->{arg}"

    fn.__name__ = name
    fn.__doc__ = description
    return fn


async def _build_gantry() -> AgentGantry:
    embedder, label, _is_real = _best_embedder()
    print(f"Embedder: {label}\n")
    gantry = AgentGantry(embedder=embedder)
    for name, desc, tags in TOOLS:
        gantry.register(_make_tool(name, desc), tags=tags)

    # A deliberately **kwargs-only tool to exercise the schema fix.
    def echo_kwargs(**kwargs: Any) -> str:
        "Echo back any keyword arguments supplied to this tool."
        return f"echo:{sorted(kwargs)}"

    gantry.register(echo_kwargs, tags=["debug"])
    await gantry.sync()
    return gantry


# ---- individual checks ---------------------------------------------------- #
async def check_p0_kwargs(gantry: AgentGantry) -> tuple[bool, str]:
    """A bare ``**kwargs`` tool must be executable with an empty arg dict."""
    result = await gantry.execute(ToolCall(tool_name="echo_kwargs", arguments={}))
    ok = result.status == ExecutionStatus.SUCCESS
    return ok, f"status={getattr(result.status, 'value', result.status)} result={result.result!r}"


async def check_default_threshold(gantry: AgentGantry) -> tuple[bool, str]:
    """retrieve_tools default (0.0) must surface tools, not filter them away."""
    tools = await gantry.retrieve_tools("send an email to my manager", limit=3)
    names = [t["function"]["name"] for t in tools]
    return (len(tools) > 0 and "send_email" in names), f"top3={names}"


async def check_toolset_invoke(gantry: AgentGantry) -> tuple[bool, str]:
    """GantryToolset.select + ToolSpec.ainvoke/invoke route through execute."""
    specs = await GantryToolset(gantry).select("convert 100 usd to eur", limit=3)
    if not specs:
        return False, "no specs selected"
    target = next((s for s in specs if s.name == "convert_currency"), specs[0])
    async_result = await target.ainvoke(arg="x")
    sync_result = target.invoke(arg="y")  # sync path, called inside the loop
    ok = async_result == f"{target.name}->x" and sync_result == f"{target.name}->y"
    return ok, f"async={async_result!r} sync={sync_result!r} (specs={[s.name for s in specs]})"


async def check_adapters(gantry: AgentGantry) -> list[tuple[str, bool, str]]:
    """Build native tools per framework; SKIP cleanly when not installed."""
    from agent_gantry.integrations import frameworks as F

    adapters = [
        ("langchain", F.for_langchain),
        ("langgraph", F.for_langgraph),
        ("llamaindex", F.for_llamaindex),
        ("crewai", F.for_crewai),
        ("pydantic_ai", F.for_pydantic_ai),
        ("openai_agents", F.for_openai_agents),
        ("smolagents", F.for_smolagents),
        ("haystack", F.for_haystack),
        ("agno", F.for_agno),
        ("autogen", F.for_autogen),
    ]
    rows: list[tuple[str, bool, str]] = []
    for name, fn in adapters:
        try:
            native = await fn(gantry, "deploy the payments service", limit=2)
            rows.append((name, True, f"built {len(native)} native tool(s)"))
        except ImportError as exc:
            # Not installed: the lazy-import contract held — this is a SKIP,
            # not a failure. (pip hint should be present.)
            rows.append((name, None, f"SKIP (not installed): {str(exc).splitlines()[0][:60]}"))  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            rows.append((name, False, f"FAIL: {type(exc).__name__}: {exc}"))
    return rows


async def check_multi_turn(gantry: AgentGantry) -> tuple[bool, str]:
    """ToolRefresher must re-select a different tool as the task pivots.

    This is a *user-driven* pivot scenario (each turn the user asks for a new
    thing), which is exactly what the refresher's default query generator —
    ``fallback_chain(last_user_text, last_tool_result)`` — handles. For an
    autonomous tool *pipeline* (previous output drives the next tool) you would
    instead pass ``query_generator=fallback_chain(last_tool_result, ...)``.
    """
    refresher = ToolRefresher(gantry, limit=3)
    turns = [
        ("what's the weather in Paris?", "get_current_weather"),
        ("now email the forecast to my team", "send_email"),
        ("convert 50 dollars to euros for the trip", "convert_currency"),
        ("add 'pack umbrella' to my todo list", "create_todo"),
    ]
    messages: list[dict[str, Any]] = []
    top_picks: list[str] = []
    in_top3 = 0
    for utterance, gold in turns:
        messages.append({"role": "user", "content": utterance})
        schemas = await refresher.refresh(messages)
        names = [s["function"]["name"] for s in schemas]
        if names:
            top_picks.append(names[0])
            # simulate executing the top tool, feeding the result back
            messages.append({"role": "assistant", "content": f"calling {names[0]}"})
            messages.append({"role": "tool", "name": names[0], "content": "done"})
        if gold in names:
            in_top3 += 1
    distinct = len(set(top_picks))
    pivoted = distinct >= 2
    accurate = in_top3 >= 3  # at least 3/4 turns surface the right tool
    used = refresher.tools_used
    ok = pivoted and accurate and len(used) >= 2
    return ok, (
        f"picks={top_picks} distinct={distinct} gold_in_top3={in_top3}/4 "
        f"tools_used={used}"
    )


# ---- runner --------------------------------------------------------------- #
async def check_autonomous_pipeline() -> tuple[bool, str]:
    """ToolRefresher must chain tools in an autonomous run with NO new user input.

    The agent is given one goal, then runs a pipeline; each tool's *result*
    must drive selection of the next tool (the recency-aware default reads the
    latest tool result when there is no newer user message). This is the
    autonomous-agent counterpart to the conversational pivot check.
    """
    embedder, _label, is_real = _best_embedder()
    if not is_real:
        return None, "SKIP: needs a real embedder (SimpleEmbedder is a hash toy)"  # type: ignore[return-value]
    g = AgentGantry(embedder=embedder)
    pipeline = [
        ("fetch_raw_data", "Fetch raw unprocessed data from the source system.", ["data"]),
        ("clean_dataset", "Clean and normalize a raw dataset, removing nulls and duplicates.", ["data"]),
        ("train_model", "Train a machine learning model on a cleaned dataset.", ["ml"]),
        ("evaluate_model", "Evaluate a trained machine learning model's accuracy metrics.", ["ml"]),
        ("generate_report", "Generate a written report summarizing evaluation results.", ["report"]),
    ]
    # A couple of distractor tools so selection isn't trivial.
    distractors = [
        ("send_email", "Compose and send an email message to a recipient.", ["email"]),
        ("get_weather", "Get the current weather for a city.", ["weather"]),
    ]
    for name, desc, tags in pipeline + distractors:
        g.register(_make_tool(name, desc), tags=tags)
    await g.sync()

    refresher = ToolRefresher(g, limit=3)  # default = latest_activity (recency-aware)

    # One user goal, then NO further user messages — only tool results feed back.
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Build and evaluate a model from the raw source data."}
    ]
    # Each result points FORWARD at the next stage (describing what is needed
    # next, not what was just done) — how a real pipeline step hands off.
    step_results = {
        "fetch_raw_data": "the records contain missing nulls and duplicate rows that must be cleaned and normalized",
        "clean_dataset": "the prepared training set is ready to fit and train a machine learning model",
        "train_model": "the fitted model now needs its accuracy and performance metrics evaluated",
        "evaluate_model": "please write a summary report describing the evaluation findings",
    }
    expected_order = ["fetch_raw_data", "clean_dataset", "train_model", "evaluate_model", "generate_report"]

    picks: list[str] = []
    for _ in expected_order:
        schemas = await refresher.refresh(messages)
        names = [s["function"]["name"] for s in schemas]
        pick = names[0] if names else None
        picks.append(pick or "—")
        if pick is None:
            break
        # Autonomously advance: append the assistant call + tool result (NO user msg).
        result_text = step_results.get(pick, f"{pick} completed")
        messages.append({"role": "assistant", "content": f"calling {pick}"})
        messages.append({"role": "tool", "name": pick, "content": result_text})

    # Result-driven chaining should advance through the pipeline. Require the
    # first step correct and the run to traverse most of the distinct stages.
    distinct_stages = len(set(picks) & set(expected_order))
    ok = picks[0] == "fetch_raw_data" and distinct_stages >= 4
    return ok, f"picks={picks} distinct_pipeline_stages={distinct_stages}/5"


async def run() -> dict[str, Any]:
    gantry = await _build_gantry()

    core_checks = [
        ("P0: **kwargs tool executable", await check_p0_kwargs(gantry)),
        ("P0: default threshold surfaces tools", await check_default_threshold(gantry)),
        ("core: GantryToolset select + invoke", await check_toolset_invoke(gantry)),
        ("multi-turn (conversational) pivots", await check_multi_turn(gantry)),
        ("multi-turn (autonomous) pipeline chains", await check_autonomous_pipeline()),
    ]

    print("=== CORE CHECKS ===")
    all_core_passed = True
    for label, (ok, detail) in core_checks:
        # ok is True (pass), False (fail), or None (skipped — not a failure).
        if ok is False:
            all_core_passed = False
        tag = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        print(f"  [{tag}] {label:<42} {detail}")

    print("\n=== FRAMEWORK ADAPTERS ===")
    adapter_rows = await check_adapters(gantry)
    adapter_failed = False
    for name, ok, detail in adapter_rows:
        tag = "OK  " if ok else ("SKIP" if ok is None else "FAIL")
        if ok is False:
            adapter_failed = True
        print(f"  [{tag}] {name:<16} {detail}")

    ok_count = sum(1 for _, ok, _ in adapter_rows if ok is True)
    skip_count = sum(1 for _, ok, _ in adapter_rows if ok is None)
    print(
        f"\nAdapters: {ok_count} built, {skip_count} skipped (not installed), "
        f"{'0' if not adapter_failed else 'SOME'} failed."
    )
    print(f"Core checks: {'ALL PASSED' if all_core_passed else 'FAILURES PRESENT'}")

    return {
        "all_core_passed": all_core_passed,
        "core_checks": {label: ok for label, (ok, _) in core_checks},
        "adapters_built": ok_count,
        "adapters_skipped": skip_count,
        "adapters_failed": adapter_failed,
    }


def main() -> int:
    summary = asyncio.run(run())
    return 0 if (summary["all_core_passed"] and not summary["adapters_failed"]) else 1


if __name__ == "__main__":
    sys.exit(main())
