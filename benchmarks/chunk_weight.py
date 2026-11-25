"""
How much should a match on a piece of a long description count for?

A tool with a long description is indexed a second time in pieces, and scored
by the best of its own vector and its pieces'. That gives a long-winded tool
many chances to match a query, each on one passage of incidental text, where
a short tool has one. This script measures what that costs and what weighting
the pieces down buys.

It takes the retrieval eval's catalogue (``tests/retrieval``), appends
sixteen sentences of ordinary API small print (authentication, paging, rate
limits, currencies, timestamps) to a share of the tools that are *not* the
answer to any labelled query, and runs the labelled tool queries with chunking
off, and with it on at each weight.

    python benchmarks/chunk_weight.py
    python benchmarks/chunk_weight.py --weights 1.0 0.9 0.8 --shares 0.3

The numbers quoted in ``benchmarks/README.md`` are from this script with
``all-MiniLM-L6-v2``. It needs sentence-transformers and the ``mcp`` package.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dotfetch.core.router as router_module
from dotfetch import DotFetch
from dotfetch.core.chunks import tool_chunks
from dotfetch.evaluation import RetrievalReport, evaluate_retrieval, tool_retriever
from dotfetch.schema.config import DotFetchConfig, RoutingConfig
from dotfetch.schema.tool import ToolDefinition
from tests.retrieval import catalogue

SMALL_PRINT = [
    "Requests are authenticated with the workspace token and need the matching scope.",
    "Results are paginated; pass the cursor from the previous page to fetch the next one.",
    "The call is rate limited per workspace and returns a retry hint when the limit is reached.",
    "Timestamps are ISO 8601 in UTC unless a timezone is supplied explicitly.",
    "Use this when the user asks for it directly, and not as a first step of another task.",
    "A failed call returns a structured error with a code, a message and the offending field.",
    "The operation is idempotent when the same request key is supplied twice.",
    "Large responses are truncated; narrow the filter if the answer is cut short.",
    "Filters accept field:value pairs joined with AND, OR and NOT, with quotes around phrases.",
    "Do not use this for bulk changes; there is a separate batch endpoint for those.",
    "Archived items are excluded unless include_archived is set to true.",
    "Returns the identifiers needed by the follow-up calls, along with a short summary.",
    "Permissions are checked against the calling user, not the integration.",
    "Field names are case sensitive and unknown fields are rejected, not ignored.",
    "Amounts are in minor units of the currency given, and never floating point.",
    "Sorting defaults to most recently updated first and can be reversed.",
]


def _catalogue(share: float, seed: int) -> tuple[list[ToolDefinition], int]:
    """The eval's tools, with ``share`` of the non-answers given long descriptions."""
    rng = random.Random(seed)
    answers = {e.split(".", 1)[-1] for case in catalogue.TOOL_CASES for e in case.expected}
    tools: list[ToolDefinition] = []
    lengthened = 0
    for tool in [*catalogue.native_tools(), *catalogue.mcp_tools()]:
        if tool.name not in answers and not tool_chunks(tool) and rng.random() < share:
            lines = SMALL_PRINT[:]
            rng.shuffle(lines)
            text = f"{tool.description.rstrip()} {' '.join(lines)}"[:2000]
            tool = tool.model_copy(update={"description": text})
            lengthened += 1
        tools.append(tool)
    return tools, lengthened


async def _measure(tools: list[ToolDefinition], *, chunk: bool, weight: float) -> RetrievalReport:
    from dotfetch.adapters.embedders.sentence_transformers import (
        SentenceTransformersEmbedder,
    )

    router_module.CHUNK_SCORE_WEIGHT = weight
    fetcher = DotFetch(
        config=DotFetchConfig(routing=RoutingConfig(chunk_long_text=chunk)),
        embedder=SentenceTransformersEmbedder(),
    )
    try:
        for tool in tools:
            await fetcher.add_tool(tool)
        await fetcher.sync()
        return await evaluate_retrieval(
            tool_retriever(fetcher, score_threshold=0.0), catalogue.TOOL_CASES, k=5
        )
    finally:
        await fetcher.close()


async def _run(shares: list[float], weights: list[float], seed: int) -> None:
    shipped = router_module.CHUNK_SCORE_WEIGHT
    answerable = sum(1 for case in catalogue.TOOL_CASES if case.expected)
    print(f"{answerable} answerable tool queries; shipped weight {shipped}")
    print(f"{'long distractors':>18}  {'chunks':>12}  {'recall@1':>8}  {'recall@5':>8}")
    try:
        for share in shares:
            tools, lengthened = _catalogue(share, seed)
            runs = [("off", False, 1.0), *((f"weight {w:g}", True, w) for w in weights)]
            for label, chunk, weight in runs:
                report = await _measure(tools, chunk=chunk, weight=weight)
                print(
                    f"{lengthened:>18}  {label:>12}  "
                    f"{report.recall_at_1:>8.3f}  {report.recall_at_k:>8.3f}",
                    flush=True,
                )
    finally:
        router_module.CHUNK_SCORE_WEIGHT = shipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shares", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    parser.add_argument("--weights", type=float, nargs="+", default=[1.0, 0.95, 0.9, 0.85, 0.8])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    asyncio.run(_run(args.shares, args.weights, args.seed))


if __name__ == "__main__":
    main()
