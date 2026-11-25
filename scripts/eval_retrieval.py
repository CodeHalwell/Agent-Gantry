#!/usr/bin/env python3
"""Measure retrieval quality on the labelled eval catalogue.

Run from the repository root::

    python scripts/eval_retrieval.py                 # hash embedder, as CI runs it
    python scripts/eval_retrieval.py --check         # fail if worse than the baseline
    python scripts/eval_retrieval.py --embedder sentence-transformers --by-kind --misses
    python scripts/eval_retrieval.py --score-threshold 0.3

The committed baseline (``tests/retrieval/baseline.json``) is measured with
``SimpleEmbedder``, which hashes tokens and understands no meaning. It is there
because it is deterministic, so CI can hold a line with it. For numbers that
say how good retrieval is, pass a real ``--embedder``.

To measure a selector or a reranker, build the fetcher yourself and call
``tests.retrieval.harness.measure(selector=..., reranker=...)``: those backends
need credentials or model downloads this script does not assume.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotfetch.evaluation import compare_to_baseline, format_reports  # noqa: E402
from tests.retrieval.harness import (  # noqa: E402
    BASELINE_PATH,
    calibrate,
    flatten,
    load_baseline,
    measure,
)


def _embedder(name: str) -> Any:
    if name == "simple":
        from dotfetch.adapters.embedders.simple import SimpleEmbedder

        return SimpleEmbedder()
    if name == "sentence-transformers":
        from dotfetch.adapters.embedders.sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        return SentenceTransformersEmbedder(model="all-MiniLM-L6-v2")
    from dotfetch.adapters.embedders.nomic import NomicEmbedder

    return NomicEmbedder()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--embedder", choices=["simple", "sentence-transformers", "nomic"], default="simple"
    )
    parser.add_argument("--k", type=int, default=5, help="limit per query, and the k in recall@k")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--by-kind", action="store_true", help="also show a row per case kind")
    parser.add_argument("--misses", action="store_true", help="list the cases that went wrong")
    parser.add_argument("--check", action="store_true", help="fail if worse than the baseline")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="also find the score threshold each modality's labelled set supports",
    )
    parser.add_argument("--update-baseline", action="store_true", help="rewrite the baseline")
    args = parser.parse_args()

    embedder = _embedder(args.embedder)
    reports = asyncio.run(
        measure(k=args.k, score_threshold=args.score_threshold, embedder=embedder)
    )
    print(f"embedder: {embedder.model_name}   score_threshold: {args.score_threshold}")
    print(format_reports(reports, by_kind=args.by_kind))

    if args.misses:
        for report in reports:
            for outcome in report.misses():
                got = ", ".join(outcome.returned[:3]) or "(nothing)"
                want = ", ".join(sorted(outcome.case.expected)) or "(a decline)"
                print(f"  [{report.modality}/{outcome.case.kind}] {outcome.case.query!r}")
                print(f"      wanted {want}; got {got}")

    if args.calibrate:
        print("\nthreshold calibration (recall@1 and declines, at 0.0 -> at the threshold):")
        for found in asyncio.run(calibrate(k=args.k, embedder=_embedder(args.embedder))):
            window = (
                "no clean window"
                if found.window is None
                else f"window {found.window[0]:.2f}-{found.window[1]:.2f}"
            )
            print(
                f"  {found.report.modality:<12} threshold {found.threshold:.2f}  ({window})  "
                f"recall@1 {found.at_zero.recall_at_1:.3f} -> {found.report.recall_at_1:.3f}  "
                f"declined {found.at_zero.decline_correctness:.3f} -> "
                f"{found.report.decline_correctness:.3f}"
            )

    comparable = args.embedder == "simple" and args.score_threshold == 0.0
    if (args.check or args.update_baseline) and not comparable:
        print("The baseline is measured with --embedder simple at threshold 0.0.", file=sys.stderr)
        return 2
    if args.update_baseline:
        BASELINE_PATH.write_text(
            json.dumps(
                {"embedder": embedder.model_name, "k": args.k, "metrics": flatten(reports)},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"baseline written to {BASELINE_PATH}")
    elif args.check:
        baseline = load_baseline()
        if baseline["k"] != args.k:
            print(f"The baseline was measured at k={baseline['k']}.", file=sys.stderr)
            return 2
        regressions = compare_to_baseline(flatten(reports), baseline["metrics"])
        if regressions:
            print("\nWorse than the baseline:", file=sys.stderr)
            for line in regressions:
                print(f"  {line}", file=sys.stderr)
            return 1
        print("\nNo metric is worse than the baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
