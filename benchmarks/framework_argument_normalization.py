"""Measure adapter argument normalization without SDKs, network, or model calls.

Run: python benchmarks/framework_argument_normalization.py --iterations 200
Results describe this boundary only, not total tool or agent latency.
"""

from __future__ import annotations

import argparse
import timeit
import tracemalloc
from datetime import datetime
from typing import Any

from agent_gantry.integrations.frameworks.base import _json_native


def cases() -> dict[str, tuple[Any, dict[str, Any]]]:
    """Large unchanged payloads and a nested payload needing conversion."""
    values = {f"field_{i}": i for i in range(10_000)}
    return {
        "free-form object": (values, {"type": "object", "additionalProperties": True}),
        "typed mapping": (
            values,
            {"type": "object", "additionalProperties": {"type": "integer"}},
        ),
        "nested dates": (
            [{"when": datetime(2026, 10, 16)} for _ in range(100)],
            {"items": {"properties": {"when": {"format": "date-time"}}}},
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    for name, (value, schema) in cases().items():
        elapsed = min(
            timeit.repeat(lambda: _json_native(value, schema), number=args.iterations, repeat=3)
        )
        tracemalloc.start()
        _json_native(value, schema)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"{name:18s} {elapsed / args.iterations * 1e6:9.1f} us/call  {peak:8,d} peak bytes")


if __name__ == "__main__":
    main()
