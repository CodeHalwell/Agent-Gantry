"""
Tripping a circuit breaker.

A tool that always fails is called until its breaker opens; the next call is
refused with ``ExecutionStatus.CIRCUIT_OPEN`` instead of reaching the tool.
No API key needed.

Run with::

    python examples/execution/circuit_breaker_demo.py
"""

import asyncio

from dotfetch import DotFetch
from dotfetch.schema.config import DotFetchConfig, ExecutionConfig
from dotfetch.schema.execution import ExecutionStatus, ToolCall


async def main():
    # 1. Configure dotfetch with a sensitive circuit breaker
    # Threshold = 2 failures -> Open Circuit
    config = DotFetchConfig(
        execution=ExecutionConfig(
            circuit_breaker_threshold=2,
            # Seconds the circuit stays open before a retry is allowed.
            # This demo stops as soon as the circuit opens, so it never elapses.
            circuit_breaker_timeout_s=5,
        )
    )
    fetcher = DotFetch(config=config)
    try:
        # 2. Register a flaky tool
        @fetcher.register(examples=["call the flaky API", "hit the upstream service"])
        def flaky_api() -> str:
            """Always fails."""
            raise ConnectionError("API is down!")

        await fetcher.sync()

        print("--- Circuit Breaker Demo ---")

        call = ToolCall(tool_name="flaky_api", arguments={})

        # 3. Trigger failures to open the circuit
        for i in range(1, 5):
            print(f"\nAttempt {i}:")
            result = await fetcher.execute(call)

            print(f"Status: {result.status.value}")
            if result.error:
                print(f"Error: {result.error}")

            if result.status == ExecutionStatus.CIRCUIT_OPEN:
                print(">>> CIRCUIT BREAKER IS OPEN! Execution blocked to protect system. <<<")
                break
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
