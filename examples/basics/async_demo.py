"""
Native async tool execution.

Registers an ``async def`` tool and executes it through ``gantry.execute``,
which awaits it directly; the timing printed shows the sleep really ran.
Runs offline with no API key.

Run: python examples/basics/async_demo.py
"""

import asyncio
import time

from agent_gantry import AgentGantry


async def main():
    gantry = AgentGantry()

    try:
        # 1. Register an Async Tool
        # Agent-Gantry natively supports async functions
        @gantry.register(examples=["wait for two seconds", "pause for a moment before continuing"])
        async def slow_operation(seconds: int) -> str:
            """Simulates a slow async operation."""
            print(f"[Tool] Starting sleep for {seconds} seconds...")
            await asyncio.sleep(seconds)
            print("[Tool] Woke up!")
            return f"Slept for {seconds} seconds"

        await gantry.sync()

        print("--- Executing Async Tool ---")
        start_time = time.perf_counter()

        from agent_gantry.schema.execution import ToolCall

        call = ToolCall(tool_name="slow_operation", arguments={"seconds": 2})

        # The executor awaits the async function automatically
        result = await gantry.execute(call)

        end_time = time.perf_counter()
        print(f"Result: {result.result}")
        print(f"Total Execution Time: {end_time - start_time:.2f}s")
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
