import asyncio
import time

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import BatchToolCall, ToolCall


async def main():
    gantry = AgentGantry()

    @gantry.register
    async def worker(id: int, duration: int) -> str:
        """Simulates work."""
        print(f"[Worker {id}] Starting {duration}s task...")
        await asyncio.sleep(duration)
        print(f"[Worker {id}] Done!")
        return f"Worker {id} finished"

    await gantry.sync()

    print("--- Batch Execution Demo ---")

    # Create 3 calls that take 1 second each
    calls = [
        ToolCall(tool_name="worker", arguments={"id": 1, "duration": 1}),
        ToolCall(tool_name="worker", arguments={"id": 2, "duration": 1}),
        ToolCall(tool_name="worker", arguments={"id": 3, "duration": 1}),
    ]

    batch = BatchToolCall(calls=calls, execution_strategy="parallel")

    print("Executing 3 tasks (1s each) in parallel...")
    start = time.perf_counter()

    result = await gantry.execute_batch(batch)

    end = time.perf_counter()

    print(f"\nTotal Time: {end - start:.2f}s")
    print(f"Successful: {result.successful_count}/{len(calls)}")

    # If sequential, it would take ~3 seconds. Parallel takes ~1 second.
    if (end - start) < 1.5:
        print(">>> Success: Tasks ran in parallel!")
    else:
        print(">>> Warning: Tasks ran sequentially.")

if __name__ == "__main__":
    asyncio.run(main())
