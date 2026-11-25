"""
How tool health affects retrieval.

A tool that keeps failing trips its circuit breaker; ``ToolQuery`` then drops
it from results by default (``exclude_unhealthy=True``) and keeps it only when
you ask. No API key needed.

Run with::

    python examples/routing/health_aware_routing_demo.py
"""

import asyncio

from dotfetch import DotFetch
from dotfetch.schema.config import DotFetchConfig, ExecutionConfig
from dotfetch.schema.execution import ToolCall
from dotfetch.schema.query import ConversationContext, ToolQuery


async def main():
    # Configure low threshold for circuit breaker
    config = DotFetchConfig(
        execution=ExecutionConfig(circuit_breaker_threshold=2, circuit_breaker_timeout_s=60)
    )
    fetcher = DotFetch(config=config)
    try:

        @fetcher.register(examples=["run the fragile tool", "try the tool that keeps breaking"])
        def fragile_tool() -> str:
            """A tool that breaks easily."""
            raise Exception("Broken!")

        @fetcher.register(examples=["run the robust tool", "use the tool that always works"])
        def robust_tool() -> str:
            """A tool that works."""
            return "Works!"

        await fetcher.sync()

        print("--- Health-Aware Routing Demo ---")

        # 1. Break the fragile tool until the circuit opens. execute() never
        #    raises for a tool failure; it reports it in the returned ToolResult.
        #    Two failures trip the breaker (threshold=2); the third call is
        #    refused with status CIRCUIT_OPEN and does not count as a failure.
        print("Breaking 'fragile_tool'...")
        call = ToolCall(tool_name="fragile_tool", arguments={})
        for _ in range(3):
            await fetcher.execute(call)

        # 2. Query with exclude_unhealthy=False
        print("\n1. Query (exclude_unhealthy=False):")
        # We manually construct the query to control flags
        ctx = ConversationContext(query="tool")
        query = ToolQuery(context=ctx, exclude_unhealthy=False, score_threshold=0.0)

        result = await fetcher.retrieve(query)
        names = [t.tool.name for t in result.tools]
        print(f"Found: {names}")
        # Should include fragile_tool

        # 3. Query with exclude_unhealthy=True (Default)
        print("\n2. Query (exclude_unhealthy=True):")
        # Note: DotFetch.retrieve defaults exclude_unhealthy to True in ToolQuery model
        query = ToolQuery(context=ctx, exclude_unhealthy=True, score_threshold=0.0)

        result = await fetcher.retrieve(query)
        names = [t.tool.name for t in result.tools]
        print(f"Found: {names}")

        tool_def = await fetcher.get_tool("fragile_tool")
        print("\nTool Health Status:")
        print(f" - Failures: {tool_def.health.consecutive_failures}")
        print(f" - Circuit Open: {tool_def.health.circuit_breaker_open}")
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
