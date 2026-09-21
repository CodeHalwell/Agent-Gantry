"""
Seeing retrieval and execution telemetry on the console.

``ConsoleTelemetryAdapter`` emits every span and event as a log record on the
``agent_gantry`` logger. The package never configures logging itself, so
nothing appears until you opt in with ``enable_console_logging()`` — without
that call this demo would print only its own headings. No API key needed.

Run with::

    python examples/observability/telemetry_demo.py
"""

import asyncio

from agent_gantry import AgentGantry, enable_console_logging
from agent_gantry.observability.console import ConsoleTelemetryAdapter


async def main():
    # 1. Opt in to console output, then hand Gantry the console adapter.
    # The adapter only logs; this call is what attaches a handler.
    enable_console_logging()
    telemetry = ConsoleTelemetryAdapter()
    gantry = AgentGantry(telemetry=telemetry)
    try:

        @gantry.register(examples=["calculate tax for $100", "how much tax do I owe on 250 pounds"])
        def calculate_tax(amount: float) -> float:
            """Calculates tax for a given amount."""
            return amount * 0.15

        await gantry.sync()

        print("--- Starting Telemetry Demo ---")
        print("Watch the console for telemetry events...\n")

        # 2. Perform Retrieval
        # This should trigger a 'tool_retrieval' span and record a retrieval event
        await gantry.retrieve_tools("calculate tax for $100")

        # 3. Perform Execution
        # This should trigger a 'tool_execution' span and record an execution event
        from agent_gantry.schema.execution import ToolCall

        call = ToolCall(tool_name="calculate_tax", arguments={"amount": 100.0})
        await gantry.execute(call)

        print("\n--- Demo Complete ---")
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
