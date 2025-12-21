import asyncio

from agent_gantry import AgentGantry
from agent_gantry.observability.console import ConsoleTelemetryAdapter


async def main():
    # 1. Initialize with Console Telemetry
    # This will print detailed logs of retrieval and execution events to the console.
    telemetry = ConsoleTelemetryAdapter()
    gantry = AgentGantry(telemetry=telemetry)

    @gantry.register
    def calculate_tax(amount: float) -> float:
        """Calculates tax for a given amount."""
        return amount * 0.15

    await gantry.sync()

    print("--- Starting Telemetry Demo ---")
    print("Watch the console for telemetry events...\n")

    # 2. Perform Retrieval
    # This should trigger a 'tool_retrieval' span and record a retrieval event
    tools = await gantry.retrieve_tools("calculate tax for $100")

    # 3. Perform Execution
    # This should trigger a 'tool_execution' span and record an execution event
    from agent_gantry.schema.execution import ToolCall
    call = ToolCall(tool_name="calculate_tax", arguments={"amount": 100.0})
    await gantry.execute(call)

    print("\n--- Demo Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
