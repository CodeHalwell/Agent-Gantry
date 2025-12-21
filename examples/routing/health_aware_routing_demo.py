import asyncio

from agent_gantry import AgentGantry
from agent_gantry.schema.config import AgentGantryConfig, ExecutionConfig
from agent_gantry.schema.execution import ToolCall
from agent_gantry.schema.query import ConversationContext, ToolQuery


async def main():
    # Configure low threshold for circuit breaker
    config = AgentGantryConfig(
        execution=ExecutionConfig(
            circuit_breaker_threshold=2,
            circuit_breaker_timeout_s=60
        )
    )
    gantry = AgentGantry(config=config)

    @gantry.register
    def fragile_tool() -> str:
        """A tool that breaks easily."""
        raise Exception("Broken!")

    @gantry.register
    def robust_tool() -> str:
        """A tool that works."""
        return "Works!"

    await gantry.sync()

    print("--- Health-Aware Routing Demo ---")

    # 1. Break the fragile tool until circuit opens
    print("Breaking 'fragile_tool'...")
    call = ToolCall(tool_name="fragile_tool", arguments={})
    for _ in range(3):
        try:
            await gantry.execute(call)
        except:
            pass

    # 2. Query with exclude_unhealthy=False
    print("\n1. Query (exclude_unhealthy=False):")
    # We manually construct the query to control flags
    ctx = ConversationContext(query="tool")
    query = ToolQuery(context=ctx, exclude_unhealthy=False, score_threshold=0.0)

    result = await gantry.retrieve(query)
    names = [t.tool.name for t in result.tools]
    print(f"Found: {names}")
    # Should include fragile_tool

    # 3. Query with exclude_unhealthy=True (Default)
    print("\n2. Query (exclude_unhealthy=True):")
    # Note: AgentGantry.retrieve defaults exclude_unhealthy to True in ToolQuery model
    query = ToolQuery(context=ctx, exclude_unhealthy=True, score_threshold=0.0)

    result = await gantry.retrieve(query)
    names = [t.tool.name for t in result.tools]
    print(f"Found: {names}")

    tool_def = await gantry.get_tool("fragile_tool")
    print("\nTool Health Status:")
    print(f" - Failures: {tool_def.health.consecutive_failures}")
    print(f" - Circuit Open: {tool_def.health.circuit_breaker_open}")

if __name__ == "__main__":
    asyncio.run(main())
