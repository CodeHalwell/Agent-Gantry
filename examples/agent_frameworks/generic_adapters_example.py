import asyncio

from dotenv import load_dotenv

from agent_gantry import AgentGantry
from agent_gantry.integrations.framework_adapters import fetch_framework_tools

load_dotenv()

async def main():
    # 1. Initialize Agent-Gantry
    gantry = AgentGantry()

    @gantry.register
    def get_market_data(ticker: str):
        """Get real-time market data for a ticker."""
        return {"ticker": ticker, "price": 250.45, "volume": "1.2M"}

    await gantry.sync()

    # 2. Define the query
    user_query = "What is the market data for MSFT?"

    # 3. Use the framework adapter
    # This helper returns the schema shape expected by the framework
    # (Currently OpenAI-style function calling for all supported frameworks)
    google_adk_tools = await fetch_framework_tools(
        gantry,
        user_query,
        framework="google_adk",
        limit=3
    )

    strands_tools = await fetch_framework_tools(
        gantry,
        user_query,
        framework="strands",
        limit=3
    )

    print(f"Retrieved {len(google_adk_tools)} tools for Google ADK")
    print(f"Retrieved {len(strands_tools)} tools for Strands")

    # 4. Example of how you would typically use these in a generic framework
    # Most modern frameworks accept the OpenAI tool schema format.
    for tool in google_adk_tools:
        name = tool["function"]["name"]
        print(f"Integrating tool: {name}")

        # The framework would then call back to Gantry for execution:
        # result = await gantry.execute(ToolCall(tool_name=name, arguments=args))

if __name__ == "__main__":
    asyncio.run(main())
