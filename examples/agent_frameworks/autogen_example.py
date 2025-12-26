"""
AutoGen (AG2) Integration Example with Agent-Gantry.

This example demonstrates how to use Agent-Gantry with AutoGen v0.4+ (AG2).
AutoGen v0.4 introduced breaking changes with a new event-driven architecture.

Requirements:
    pip install autogen-agentchat>=0.7.5 autogen-ext[openai]

Compatibility:
    - AutoGen v0.4+ (AG2): ✅ Compatible
    - AutoGen v0.2: ❌ Not compatible (breaking changes)

Migration Notes:
    AutoGen v0.4 uses a new async, event-driven API:
    - autogen_agentchat: High-level agent chat API
    - autogen_ext: Extensions for model providers
    - autogen_core: Low-level event-driven framework
"""

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall

load_dotenv()

async def main():
    # 1. Initialize Agent-Gantry
    print("🚀 Initializing Agent-Gantry with AutoGen v0.4+ (AG2)...\n")
    gantry = AgentGantry()

    @gantry.register
    def get_system_load():
        """Get the current system CPU load."""
        return "CPU Load: 15%"

    @gantry.register
    def get_memory_usage():
        """Get the current memory usage."""
        return "Memory Usage: 45%"

    await gantry.sync()
    print(f"✅ Registered {await gantry.count_tools()} tools in Agent-Gantry\n")

    # 2. Retrieve tools from Gantry using semantic search
    user_query = "Check the system load and report back."
    print(f"🔍 Retrieving tools for: '{user_query}'")
    # Lowering threshold for SimpleEmbedder compatibility in this example
    retrieved_tools = await gantry.retrieve_tools(user_query, limit=3, score_threshold=0.1)
    print(f"✅ Retrieved {len(retrieved_tools)} relevant tools\n")

    # 3. Wrap Gantry tools for AutoGen (AG2)
    def make_autogen_tool(tool_name: str, tool_desc: str, gantry_instance: AgentGantry):
        """
        Factory function to properly bind tool name to AutoGen tool wrapper.

        AutoGen v0.4+ requires async functions with proper metadata.
        """
        async def tool_wrapper() -> str:
            """Execute tool via Agent-Gantry."""
            try:
                result = await gantry_instance.execute(
                    ToolCall(tool_name=tool_name, arguments={})
                )
                if result.status == "success":
                    return str(result.result)
                else:
                    return f"Error: {result.error}"
            except Exception as e:
                return f"Tool execution failed: {str(e)}"

        tool_wrapper.__name__ = tool_name
        tool_wrapper.__doc__ = tool_desc
        return tool_wrapper

    autogen_tools = []
    for ts in retrieved_tools:
        name = ts["function"]["name"]
        desc = ts["function"].get("description", f"Execute {name}")
        autogen_tools.append(make_autogen_tool(name, desc, gantry))
        print(f"  📦 Wrapped tool: {name}")

    # 4. Setup AutoGen Agent with v0.4+ API
    print("\n🤖 Setting up AutoGen AssistantAgent...")
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=autogen_tools,
        system_message="You are a helpful assistant. Use the available tools to answer questions accurately."
    )

    # 5. Run Conversation with AutoGen v0.4+ streaming
    print("\n" + "="*60)
    print("🎯 Running AutoGen (AG2) Agent with Agent-Gantry")
    print("="*60 + "\n")

    await Console(assistant.run_stream(task=user_query))

if __name__ == "__main__":
    asyncio.run(main())
