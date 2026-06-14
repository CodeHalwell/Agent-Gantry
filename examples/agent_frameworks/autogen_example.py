"""
AutoGen (AG2) Integration Example with Agent-Gantry.

This example demonstrates how to use Agent-Gantry with AutoGen v0.4+ (AG2).
AutoGen v0.4 introduced breaking changes with a new event-driven architecture.

Requirements:
    pip install autogen-agentchat>=0.7.5 autogen-ext[openai]

Compatibility:
    - AutoGen v0.4+ (AG2): ✅ Compatible
    - AutoGen v0.2: ❌ Not compatible (breaking changes)

Migration Notes (AG2 → Microsoft Agent Framework):
    Microsoft Agent Framework (MAF) is Microsoft's stated successor direction to AutoGen.
    For teams evaluating MAF, see ``examples/agent_frameworks/agent_framework_example.py``
    for the idiomatic Gantry + MAF integration pattern, including multi-agent workflows.
    This example remains fully supported for teams staying on AG2.

Migration Notes (AutoGen v0.2 → v0.4):
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
from agent_gantry.autogen import for_autogen

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
    print(f"✅ Registered {gantry.tool_count} tools in Agent-Gantry\n")

    # 2. Define the query (tool selection happens in step 3 via for_autogen).
    user_query = "Check the system load and report back."
    print(f"🔍 Retrieving tools for: '{user_query}'")

    # 3. Get AutoGen-ready callables for the selected tools. for_autogen returns
    #    {name, description, callable} dicts; AutoGen's `tools=` wants callables.
    #    Each callable carries a real signature and routes through gantry.execute.
    selected = await for_autogen(gantry, user_query, limit=3, score_threshold=0.1)
    autogen_tools = [entry["callable"] for entry in selected]
    for entry in selected:
        print(f"  📦 Wrapped tool: {entry['name']}")

    # 4. Setup AutoGen Agent with v0.4+ API
    print("\n🤖 Setting up AutoGen AssistantAgent...")
    model_client = OpenAIChatCompletionClient(model="gpt-5.5")

    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=autogen_tools,
        system_message="You are a helpful assistant. Use the available tools to answer questions accurately.",
    )

    # 5. Run Conversation with AutoGen v0.4+ streaming
    print("\n" + "=" * 60)
    print("🎯 Running AutoGen (AG2) Agent with Agent-Gantry")
    print("=" * 60 + "\n")

    await Console(assistant.run_stream(task=user_query))


if __name__ == "__main__":
    asyncio.run(main())
