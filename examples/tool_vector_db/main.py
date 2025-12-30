"""
Simplest Agent-Gantry Example: 50 Tools with OpenAI Responses API

This example demonstrates:
1. Registering 50 tools with Agent-Gantry
2. Semantic retrieval of relevant tools using @with_semantic_tools decorator
3. Tool execution via OpenAI Responses API

Key Feature: ZERO-CONFIG AUTO-SYNC
    - First run: Tools are embedded and stored in LanceDB
    - Subsequent runs: Fingerprints are checked, only changed tools re-embedded
    - No manual sync() calls needed!

Requirements:
    pip install agent-gantry[nomic] openai python-dotenv

Usage:
    export OPENAI_API_KEY=your-key-here
    python main.py
"""

from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

from agent_gantry.integrations.semantic_tools import with_semantic_tools
from agent_gantry.schema.execution import ToolCall

# Import our 50 tools - they're registered at import time
from tools import tools


async def main():
    # Ensure API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable")
        return

    # Create OpenAI client
    client = AsyncOpenAI()

    # Define a chat function with semantic tool injection
    # The @with_semantic_tools decorator:
    #   1. Automatically syncs tools on first use (with smart change detection)
    #   2. Retrieves semantically relevant tools based on the prompt
    #   3. Injects them into the function call
    @with_semantic_tools(
        tools,
        limit=10,  # Retrieve top 10 relevant tools
        dialect="openai_responses",
        prompt_param="prompt",
    )
    async def chat(prompt: str, tools: list | None = None):
        """Chat with the LLM using semantically retrieved tools."""
        if tools:
            print(f"\nRetrieved {len(tools)} relevant tools (out of {len(tools)}→50):")
            for t in tools:
                print(f"  • {t['name']}")
            print()
        return await client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            tools=tools,
        )

    # Ask a question - everything happens automatically!
    query = "Calculate the mean of [10, 20, 30, 40, 50] and convert 100 meters to feet"
    
    print(f"Query: {query}")
    print("-" * 60)
    
    response = await chat(query)
    
    # Process tool calls and execute them
    tool_results = []
    for item in response.output:
        if item.type == "function_call":
            tool_name = item.name
            arguments = json.loads(item.arguments)
            
            print(f"Tool call: {tool_name}({arguments})")
            
            # Execute the tool using Agent-Gantry
            call = ToolCall(tool_name=tool_name, arguments=arguments)
            result = await tools.execute(call)
            
            print(f"  Result: {result.result}")
            tool_results.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": str(result.result),
            })
    
    # Send tool results back to get final response
    if tool_results:
        final_response = await client.responses.create(
            model="gpt-4o-mini",
            input=tool_results,
            previous_response_id=response.id,
        )
        print(f"\nFinal Response: {final_response.output_text}")
    else:
        print(f"\nResponse: {response.output_text}")


if __name__ == "__main__":
    asyncio.run(main())
