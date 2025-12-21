import asyncio
import os
import json
from typing import Any

from dotenv import load_dotenv
from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall
from agent_gantry.schema.query import ConversationContext, ToolQuery

# Load environment variables
load_dotenv()

async def main():
    print("=== Agent-Gantry + Google GenAI (Gemini) Integration Demo ===\n")

    # 1. Check for API Key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment.")
        print("   Please set it in your .env file.")
        return

    # 2. Initialize Gantry
    gantry = AgentGantry()

    # 3. Register Tools
    @gantry.register(tags=["search"])
    def search_knowledge_base(query: str) -> str:
        """Search the internal knowledge base for documents."""
        return f"Found 2 documents for '{query}': [Doc A, Doc B]"

    await gantry.sync()
    print(f"✅ Registered {gantry.tool_count} tools\n")

    # 4. Initialize Google GenAI
    import google.generativeai as genai
    from google.generativeai.types import FunctionDeclaration, Tool
    
    genai.configure(api_key=api_key)

    # --- Scenario: Dynamic Retrieval with Gemini Schema ---
    print("--- Scenario: Dynamic Retrieval with Gemini Schema ---")
    user_query = "Find documents about project alpha"
    print(f"User Query: '{user_query}'")

    # A. Retrieve Tools
    retrieval_result = await gantry.retrieve(ToolQuery(
        context=ConversationContext(query=user_query),
        limit=1
    ))

    # B. Convert to Gemini Schema
    # Agent-Gantry provides `to_gemini_schema()` which returns a dict compatible with FunctionDeclaration
    gemini_tools = []
    for t in retrieval_result.tools:
        schema = t.tool.to_gemini_schema()
        
        # Create Gemini FunctionDeclaration object
        func_decl = FunctionDeclaration(
            name=schema["name"],
            description=schema["description"],
            parameters=schema["parameters"]
        )
        gemini_tools.append(func_decl)

    # Wrap in Tool object
    toolbox = Tool(function_declarations=gemini_tools)
    
    print(f"Gantry retrieved {len(gemini_tools)} tool(s)")

    # C. Call Gemini
    model = genai.GenerativeModel('gemini-1.5-flash', tools=[toolbox])
    chat = model.start_chat(enable_automatic_function_calling=True) # Or manual handling
    
    # Note: automatic_function_calling executes the function internally if you provide the implementation map.
    # Since we want Gantry to execute it (for security/telemetry), we usually handle it manually.
    # But for this demo, we'll show how to inspect the response part.
    
    response = chat.send_message(user_query)
    
    # Inspect response for function calls
    for part in response.parts:
        if fn := part.function_call:
            print(f"Gemini decided to call: {fn.name}({fn.args})")
            
            # Execute securely via Gantry
            # Note: fn.args is a ProtoStruct, convert to dict
            args_dict = dict(fn.args)
            
            result = await gantry.execute(ToolCall(
                tool_name=fn.name,
                arguments=args_dict
            ))
            print(f"Execution Result: {result.result}")
            
            # In a real loop, you would send this result back to Gemini
            # response = chat.send_message(
            #     Part.from_function_response(name=fn.name, response={"result": result.result})
            # )

    # --- Scenario: Using the Decorator ---
    print("\n--- Scenario: Using @with_semantic_tools Decorator ---")
    from agent_gantry.integrations.decorator import with_semantic_tools

    # Note: The decorator returns a list of dicts (OpenAI format by default).
    # For Gemini, we need to manually convert these or use the 'gemini' dialect if supported.
    # Currently, the decorator's 'gemini' dialect returns OpenAI-compatible dicts because 
    # Gemini's Python SDK often prefers FunctionDeclaration objects.
    # Here we show how to adapt the decorator output for Gemini.

    @with_semantic_tools(gantry, limit=1)
    async def chat_with_gemini(prompt: str, tools: list = None):
        # Convert the injected 'tools' (list of dicts) to Gemini Tool objects
        if tools:
            print(f"   [Decorator] Injected {len(tools)} tools")
            gemini_funcs = []
            for t in tools:
                # t is {'type': 'function', 'function': {...}}
                f_spec = t['function']
                gemini_funcs.append(FunctionDeclaration(
                    name=f_spec['name'],
                    description=f_spec['description'],
                    parameters=f_spec['parameters']
                ))
            toolbox = Tool(function_declarations=gemini_funcs)
            model = genai.GenerativeModel('gemini-1.5-flash', tools=[toolbox])
        else:
            model = genai.GenerativeModel('gemini-1.5-flash')
            
        chat = model.start_chat()
        return await chat.send_message_async(prompt)

    query_dec = "Find documents about project beta"
    print(f"User Query: '{query_dec}'")
    
    response_dec = await chat_with_gemini(prompt=query_dec)
    
    for part in response_dec.parts:
        if fn := part.function_call:
            print(f"Gemini decided to call: {fn.name}")

if __name__ == "__main__":
    asyncio.run(main())
