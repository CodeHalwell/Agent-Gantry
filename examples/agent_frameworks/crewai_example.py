import asyncio
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

from agent_gantry import AgentGantry
from agent_gantry.integrations.framework_adapters import fetch_framework_tools
from agent_gantry.schema.execution import ToolCall

load_dotenv()

async def main():
    # 1. Initialize Agent-Gantry
    gantry = AgentGantry()

    @gantry.register
    def get_customer_info(email: str):
        """Retrieve customer details from the CRM."""
        return {"name": "John Doe", "tier": "Gold", "email": email}

    await gantry.sync()

    # 2. Fetch tools for the task
    user_query = "Get info for customer john@example.com"
    # Lowering threshold for SimpleEmbedder compatibility in this example
    tools_schema = await fetch_framework_tools(gantry, user_query, framework="crew_ai", score_threshold=0.1)

    # 3. Wrap Gantry tools for CrewAI
    from crewai.tools import tool

    def make_crew_tool(tool_name: str, tool_desc: str, gantry_instance: AgentGantry):
        """Factory function to properly bind tool name to CrewAI tool wrapper."""
        @tool(tool_name)
        async def tool_wrapper(**kwargs):
            result = await gantry_instance.execute(ToolCall(tool_name=tool_name, arguments=kwargs))
            return result.result if result.status == "success" else result.error
        tool_wrapper.__doc__ = tool_desc
        return tool_wrapper

    crew_tools = []
    for ts in tools_schema:
        name = ts["function"]["name"]
        desc = ts["function"]["description"]
        
        if name == "get_customer_info":
            crew_tools.append(make_crew_tool(name, desc, gantry))

    # 4. Define CrewAI Agent
    llm = ChatOpenAI(model="gpt-4o")
    
    researcher = Agent(
        role='Customer Success Researcher',
        goal='Find and analyze customer information',
        backstory='You are an expert in CRM systems and customer data.',
        tools=crew_tools,
        llm=llm,
        verbose=True
    )

    # 5. Define Task
    task = Task(
        description=f"Research the customer with query: {user_query}",
        expected_output="A summary of the customer's profile and tier.",
        agent=researcher
    )

    # 6. Run Crew
    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential
    )

    print("--- Starting CrewAI with Agent-Gantry ---")
    result = await crew.kickoff_async()
    print(f"\nCrewAI Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
