import asyncio

from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agent_gantry import AgentGantry
from agent_gantry.crewai import CrewAIAdapter

load_dotenv()


async def main():
    # 1. Initialize Agent-Gantry
    gantry = AgentGantry()

    @gantry.register
    def get_customer_info(email: str):
        """Retrieve customer details from the CRM."""
        return {"name": "John Doe", "tier": "Gold", "email": email}

    await gantry.sync()

    # 2. Select relevant tools and get them as native CrewAI BaseTools in one
    #    call — CrewAIAdapter.select handles retrieval, conversion, and
    #    execution wiring (no manual factory or name-based branching needed).
    user_query = "Get info for customer john@example.com"
    # Lowering threshold for SimpleEmbedder compatibility in this example
    crew_tools = await CrewAIAdapter(gantry).select(
        user_query, limit=1, score_threshold=0.1
    )

    # 4. Define CrewAI Agent
    llm = ChatOpenAI(model="gpt-5.5")

    researcher = Agent(
        role="Customer Success Researcher",
        goal="Find and analyze customer information",
        backstory="You are an expert in CRM systems and customer data.",
        tools=crew_tools,
        llm=llm,
        verbose=True,
    )

    # 5. Define Task
    task = Task(
        description=f"Research the customer with query: {user_query}",
        expected_output="A summary of the customer's profile and tier.",
        agent=researcher,
    )

    # 6. Run Crew
    crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential)

    print("--- Starting CrewAI with Agent-Gantry ---")
    result = await crew.kickoff_async()
    print(f"\nCrewAI Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
