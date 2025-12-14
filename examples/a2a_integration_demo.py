"""
A2A (Agent-to-Agent) Integration Demo.

This example demonstrates:
1. Consuming external A2A agents as tools
2. Exposing AgentGantry as an A2A agent
3. Agent Card generation
"""

import asyncio

from agent_gantry import AgentGantry
from agent_gantry.schema.config import A2AAgentConfig


async def demo_a2a_client():
    """Demonstrate consuming an external A2A agent."""
    print("\n=== A2A Client Demo ===\n")

    # Initialize AgentGantry
    gantry = AgentGantry()

    # Register some local tools
    @gantry.register(tags=["math"])
    def calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers."""
        return a + b

    @gantry.register(tags=["math"])
    def calculate_product(a: int, b: int) -> int:
        """Calculate the product of two numbers."""
        return a * b

    await gantry.sync()

    # Configure an external A2A agent
    # Note: This is a hypothetical external agent - replace with actual agent URL
    external_agent = A2AAgentConfig(
        name="translation-agent",
        url="https://translation-agent.example.com",
        namespace="external",
    )

    print(f"Configured external A2A agent: {external_agent.name}")
    print(f"Agent URL: {external_agent.url}")

    # In a real scenario, you would discover and register the agent:
    # count = await gantry.add_a2a_agent(external_agent)
    # print(f"Discovered {count} skills from external agent")

    # List all tools including external agent skills
    tools = await gantry.list_tools()
    print(f"\nTotal tools available: {len(tools)}")

    for tool in tools:
        print(f"  - {tool.name} ({tool.source.value}): {tool.description[:60]}...")


async def demo_agent_card():
    """Demonstrate Agent Card generation."""
    print("\n=== Agent Card Demo ===\n")

    from agent_gantry.servers.a2a_server import generate_agent_card

    # Initialize AgentGantry with some tools
    gantry = AgentGantry()

    @gantry.register(tags=["data", "query"])
    def query_database(query: str) -> str:
        """Query a database and return results."""
        return f"Results for: {query}"

    @gantry.register(tags=["communication"])
    def send_notification(message: str, recipient: str) -> str:
        """Send a notification to a recipient."""
        return f"Sent '{message}' to {recipient}"

    await gantry.sync()

    # Generate Agent Card
    agent_card = generate_agent_card(gantry, "http://localhost:8080")

    print(f"Agent Name: {agent_card.name}")
    print(f"Description: {agent_card.description}")
    print(f"URL: {agent_card.url}")
    print(f"Version: {agent_card.version}")
    print(f"\nSkills ({len(agent_card.skills)}):")

    for skill in agent_card.skills:
        print(f"\n  Skill ID: {skill.id}")
        print(f"  Name: {skill.name}")
        print(f"  Description: {skill.description}")
        print(f"  Input Modes: {', '.join(skill.input_modes)}")
        print(f"  Output Modes: {', '.join(skill.output_modes)}")

    # The agent card would be served at /.well-known/agent.json
    print("\n\nAgent Card JSON:")
    import json

    print(json.dumps(agent_card.model_dump(), indent=2))


async def demo_a2a_server():
    """Demonstrate starting an A2A server."""
    print("\n=== A2A Server Demo ===\n")

    # Note: This requires FastAPI and uvicorn to be installed
    # pip install fastapi uvicorn

    gantry = AgentGantry()

    @gantry.register(tags=["analysis"])
    def analyze_data(data: str) -> str:
        """Analyze provided data and return insights."""
        return f"Analysis of '{data}': [insights here]"

    await gantry.sync()

    print("To start the A2A server, use:")
    print("\n  gantry.serve_a2a(host='0.0.0.0', port=8080)")
    print("\nThe server will expose:")
    print("  - Agent Card: http://localhost:8080/.well-known/agent.json")
    print("  - Task endpoint: http://localhost:8080/tasks/send")
    print("\nSkills available:")
    print("  - tool_discovery: Find relevant tools using semantic search")
    print("  - tool_execution: Execute tools with retries and circuit breakers")

    # Uncomment to actually run the server:
    # gantry.serve_a2a(host="0.0.0.0", port=8080)


async def demo_skill_to_tool_mapping():
    """Demonstrate how A2A skills are mapped to tools."""
    print("\n=== Skill to Tool Mapping Demo ===\n")

    from agent_gantry.providers.a2a_client import A2AClient
    from agent_gantry.schema.a2a import AgentCard, AgentSkill

    # Create a mock agent card
    mock_card = AgentCard(
        name="DataAgent",
        description="Agent for data processing",
        url="http://data-agent.example.com",
        skills=[
            AgentSkill(
                id="extract_data",
                name="Data Extraction",
                description="Extract structured data from unstructured text",
                input_modes=["text"],
                output_modes=["json"],
            ),
            AgentSkill(
                id="transform_data",
                name="Data Transformation",
                description="Transform data from one format to another",
                input_modes=["json", "csv"],
                output_modes=["json", "csv"],
            ),
        ],
    )

    config = A2AAgentConfig(
        name="data-agent",
        url="http://data-agent.example.com",
        namespace="data_ops",
    )

    client = A2AClient(config)
    # WARNING: Directly setting a private attribute (_agent_card) is for demonstration purposes only.
    # Do NOT use this pattern in production code. Always use public APIs when available.
    client._agent_card = mock_card  # Set directly for demo

    # Convert skills to tools
    tools = await client.list_tools()

    print(f"Agent: {mock_card.name}")
    print(f"Skills: {len(mock_card.skills)}")
    print(f"Tools created: {len(tools)}")
    print("\nMapping:")

    for skill, tool in zip(mock_card.skills, tools):
        print(f"\n  Skill: {skill.id}")
        print(f"  → Tool Name: {tool.name}")
        print(f"  → Namespace: {tool.namespace}")
        print(f"  → Source: {tool.source.value}")
        print(f"  → Source URI: {tool.source_uri}")
        print(f"  → Metadata:")
        for key, value in tool.metadata.items():
            print(f"      {key}: {value}")


async def main():
    """Run all A2A demos."""
    print("=" * 70)
    print("Agent-Gantry A2A Integration Demo")
    print("=" * 70)

    await demo_agent_card()
    await demo_a2a_client()
    await demo_skill_to_tool_mapping()
    await demo_a2a_server()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
