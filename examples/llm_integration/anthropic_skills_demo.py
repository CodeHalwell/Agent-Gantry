"""
Anthropic Skills Demo with Agent-Gantry.

This example demonstrates how to use the Skills abstraction with Agent-Gantry
to create reusable, composable skills that combine multiple tools.

Skills inject instructions into the system prompt and provide associated
tools to Claude, enabling coherent multi-tool workflows.

Requirements:
    pip install anthropic agent-gantry

Environment:
    Set ANTHROPIC_API_KEY in your environment
"""

import asyncio
import os

from dotenv import load_dotenv

from agent_gantry import AgentGantry
from agent_gantry.integrations.anthropic_skills import (
    create_skills_client,
)

load_dotenv()


async def demo_basic_skills():
    """Demonstrate basic skill creation and usage."""
    print("=" * 80)
    print("Demo 1: Basic Skills API")
    print("=" * 80)
    print()

    # Initialize Agent-Gantry with tools
    gantry = AgentGantry()

    @gantry.register
    def get_order(order_id: str) -> dict:
        """Get order details by ID."""
        # Mock order data
        return {
            "order_id": order_id,
            "customer": "John Doe",
            "items": ["Widget A", "Gadget B"],
            "total": 49.99,
            "status": "shipped",
        }

    @gantry.register
    def process_refund(order_id: str, amount: float) -> str:
        """Process a refund for an order."""
        return f"Refund of ${amount} processed for order {order_id}"

    @gantry.register
    def send_email(to: str, subject: str, body: str) -> str:
        """Send an email to a customer."""
        return f"Email sent to {to}: {subject}"

    await gantry.sync()

    # Create Skills client
    client = await create_skills_client(gantry=gantry)

    # Register a customer support skill
    client.skills.register(
        name="customer_support",
        description="Handle customer support inquiries including refunds and order tracking",
        instructions="""
        You are a customer support assistant. Use the available tools to:
        1. Look up order details using get_order
        2. Process refunds using process_refund
        3. Send confirmation emails using send_email

        Always be polite and helpful. Verify order details before processing refunds.
        """,
        tools=["get_order", "process_refund", "send_email"],
        examples=[
            {
                "input": "I need a refund for order #12345",
                "output": "I'll help you with that refund right away.",
            }
        ],
    )

    # Use the skill
    print("🎯 Using customer_support skill:")
    print()

    response = await client.create_message(
        model="claude-sonnet-4-5",
        messages=[
            {
                "role": "user",
                "content": "I need a refund for order #12345. The total was $49.99.",
            }
        ],
        skills=["customer_support"],
        max_tokens=2048,
    )

    # Process tool calls
    for block in response.content:
        if hasattr(block, "type"):
            if block.type == "text":
                print(f"💬 Claude: {block.text}")
            elif block.type == "tool_use":
                print(f"🔧 Tool: {block.name}({block.input})")

    # Execute tools if needed
    tool_results = await client.execute_tool_calls(response)
    if tool_results:
        print()
        print("📊 Tool Results:")
        for result in tool_results:
            print(f"  • {result['content']}")

    print()


async def demo_multi_skill_workflow():
    """Demonstrate using multiple skills together."""
    print("=" * 80)
    print("Demo 2: Multi-Skill Workflow")
    print("=" * 80)
    print()

    gantry = AgentGantry()

    # Register data analysis tools
    @gantry.register
    def query_database(query: str) -> list[dict]:
        """Query the database and return results."""
        # Mock data
        return [
            {"name": "Product A", "sales": 1500, "revenue": 45000},
            {"name": "Product B", "sales": 2300, "revenue": 69000},
            {"name": "Product C", "sales": 800, "revenue": 24000},
        ]

    @gantry.register
    def calculate_metrics(data: list[dict], metric: str) -> float:
        """Calculate statistical metrics from data."""
        if metric == "average_revenue":
            return sum(d["revenue"] for d in data) / len(data)
        elif metric == "total_sales":
            return sum(d["sales"] for d in data)
        return 0.0

    @gantry.register
    def create_chart(data: list[dict], chart_type: str) -> str:
        """Create a visualization of the data."""
        return f"Created {chart_type} chart with {len(data)} data points"

    await gantry.sync()

    client = await create_skills_client(gantry=gantry)

    # Register multiple skills
    client.skills.register(
        name="data_analysis",
        description="Analyze business data and calculate metrics",
        instructions="""
        Use query_database to fetch data, then use calculate_metrics to compute statistics.
        Always explain the insights from the data.
        """,
        tools=["query_database", "calculate_metrics"],
    )

    client.skills.register(
        name="data_visualization",
        description="Create visual representations of data",
        instructions="""
        Use create_chart to generate visualizations based on the data provided.
        Choose appropriate chart types based on the data and question.
        """,
        tools=["create_chart"],
    )

    # Use multiple skills together
    print("🎯 Using data_analysis and data_visualization skills:")
    print()

    response = await client.create_message(
        model="claude-sonnet-4-5",
        messages=[
            {
                "role": "user",
                "content": "What are our top performing products and can you create a chart?",
            }
        ],
        skills=["data_analysis", "data_visualization"],
        max_tokens=2048,
    )

    for block in response.content:
        if hasattr(block, "type") and block.type == "text":
            print(f"💬 Claude: {block.text}")

    print()


async def demo_skill_from_gantry_tools():
    """Demonstrate creating skills directly from Agent-Gantry tools."""
    print("=" * 80)
    print("Demo 3: Auto-Register Skills from Agent-Gantry")
    print("=" * 80)
    print()

    gantry = AgentGantry()

    # Register tools with semantic tags
    @gantry.register(tags=["math", "calculation"])
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @gantry.register(tags=["math", "calculation"])
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    @gantry.register(tags=["math", "calculation"])
    def divide(a: float, b: float) -> float:
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    await gantry.sync()

    client = await create_skills_client(gantry=gantry)

    # Register skill using Agent-Gantry tool names
    client.register_skill_from_gantry_tools(
        skill_name="math_operations",
        description="Perform mathematical calculations",
        instructions="""
        You have access to basic math operations: add, multiply, and divide.
        Break down complex calculations into simple steps using these tools.
        Always check for edge cases like division by zero.
        """,
        tool_names=["add", "multiply", "divide"],
        examples=[
            {
                "input": "Calculate (5 + 3) * 2",
                "steps": [
                    "Use add to get 5 + 3 = 8",
                    "Use multiply to get 8 * 2 = 16",
                ],
            }
        ],
    )

    print("🎯 Using math_operations skill:")
    print()

    response = await client.create_message(
        model="claude-sonnet-4-5",
        messages=[
            {
                "role": "user",
                "content": "Calculate (15 + 25) * 2 / 4",
            }
        ],
        skills=["math_operations"],
        max_tokens=1024,
    )

    for block in response.content:
        if hasattr(block, "type"):
            if block.type == "text":
                print(f"💬 Claude: {block.text}")
            elif block.type == "tool_use":
                print(f"🔧 {block.name}({block.input})")

    print()


async def demo_skill_registry():
    """Demonstrate skill registry management."""
    print("=" * 80)
    print("Demo 4: Skill Registry Management")
    print("=" * 80)
    print()

    client = await create_skills_client()

    # Register multiple skills
    client.skills.register(
        name="research",
        description="Research and gather information",
        instructions="Use search and summarization tools",
        tools=["search", "summarize"],
    )

    client.skills.register(
        name="writing",
        description="Write and edit content",
        instructions="Use writing and editing tools",
        tools=["write_draft", "edit_text"],
    )

    client.skills.register(
        name="analysis",
        description="Analyze data and generate insights",
        instructions="Use analysis and visualization tools",
        tools=["analyze_data", "create_chart"],
    )

    # List all skills
    print("📋 Registered Skills:")
    for skill in client.skills.list_skills():
        print(f"  • {skill.name}: {skill.description}")
        print(f"    Tools: {', '.join(skill.tools)}")
    print()

    # Get specific skill
    research_skill = client.skills.get("research")
    if research_skill:
        print(f"🔍 Details for '{research_skill.name}':")
        print(f"  Description: {research_skill.description}")
        print(f"  Instructions: {research_skill.instructions}")
        print(f"  Tools: {research_skill.tools}")
    print()

    # Convert to Anthropic schema
    print("📝 Anthropic Schema (first skill):")
    schema = client.skills.to_anthropic_schema()[0]
    import json
    print(json.dumps(schema, indent=2))
    print()


async def main():
    """Run all demos."""
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not set in environment")
        print("Set it to run the demos:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        return

    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "Anthropic Skills API Demo" + " " * 33 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    try:
        await demo_basic_skills()
        await demo_multi_skill_workflow()
        await demo_skill_from_gantry_tools()
        await demo_skill_registry()

        print("=" * 80)
        print("✅ All demos completed successfully!")
        print("=" * 80)
        print()
        print("Key Takeaways:")
        print("  • Skills provide reusable, composable tool workflows")
        print("  • Combine multiple Agent-Gantry tools into cohesive skills")
        print("  • Skills include instructions and examples for better results")
        print("  • Easy integration with existing Agent-Gantry tools")
        print("  • Full registry management for organizing skills")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
