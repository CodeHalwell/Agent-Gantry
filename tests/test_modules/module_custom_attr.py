"""Test module with custom attribute name."""

from agent_gantry import AgentGantry

# Create a gantry instance with a custom attribute name
my_custom_tools = AgentGantry()


@my_custom_tools.register
def custom_tool(x: int) -> str:
    """Tool with custom attribute name."""
    return f"Result: {x}"
