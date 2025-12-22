"""Test module with duplicate tool name for testing."""

from agent_gantry import AgentGantry

# Create a gantry instance with a tool that has same name as module_a
tools = AgentGantry()


@tools.register
def tool_a1(x: int) -> int:
    """Duplicate tool from module C."""
    return x * 3  # Different implementation
