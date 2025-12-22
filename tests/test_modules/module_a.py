"""Test module A with tools for import testing."""

from agent_gantry import AgentGantry

# Create a gantry instance with some tools
tools = AgentGantry()


@tools.register
def tool_a1(x: int) -> int:
    """First tool from module A."""
    return x * 2


@tools.register
def tool_a2(y: str) -> str:
    """Second tool from module A."""
    return y.upper()
