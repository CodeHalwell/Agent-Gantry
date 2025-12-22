"""Test module B with tools for import testing."""

from agent_gantry import AgentGantry

# Create a gantry instance with some tools
tools = AgentGantry()


@tools.register
def tool_b1(a: float) -> float:
    """First tool from module B."""
    return a * 3.14


@tools.register
def tool_b2(b: list[int]) -> int:
    """Second tool from module B."""
    return sum(b)
