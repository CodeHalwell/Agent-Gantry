"""A duplicate of ``module_a``'s tool, under a non-default attribute.

Exists so duplicate detection can be tested across module specs that name
*different* attributes, which the CLI once collected in separate passes.
"""

from agent_gantry import AgentGantry

other_tools = AgentGantry()


@other_tools.register
def tool_a1(x: int) -> int:
    """Duplicate tool from module D, under a custom attribute."""
    return x * 5
