"""Test module A with tools for import testing."""

from dotfetch import DotFetch

# Create a fetcher instance with some tools
tools = DotFetch()


@tools.register
def tool_a1(x: int) -> int:
    """First tool from module A."""
    return x * 2


@tools.register
def tool_a2(y: str) -> str:
    """Second tool from module A."""
    return y.upper()
