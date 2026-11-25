"""Test module with duplicate tool name for testing."""

from dotfetch import DotFetch

# Create a fetcher instance with a tool that has same name as module_a
tools = DotFetch()


@tools.register
def tool_a1(x: int) -> int:
    """Duplicate tool from module C."""
    return x * 3  # Different implementation
