"""Test module with custom attribute name."""

from dotfetch import DotFetch

# Create a fetcher instance with a custom attribute name
my_custom_tools = DotFetch()


@my_custom_tools.register
def custom_tool(x: int) -> str:
    """Tool with custom attribute name."""
    return f"Result: {x}"
