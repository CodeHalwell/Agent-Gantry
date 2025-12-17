"""
Servers for Agent-Gantry.

MCP and A2A server implementations.
"""

from agent_gantry.servers.a2a_server import create_a2a_server, generate_agent_card

__all__ = ["create_a2a_server", "generate_agent_card"]

# Optional MCP server (requires mcp package)
try:
    from agent_gantry.servers.mcp_server import (  # noqa: F401
        MCPServer,
        create_mcp_server,
    )

    __all__.extend(["MCPServer", "create_mcp_server"])
except ImportError:
    # MCP server is optional; ignore if the required package is not installed.
    pass
