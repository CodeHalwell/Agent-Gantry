"""
Main CLI entry point for Agent-Gantry.
"""

from __future__ import annotations

import sys


def main() -> int:
    """
    Main entry point for the Agent-Gantry CLI.

    Returns:
        Exit code
    """
    print("Agent-Gantry CLI")
    print("================")
    print()
    print("Available commands:")
    print("  serve     - Run as MCP server (stdio or SSE)")
    print("  list      - List registered tools")
    print("  search    - Search for tools")
    print("  benchmark - Run performance benchmarks")
    print()
    print("Use 'agent-gantry <command> --help' for more information.")
    print()
    print("Note: Full CLI implementation coming in a future release.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
