"""
Main CLI entry point for Agent-Gantry.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from agent_gantry import AgentGantry
from agent_gantry.schema.query import ConversationContext, ToolQuery


def _load_demo_tools(gantry: AgentGantry) -> None:
    """Register a small set of demo tools for CLI usage."""

    @gantry.register(tags=["email", "communication"])
    def send_email(to: str, subject: str, body: str) -> str:
        """Send an email with a subject and body."""
        return f"Email sent to {to}"

    @gantry.register(tags=["report", "analytics"])
    def generate_report(report_type: str, start_date: str, end_date: str) -> str:
        """Generate a report for the given date range."""
        return f"Report {report_type} from {start_date} to {end_date}"

    @gantry.register(tags=["finance", "customer"])
    def process_refund(order_id: str, amount: float) -> str:
        """Process a refund for a given order."""
        return f"Refund {amount} for {order_id}"


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the Agent-Gantry CLI.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(prog="agent-gantry", description="Agent-Gantry CLI")
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List registered tools")
    list_parser.add_argument("--namespace", default=None, help="Namespace filter")

    search_parser = subparsers.add_parser("search", help="Search for relevant tools")
    search_parser.add_argument("query", help="Natural language query")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum tools to return")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    gantry = AgentGantry()
    _load_demo_tools(gantry)
    asyncio.run(gantry.sync())

    if args.command == "list":
        tools = asyncio.run(gantry.list_tools(namespace=args.namespace))
        for tool in tools:
            print(f"{tool.namespace}.{tool.name}: {tool.description}")
        return 0

    if args.command == "search":
        context = ConversationContext(query=args.query)
        query = ToolQuery(context=context, limit=args.limit)
        result = asyncio.run(gantry.retrieve(query))
        if not result.tools:
            print("No tools found.")
            return 0
        for scored in result.tools:
            print(f"{scored.tool.name} ({scored.semantic_score:.2f}) - {scored.tool.description}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
