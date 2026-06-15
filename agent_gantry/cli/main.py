"""
Main CLI entry point for Agent-Gantry.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from agent_gantry import AgentGantry
from agent_gantry.schema.query import ConversationContext, ToolQuery

# Default skill install target, and Claude Code's personal skills directory
# (searched at startup). install_to() expands the leading ~ via Path.expanduser().
_DEFAULT_SKILL_TARGET = "./skills"
_CLAUDE_SKILLS_DIR = "~/.claude/skills"


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

    lint_parser = subparsers.add_parser(
        "lint",
        help="Detect tool-description authoring mistakes",
    )
    lint_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Cosine threshold above which two tools are flagged as similar (default 0.85).",
    )
    lint_parser.add_argument(
        "--tag-overlap-share",
        type=float,
        default=0.5,
        help="Tag flagged when it appears on more than this fraction of tools (default 0.5).",
    )

    sim_parser = subparsers.add_parser(
        "sim",
        help="Print the cosine similarity between two registered tools",
    )
    sim_parser.add_argument("tool_a", help="First tool name (or namespace.name)")
    sim_parser.add_argument("tool_b", help="Second tool name (or namespace.name)")

    sync_parser = subparsers.add_parser(
        "sync",
        help="Sync tool embeddings into the configured vector store",
    )
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which tools would be (re-)embedded and why, without doing it.",
    )
    sync_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-sync of all tools regardless of fingerprint match.",
    )

    skill_parser = subparsers.add_parser(
        "install-skill",
        help="Install the bundled Agent-Gantry Claude Skill into a target directory",
    )
    skill_dest = skill_parser.add_mutually_exclusive_group()
    skill_dest.add_argument(
        "--target",
        default=None,
        help=f"Destination directory (default: {_DEFAULT_SKILL_TARGET}). "
        "Mutually exclusive with --claude.",
    )
    skill_dest.add_argument(
        "--claude",
        action="store_true",
        help=f"Install into Claude's personal skills directory ({_CLAUDE_SKILLS_DIR}) "
        "so Claude Code discovers it automatically.",
    )
    skill_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing agent-gantry directory in the target.",
    )
    skill_parser.add_argument(
        "--print-path",
        action="store_true",
        help="Just print the path to the bundled skill (no copy).",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # install-skill is a pure file-copy operation; it doesn't need a
    # gantry instance or demo tool registration.
    if args.command == "install-skill":
        gantry = None  # type: ignore[assignment]
    else:
        gantry = AgentGantry()
        _load_demo_tools(gantry)
        # Defer the eager sync until we know the command actually needs it —
        # lint/sim/sync --dry-run shouldn't trigger embedding work.
        if args.command not in ("lint", "sim", "sync"):
            asyncio.run(gantry.sync())

    if args.command == "list":
        tools = asyncio.run(gantry.list_tools(namespace=args.namespace))
        for tool in tools:
            print(f"{tool.namespace}.{tool.name}: {tool.description}")
        return 0

    if args.command == "search":
        context = ConversationContext(query=args.query)
        query = ToolQuery(context=context, limit=args.limit, score_threshold=0.0)
        result = asyncio.run(gantry.retrieve(query))
        if not result.tools:
            print("No tools found.")
            return 0
        for scored in result.tools:
            print(f"{scored.tool.name} ({scored.semantic_score:.2f}) - {scored.tool.description}")
        return 0

    if args.command == "lint":
        analysis = asyncio.run(
            gantry.analyze_registry(
                similarity_threshold=args.similarity_threshold,
                tag_overlap_share=args.tag_overlap_share,
            )
        )
        print(analysis.format_text())
        return 1 if not analysis.empty else 0

    if args.command == "sim":
        try:
            score = asyncio.run(
                gantry.pairwise_similarity(args.tool_a, args.tool_b)
            )
        except LookupError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"{args.tool_a} ⇄ {args.tool_b}: {score:.4f}")
        return 0

    if args.command == "sync":
        return asyncio.run(_run_sync_command(gantry, dry_run=args.dry_run, force=args.force))

    if args.command == "install-skill":
        from agent_gantry.skills import install_to, skill_path

        if args.print_path:
            try:
                print(skill_path())
                return 0
            except FileNotFoundError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
        target = _CLAUDE_SKILLS_DIR if args.claude else (args.target or _DEFAULT_SKILL_TARGET)
        try:
            dst = install_to(target, overwrite=args.overwrite)
        except FileExistsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            print("  Re-run with --overwrite to replace.", file=sys.stderr)
            return 2
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"Installed Agent-Gantry skill to {dst}")
        return 0

    parser.print_help()
    return 0


async def _run_sync_command(
    gantry: AgentGantry,
    *,
    dry_run: bool,
    force: bool,
) -> int:
    """Run the ``gantry sync`` subcommand.

    In ``--dry-run`` mode, queries the SyncManager for the set of tools
    whose fingerprints don't match what's stored, and reports them
    without invoking the embedder.
    """
    # Touching ``ensure_synced`` triggers the embedding work we are
    # trying to avoid in dry-run mode. Use the lower-level
    # ``detect_changes`` path instead.
    await gantry._ensure_initialized()
    sync_mgr = gantry._sync_manager
    all_tools = gantry.export_tools()
    to_sync = await sync_mgr.detect_changes(all_tools, force=force)
    if dry_run:
        if not to_sync:
            print("Up to date — no tools would be (re-)embedded.")
            return 0
        print(f"{len(to_sync)} tool(s) would be (re-)embedded:")
        stored = await gantry._vector_store.get_stored_fingerprints()
        for tool in to_sync:
            tool_id = f"{tool.namespace}.{tool.name}"
            reason = "new" if tool_id not in stored else "fingerprint changed"
            print(f"  - {tool_id}: {reason}")
        return 0

    count = await gantry.sync(force=force)
    print(f"Synced {count} tool(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
