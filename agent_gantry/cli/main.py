"""
Main CLI entry point for Agent-Gantry.

Every inspection command works against a *real* registry: point ``--module``
at the Python module that holds your ``AgentGantry`` instance (``pkg.tools``
or ``pkg.tools:my_gantry``) and ``list``/``search``/``lint``/``sim``/``sync``
operate on that instance — its embedder, its vector store, its tools.
``serve-mcp`` exposes the same registry to Claude Desktop, Claude Code and
any other MCP client over stdio or HTTP. Without ``--module`` a small demo
registry is used so the commands can be tried before any code exists.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from collections.abc import Sequence

from agent_gantry import AgentGantry
from agent_gantry.schema.query import ConversationContext, ToolQuery

# Default skill install target, and Claude Code's personal skills directory
# (searched at startup). install_to() expands the leading ~ via Path.expanduser().
_DEFAULT_SKILL_TARGET = "./skills"
_CLAUDE_SKILLS_DIR = "~/.claude/skills"
_DEFAULT_MODULE_ATTR = "tools"


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


def _split_module_spec(spec: str, default_attr: str) -> tuple[str, str]:
    """Split ``pkg.mod:attr`` into ``(module, attr)``; the attr is optional."""
    module, _, attr = spec.partition(":")
    return module.strip(), (attr.strip() or default_attr)


def _import_gantry(spec: str, default_attr: str) -> AgentGantry:
    """Import the ``AgentGantry`` instance a ``--module`` spec names."""
    module_path, attr = _split_module_spec(spec, default_attr)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise SystemExit(f"error: could not import module '{module_path}': {exc}") from exc
    gantry = getattr(module, attr, None)
    if not isinstance(gantry, AgentGantry):
        found = type(gantry).__name__ if gantry is not None else "nothing"
        raise SystemExit(
            f"error: '{module_path}' has no AgentGantry at attribute '{attr}' (found {found}). "
            f"Use --module {module_path}:<attr> to name the instance."
        )
    return gantry


def build_gantry(
    modules: Sequence[str] | None,
    *,
    attr: str = _DEFAULT_MODULE_ATTR,
    config: str | None = None,
    quiet: bool = False,
) -> AgentGantry:
    """Resolve the gantry a CLI invocation should operate on.

    - One ``--module``: that module's own instance is used directly, so the
      command sees the user's configured embedder and vector store.
    - Several: their tools are collected into one fresh gantry (built from
      ``--config`` when given), the same merge ``AgentGantry.from_modules``
      performs.
    - None: the demo registry, with a note on stderr so nobody mistakes it
      for their own tools.
    """
    modules = list(modules or [])
    base_config = None
    if config:
        from agent_gantry.schema.config import AgentGantryConfig

        base_config = AgentGantryConfig.from_yaml(config)

    if len(modules) == 1 and base_config is None:
        return _import_gantry(modules[0], attr)

    if modules:
        gantry = AgentGantry(config=base_config)

        async def _collect() -> None:
            for spec in modules:
                module_path, module_attr = _split_module_spec(spec, attr)
                _import_gantry(spec, attr)  # fail early, with the CLI's own message
                await gantry.collect_tools_from_modules([module_path], module_attr=module_attr)

        asyncio.run(_collect())
        return gantry

    gantry = AgentGantry(config=base_config)
    _load_demo_tools(gantry)
    if not quiet:
        print(
            "note: no --module given; using the built-in demo tools. "
            "Pass --module pkg.tools[:attr] to inspect your own registry.",
            file=sys.stderr,
        )
    return gantry


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--module",
        "-m",
        action="append",
        default=None,
        metavar="MODULE[:ATTR]",
        help="Module holding your AgentGantry instance (repeatable). "
        f"ATTR defaults to '{_DEFAULT_MODULE_ATTR}'.",
    )
    common.add_argument(
        "--attr",
        default=_DEFAULT_MODULE_ATTR,
        help=f"Attribute name for --module entries without ':ATTR' (default {_DEFAULT_MODULE_ATTR}).",
    )
    common.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="YAML config (AgentGantryConfig) for the gantry the CLI builds.",
    )

    parser = argparse.ArgumentParser(prog="agent-gantry", description="Agent-Gantry CLI")
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", parents=[common], help="List registered tools")
    list_parser.add_argument("--namespace", default=None, help="Namespace filter")

    search_parser = subparsers.add_parser(
        "search", parents=[common], help="Search for relevant tools"
    )
    search_parser.add_argument("query", help="Natural language query")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum tools to return")
    search_parser.add_argument("--namespace", default=None, help="Namespace filter")

    lint_parser = subparsers.add_parser(
        "lint",
        parents=[common],
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
        parents=[common],
        help="Print the cosine similarity between two registered tools",
    )
    sim_parser.add_argument("tool_a", help="First tool name (or namespace.name)")
    sim_parser.add_argument("tool_b", help="Second tool name (or namespace.name)")

    sync_parser = subparsers.add_parser(
        "sync",
        parents=[common],
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
    sync_parser.add_argument(
        "--prune",
        action="store_true",
        default=None,
        help="Also delete stored tools that are no longer registered "
        "(default: the config's prune_on_sync).",
    )

    serve_parser = subparsers.add_parser(
        "serve-mcp",
        parents=[common],
        help="Expose the registry as an MCP server (stdio, Streamable HTTP or SSE)",
    )
    serve_parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse"],
        default="stdio",
        help="stdio for Claude Desktop/Claude Code; http (Streamable HTTP) or sse for remote clients.",
    )
    serve_parser.add_argument(
        "--mode",
        choices=["dynamic", "static", "hybrid"],
        default="dynamic",
        help="dynamic: two meta-tools; static: every tool; hybrid: --expose tools plus meta-tools.",
    )
    serve_parser.add_argument(
        "--expose",
        action="append",
        default=None,
        metavar="TOOL",
        help="Tool listed directly in hybrid mode (name or namespace.name; repeatable).",
    )
    serve_parser.add_argument("--name", default="agent-gantry", help="MCP server name.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host for http/sse.")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port for http/sse.")
    serve_parser.add_argument("--path", default="/mcp", help="Endpoint path for http.")

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
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the Agent-Gantry CLI.

    Returns:
        Exit code
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "install-skill":
        return _run_install_skill(args)

    # stdio MCP serving owns stdout, so the demo-registry note (stderr) is the
    # only thing the CLI may print before the protocol starts.
    gantry = build_gantry(args.module, attr=args.attr, config=args.config)

    if args.command == "list":
        tools = gantry.list_tools_sync(namespace=args.namespace)
        for tool in sorted(tools, key=lambda t: (t.namespace, t.name)):
            print(f"{tool.namespace}.{tool.name}: {tool.description}")
        return 0

    if args.command == "search":
        context = ConversationContext(query=args.query)
        query = ToolQuery(
            context=context,
            limit=args.limit,
            score_threshold=0.0,
            namespaces=[args.namespace] if args.namespace else None,
        )
        result = asyncio.run(gantry.retrieve(query))
        if not result.tools:
            print("No tools found.")
            return 0
        for scored in result.tools:
            tool = scored.tool
            print(f"{tool.namespace}.{tool.name} ({scored.semantic_score:.2f}) - {tool.description}")
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
            score = asyncio.run(gantry.pairwise_similarity(args.tool_a, args.tool_b))
        except LookupError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"{args.tool_a} ⇄ {args.tool_b}: {score:.4f}")
        return 0

    if args.command == "sync":
        return asyncio.run(
            _run_sync_command(gantry, dry_run=args.dry_run, force=args.force, prune=args.prune)
        )

    if args.command == "serve-mcp":
        return _run_serve_mcp(gantry, args)

    parser.print_help()
    return 0


def _run_install_skill(args: argparse.Namespace) -> int:
    """Run the ``install-skill`` subcommand (a pure file copy)."""
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


def _run_serve_mcp(gantry: AgentGantry, args: argparse.Namespace) -> int:
    """Run the ``serve-mcp`` subcommand until interrupted."""
    try:
        import mcp  # noqa: F401
    except ImportError:
        print(
            "error: MCP support is not installed. Install it with "
            "'pip install agent-gantry[mcp]'.",
            file=sys.stderr,
        )
        return 2
    transport = "streamable_http" if args.transport == "http" else args.transport
    if transport != "stdio":
        print(
            f"Serving MCP ({args.mode}) over {args.transport} at "
            f"http://{args.host}:{args.port}{args.path if transport == 'streamable_http' else '/sse'}",
            file=sys.stderr,
        )
    try:
        asyncio.run(
            gantry.serve_mcp(
                transport=transport,
                mode=args.mode,
                name=args.name,
                host=args.host,
                port=args.port,
                path=args.path,
                expose=args.expose,
            )
        )
    except KeyboardInterrupt:
        pass
    return 0


async def _run_sync_command(
    gantry: AgentGantry,
    *,
    dry_run: bool,
    force: bool,
    prune: bool | None = None,
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
    # ``--prune`` absent means "whatever the config says", so a config with
    # ``prune_on_sync: true`` is honoured and the dry run reports what the
    # real sync would actually do.
    effective_prune = gantry._config.prune_on_sync if prune is None else prune
    to_sync = await sync_mgr.detect_changes(all_tools, force=force)
    if dry_run:
        if not to_sync:
            print("Up to date — no tools would be (re-)embedded.")
        else:
            print(f"{len(to_sync)} tool(s) would be (re-)embedded:")
            stored = await gantry._vector_store.get_stored_fingerprints()
            for tool in to_sync:
                tool_id = f"{tool.namespace}.{tool.name}"
                reason = "new" if tool_id not in stored else "fingerprint changed"
                print(f"  - {tool_id}: {reason}")
        if effective_prune:
            wanted = {f"{t.namespace}.{t.name}" for t in all_tools}
            stale = [
                f"{t.namespace}.{t.name}"
                for t in await gantry._list_all_pages(gantry._vector_store.list_all)
                if t.namespace != "__mcp_servers__" and f"{t.namespace}.{t.name}" not in wanted
            ]
            if stale:
                print(f"{len(stale)} stale tool(s) would be pruned: {', '.join(sorted(stale))}")
            else:
                print("No stale tools to prune.")
        return 0

    count = await gantry.sync(force=force, prune=effective_prune)
    print(f"Synced {count} tool(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
