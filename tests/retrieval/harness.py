"""Build the eval catalogue into a gantry and measure every modality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_gantry import AgentGantry
from agent_gantry.evaluation import (
    RetrievalReport,
    evaluate_retrieval,
    mcp_server_retriever,
    skill_retriever,
    tool_retriever,
)
from agent_gantry.schema.tool import ToolDefinition
from tests.retrieval import catalogue

BASELINE_PATH = Path(__file__).with_name("baseline.json")


async def build_gantry(**gantry_kwargs: Any) -> AgentGantry:
    """A gantry holding the whole eval catalogue: tools, skills and MCP servers.

    Args:
        **gantry_kwargs: Passed to :class:`AgentGantry`, e.g. ``embedder``,
            ``reranker`` or ``selector``: whatever backend is being measured.

    Raises:
        ImportError: If the optional ``mcp`` package is not installed. The MCP
            tools are a fifth of the catalogue, so the numbers without them
            would not be comparable to the baseline.
    """
    gantry = AgentGantry(**gantry_kwargs)
    mcp_tools = catalogue.mcp_tools()
    for tool in [*catalogue.native_tools(), *mcp_tools]:
        await gantry.add_tool(tool)
    await gantry.sync()
    await gantry.add_skills(catalogue.skills())
    by_server: dict[str, list[ToolDefinition]] = {}
    for tool in mcp_tools:
        by_server.setdefault(tool.metadata["mcp_server"], []).append(tool)
    for name, description in catalogue.MCP_SERVERS.items():
        gantry.register_mcp_server(name, url=f"https://{name}.example/mcp", description=description)
        if name in by_server:
            # Catalogue the server as a deployment would, with the fixture
            # standing in for the live server, so it is routed to by what it
            # lists as well as by its (deliberately vague) description.
            gantry._mcp_registry.get_client(name).list_tools = _lister(by_server[name])
            await gantry.list_server_tools(name)
    await gantry.sync_mcp_servers()
    return gantry


def _lister(tools: list[ToolDefinition]) -> Any:
    async def list_tools() -> list[ToolDefinition]:
        return list(tools)

    return list_tools


async def measure(
    *, k: int = 5, score_threshold: float = 0.0, **gantry_kwargs: Any
) -> list[RetrievalReport]:
    """Run the three labelled sets and return a report for each modality."""
    gantry = await build_gantry(**gantry_kwargs)
    threshold = score_threshold or None
    try:
        return [
            await evaluate_retrieval(
                tool_retriever(gantry, score_threshold=score_threshold),
                catalogue.TOOL_CASES,
                k=k,
                modality="tools",
            ),
            await evaluate_retrieval(
                skill_retriever(gantry, score_threshold=threshold),
                catalogue.SKILL_CASES,
                k=k,
                modality="skills",
            ),
            await evaluate_retrieval(
                mcp_server_retriever(gantry, score_threshold=threshold),
                catalogue.MCP_SERVER_CASES,
                k=k,
                modality="mcp_servers",
            ),
        ]
    finally:
        await gantry.close()


def flatten(reports: list[RetrievalReport]) -> dict[str, dict[str, Any]]:
    """Metrics per modality and per case kind, keyed ``"tools"``, ``"tools/paraphrase"``, ..."""
    flat: dict[str, dict[str, Any]] = {}
    for report in reports:
        flat[report.modality] = report.to_dict()
        for part in report.by_kind().values():
            flat[part.modality] = part.to_dict()
    return flat


def load_baseline() -> dict[str, Any]:
    """The committed baseline: its embedder, ``k`` and per-modality metrics."""
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
