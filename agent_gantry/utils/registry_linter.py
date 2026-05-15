"""
Registry linter — flags common tool-description authoring mistakes.

Embedding similarity doesn't understand negation, and it doesn't
understand cross-tool references. A description that *names* another
tool — "Different from counting characters", "Prefer SHA-256 when …" —
will be semantically pulled *toward* the queries that the named tool
was meant to differentiate from. This module catches those mistakes
before they ship.

The two entry points are:

- :func:`analyze_registry` — Python API, returns a structured
  :class:`RegistryAnalysis` so callers can post-process the findings
  (pre-commit hooks, CI checks, dashboards).
- :func:`pairwise_similarity` — score two registered tools head-to-head.

The CLI mirrors both as ``gantry lint`` and ``gantry sim``.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_gantry.adapters.embedders.base import EmbeddingAdapter
    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.schema.tool import ToolDefinition


@dataclass
class CrossReferenceFinding:
    """A tool's text mentions another registered tool by name."""

    tool: str
    references: list[str]


@dataclass
class SimilarPairFinding:
    """Two tools whose searchable texts are highly similar."""

    tool_a: str
    tool_b: str
    cosine: float


@dataclass
class OverlappingTagFinding:
    """A tag that appears on more than a threshold number of tools."""

    tag: str
    tools: list[str]


@dataclass
class RegistryAnalysis:
    """Aggregated analysis output."""

    cross_references: list[CrossReferenceFinding] = field(default_factory=list)
    similar_pairs: list[SimilarPairFinding] = field(default_factory=list)
    overlapping_tags: list[OverlappingTagFinding] = field(default_factory=list)
    embedder_used: str | None = None

    @property
    def empty(self) -> bool:
        return (
            not self.cross_references
            and not self.similar_pairs
            and not self.overlapping_tags
        )

    def format_text(self) -> str:
        """Human-readable rendering for the CLI."""
        if self.empty:
            return "No issues found."
        lines: list[str] = []
        if self.cross_references:
            lines.append("Cross-references (descriptions naming other tools):")
            for f in self.cross_references:
                lines.append(f"  - {f.tool}: references {', '.join(f.references)}")
        if self.similar_pairs:
            lines.append("")
            lines.append("Similar tool pairs (cosine > threshold):")
            for f in self.similar_pairs:
                lines.append(f"  - {f.tool_a} ⇄ {f.tool_b}: {f.cosine:.3f}")
        if self.overlapping_tags:
            lines.append("")
            lines.append("Overlapping tags (low discriminative value):")
            for f in self.overlapping_tags:
                lines.append(f"  - {f.tag}: {', '.join(f.tools)}")
        return "\n".join(lines)


def _searchable_text(tool: ToolDefinition) -> str:
    return tool.to_searchable_text()


def _qualified(tool: ToolDefinition) -> str:
    return f"{tool.namespace}.{tool.name}"


def _detect_cross_references(
    tools: list[ToolDefinition],
) -> list[CrossReferenceFinding]:
    findings: list[CrossReferenceFinding] = []
    names = {t.name for t in tools}
    # Pre-compile one pattern per tool name so we don't pay the regex
    # compilation cost inside the nested loop. For a registry with N
    # tools, this turns an O(N²) compile cost into O(N).
    # Match whole-word, case-insensitive, with non-word boundaries so we
    # don't match the tool's *own* name when it's a substring of another.
    # Short names (< 3 chars) produce too many false positives — skip them.
    patterns: dict[str, re.Pattern[str]] = {
        n: re.compile(rf"(?<!\w){re.escape(n.lower())}(?!\w)")
        for n in names
        if len(n) >= 3
    }
    # Tokens to scan: description + extended_description + tags + examples.
    for tool in tools:
        text_blobs: list[str] = [tool.description or ""]
        if tool.extended_description:
            text_blobs.append(tool.extended_description)
        text_blobs.extend(tool.tags or [])
        text_blobs.extend(tool.examples or [])
        full = " ".join(text_blobs).lower()
        refs: list[str] = []
        for other, pattern in patterns.items():
            if other == tool.name:
                continue
            if pattern.search(full):
                refs.append(other)
        if refs:
            findings.append(CrossReferenceFinding(tool=tool.name, references=refs))
    return findings


def _detect_overlapping_tags(
    tools: list[ToolDefinition], *, max_share: float = 0.5
) -> list[OverlappingTagFinding]:
    if not tools:
        return []
    tag_to_tools: dict[str, list[str]] = defaultdict(list)
    for tool in tools:
        for tag in tool.tags or []:
            tag_to_tools[tag].append(tool.name)
    cutoff = max(2, math.ceil(len(tools) * max_share))
    return [
        OverlappingTagFinding(tag=tag, tools=names)
        for tag, names in sorted(tag_to_tools.items())
        if len(names) >= cutoff
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


async def _detect_similar_pairs(
    tools: list[ToolDefinition],
    embedder: EmbeddingAdapter,
    *,
    similarity_threshold: float,
) -> list[SimilarPairFinding]:
    if len(tools) < 2:
        return []
    texts = [_searchable_text(t) for t in tools]
    embeddings = await embedder.embed_batch(texts)
    findings: list[SimilarPairFinding] = []
    for i in range(len(tools)):
        for j in range(i + 1, len(tools)):
            score = _cosine(embeddings[i], embeddings[j])
            if score >= similarity_threshold:
                findings.append(
                    SimilarPairFinding(
                        tool_a=_qualified(tools[i]),
                        tool_b=_qualified(tools[j]),
                        cosine=score,
                    )
                )
    findings.sort(key=lambda f: f.cosine, reverse=True)
    return findings


async def analyze_registry(
    gantry: AgentGantry,
    *,
    similarity_threshold: float = 0.85,
    tag_overlap_share: float = 0.5,
    embedder: EmbeddingAdapter | None = None,
) -> RegistryAnalysis:
    """Analyse a gantry's registry for common authoring mistakes.

    Args:
        gantry: The :class:`AgentGantry` instance to analyse.
        similarity_threshold: Pairs of tools whose searchable-text
            cosine similarity is at least this value are flagged as
            potential merge / differentiate candidates. Defaults to
            ``0.85``.
        tag_overlap_share: Tags that appear on more than this fraction
            of the registry are flagged as having low discriminative
            value. Defaults to ``0.5``.
        embedder: Override the embedder used for similarity. Defaults
            to the gantry's configured embedder.

    Returns:
        A :class:`RegistryAnalysis` summarising the findings.
    """
    tools = gantry.list_tools_sync()
    cross_refs = _detect_cross_references(tools)
    tag_overlaps = _detect_overlapping_tags(tools, max_share=tag_overlap_share)

    eff_embedder = embedder if embedder is not None else gantry.embedder
    similar_pairs = await _detect_similar_pairs(
        tools, eff_embedder, similarity_threshold=similarity_threshold
    )
    embedder_id = getattr(eff_embedder, "model_name", type(eff_embedder).__name__)
    return RegistryAnalysis(
        cross_references=cross_refs,
        similar_pairs=similar_pairs,
        overlapping_tags=tag_overlaps,
        embedder_used=str(embedder_id),
    )


async def pairwise_similarity(
    gantry: AgentGantry,
    tool_a: str,
    tool_b: str,
    *,
    embedder: EmbeddingAdapter | None = None,
) -> float:
    """Return the cosine similarity between two registered tools' texts.

    Args:
        gantry: The gantry instance.
        tool_a: Name (or ``namespace.name``) of the first tool.
        tool_b: Name (or ``namespace.name``) of the second tool.
        embedder: Override the embedder used. Defaults to the gantry's.

    Raises:
        LookupError: If either tool is not registered.
    """
    tools = gantry.list_tools_sync()
    lookup: dict[str, ToolDefinition] = {}
    for t in tools:
        lookup[t.name] = t
        lookup[f"{t.namespace}.{t.name}"] = t
    if tool_a not in lookup:
        raise LookupError(f"Tool {tool_a!r} not found in registry.")
    if tool_b not in lookup:
        raise LookupError(f"Tool {tool_b!r} not found in registry.")
    eff_embedder = embedder if embedder is not None else gantry.embedder
    texts = [
        _searchable_text(lookup[tool_a]),
        _searchable_text(lookup[tool_b]),
    ]
    embs = await eff_embedder.embed_batch(texts)
    return _cosine(embs[0], embs[1])


__all__ = [
    "CrossReferenceFinding",
    "OverlappingTagFinding",
    "RegistryAnalysis",
    "SimilarPairFinding",
    "analyze_registry",
    "pairwise_similarity",
]
