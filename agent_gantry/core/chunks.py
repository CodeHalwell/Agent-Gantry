"""
Chunked indexing for text one vector cannot carry.

An embedder reads a bounded number of tokens: 256 word-pieces for the default
``all-MiniLM-L6-v2``, which is roughly the first thousand characters. A tool
whose description runs to several thousand, as MCP tools' often do, is
therefore searchable only by its opening lines, and real MCP servers put the
discriminating part at the end: "use this when...", the filter syntax, what the
tool must not be used for.

So long text is indexed a second time, in pieces. Each piece is stored as a
pseudo-entry in a reserved namespace, carrying a pointer to the entry it was
cut from, and retrieval scores an entry by the best of its own vector and its
pieces'. This is the pattern MCP servers already use to share the tools index
(see :mod:`agent_gantry.core.mcp_manager`): no new table, no change to the
vector-store protocol, and one vector per row as every store expects.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from agent_gantry.schema.skill import Skill
from agent_gantry.schema.tool import CHUNK_NAMESPACE, ToolDefinition

if TYPE_CHECKING:
    from agent_gantry.adapters.embedders.base import EmbeddingAdapter
    from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter

logger = logging.getLogger(__name__)

__all__ = [
    "CHUNK_AFTER",
    "CHUNK_CHARS",
    "MAX_CHUNKS",
    "MAX_SKILL_CHUNKS",
    "chunk_parent",
    "delete_skill_chunks",
    "delete_tool_chunks",
    "skill_chunks",
    "split_text",
    "sync_skill_chunks",
    "sync_tool_chunks",
    "tool_chunks",
]

#: Text longer than this is chunked. About what a 256-token embedder reads, so
#: shorter text is already carried whole by the entry's own vector.
CHUNK_AFTER = 900

#: Target size of one chunk. Comfortably inside a 256-token window with the
#: entry's name and tags in front of it.
CHUNK_CHARS = 700

#: Most chunks one entry may have. Sixteen of 700 characters cover the 10,000
#: an extended description may hold.
MAX_CHUNKS = 16

#: Most chunks one skill may have: enough for the 50,000 characters a skill's
#: content may hold.
MAX_SKILL_CHUNKS = 72

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


def split_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split ``text`` into pieces of about ``size`` characters, at sentence ends.

    A sentence longer than ``size`` is cut at a space, or mid-word if it has
    none; nothing is dropped.
    """
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(" ".join(text.split())):
        while len(sentence) > size:
            cut = sentence.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            head, sentence = sentence[:cut], sentence[cut:].lstrip()
            if current:
                pieces.append(current)
                current = ""
            pieces.append(head)
        if current and len(current) + 1 + len(sentence) > size:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _chunk_name(namespace: str, name: str, index: int) -> str:
    """Deterministic name of chunk ``index`` of ``namespace.name``."""
    digest = hashlib.sha256(f"{namespace}.{name}".encode()).hexdigest()[:16]
    return f"chunk_{digest}_{index:02d}"


def tool_chunks(tool: ToolDefinition) -> list[ToolDefinition]:
    """The chunk entries for ``tool``: none unless its text is long.

    The full text is the extended description when there is one (MCP discovery
    keeps a long description there, with only its head in ``description``),
    otherwise the description. Each chunk's own text opens with the tool's name
    and tags, so a piece from the middle of a description still says what it
    is a piece of, and carries the tool's tags, so a tag filter (such as a
    search scoped to one MCP server) finds a tool by its chunks too.
    """
    text = tool.extended_description or tool.description
    if len(text) <= CHUNK_AFTER or tool.namespace == CHUNK_NAMESPACE:
        return []
    head = " ".join(part for part in (tool.name, " ".join(tool.tags)) if part)
    return [
        ToolDefinition(
            name=_chunk_name(tool.namespace, tool.name, index),
            namespace=CHUNK_NAMESPACE,
            description=f"{head} {piece}"[:2000],
            parameters_schema={"type": "object", "properties": {}},
            tags=list(tool.tags),
            metadata={
                "entity_type": "tool_chunk",
                "parent_name": tool.name,
                "parent_namespace": tool.namespace,
                "chunk": index,
            },
        )
        for index, piece in enumerate(split_text(text)[:MAX_CHUNKS])
    ]


def chunk_parent(entry: Any) -> tuple[str, str] | None:
    """``(namespace, name)`` of the entry a chunk was cut from, or ``None``."""
    if getattr(entry, "namespace", None) != CHUNK_NAMESPACE:
        return None
    metadata = getattr(entry, "metadata", None) or {}
    name, namespace = metadata.get("parent_name"), metadata.get("parent_namespace")
    if isinstance(name, str) and isinstance(namespace, str):
        return namespace, name
    return None


async def delete_tool_chunks(
    store: VectorStoreAdapter, name: str, namespace: str, *, start: int = 0
) -> int:
    """Delete the chunks of ``namespace.name`` from index ``start`` up.

    Chunk indices are contiguous, so this stops at the first one that is not
    there: one request for an entry that never had chunks.
    """
    removed = 0
    for index in range(start, MAX_CHUNKS):
        try:
            deleted = await store.delete(_chunk_name(namespace, name, index), CHUNK_NAMESPACE)
        except Exception:
            logger.debug("Error deleting a tool chunk from the vector store", exc_info=True)
            break
        if deleted is not True:
            break
        removed += 1
    return removed


async def sync_tool_chunks(
    store: VectorStoreAdapter,
    embedder: EmbeddingAdapter,
    tools: Sequence[ToolDefinition],
    *,
    previously_stored: set[str] | None = None,
) -> int:
    """Index the chunks of ``tools``, and drop chunks a shorter text no longer has.

    Args:
        store: The tools vector store.
        embedder: The embedder the tools themselves were embedded with.
        tools: Tools just synced.
        previously_stored: ``namespace.name`` of the tools that were in the
            store before this sync. Only those can have chunks left over from
            an earlier, longer text; ``None`` means "check all of them".

    Returns:
        Number of chunks stored.
    """
    chunks: list[ToolDefinition] = []
    counts: dict[tuple[str, str], int] = {}
    for tool in tools:
        own = tool_chunks(tool)
        counts[(tool.namespace, tool.name)] = len(own)
        chunks.extend(own)

    stored = 0
    if chunks:
        embeddings = await embedder.embed_batch([chunk.description for chunk in chunks])
        stored = await store.add_tools(chunks, embeddings, upsert=True)

    for (namespace, name), count in counts.items():
        if previously_stored is None or f"{namespace}.{name}" in previously_stored:
            await delete_tool_chunks(store, name, namespace, start=count)
    return stored


def skill_chunks(skill: Skill) -> list[Skill]:
    """The chunk entries for ``skill``: its content, in pieces.

    A skill's own vector is made from its name, description, tags and summary
    and deliberately leaves the content out: fifty thousand characters cannot
    go through one embedding, and what a skill is *for* lives in its
    description. But that made a skill findable only by what its one-line
    description happens to say, while the detail someone would search for
    ("filter logs with service:checkout") sits in the body. The body is
    therefore indexed here, piece by piece, whatever its length.

    A chunk keeps the skill's category and tags, so the filters a search
    applies to skills apply to their chunks alike.
    """
    if skill.namespace == CHUNK_NAMESPACE or skill.content.strip() == skill.description.strip():
        return []
    return [
        Skill(
            name=_chunk_name(skill.namespace, skill.name, index),
            namespace=CHUNK_NAMESPACE,
            description=f"Part of {skill.name}: {piece}"[:2000],
            content=piece,
            category=skill.category,
            tags=list(skill.tags),
            metadata={
                "entity_type": "skill_chunk",
                "parent_name": skill.name,
                "parent_namespace": skill.namespace,
                "chunk": index,
            },
        )
        for index, piece in enumerate(split_text(skill.content)[:MAX_SKILL_CHUNKS])
    ]


async def delete_skill_chunks(store: Any, name: str, namespace: str, *, start: int = 0) -> int:
    """Delete the chunks of skill ``namespace.name`` from index ``start`` up."""
    removed = 0
    for index in range(start, MAX_SKILL_CHUNKS):
        try:
            deleted = await store.delete_skill(_chunk_name(namespace, name, index), CHUNK_NAMESPACE)
        except Exception:
            logger.debug("Error deleting a skill chunk from the vector store", exc_info=True)
            break
        if deleted is not True:
            break
        removed += 1
    return removed


async def sync_skill_chunks(store: Any, embedder: EmbeddingAdapter, skills: Sequence[Skill]) -> int:
    """Index the chunks of ``skills`` and drop any a shorter body no longer has.

    Returns:
        Number of chunks stored.
    """
    chunks: list[Skill] = []
    counts: dict[tuple[str, str], int] = {}
    for skill in skills:
        own = skill_chunks(skill)
        counts[(skill.namespace, skill.name)] = len(own)
        chunks.extend(own)
    stored = 0
    if chunks:
        embeddings = await embedder.embed_batch([chunk.to_embedding_text() for chunk in chunks])
        stored = await store.add_skills(chunks, embeddings, upsert=True)
    for (namespace, name), count in counts.items():
        await delete_skill_chunks(store, name, namespace, start=count)
    return stored
