"""
Registering A2A agents and routing a query to one.

The counterpart, for A2A agents, of :mod:`agent_gantry.core.mcp_manager` and
:mod:`agent_gantry.core.mcp_router`. Before it, the only way to bring an A2A
agent in was to fetch its card and turn every skill into a tool: there was no
way to register an agent without contacting it, nothing indexed the agent
itself, and so no way to ask which agent a request belongs to.

An agent is embedded as a pseudo-tool in a reserved namespace of the tools
index, exactly as MCP servers are, with the skills on its card indexed in
pieces beside it so that it is found by what it can do and not only by what
its description says.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING

from agent_gantry.adapters.embedders.base import embed_query
from agent_gantry.core.chunks import CHUNK_CHARS
from agent_gantry.schema.a2a import A2AAgentDefinition
from agent_gantry.schema.tool import A2A_AGENT_NAMESPACE, ToolDefinition

if TYPE_CHECKING:
    from agent_gantry.adapters.embedders.base import EmbeddingAdapter
    from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
    from agent_gantry.core.sync_manager import SyncManager

logger = logging.getLogger(__name__)

#: ``metadata["entity_type"]`` of an agent's own entry, and of a piece of its
#: skills. The router credits both to the agent.
AGENT_ENTITY = "a2a_agent"
SKILLS_ENTITY = "a2a_agent_skills"

#: Most pieces one agent's skills are indexed in.
MAX_SKILL_PIECES = 24

_DESCRIPTION_MAX = 2000

#: Store metadata key recording which embedder the agent entries were made with.
_EMBEDDER_KEY = "a2a_agents_embedder_id"


def _digest(agent: A2AAgentDefinition) -> str:
    return hashlib.sha256(agent.qualified_name.encode()).hexdigest()


def _entries(agent: A2AAgentDefinition) -> list[ToolDefinition]:
    """The index entries for ``agent``: its own, then its skills in pieces."""
    stem = re.sub(r"[^a-z0-9_]+", "_", agent.qualified_name.lower()).strip("_")[:100]
    metadata = {"agent_name": agent.name, "agent_namespace": agent.namespace}
    entries = [
        ToolDefinition(
            name=f"a2a_agent_{stem}_{_digest(agent)[:8]}",
            namespace=A2A_AGENT_NAMESPACE,
            description=agent.to_searchable_text()[:_DESCRIPTION_MAX],
            parameters_schema={"type": "object", "properties": {}},
            metadata={"entity_type": AGENT_ENTITY, **metadata},
        )
    ]
    pieces: list[str] = []
    current = ""
    for skill in agent.skills:
        if current and len(current) + len(skill) + 2 > CHUNK_CHARS:
            pieces.append(current)
            current = ""
        current = f"{current}; {skill}" if current else skill
    if current:
        pieces.append(current)
    for index, piece in enumerate(pieces[:MAX_SKILL_PIECES]):
        entries.append(
            ToolDefinition(
                name=f"a2a_skills_{_digest(agent)[:16]}_{index:02d}",
                namespace=A2A_AGENT_NAMESPACE,
                description=f"{agent.name} skills: {piece}"[:_DESCRIPTION_MAX],
                parameters_schema={"type": "object", "properties": {}},
                metadata={"entity_type": SKILLS_ENTITY, "chunk": index, **metadata},
            )
        )
    return entries


class A2AManager:
    """Holds registered A2A agents, indexes them and routes queries to them."""

    def __init__(
        self,
        vector_store: VectorStoreAdapter,
        embedder: EmbeddingAdapter,
        sync_manager: SyncManager,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._sync_manager = sync_manager
        self._agents: dict[str, A2AAgentDefinition] = {}
        # Counted, not flagged: a registration made while a sync is awaiting
        # the embedder is not in that sync's snapshot, and a flag cleared at
        # the end of the sync would lose it for good.
        self._generation = 0
        self._synced_generation = 0

    def register(self, agent: A2AAgentDefinition) -> None:
        """Register ``agent``, replacing one of the same qualified name."""
        self._agents[agent.qualified_name] = agent
        self._generation += 1
        logger.info(f"Registered A2A agent: {agent.qualified_name}")

    def get(self, name: str, namespace: str = "default") -> A2AAgentDefinition | None:
        """The registered agent ``namespace.name``, if there is one."""
        return self._agents.get(f"{namespace}.{name}")

    def remove(self, name: str, namespace: str = "default") -> bool:
        """Forget ``namespace.name``; its index entries go at the next sync."""
        removed = self._agents.pop(f"{namespace}.{name}", None) is not None
        if removed:
            self._generation += 1
        return removed

    def list(self, namespace: str | None = None) -> list[A2AAgentDefinition]:
        """Registered agents, optionally those of one namespace."""
        return [a for a in self._agents.values() if namespace is None or a.namespace == namespace]

    @property
    def needs_sync(self) -> bool:
        """Whether something was registered or removed since the last sync began."""
        return self._generation != self._synced_generation

    async def sync(self, batch_size: int = 100, force: bool = False) -> int:
        """Embed new or changed agents, and drop the entries of removed ones.

        Returns:
            Number of agents synced.
        """
        generation = self._generation
        entries = [entry for agent in self._agents.values() for entry in _entries(agent)]

        # The agents' own record of which embedder made their vectors. The
        # store's shared marker is the tools': a tool sync after a change of
        # embedding model brings it up to date, and agent entries, whose
        # fingerprints have not changed, would then be skipped and searched
        # with the new model against the old model's vectors.
        embedder_id = self._sync_manager.get_embedder_id()
        get_metadata = getattr(self._vector_store, "get_metadata", None)
        set_metadata = getattr(self._vector_store, "set_metadata", None)
        stored_id = await get_metadata(_EMBEDDER_KEY) if callable(get_metadata) else None
        if isinstance(stored_id, str) and stored_id != embedder_id:
            force = True

        synced = 0
        if entries:
            to_sync = await self._sync_manager.detect_changes(entries, force)
            for i in range(0, len(to_sync), batch_size):
                batch = to_sync[i : i + batch_size]
                # An entry's description is exactly the text to search by.
                embeddings = await self._embedder.embed_batch([e.description for e in batch])
                await self._vector_store.add_tools(batch, embeddings, upsert=True)
                synced += sum(1 for e in batch if e.metadata["entity_type"] == AGENT_ENTITY)
        await self._prune({entry.name for entry in entries})
        if callable(set_metadata):
            await set_metadata(_EMBEDDER_KEY, embedder_id)
        # What this sync covered, not "everything": see ``_generation``.
        self._synced_generation = generation
        return synced

    async def _prune(self, wanted: set[str]) -> None:
        """Delete index entries no registered agent owns."""
        offset = 0
        stored: list[ToolDefinition] = []
        while True:
            page = list(
                await self._vector_store.list_all(
                    namespace=A2A_AGENT_NAMESPACE, limit=1000, offset=offset
                )
            )
            stored.extend(page)
            if len(page) < 1000:
                break
            offset += 1000
        for entry in stored:
            if entry.name not in wanted:
                await self._vector_store.delete(entry.name, A2A_AGENT_NAMESPACE)

    async def route(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float | None = None,
        namespaces: list[str] | None = None,
    ) -> list[tuple[A2AAgentDefinition, float]]:
        """The registered agents most relevant to ``query``, best first.

        An agent is scored by the best of its own entry and the pieces of its
        skills. Results come back best first, so the first row seen for an
        agent is that best, and the window is widened when one agent's pieces
        crowd the others out of it.
        """
        query_embedding = await embed_query(self._embedder, query)
        fetch = limit * 4
        found: list[tuple[A2AAgentDefinition, float]] = []
        for _ in range(3):
            candidates = await self._vector_store.search(
                query_vector=query_embedding,
                limit=fetch,
                filters={"namespace": [A2A_AGENT_NAMESPACE]},
                score_threshold=score_threshold,
            )
            found = []
            seen: set[str] = set()
            for entry, score, *_ in candidates:
                if entry.metadata.get("entity_type") not in (AGENT_ENTITY, SKILLS_ENTITY):
                    continue
                key = f"{entry.metadata.get('agent_namespace')}.{entry.metadata.get('agent_name')}"
                agent = self._agents.get(key)
                if agent is None or key in seen:
                    continue
                if namespaces and agent.namespace not in namespaces:
                    continue
                seen.add(key)
                found.append((agent, float(score)))
            if len(found) >= limit or len(candidates) < fetch:
                break
            fetch *= 4
        return found[:limit]
