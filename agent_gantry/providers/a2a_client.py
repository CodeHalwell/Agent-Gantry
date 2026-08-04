"""
A2A Client for consuming external A2A agents as tools.

Fetches agent cards, maps skills to tools, and executes tasks.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from agent_gantry.schema.a2a import AgentCard, TaskMessage, TaskMessagePart, TaskRequest
from agent_gantry.schema.tool import ToolDefinition, ToolSource

if TYPE_CHECKING:
    from agent_gantry.schema.config import A2AAgentConfig

logger = logging.getLogger(__name__)
_SANITIZE_PATTERN = re.compile(r"[^a-z0-9_]+")


class A2AClient:
    """
    Client for interacting with A2A protocol agents.

    Discovers agent capabilities via Agent Card and executes tasks.
    """

    def __init__(self, config: A2AAgentConfig) -> None:
        """
        Initialize A2A client.

        Args:
            config: Configuration for the A2A agent to connect to
        """
        self.config = config
        self._agent_card: AgentCard | None = None
        self._base_url = config.url.rstrip("/")
        # Persistent HTTP client(s) so repeated task sends reuse connections
        # (keep-alive) instead of paying DNS + TCP + TLS per call. httpx
        # AsyncClient connections bind to the event loop they were created on,
        # so keep one client per running loop (bounded in practice — see
        # core/rate_limiter.py for the same pattern). Entries hold a weakref
        # to the owning loop so close() can dispose of clients on other live
        # loops instead of abandoning them.
        self._http_clients: dict[int, tuple[Any, Any]] = {}

    def _get_http_client(self) -> Any:
        """Return a persistent ``httpx.AsyncClient`` for the running loop."""
        import asyncio
        import weakref

        import httpx

        loop = asyncio.get_running_loop()
        entry = self._http_clients.get(id(loop))
        if entry is not None:
            loop_ref, client = entry
            if loop_ref() is loop and not client.is_closed:
                return client
            # Stale entry (dead loop whose id() was reused, or a closed
            # client). The old client can't be aclose()'d here — it belongs
            # to a gone loop — so its sockets are reclaimed by httpx's
            # finalizer. Call close() before abandoning a loop to release
            # them deterministically.
            if not client.is_closed:
                logger.debug(
                    "Dropping A2A HTTP client bound to a dead event loop for "
                    "agent '%s'; connections will be reclaimed by GC",
                    self.config.name,
                )
        client = httpx.AsyncClient()
        self._http_clients[id(loop)] = (weakref.ref(loop), client)
        # Opportunistic cleanup of entries whose loops were garbage-collected
        # (same GC-reclaim caveat as above for any still-open clients).
        for stale in [k for k, (ref, _c) in self._http_clients.items() if ref() is None]:
            self._http_clients.pop(stale, None)
        return client

    async def close(self) -> None:
        """Close any persistent HTTP clients. Safe to call multiple times.

        The current loop's client is closed inline. Clients bound to other
        loops can't be awaited from here: if their loop is still running,
        ``aclose()`` is scheduled onto it thread-safely; if the loop is gone,
        the reference is dropped and connections are reclaimed by GC. Those
        scheduled closes are not awaited — close() returns before other
        loops' clients have finished closing.
        """
        import asyncio

        current = asyncio.get_running_loop()
        for key, (loop_ref, client) in list(self._http_clients.items()):
            self._http_clients.pop(key, None)
            loop = loop_ref()
            if loop is current:
                await client.aclose()
            elif loop is not None and not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(
                        lambda c=client, lo=loop: lo.create_task(c.aclose())
                    )
                except RuntimeError:
                    # Loop shut down between the check and the call — drop.
                    pass

    async def discover(self) -> AgentCard:
        """
        Fetch the agent card from .well-known/agent.json.

        Returns:
            Agent card describing the agent's capabilities

        Raises:
            RuntimeError: If discovery fails
        """
        try:
            client = self._get_http_client()
            response = await client.get(
                f"{self._base_url}/.well-known/agent.json",
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            self._agent_card = AgentCard(**data)
            logger.info(
                f"Discovered A2A agent: {self._agent_card.name} "
                f"with {len(self._agent_card.skills)} skills"
            )
            return self._agent_card
        except Exception as e:
            raise RuntimeError(f"Failed to discover A2A agent at {self._base_url}: {e}") from e

    def _generate_tool_name(self, skill_id: str) -> str:
        """
        Generate a unique tool name for an A2A skill.

        Args:
            skill_id: The skill ID

        Returns:
            Tool name in the format: a2a_{agent_name}_{skill_id}
        """
        agent_part = _SANITIZE_PATTERN.sub("_", self.config.name.lower()).strip("_")
        skill_part = _SANITIZE_PATTERN.sub("_", skill_id.lower()).strip("_")
        return f"a2a_{agent_part}_{skill_part}"

    def _skill_to_tool(self, skill: Any) -> ToolDefinition:
        """
        Convert an agent skill to a ToolDefinition.

        Args:
            skill: Agent skill from the agent card

        Returns:
            ToolDefinition for the skill
        """
        # Generate a unique tool name based on agent and skill
        tool_name = self._generate_tool_name(skill.id)

        # Build parameter schema (for now, accept text input)
        parameters_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"Input for {skill.name}",
                }
            },
            "required": ["query"],
        }

        return ToolDefinition(
            name=tool_name,
            description=skill.description,
            parameters_schema=parameters_schema,
            namespace=self.config.namespace,
            source=ToolSource.A2A_AGENT,
            source_uri=f"a2a://{self.config.name}",
            metadata={
                "a2a_agent": self.config.name,
                "a2a_url": self._base_url,
                "skill_id": skill.id,
                "skill_name": skill.name,
                "input_modes": skill.input_modes,
                "output_modes": skill.output_modes,
            },
        )

    async def list_tools(self) -> list[ToolDefinition]:
        """
        List all tools (skills) provided by this A2A agent.

        Returns:
            List of tool definitions for each skill

        Raises:
            RuntimeError: If agent card not discovered yet
        """
        if self._agent_card is None:
            await self.discover()

        if self._agent_card is None:
            raise RuntimeError("Failed to discover agent card")

        return [self._skill_to_tool(skill) for skill in self._agent_card.skills]

    async def send_task(
        self,
        skill_id: str,
        query: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Send a task to the A2A agent.

        Args:
            skill_id: ID of the skill to invoke
            query: The query/input for the task
            metadata: Optional metadata for the task

        Returns:
            Task response as a dictionary

        Raises:
            RuntimeError: If task execution fails
        """
        try:
            # Build task request
            task_request = TaskRequest(
                skill_id=skill_id,
                messages=[
                    TaskMessage(
                        role="user",
                        parts=[TaskMessagePart(type="text", text=query)],
                    )
                ],
                metadata=metadata or {},
            )

            # Send JSON-RPC request over the persistent connection
            client = self._get_http_client()
            response = await client.post(
                f"{self._base_url}/tasks/send",
                json={
                    "jsonrpc": "2.0",
                    "method": "tasks/send",
                    "params": task_request.model_dump(),
                    "id": 1,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()

            # Handle JSON-RPC response
            if "error" in result:
                raise RuntimeError(f"A2A task error: {result['error']}")

            task_result: dict[str, Any] = result.get("result", {})
            return task_result

        except Exception as e:
            raise RuntimeError(f"Failed to send task to A2A agent {self.config.name}: {e}") from e
