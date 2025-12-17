"""
A2A Server implementation for Agent-Gantry.

Exposes AgentGantry as an A2A agent with tool discovery and execution skills.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent_gantry.schema.a2a import AgentCard, AgentSkill, TaskResponse

if TYPE_CHECKING:
    from agent_gantry import AgentGantry

logger = logging.getLogger(__name__)


def generate_agent_card(gantry: AgentGantry, base_url: str) -> AgentCard:
    """
    Generate an Agent Card for the AgentGantry instance.

    Args:
        gantry: AgentGantry instance
        base_url: Base URL where the A2A server is hosted

    Returns:
        Agent card describing AgentGantry's capabilities
    """
    return AgentCard(
        name="AgentGantry",
        description=(
            f"Intelligent tool routing and execution service with "
            f"{gantry.tool_count} tools available"
        ),
        url=base_url,
        version="1.0.0",
        skills=[
            AgentSkill(
                id="tool_discovery",
                name="Tool Discovery",
                description=(
                    "Find relevant tools for a given task using semantic search. "
                    "Returns a list of tools with their names, descriptions, and schemas."
                ),
                input_modes=["text"],
                output_modes=["text"],
            ),
            AgentSkill(
                id="tool_execution",
                name="Tool Execution",
                description=(
                    "Execute a registered tool by name with provided arguments. "
                    "Supports retries, timeouts, and circuit breakers."
                ),
                input_modes=["text"],
                output_modes=["text"],
            ),
        ],
        authentication=None,  # Basic authentication can be added
        provider={
            "organization": "Agent-Gantry",
            "url": "https://github.com/CodeHalwell/Agent-Gantry",
        },
    )


def create_a2a_server(gantry: AgentGantry, base_url: str = "http://localhost:8080") -> Any:
    """
    Create a FastAPI application serving the A2A protocol.

    Args:
        gantry: AgentGantry instance to expose
        base_url: Base URL for the server

    Returns:
        FastAPI application instance

    Raises:
        ImportError: If FastAPI is not installed
    """
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as e:
        raise ImportError(
            "FastAPI is required for A2A server. Install with: pip install fastapi uvicorn"
        ) from e

    app = FastAPI(
        title="AgentGantry A2A Server",
        description="Agent-to-Agent protocol server for AgentGantry",
        version="1.0.0",
    )

    # Generate agent card
    agent_card = generate_agent_card(gantry, base_url)

    @app.get("/.well-known/agent.json")
    async def get_agent_card() -> dict[str, Any]:
        """Serve the Agent Card."""
        return agent_card.model_dump()

    @app.post("/tasks/send")
    async def send_task(request: dict[str, Any]) -> dict[str, Any]:
        """
        Handle JSON-RPC task requests.

        Expects JSON-RPC 2.0 format with method="tasks/send".
        """
        try:
            # Validate JSON-RPC structure
            if request.get("jsonrpc") != "2.0":
                raise HTTPException(status_code=400, detail="Invalid JSON-RPC version")

            if request.get("method") != "tasks/send":
                raise HTTPException(status_code=400, detail="Invalid method")

            params = request.get("params", {})
            skill_id = params.get("skill_id")
            messages = params.get("messages", [])

            if not skill_id:
                raise HTTPException(status_code=400, detail="Missing skill_id")

            # Extract text from messages
            query_text = ""
            for message in messages:
                for part in message.get("parts", []):
                    if part.get("type") == "text" and part.get("text"):
                        query_text = part["text"]
                        break
                if query_text:
                    break

            if not query_text:
                raise HTTPException(
                    status_code=400,
                    detail="No text content found in message parts. Expected at least one message with a text part."
                )

            # Route to appropriate skill
            if skill_id == "tool_discovery":
                # Use semantic retrieval to find relevant tools
                result = await handle_tool_discovery(gantry, query_text)
            elif skill_id == "tool_execution":
                # Parse and execute tool
                result = await handle_tool_execution(gantry, query_text)
            else:
                raise HTTPException(status_code=404, detail=f"Unknown skill: {skill_id}")

            # Return JSON-RPC response
            return {
                "jsonrpc": "2.0",
                "result": TaskResponse(
                    status="success",
                    result=result,
                ).model_dump(),
                "id": request.get("id"),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error handling A2A task: {e}")
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e),
                },
                "id": request.get("id"),
            }

    return app


async def handle_tool_discovery(gantry: AgentGantry, query: str) -> dict[str, Any]:
    """
    Handle tool_discovery skill.

    Args:
        gantry: AgentGantry instance
        query: Search query

    Returns:
        Dictionary with discovered tools (serialized as dicts)
    """
    # Retrieve relevant tools using semantic search
    tools_raw = await gantry.retrieve_tools(query, limit=5)

    # Convert to serializable format (ensure all are dicts)
    tools = [
        tool if isinstance(tool, dict)
        else tool.model_dump() if hasattr(tool, "model_dump")
        else dict(tool) if hasattr(tool, "__dict__")
        else tool
        for tool in tools_raw
    ]

    return {
        "query": query,
        "tools_found": len(tools),
        "tools": tools,
    }


async def handle_tool_execution(gantry: AgentGantry, query: str) -> dict[str, Any]:
    """
    Handle tool_execution skill.

    Args:
        gantry: AgentGantry instance
        query: Query containing tool name and arguments (parsed from text)

    Returns:
        Dictionary with execution result

    Note:
        This is a simplified implementation. In production, you'd want
        structured input for tool name and arguments.
    """

    # Simple parsing: expect format like "tool_name with arg1=value1, arg2=value2"
    # For demo purposes, we'll just return a message about needing structured input
    return {
        "message": (
            "Tool execution via A2A requires structured input. "
            "Please use the tool_discovery skill first to find tools, "
            "then call them directly with proper argument schemas."
        ),
        "query": query,
    }
