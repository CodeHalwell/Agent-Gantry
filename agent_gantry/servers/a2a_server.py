"""
A2A Server implementation for Agent-Gantry.

Exposes AgentGantry as an A2A agent with tool discovery and execution skills.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from agent_gantry.schema.a2a import AgentCard, AgentSkill, TaskResponse
from agent_gantry.schema.execution import ToolCall

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


def _invalid_params(request_id: Any, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "error": {"code": -32602, "message": message}, "id": request_id}


def _first_text(messages: Any) -> str:
    """The first non-empty text part in ``messages``; ``ValueError`` on a malformed shape."""
    if not isinstance(messages, list):
        raise ValueError("'messages' must be a list")
    for message in messages:
        parts = message.get("parts", []) if isinstance(message, dict) else None
        if not isinstance(parts, list):
            raise ValueError("each message must be an object with a 'parts' list")
        for part in parts:
            if not isinstance(part, dict):
                raise ValueError("each part must be an object")
            text = part.get("text")
            if part.get("type") == "text" and isinstance(text, str) and text:
                return text
    raise ValueError("no text part found; expected at least one message with a text part")


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
            "FastAPI is required for A2A server. Install with: pip install fastapi uvicorn (or uv add fastapi uvicorn)"
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
        """Handle a JSON-RPC 2.0 ``tasks/send`` request."""
        request_id = request.get("id")
        if request.get("jsonrpc") != "2.0":
            raise HTTPException(status_code=400, detail="Invalid JSON-RPC version")
        if request.get("method") != "tasks/send":
            raise HTTPException(status_code=400, detail="Invalid method")

        params = request.get("params")
        if not isinstance(params, dict):
            return _invalid_params(request_id, "'params' must be an object")
        skill_id = params.get("skill_id")
        if not skill_id:
            return _invalid_params(request_id, "Missing skill_id")
        try:
            query_text = _first_text(params.get("messages", []))
        except ValueError as exc:
            return _invalid_params(request_id, str(exc))

        try:
            if skill_id == "tool_discovery":
                result = await handle_tool_discovery(gantry, query_text)
            elif skill_id == "tool_execution":
                result = await handle_tool_execution(gantry, query_text)
            else:
                raise HTTPException(status_code=404, detail=f"Unknown skill: {skill_id}")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error handling A2A task: {e}")
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": "Internal error", "data": str(e)},
                "id": request_id,
            }

        # A failed execution is a completed task whose status says so; the
        # inner status and error travel up into the envelope.
        response = TaskResponse(
            status=result.get("status", "success"), result=result, error=result.get("error")
        )
        return {"jsonrpc": "2.0", "result": response.model_dump(), "id": request_id}

    return app


async def handle_tool_discovery(gantry: AgentGantry, query: str) -> dict[str, Any]:
    """Semantic search for tools relevant to ``query``, as OpenAI-style schemas."""
    tools = await gantry.retrieve_tools(query, limit=5)
    return {"query": query, "tools_found": len(tools), "tools": tools}


async def handle_tool_execution(gantry: AgentGantry, query: str) -> dict[str, Any]:
    """
    Handle tool_execution skill.

    Args:
        gantry: AgentGantry instance
        query: JSON text carrying ``tool_name`` and ``arguments``

    Returns:
        Dictionary with execution result
    """
    try:
        data = json.loads(query)

        if not isinstance(data, dict):
            raise ValueError("Input must be a JSON object")

        tool_name = data.get("tool_name") or data.get("name")
        if not tool_name:
            raise ValueError("Missing 'tool_name' in JSON input")

        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("'arguments' must be a JSON object")

        result = await gantry.execute(ToolCall(tool_name=tool_name, arguments=arguments))

        return {
            "status": result.status.value,
            "result": result.result,
            "error": result.error,
            "tool_name": result.tool_name,
            "latency_ms": result.latency_ms,
        }

    except json.JSONDecodeError:
        return {
            "status": "error",
            "error": "Invalid input format. Expected JSON object with 'tool_name' and 'arguments'.",
            "message": (
                "Tool execution via A2A requires structured JSON input. "
                "Format: {\"tool_name\": \"name\", \"arguments\": {\"arg1\": \"value1\"}}"
            ),
            "query": query,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "query": query,
        }
