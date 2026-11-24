"""
A2A (Agent-to-Agent) protocol models.

Models for agent discovery, skill definitions, and task communication.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_gantry.schema.base import reject_newlines


class AgentSkill(BaseModel):
    """
    A skill provided by an A2A agent.

    Skills map to tool capabilities that can be executed via task requests.
    """

    id: str = Field(..., description="Unique identifier for the skill")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Detailed description of what the skill does")
    input_modes: list[str] = Field(
        default_factory=lambda: ["text"], description="Supported input types"
    )
    output_modes: list[str] = Field(
        default_factory=lambda: ["text"], description="Supported output types"
    )

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    _reject_newline_identifiers = field_validator("id", "name")(reject_newlines)


class AgentCard(BaseModel):
    """
    Agent Card following the A2A protocol specification.

    Served at /.well-known/agent.json to advertise agent capabilities.
    """

    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    url: str = Field(..., description="Base URL for the agent")
    version: str = Field(default="1.0.0", description="Agent version")

    skills: list[AgentSkill] = Field(
        default_factory=list, description="Skills provided by this agent"
    )

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    _reject_newline_identifiers = field_validator("name")(reject_newlines)

    authentication: dict[str, Any] | None = Field(
        default=None, description="Authentication configuration"
    )
    provider: dict[str, str] = Field(default_factory=dict, description="Provider information")


class TaskMessagePart(BaseModel):
    """A part of a task message."""

    type: str = Field(default="text", description="Message part type")
    text: str | None = Field(default=None, description="Text content if type is text")
    data: Any | None = Field(default=None, description="Additional data")


class TaskMessage(BaseModel):
    """A message in a task request."""

    role: str = Field(..., description="Role (e.g., 'user', 'assistant')")
    parts: list[TaskMessagePart] = Field(default_factory=list, description="Message parts")


class TaskRequest(BaseModel):
    """Request to execute a task on an A2A agent."""

    skill_id: str = Field(..., description="ID of the skill to execute")
    messages: list[TaskMessage] = Field(default_factory=list, description="Conversation messages")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TaskResponse(BaseModel):
    """Response from an A2A agent task execution."""

    status: str = Field(..., description="Status (success, error, etc.)")
    result: Any | None = Field(default=None, description="Task result")
    error: str | None = Field(default=None, description="Error message if failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Response metadata")


#: Prefix of the tag every tool converted from an A2A agent's skill carries.
A2A_AGENT_TAG_PREFIX = "a2a:"


def a2a_agent_tag(agent_name: str) -> str:
    """The tag carried by every tool converted from a skill of ``agent_name``.

    As :func:`agent_gantry.schema.mcp.mcp_server_tag` does for MCP servers: it
    puts the agent's name in the tool's searchable text, makes the agent
    filterable at the vector store, and tells a selector where the tool came
    from. It matters more here than there: every tool converted from an A2A
    skill has the same parameter schema, one ``query`` string, so the
    description and this tag are all a retriever has to tell them apart.
    """
    return f"{A2A_AGENT_TAG_PREFIX}{agent_name}"


class A2AAgentDefinition(BaseModel):
    """An A2A agent as the gantry knows it, for routing a query to an agent.

    The counterpart of :class:`~agent_gantry.schema.mcp.MCPServerDefinition`.
    It can be registered by hand, before the agent has been contacted, or
    filled from the agent's card when it is discovered.

    Attributes:
        name: The name the gantry knows the agent by.
        namespace: Namespace for the agent and the tools its skills become.
        url: Base URL of the agent.
        description: What the agent does.
        tags: Tags for categorising the agent.
        examples: Example requests the agent handles.
        skills: ``"name: description"`` of each skill on the agent's card,
            when it has been discovered. Indexed in pieces beside the
            description, so the agent is found by what it can do.
        input_modes: Content types the agent accepts, from its card.
        output_modes: Content types the agent produces, from its card.
    """

    name: str = Field(..., min_length=1, max_length=128)
    namespace: str = Field(default="default")
    url: str
    description: str = Field(..., min_length=10, max_length=2000)
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    input_modes: list[str] = Field(default_factory=list)
    output_modes: list[str] = Field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        """``namespace.name``."""
        return f"{self.namespace}.{self.name}"

    def to_searchable_text(self) -> str:
        """The agent's own text for embedding: description, tags and examples.

        Its skills are indexed separately, in pieces; see
        :mod:`agent_gantry.core.a2a_manager`.
        """
        parts = [f"Agent: {self.name}", f"Description: {self.description}"]
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        if self.examples:
            parts.append(f"Examples: {', '.join(self.examples)}")
        return " ".join(parts)
