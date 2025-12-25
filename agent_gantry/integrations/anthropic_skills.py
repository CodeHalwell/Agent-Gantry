"""
Anthropic Skills API support for Agent-Gantry.

The Skills API allows you to create reusable, composable skills that Claude can use.
This module provides integration between Anthropic Skills and Agent-Gantry tools.

Beta version: skills-2025-10-02
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall
from agent_gantry.schema.query import ConversationContext, ToolQuery


@dataclass
class Skill:
    """
    A reusable skill that can be invoked by Claude.

    Skills are higher-level abstractions that can combine multiple tools
    and reasoning steps.
    """

    name: str
    description: str
    instructions: str
    tools: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert skill to Anthropic Skills API format."""
        schema: dict[str, Any] = {
            "type": "skill",
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
        }

        if self.tools:
            schema["tools"] = self.tools

        if self.examples:
            schema["examples"] = self.examples

        if self.metadata:
            schema["metadata"] = self.metadata

        return schema


class SkillRegistry:
    """Registry for managing Anthropic skills."""

    def __init__(self) -> None:
        """Initialize the skill registry."""
        self._skills: dict[str, Skill] = {}

    def register(
        self,
        name: str,
        description: str,
        instructions: str,
        tools: list[str] | None = None,
        examples: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Skill:
        """
        Register a new skill.

        Args:
            name: Skill name (must be unique)
            description: Brief description of what the skill does
            instructions: Detailed instructions for using the skill
            tools: List of tool names this skill can use
            examples: Example usage scenarios
            metadata: Additional metadata

        Returns:
            The registered skill

        Raises:
            ValueError: If skill name already exists
        """
        if name in self._skills:
            raise ValueError(f"Skill '{name}' already registered")

        skill = Skill(
            name=name,
            description=description,
            instructions=instructions,
            tools=tools or [],
            examples=examples or [],
            metadata=metadata or {},
        )

        self._skills[name] = skill
        return skill

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        """List all registered skills."""
        return list(self._skills.values())

    def to_anthropic_schema(self) -> list[dict[str, Any]]:
        """Convert all skills to Anthropic format."""
        return [skill.to_anthropic_schema() for skill in self._skills.values()]

    def clear(self) -> None:
        """Clear all registered skills."""
        self._skills.clear()


class SkillsClient:
    """
    Anthropic client with Skills API support.

    Provides easy integration between Anthropic Skills and Agent-Gantry tools.
    """

    def __init__(
        self,
        api_key: str | None = None,
        gantry: AgentGantry | None = None,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        """
        Initialize the Skills client.

        Args:
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            gantry: AgentGantry instance for tool execution
            skill_registry: Optional skill registry (creates new if not provided)
        """
        from anthropic import AsyncAnthropic

        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "API key required. Set ANTHROPIC_API_KEY or pass api_key parameter."
            )

        self._gantry = gantry
        self._skills = skill_registry or SkillRegistry()

        # Initialize client with Skills API beta header
        self._client = AsyncAnthropic(
            api_key=self._api_key,
            default_headers={"anthropic-beta": "skills-2025-10-02"},
        )

    @property
    def skills(self) -> SkillRegistry:
        """Access the skill registry."""
        return self._skills

    async def create_message(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        skills: list[str] | Literal["all"] | None = None,
        auto_retrieve_tools: bool = True,
        query: str | None = None,
        tool_limit: int = 10,
        **kwargs: Any,
    ) -> Any:
        """
        Create a message with Skills API support.

        Args:
            model: Model identifier (e.g., "claude-3-5-sonnet-20241022")
            messages: Message history
            max_tokens: Maximum tokens to generate
            skills: List of skill names to enable, "all" for all skills, or None
            auto_retrieve_tools: Whether to automatically retrieve tools from Agent-Gantry
            query: Query for tool retrieval (defaults to last user message)
            tool_limit: Maximum number of tools to retrieve
            **kwargs: Additional arguments passed to messages.create()

        Returns:
            Anthropic message response
        """
        # Determine which skills to include
        skill_schemas: list[dict[str, Any]] = []
        if skills == "all":
            skill_schemas = self._skills.to_anthropic_schema()
        elif skills:
            for skill_name in skills:
                skill = self._skills.get(skill_name)
                if skill:
                    skill_schemas.append(skill.to_anthropic_schema())

        # Extract query from messages if not provided
        if not query and auto_retrieve_tools:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        query = content
                        break

        # Retrieve tools if enabled and gantry available
        tools: list[dict[str, Any]] = []
        if auto_retrieve_tools and self._gantry and query:
            retrieval_result = await self._gantry.retrieve(
                ToolQuery(
                    context=ConversationContext(query=query),
                    limit=tool_limit,
                )
            )
            tools = [t.tool.to_anthropic_schema() for t in retrieval_result.tools]

        # Create message with skills and tools
        response = await self._client.messages.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            skills=skill_schemas if skill_schemas else None,
            tools=tools if tools else [],
            **kwargs,
        )

        return response

    async def execute_tool_calls(
        self,
        response: Any,
    ) -> list[dict[str, Any]]:
        """
        Execute tool calls from a Skills API response.

        Args:
            response: Anthropic message response

        Returns:
            List of tool results in Anthropic format
        """
        if not self._gantry:
            raise ValueError("AgentGantry instance required for tool execution")

        tool_results = []
        for block in response.content:
            if hasattr(block, "type") and block.type == "tool_use":
                # Execute via Agent-Gantry
                result = await self._gantry.execute(
                    ToolCall(
                        tool_name=block.name,
                        arguments=block.input,
                    )
                )

                # Format result for Anthropic
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result.result) if result.status == "success" else f"Error: {result.error}",
                })

        return tool_results

    def register_skill_from_gantry_tools(
        self,
        skill_name: str,
        description: str,
        instructions: str,
        tool_names: list[str],
        examples: list[dict[str, Any]] | None = None,
    ) -> Skill:
        """
        Register a skill that uses Agent-Gantry tools.

        Args:
            skill_name: Name for the skill
            description: Brief description
            instructions: Detailed instructions for Claude
            tool_names: List of Agent-Gantry tool names this skill uses
            examples: Optional usage examples

        Returns:
            The registered skill
        """
        return self._skills.register(
            name=skill_name,
            description=description,
            instructions=instructions,
            tools=tool_names,
            examples=examples or [],
        )


async def create_skills_client(
    api_key: str | None = None,
    gantry: AgentGantry | None = None,
) -> SkillsClient:
    """
    Convenience function to create a Skills API client.

    Args:
        api_key: Anthropic API key
        gantry: AgentGantry instance for tool execution

    Returns:
        Configured SkillsClient

    Example:
        >>> client = await create_skills_client(gantry=gantry)
        >>> client.skills.register(
        ...     name="customer_support",
        ...     description="Handle customer support inquiries",
        ...     instructions="Use these tools to help customers with refunds, tracking, and issues",
        ...     tools=["get_order", "process_refund", "send_email"],
        ... )
        >>> response = await client.create_message(
        ...     model="claude-3-5-sonnet-20241022",
        ...     messages=[{"role": "user", "content": "I need a refund for order #12345"}],
        ...     skills=["customer_support"],
        ... )
    """
    return SkillsClient(api_key=api_key, gantry=gantry)
