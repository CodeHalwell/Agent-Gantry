"""
Anthropic Skills support for Agent-Gantry.

Skills allow you to create reusable, composable abstractions that combine
multiple tools with instructions and examples. When used in a message,
skill instructions are injected into the system prompt and the associated
tools are made available to Claude.

This provides a way to organize related tools into cohesive capabilities
that Claude can reason about and use effectively.
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
    Anthropic client with Skills support.

    Skills are implemented by injecting skill instructions into the system
    prompt and providing the associated tools to Claude. This allows for
    reusable, composable tool workflows without requiring a special API.
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

        # Initialize client
        self._client = AsyncAnthropic(api_key=self._api_key)

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
        system: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create a message with Skills support.

        Skills are implemented by:
        1. Injecting skill instructions into the system prompt
        2. Including tools associated with skills

        Args:
            model: Model identifier (e.g., "claude-sonnet-4-5")
            messages: Message history
            max_tokens: Maximum tokens to generate
            skills: List of skill names to enable, "all" for all skills, or None
            auto_retrieve_tools: Whether to automatically retrieve tools from Agent-Gantry
            query: Query for tool retrieval (defaults to last user message)
            tool_limit: Maximum number of tools to retrieve
            system: Optional system prompt (will be combined with skill instructions)
            **kwargs: Additional arguments passed to messages.create()

        Returns:
            Anthropic message response
        """
        # Determine which skills to include
        selected_skills: list[Skill] = []
        skill_tool_names: set[str] = set()

        if skills == "all":
            selected_skills = self._skills.list_skills()
        elif skills:
            for skill_name in skills:
                skill = self._skills.get(skill_name)
                if skill:
                    selected_skills.append(skill)

        # Collect tool names from skills
        for skill in selected_skills:
            skill_tool_names.update(skill.tools)

        # Build system prompt with skill instructions
        system_parts: list[str] = []
        if system:
            system_parts.append(system)

        if selected_skills:
            system_parts.append("\n--- Skills ---\n")
            for skill in selected_skills:
                skill_section = f"## {skill.name}\n"
                skill_section += f"Description: {skill.description}\n\n"
                skill_section += f"Instructions:\n{skill.instructions}\n"
                if skill.tools:
                    skill_section += f"\nAvailable tools: {', '.join(skill.tools)}\n"
                if skill.examples:
                    skill_section += "\nExamples:\n"
                    for example in skill.examples:
                        if "input" in example:
                            skill_section += f"  Input: {example['input']}\n"
                        if "output" in example:
                            skill_section += f"  Output: {example['output']}\n"
                        if "steps" in example:
                            skill_section += "  Steps:\n"
                            for step in example["steps"]:
                                skill_section += f"    - {step}\n"
                system_parts.append(skill_section)

        final_system = "\n".join(system_parts) if system_parts else None

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
        if self._gantry:
            # If skills define specific tools, get those
            if skill_tool_names:
                # Get all registered tools from gantry that match skill tool names
                all_tools = await self._gantry.list_tools()
                for tool_def in all_tools:
                    if tool_def.name in skill_tool_names:
                        tools.append(tool_def.to_anthropic_schema())
            elif auto_retrieve_tools and query:
                # Fall back to semantic retrieval
                retrieval_result = await self._gantry.retrieve(
                    ToolQuery(
                        context=ConversationContext(query=query),
                        limit=tool_limit,
                    )
                )
                tools = [t.tool.to_anthropic_schema() for t in retrieval_result.tools]

        # Build the request kwargs
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            **kwargs,
        }

        if final_system:
            request_kwargs["system"] = final_system

        if tools:
            request_kwargs["tools"] = tools

        # Create message
        response = await self._client.messages.create(**request_kwargs)

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
