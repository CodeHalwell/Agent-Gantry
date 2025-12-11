"""
Conversation context manager for Agent-Gantry.

Manages conversation state for context-aware routing.
"""

from __future__ import annotations

from agent_gantry.schema.query import ConversationContext
from agent_gantry.schema.tool import ToolCapability


class ConversationContextManager:
    """
    Manages conversation context for tool routing.

    Tracks:
    - Recent messages
    - Tools that have been used
    - Tools that have failed
    - User capabilities
    """

    def __init__(
        self,
        max_recent_messages: int = 10,
        user_capabilities: list[ToolCapability] | None = None,
    ) -> None:
        """
        Initialize the context manager.

        Args:
            max_recent_messages: Maximum number of recent messages to keep
            user_capabilities: User's allowed capabilities
        """
        self._max_messages = max_recent_messages
        self._recent_messages: list[dict[str, str]] = []
        self._tools_used: list[str] = []
        self._tools_failed: list[str] = []
        self._conversation_summary: str | None = None
        self._user_capabilities = user_capabilities or list(ToolCapability)

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        self._recent_messages.append({"role": role, "content": content})
        if len(self._recent_messages) > self._max_messages:
            self._recent_messages.pop(0)

    def record_tool_used(self, tool_name: str) -> None:
        """
        Record that a tool was used.

        Args:
            tool_name: Name of the tool that was used
        """
        if tool_name not in self._tools_used:
            self._tools_used.append(tool_name)

    def record_tool_failed(self, tool_name: str) -> None:
        """
        Record that a tool failed.

        Args:
            tool_name: Name of the tool that failed
        """
        if tool_name not in self._tools_failed:
            self._tools_failed.append(tool_name)

    def set_summary(self, summary: str) -> None:
        """
        Set the conversation summary.

        Args:
            summary: Summary of the conversation
        """
        self._conversation_summary = summary

    def build_context(self, query: str) -> ConversationContext:
        """
        Build a conversation context for a query.

        Args:
            query: The current query

        Returns:
            ConversationContext for routing
        """
        return ConversationContext(
            query=query,
            conversation_summary=self._conversation_summary,
            recent_messages=self._recent_messages.copy(),
            tools_already_used=self._tools_used.copy(),
            tools_failed=self._tools_failed.copy(),
            user_capabilities=self._user_capabilities,
        )

    def reset(self) -> None:
        """Reset the conversation context."""
        self._recent_messages = []
        self._tools_used = []
        self._tools_failed = []
        self._conversation_summary = None
