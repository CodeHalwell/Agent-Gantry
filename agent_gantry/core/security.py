"""
Security policy and permission checking for Agent-Gantry.

Implements zero-trust security controls including:
- SecurityPolicy: pattern-based rules for tool access
- PermissionChecker: capability-based access control
- Input validation helpers
"""

from __future__ import annotations

import fnmatch
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_gantry.schema.tool import ToolCapability, ToolDefinition


class ConfirmationRequiredError(Exception):
    """Raised when a tool requires human confirmation."""

    pass


class PermissionDeniedError(Exception):
    """Raised when a tool execution is not permitted."""

    pass


# Backwards compatibility aliases (deprecated)
ConfirmationRequired = ConfirmationRequiredError
PermissionDenied = PermissionDeniedError


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


class SecurityPolicy:
    """
    Rules of Engagement for tools.

    Enforces pattern-based policies for tool confirmation and access control.
    """

    def __init__(
        self,
        require_confirmation: list[str] | None = None,
        allowed_domains: list[str] | None = None,
        max_requests_per_minute: int = 60,
    ) -> None:
        """
        Initialize security policy.

        Args:
            require_confirmation: List of tool name patterns requiring confirmation
            allowed_domains: List of allowed domains for external API access
            max_requests_per_minute: Maximum requests per minute
        """
        self.require_confirmation = require_confirmation or [
            "delete_*",
            "payment_*",
            "drop_*",
            "refund_*",
        ]
        self.allowed_domains = allowed_domains or []
        self.max_requests_per_minute = max_requests_per_minute

    def check_permission(self, tool_name: str, arguments: dict[str, str]) -> None:
        """
        Check if tool execution is permitted.

        Raises:
            ConfirmationRequiredError: If tool requires human approval
            PermissionDeniedError: If execution is not permitted

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool
        """
        for pattern in self.require_confirmation:
            if fnmatch.fnmatch(tool_name, pattern):
                raise ConfirmationRequiredError(f"Tool {tool_name} requires human approval.")


class PermissionChecker:
    """Enforce capability-based access control."""

    def __init__(self, user_capabilities: list[ToolCapability]) -> None:
        """
        Initialize permission checker.

        Args:
            user_capabilities: List of capabilities the user has
        """
        self.allowed = set(user_capabilities)

    def can_use(self, tool: ToolDefinition) -> tuple[bool, str | None]:
        """
        Check if user can use the given tool.

        Args:
            tool: Tool to check permissions for

        Returns:
            Tuple of (can_use, error_message)
        """
        required = set(tool.capabilities)
        missing = required - self.allowed
        if missing:
            return False, f"Missing capabilities: {', '.join(c.value for c in missing)}"
        return True, None

    def filter_tools(self, tools: list[ToolDefinition]) -> list[ToolDefinition]:
        """
        Filter tools based on user capabilities.

        Args:
            tools: List of tools to filter

        Returns:
            List of tools the user can access
        """
        return [t for t in tools if self.can_use(t)[0]]


def validate_tool_name(name: str) -> tuple[bool, str | None]:
    """
    Validate tool name format.

    Args:
        name: Tool name to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not re.match(r"^[a-z][a-z0-9_]{0,127}$", name):
        return False, "Name must be lowercase alphanumeric with underscores, 1-128 chars"
    return True, None


def validate_description(desc: str) -> tuple[bool, str | None]:
    """
    Validate tool description for suspicious patterns.

    Args:
        desc: Description to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    suspicious_patterns = [
        r"\{\{.*\}\}",
        r"<script",
        r"javascript:",
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, desc, re.IGNORECASE):
            return False, "Description contains suspicious pattern"
    return True, None
