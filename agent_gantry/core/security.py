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
import time
import typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_gantry.schema.tool import ToolCapability, ToolDefinition


class ConfirmationRequiredError(Exception):
    """Raised when a tool requires human confirmation."""

    pass


class PermissionDeniedError(Exception):
    """Raised when a tool execution is not permitted."""

    pass


# Backwards compatibility aliases (deprecated — will be removed in 1.0)
import warnings as _warnings


def __getattr__(name: str) -> type:
    _deprecated = {
        "ConfirmationRequired": ConfirmationRequiredError,
        "PermissionDenied": PermissionDeniedError,
    }
    if name in _deprecated:
        _warnings.warn(
            f"{name} is deprecated, use {_deprecated[name].__name__} instead. "
            "This alias will be removed in version 1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _deprecated[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


def accepts_confirmation_approved(policy: typing.Any) -> bool:
    """Whether ``policy.check_permission`` accepts ``confirmation_approved``.

    Callers that honour a human approval (the executor's
    ``ToolCall(require_confirmation=False)`` path, the Agent Framework
    approval middleware's replay path) pass the keyword only when the
    policy's signature declares it (or takes ``**kwargs``), so duck-typed
    policies predating the keyword keep working — their pattern gate then
    simply stays un-approvable, exactly as before the keyword existed.
    Inspecting the signature once is what decides; never call-and-retry,
    which would re-run rate-limit accounting on a mismatch.
    """
    check = getattr(policy, "check_permission", None)
    if not callable(check):
        return False
    import inspect

    try:
        parameters = inspect.signature(check).parameters.values()
    except (TypeError, ValueError):  # C callables / exotic doubles
        return False
    return any(
        p.name == "confirmation_approved" or p.kind is inspect.Parameter.VAR_KEYWORD
        for p in parameters
    )


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
        self._request_timestamps: list[float] = []

    def check_permission(
        self,
        tool_name: str,
        arguments: dict[str, typing.Any],
        *,
        confirmation_approved: bool = False,
    ) -> None:
        """
        Check if tool execution is permitted.

        Raises:
            ConfirmationRequiredError: If tool requires human approval
            PermissionDeniedError: If execution is not permitted

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool
            confirmation_approved: When ``True``, the caller vouches that a
                human already approved this specific call, so the
                ``require_confirmation`` pattern gate is skipped. The
                ``allowed_domains`` denial check still runs — approval is
                not a policy bypass. The rate limit is also skipped for an
                approved call: it's the replay of a call already counted
                when it first requested confirmation, not a new request, so
                counting it again would let one logical call consume two
                slots of the budget (and, with a small enough limit, make
                confirmation-gated tools permanently unexecutable). Set by
                the executor when a call carries
                ``ToolCall(require_confirmation=False)``.
        """
        if self.max_requests_per_minute > 0 and not confirmation_approved:
            now = time.time()
            # ⚡ Bolt: Fast sliding window cleanup using index slice instead of O(N) comprehension
            split_idx = 0
            for t in self._request_timestamps:
                if now - t < 60:
                    break
                split_idx += 1
            if split_idx > 0:
                self._request_timestamps = self._request_timestamps[split_idx:]

            if len(self._request_timestamps) >= self.max_requests_per_minute:
                raise PermissionDeniedError(
                    f"Rate limit exceeded: maximum {self.max_requests_per_minute} requests per minute allowed."
                )
            self._request_timestamps.append(now)

        if not confirmation_approved:
            for pattern in self.require_confirmation:
                if fnmatch.fnmatch(tool_name, pattern):
                    raise ConfirmationRequiredError(
                        f"Tool {tool_name} requires human approval."
                    )

        # 2. Check allowed domains if they are configured
        if self.allowed_domains:
            for str_val in self._extract_all_strings(arguments):
                domains = self._extract_domains(str_val)
                for domain in domains:
                    if not self._is_domain_allowed(domain):
                        raise PermissionDeniedError(
                            f"Execution denied: Domain '{domain}' is not in allowed_domains."
                        )

    def _extract_all_strings(self, data: typing.Any) -> typing.Iterator[str]:
        """Recursively extract all string values from a data structure."""
        if isinstance(data, str):
            yield data
        elif isinstance(data, dict):
            for value in data.values():
                yield from self._extract_all_strings(value)
        elif isinstance(data, list) or isinstance(data, tuple):
            for item in data:
                yield from self._extract_all_strings(item)

    def _extract_domains(self, value: str) -> set[str]:
        """Extract potential domains from a string value."""
        import re
        import urllib.parse

        domains = set()

        # Match URLs with explicit protocol schemes (http, https, ftp, ftps)
        # and protocol-relative URLs (//example.com/path) securely,
        # avoiding matching inline comments
        url_pattern = r"(?:https?|ftps?|file)://[^\s\"\'<>]+|//(?:[a-zA-Z0-9][-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}|localhost)\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*"
        for url_match in re.finditer(url_pattern, value, re.IGNORECASE):
            try:
                url = url_match.group(0)
                # Repeatedly unquote to handle double/triple encoding bypasses
                prev_url = ""
                for _ in range(5):
                    if url == prev_url:
                        break
                    prev_url = url
                    url = urllib.parse.unquote(url)

                # Normalize backslashes to forward slashes to prevent SSRF bypasses
                # (e.g. evil.com\@example.com)
                url = url.replace("\\", "/")

                parsed = urllib.parse.urlparse(url)

                try:
                    # Accessing port triggers parsing that catches invalid ports
                    # like http://example.com:evil.com
                    _ = parsed.port
                except ValueError:
                    # Invalid port, fail the validation to avoid SSRF bypasses
                    # where downstream clients parse the malformed port differently
                    domains.add("<invalid_domain>")
                    continue

                if parsed.scheme.lower() == "file" or not parsed.hostname:
                    domains.add("<invalid_domain>")
                else:
                    domains.add(parsed.hostname)
            except Exception:
                pass

        # Block data URIs that reference external resources
        if re.search(r"data:\s*[^;,]+", value, re.IGNORECASE) and "data:" in value.lower():
            # data URIs themselves don't have domains, but flag if used
            # in combination with domain references
            domains.add("<invalid_domain>")

        # We deliberately don't extract plain strings that look like "example.com"
        # because this will flag filenames (e.g. "main.py") and block valid tool calls.

        return domains

    def _is_domain_allowed(self, domain: str) -> bool:
        """Check if a domain matches the allowed list."""
        for allowed in self.allowed_domains:
            if allowed.startswith("*."):
                suffix = allowed[2:]
                # Match exactly the suffix or subdomains
                if domain == suffix or domain.endswith("." + suffix):
                    return True
            elif domain == allowed:
                return True
        return False


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
    if not re.match(r"^[a-z][a-z0-9_]{0,127}\Z", name):
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
        if re.search(pattern, desc, re.IGNORECASE | re.DOTALL):
            return False, "Description contains suspicious pattern"
    return True, None
