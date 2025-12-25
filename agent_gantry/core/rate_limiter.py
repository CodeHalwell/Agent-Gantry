"""
Rate limiting for tool execution.

Provides rate limiting capabilities to prevent abuse and manage resource consumption.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_gantry.schema.config import RateLimitConfig


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RateLimiter:
    """
    Rate limiter for tool execution.

    Supports multiple strategies:
    - Sliding window: Track calls in a time window
    - Token bucket: Refill tokens at a rate
    - Fixed window: Fixed time buckets
    """

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        """
        Initialize the rate limiter.

        Args:
            config: Rate limiting configuration
        """
        self._config = config or RateLimitConfig()

        # Sliding window: deque of timestamps per key
        self._call_history: dict[str, deque[float]] = defaultdict(lambda: deque())

        # Token bucket: tokens and last refill time per key
        self._tokens: dict[str, float] = defaultdict(lambda: float(self._config.max_calls_per_minute))
        self._last_refill: dict[str, float] = defaultdict(time.time)

        # Fixed window: call count and window start per key
        self._window_calls: dict[str, int] = defaultdict(int)
        self._window_start: dict[str, float] = defaultdict(time.time)

        # Concurrent execution tracking
        self._concurrent: dict[str, int] = defaultdict(int)
        self._concurrent_lock = asyncio.Lock()

    def _get_key(self, tool_name: str, namespace: str = "default") -> str:
        """Get rate limit key based on configuration."""
        if self._config.per_namespace:
            return namespace
        elif self._config.per_tool:
            return f"{namespace}.{tool_name}"
        else:
            return "global"

    async def acquire(
        self,
        tool_name: str,
        namespace: str = "default",
    ) -> None:
        """
        Acquire permission to execute a tool.

        Args:
            tool_name: Tool name
            namespace: Tool namespace

        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        if not self._config.enabled:
            return

        key = self._get_key(tool_name, namespace)

        # Check concurrent limit
        async with self._concurrent_lock:
            if self._concurrent[key] >= self._config.max_concurrent:
                raise RateLimitExceeded(
                    f"Concurrent execution limit ({self._config.max_concurrent}) exceeded for {key}",
                    retry_after=1.0,
                )

        # Check rate limit based on strategy
        if self._config.strategy == "sliding_window":
            await self._sliding_window_check(key)
        elif self._config.strategy == "token_bucket":
            await self._token_bucket_check(key)
        elif self._config.strategy == "fixed_window":
            await self._fixed_window_check(key)

        # Increment concurrent counter
        async with self._concurrent_lock:
            self._concurrent[key] += 1

    async def release(
        self,
        tool_name: str,
        namespace: str = "default",
    ) -> None:
        """
        Release a tool execution slot.

        Args:
            tool_name: Tool name
            namespace: Tool namespace
        """
        if not self._config.enabled:
            return

        key = self._get_key(tool_name, namespace)

        # Decrement concurrent counter
        async with self._concurrent_lock:
            self._concurrent[key] = max(0, self._concurrent[key] - 1)

    async def _sliding_window_check(self, key: str) -> None:
        """Check sliding window rate limit."""
        now = time.time()
        history = self._call_history[key]

        # Remove calls older than 1 minute
        minute_ago = now - 60
        while history and history[0] < minute_ago:
            history.popleft()

        # Check minute limit
        if len(history) >= self._config.max_calls_per_minute:
            retry_after = 60 - (now - history[0])
            raise RateLimitExceeded(
                f"Rate limit exceeded: {len(history)}/{self._config.max_calls_per_minute} calls per minute",
                retry_after=retry_after,
            )

        # Remove calls older than 1 hour
        hour_ago = now - 3600
        while history and history[0] < hour_ago:
            history.popleft()

        # Check hour limit
        if len(history) >= self._config.max_calls_per_hour:
            retry_after = 3600 - (now - history[0])
            raise RateLimitExceeded(
                f"Rate limit exceeded: {len(history)}/{self._config.max_calls_per_hour} calls per hour",
                retry_after=retry_after,
            )

        # Record this call
        history.append(now)

    async def _token_bucket_check(self, key: str) -> None:
        """Check token bucket rate limit."""
        now = time.time()

        # Refill tokens based on elapsed time
        elapsed = now - self._last_refill[key]
        refill_rate = self._config.max_calls_per_minute / 60  # tokens per second
        new_tokens = elapsed * refill_rate
        max_tokens = self._config.burst_size or self._config.max_calls_per_minute

        self._tokens[key] = min(max_tokens, self._tokens[key] + new_tokens)
        self._last_refill[key] = now

        # Check if we have tokens
        if self._tokens[key] < 1:
            retry_after = 1 / refill_rate
            raise RateLimitExceeded(
                f"Rate limit exceeded: no tokens available (refills at {refill_rate:.2f}/s)",
                retry_after=retry_after,
            )

        # Consume a token
        self._tokens[key] -= 1

    async def _fixed_window_check(self, key: str) -> None:
        """Check fixed window rate limit."""
        now = time.time()

        # Check if we need to reset the window (every minute)
        if now - self._window_start[key] >= 60:
            self._window_calls[key] = 0
            self._window_start[key] = now

        # Check limit
        if self._window_calls[key] >= self._config.max_calls_per_minute:
            window_end = self._window_start[key] + 60
            retry_after = window_end - now
            raise RateLimitExceeded(
                f"Rate limit exceeded: {self._window_calls[key]}/{self._config.max_calls_per_minute} calls in window",
                retry_after=retry_after,
            )

        # Increment counter
        self._window_calls[key] += 1

    def get_stats(self, tool_name: str | None = None, namespace: str = "default") -> dict[str, Any]:
        """
        Get rate limiting statistics.

        Args:
            tool_name: Optional tool name to get stats for
            namespace: Tool namespace

        Returns:
            Dictionary of statistics
        """
        if tool_name:
            key = self._get_key(tool_name, namespace)
            return {
                "key": key,
                "concurrent": self._concurrent.get(key, 0),
                "calls_last_minute": len([
                    t for t in self._call_history.get(key, [])
                    if time.time() - t < 60
                ]),
                "calls_last_hour": len(self._call_history.get(key, [])),
                "tokens": self._tokens.get(key, 0) if self._config.strategy == "token_bucket" else None,
            }
        else:
            # Global stats
            return {
                "total_keys": len(self._call_history),
                "total_concurrent": sum(self._concurrent.values()),
                "config": {
                    "strategy": self._config.strategy,
                    "max_calls_per_minute": self._config.max_calls_per_minute,
                    "max_calls_per_hour": self._config.max_calls_per_hour,
                    "max_concurrent": self._config.max_concurrent,
                },
            }

    async def reset(self, tool_name: str | None = None, namespace: str = "default") -> None:
        """
        Reset rate limit counters.

        Args:
            tool_name: Optional tool name to reset (resets all if None)
            namespace: Tool namespace
        """
        if tool_name:
            key = self._get_key(tool_name, namespace)
            self._call_history[key].clear()
            self._tokens[key] = float(self._config.max_calls_per_minute)
            self._last_refill[key] = time.time()
            self._window_calls[key] = 0
            self._window_start[key] = time.time()
            async with self._concurrent_lock:
                self._concurrent[key] = 0
        else:
            self._call_history.clear()
            self._tokens.clear()
            self._last_refill.clear()
            self._window_calls.clear()
            self._window_start.clear()
            async with self._concurrent_lock:
                self._concurrent.clear()
