"""
Decorator for semantic tool selection in LLM generate functions.

This module provides decorators that wrap LLM client generate functions to
automatically perform semantic tool selection using Agent Gantry before
forwarding to the underlying LLM API.

Example:
    from agent_gantry import AgentGantry
    from agent_gantry.integrations.decorator import with_semantic_tools

    gantry = AgentGantry()

    # Register tools with gantry...

    @with_semantic_tools(gantry)
    async def generate(prompt: str, *, tools: list | None = None) -> str:
        # Your LLM client logic here
        return await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
        )
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

P = ParamSpec("P")
R = TypeVar("R")

# Module-level logger to avoid repeated instantiation
_logger = logging.getLogger(__name__)


class SemanticToolSelector:
    """
    A wrapper that provides semantic tool selection for LLM generate functions.

    This class intercepts calls to a wrapped function, extracts the prompt,
    uses Agent Gantry to semantically select relevant tools, and injects them
    into the function call.

    Attributes:
        gantry: The AgentGantry instance for tool retrieval.
        prompt_param: The parameter name containing the user prompt.
        tools_param: The parameter name for passing tools to the LLM.
        limit: Maximum number of tools to retrieve.
        dialect: The schema dialect for tool conversion (openai, anthropic, gemini).
        auto_sync: Whether to automatically sync tools before retrieval.
        score_threshold: Minimum score threshold for tool selection.
    """

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        prompt_param: str = "prompt",
        tools_param: str = "tools",
        limit: int = 5,
        dialect: str = "openai",
        auto_sync: bool = True,
        score_threshold: float = 0.5,
    ) -> None:
        """
        Initialize the semantic tool selector.

        Args:
            gantry: The AgentGantry instance for tool retrieval.
            prompt_param: The parameter name containing the user prompt.
            tools_param: The parameter name for passing tools to the LLM.
            limit: Maximum number of tools to retrieve (default: 5).
            dialect: Schema dialect for tool conversion (default: "openai").
            auto_sync: Whether to sync tools before retrieval (default: True).
            score_threshold: Minimum score threshold (default: 0.5).
        """
        self._gantry = gantry
        self._prompt_param = prompt_param
        self._tools_param = tools_param
        self._limit = limit
        self._dialect = dialect
        self._auto_sync = auto_sync
        self._score_threshold = score_threshold

    async def _retrieve_tools(self, prompt: str) -> list[dict[str, Any]]:
        """
        Retrieve semantically relevant tools for the given prompt.

        Args:
            prompt: The user prompt to match tools against.

        Returns:
            List of tools in the specified dialect format.
        """
        from agent_gantry.schema.query import ConversationContext, ToolQuery

        if self._auto_sync:
            await self._gantry.sync()

        context = ConversationContext(query=prompt)
        query = ToolQuery(
            context=context,
            limit=self._limit,
            score_threshold=self._score_threshold,
        )

        result = await self._gantry.retrieve(query)

        # Convert to the appropriate dialect
        if self._dialect == "anthropic":
            return result.to_anthropic_tools()
        # OpenAI and Gemini use the same format (OpenAI-style function calling)
        # Default to OpenAI format for "openai", "gemini", and any unknown dialects
        return result.to_openai_tools()

    def _extract_prompt(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        sig: inspect.Signature,
    ) -> str | None:
        """
        Extract the prompt from function arguments.

        Supports extracting from:
        - A parameter named by `prompt_param`
        - OpenAI-style `messages` parameter (extracts last user message)
        - Anthropic-style `messages` parameter (extracts last user message)

        Args:
            args: Positional arguments to the function.
            kwargs: Keyword arguments to the function.
            sig: The function signature.

        Returns:
            The extracted prompt string, or None if not found.
        """
        # Build a mapping of parameter names to values
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        params = bound.arguments

        # Helper to extract user message from messages list
        def extract_from_messages(messages: Any) -> str | None:
            if isinstance(messages, list) and messages:
                # Find the last user message
                for msg in reversed(messages):
                    if isinstance(msg, dict):
                        role = msg.get("role", "")
                        if role == "user":
                            content = msg.get("content", "")
                            if isinstance(content, str):
                                return content
                            # Handle content as list (multi-modal)
                            if isinstance(content, list):
                                for part in content:
                                    if isinstance(part, dict) and part.get("type") == "text":
                                        text = part.get("text", "")
                                        if isinstance(text, str):
                                            return text
            return None

        # Try direct prompt parameter
        if self._prompt_param in params:
            value = params[self._prompt_param]
            # If the prompt_param points to a messages list, extract from it
            if isinstance(value, list):
                extracted = extract_from_messages(value)
                if extracted:
                    return extracted
            # Otherwise, return as string
            if isinstance(value, str):
                return value
            return str(value)

        # Try OpenAI/Anthropic-style messages if not already handled
        if "messages" in params and self._prompt_param != "messages":
            extracted = extract_from_messages(params["messages"])
            if extracted:
                return extracted

        return None

    def wrap_async(
        self, func: Callable[P, Awaitable[R]]
    ) -> Callable[P, Awaitable[R]]:
        """
        Wrap an async function with semantic tool selection.

        Note: This wrapper mutates the kwargs dictionary by adding tools
        to it when they are successfully retrieved. The original kwargs
        dictionary passed by the caller may be modified.

        Args:
            func: The async function to wrap.

        Returns:
            Wrapped async function.
        """
        sig = inspect.signature(func)

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            prompt = self._extract_prompt(args, kwargs, sig)

            if prompt and self._tools_param not in kwargs:
                try:
                    tools = await self._retrieve_tools(prompt)
                    if tools:
                        kwargs[self._tools_param] = tools
                except Exception as e:
                    # If tool retrieval fails, call function without tools
                    _logger.warning(
                        "Tool retrieval failed, proceeding without tools: %s", e
                    )

            return await func(*args, **kwargs)

        return wrapper

    def wrap_sync(self, func: Callable[P, R]) -> Callable[P, R]:
        """
        Wrap a sync function with semantic tool selection.

        Note: This runs async retrieval synchronously. In threaded environments
        or when an event loop is already running, this may cause issues.
        For best compatibility, prefer using async functions.

        This wrapper mutates the kwargs dictionary by adding tools
        to it when they are successfully retrieved. The original kwargs
        dictionary passed by the caller may be modified.

        Args:
            func: The sync function to wrap.

        Returns:
            Wrapped sync function.
        """
        import warnings

        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            prompt = self._extract_prompt(args, kwargs, sig)

            if prompt and self._tools_param not in kwargs:
                # Run async retrieval in sync context using asyncio.run()
                # This creates a new event loop which is safer than reusing existing ones
                try:
                    # Check if we're already in an async context
                    try:
                        asyncio.get_running_loop()
                        # If we get here, there's already a running loop
                        warnings.warn(
                            "with_semantic_tools sync wrapper is being used inside an "
                            "async context. This may cause issues. Consider using an "
                            "async function instead.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        # Use nest_asyncio pattern or fall back to thread pool
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run, self._retrieve_tools(prompt)
                            )
                            tools = future.result()
                    except RuntimeError:
                        # No running loop, safe to use asyncio.run()
                        tools = asyncio.run(self._retrieve_tools(prompt))

                    if tools:
                        kwargs[self._tools_param] = tools
                except Exception as e:
                    # If tool retrieval fails, call function without tools
                    _logger.warning(
                        "Tool retrieval failed, proceeding without tools: %s", e
                    )

            return func(*args, **kwargs)

        return wrapper

    def __call__(
        self, func: Callable[..., Any]
    ) -> Callable[..., Any]:
        """
        Wrap a function with semantic tool selection.

        Automatically detects async vs sync functions.

        Args:
            func: The function to wrap.

        Returns:
            Wrapped function with semantic tool selection.
        """
        if asyncio.iscoroutinefunction(func):
            return self.wrap_async(func)
        return self.wrap_sync(func)


@overload
def with_semantic_tools(
    gantry_or_func: AgentGantry,
    *,
    prompt_param: str = ...,
    tools_param: str = ...,
    limit: int = ...,
    dialect: str = ...,
    auto_sync: bool = ...,
    score_threshold: float = ...,
) -> SemanticToolSelector: ...


@overload
def with_semantic_tools(
    gantry_or_func: Callable[P, R],
) -> Callable[P, R]: ...


def with_semantic_tools(
    gantry_or_func: AgentGantry | Callable[..., Any],
    *,
    prompt_param: str = "prompt",
    tools_param: str = "tools",
    limit: int = 5,
    dialect: str = "openai",
    auto_sync: bool = True,
    score_threshold: float = 0.5,
) -> SemanticToolSelector | Callable[..., Any]:
    """
    Decorator for automatic semantic tool selection in LLM generate functions.

    This decorator intercepts calls to the wrapped function, extracts the prompt,
    uses Agent Gantry to semantically select relevant tools, and injects them
    into the function call via the specified tools parameter.

    Can be used in two ways:

    1. With explicit gantry instance:
        @with_semantic_tools(gantry, limit=3)
        async def generate(prompt: str, *, tools: list | None = None) -> Response:
            ...

    2. As a class method with gantry instance:
        selector = with_semantic_tools(gantry, dialect="anthropic")

        @selector
        async def call_claude(messages: list, *, tools: list | None = None):
            ...

    Args:
        gantry_or_func: Either an AgentGantry instance or the function to wrap.
        prompt_param: Parameter name for the prompt (default: "prompt").
                     Also supports OpenAI/Anthropic "messages" format.
        tools_param: Parameter name for injecting tools (default: "tools").
        limit: Maximum tools to retrieve (default: 5).
        dialect: Tool schema format - "openai", "anthropic", "gemini" (default: "openai").
        auto_sync: Whether to sync tools before retrieval (default: True).
        score_threshold: Minimum relevance score for tools (default: 0.5).

    Returns:
        SemanticToolSelector instance or wrapped function.

    Example:
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.decorator import with_semantic_tools
        from openai import OpenAI

        gantry = AgentGantry()

        @gantry.register
        def get_weather(city: str) -> str:
            '''Get current weather for a city.'''
            return f"Weather in {city}: Sunny"

        @gantry.register
        def search_web(query: str) -> str:
            '''Search the web for information.'''
            return f"Results for: {query}"

        client = OpenAI()

        @with_semantic_tools(gantry, limit=3)
        async def generate(prompt: str, *, tools: list | None = None):
            return client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
            )

        # When called, the decorator will:
        # 1. Extract the prompt
        # 2. Use Agent Gantry to find relevant tools
        # 3. Inject them as the 'tools' parameter
        response = await generate("What's the weather in Paris?")

    Architectural Notes:
        - The decorator preserves the original function signature
        - Tools are only injected if not already provided
        - Works with both sync and async functions
        - Supports OpenAI messages format for prompt extraction
        - Context is cached per request, not globally

    Tradeoffs:
        - Adds latency for semantic retrieval on each call
        - Sync wrapper may not work well in existing event loops
        - Tool selection is based only on the prompt, not full conversation
    """
    from agent_gantry.core.gantry import AgentGantry

    # If gantry_or_func is an AgentGantry instance, return a selector
    if isinstance(gantry_or_func, AgentGantry):
        return SemanticToolSelector(
            gantry_or_func,
            prompt_param=prompt_param,
            tools_param=tools_param,
            limit=limit,
            dialect=dialect,
            auto_sync=auto_sync,
            score_threshold=score_threshold,
        )

    # Otherwise, we need a default gantry (error case)
    raise TypeError(
        "with_semantic_tools requires an AgentGantry instance as the first argument. "
        "Usage: @with_semantic_tools(gantry)"
    )


# Convenience class for method-style usage
class SemanticToolsDecorator:
    """
    A reusable decorator factory for semantic tool selection.

    This class provides a more object-oriented approach to using the decorator,
    allowing configuration to be set once and reused across multiple functions.

    Example:
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.decorator import SemanticToolsDecorator

        gantry = AgentGantry()
        # ... register tools ...

        decorator = SemanticToolsDecorator(gantry, dialect="openai", limit=5)

        @decorator.wrap
        async def generate_openai(prompt: str, *, tools=None):
            ...

        @decorator.wrap
        async def generate_azure(messages: list, *, tools=None):
            ...
    """

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        prompt_param: str = "prompt",
        tools_param: str = "tools",
        limit: int = 5,
        dialect: str = "openai",
        auto_sync: bool = True,
        score_threshold: float = 0.5,
    ) -> None:
        """
        Initialize the decorator factory.

        Args:
            gantry: The AgentGantry instance for tool retrieval.
            prompt_param: Default parameter name for the prompt.
            tools_param: Default parameter name for tools.
            limit: Default maximum tools to retrieve.
            dialect: Default schema dialect.
            auto_sync: Whether to auto-sync by default.
            score_threshold: Default score threshold.
        """
        self._gantry = gantry
        self._prompt_param = prompt_param
        self._tools_param = tools_param
        self._limit = limit
        self._dialect = dialect
        self._auto_sync = auto_sync
        self._score_threshold = score_threshold

    def wrap(
        self,
        func: Callable[P, R] | None = None,
        *,
        prompt_param: str | None = None,
        tools_param: str | None = None,
        limit: int | None = None,
        dialect: str | None = None,
    ) -> Any:
        """
        Wrap a function with semantic tool selection.

        Can be used as @decorator.wrap or @decorator.wrap(limit=3).

        Args:
            func: The function to wrap (when used without parentheses).
            prompt_param: Override prompt parameter name.
            tools_param: Override tools parameter name.
            limit: Override tool limit.
            dialect: Override schema dialect.

        Returns:
            Wrapped function or decorator.
        """
        selector = SemanticToolSelector(
            self._gantry,
            prompt_param=prompt_param or self._prompt_param,
            tools_param=tools_param or self._tools_param,
            limit=limit if limit is not None else self._limit,
            dialect=dialect or self._dialect,
            auto_sync=self._auto_sync,
            score_threshold=self._score_threshold,
        )

        if func is not None:
            return selector(func)

        def decorator(fn: Callable[P, R]) -> Any:
            return selector(fn)

        return decorator
