"""
Decorator for semantic tool selection in LLM generate functions.

This module provides decorators that wrap LLM client generate functions to
automatically perform semantic tool selection using Agent Gantry before
forwarding to the underlying LLM API.

Example:
    from agent_gantry import AgentGantry, with_semantic_tools

    gantry = AgentGantry()

    # Register tools with gantry...

    @with_semantic_tools(gantry)
    async def generate(prompt: str, *, tools: list | None = None) -> str:
        # Your LLM client logic here
        response = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
        )
        return response.choices[0].message.content
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

P = ParamSpec("P")
R = TypeVar("R")

# Module-level logger to avoid repeated instantiation
_logger = logging.getLogger(__name__)

# Context-local default gantry instance for thread-safe usage
# Uses contextvars to provide isolation between async tasks and threads
_gantry_context: ContextVar[AgentGantry | None] = ContextVar("default_gantry", default=None)


def set_default_gantry(gantry: AgentGantry) -> None:
    """
    Set the default AgentGantry instance for the current context.

    This allows using @with_semantic_tools without explicitly passing
    a gantry instance, making the decorator simpler to use.

    **Thread Safety:** Uses contextvars for proper isolation between
    async tasks and threads. Each context has its own gantry instance.

    Args:
        gantry: The AgentGantry instance to use as default

    Example:
        >>> from agent_gantry import AgentGantry, with_semantic_tools, set_default_gantry
        >>>
        >>> gantry = AgentGantry()
        >>> set_default_gantry(gantry)
        >>>
        >>> @with_semantic_tools(limit=3)  # No need to pass gantry!
        ... async def generate(prompt: str, *, tools=None):
        ...     ...
    """
    _gantry_context.set(gantry)


def get_default_gantry() -> AgentGantry | None:
    """
    Get the default AgentGantry instance for the current context.

    Returns:
        The default gantry or None if not set
    """
    return _gantry_context.get()


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
        auto_sync: Deprecated and ignored — the gantry always ensures an
            (incremental) sync before retrieval.
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
        score_threshold: float = 0.0,
        dialect_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the semantic tool selector.

        Args:
            gantry: The AgentGantry instance for tool retrieval.
            prompt_param: The parameter name containing the user prompt.
            tools_param: The parameter name for passing tools to the LLM.
            limit: Maximum number of tools to retrieve (default: 5).
            dialect: Schema dialect for tool conversion (default: "openai").
            auto_sync: Deprecated and ignored — ``AgentGantry.retrieve()``
                always ensures an incremental sync before routing. Passing
                ``False`` warns and changes nothing.
            score_threshold: Minimum score threshold (default: 0.0 — no
                filtering, matching every framework adapter in
                ``agent_gantry.integrations.frameworks``. This intentionally
                differs from the raw ``ToolQuery`` schema default of 0.5 — see
                ``agent_gantry.schema.query.ToolQuery.score_threshold`` for why
                a non-zero default is a silent-drop trap for convenience APIs).
            dialect_options: Options forwarded to the dialect adapter on every
                call — e.g. ``{"strict": True}`` for OpenAI structured outputs.
        """
        self._gantry = gantry
        self._prompt_param = prompt_param
        self._tools_param = tools_param
        self._limit = limit
        self._dialect = dialect
        self._auto_sync = auto_sync
        self._score_threshold = score_threshold
        self._dialect_options = dialect_options or {}
        if not auto_sync:
            import warnings

            warnings.warn(
                "auto_sync=False has no effect: AgentGantry.retrieve() always "
                "calls ensure_synced() before routing, so tools are synced "
                "regardless of this flag. The parameter is accepted for "
                "backward compatibility and will be removed in a future "
                "release; sync is incremental (fingerprint-based), so leaving "
                "it enabled is cheap.",
                DeprecationWarning,
                stacklevel=3,
            )

    async def _retrieve_tools(self, prompt: str) -> list[dict[str, Any]]:
        """
        Retrieve semantically relevant tools for the given prompt.

        The gantry now handles auto-sync internally with smart fingerprint-based
        change detection, so we no longer need to explicitly sync here.

        Args:
            prompt: The user prompt to match tools against.

        Returns:
            List of tools in the specified dialect format.
        """
        from agent_gantry.schema.query import ConversationContext, ToolQuery

        # Note: sync() is now called automatically by gantry.retrieve()
        # with smart fingerprinting - no need to explicitly call it here

        context = ConversationContext(query=prompt)
        query = ToolQuery(
            context=context,
            limit=self._limit,
            score_threshold=self._score_threshold,
        )

        result = await self._gantry.retrieve(query)

        # Use the dialect registry for extensible provider support
        return result.to_dialect(self._dialect, **self._dialect_options)

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

    def _record_usage_sync(self, response: Any) -> None:
        """Report usage from a synchronously-returned response, best effort.

        The sync wrapper has no loop to await on, and may itself be running
        inside someone else's, so the recording is bridged the same way the
        retrieval above is. Without this the whole telemetry path was dead for
        sync callers -- and ``__call__`` routes every non-coroutine function
        here, so that is the default for anyone wrapping a blocking SDK client.
        """
        telemetry = getattr(self._gantry, "telemetry", None)
        if telemetry is None:
            return
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self._record_usage(response))
                return

            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.submit(asyncio.run, self._record_usage(response)).result()
        except Exception as exc:  # never fail a user's call over accounting
            _logger.debug("Token usage recording skipped: %s", exc)

    async def _record_usage(self, response: Any) -> None:
        """Report provider token usage for ``response``, best effort.

        The flagship claim of semantic tool selection is a smaller prompt, but
        nothing in the library measured it: ``TokenUsageEvent`` was defined and
        never constructed, and ``record_token_usage`` was never called outside
        tests. Every provider this decorator targets returns a ``usage`` block,
        so the *actual* cost of each call is recorded here.

        Savings are deliberately not inferred: computing them needs a real
        baseline (the same prompt with every tool injected), and
        ``agent_gantry.metrics.token_usage`` refuses approximate estimators so
        reported numbers stay auditable. Callers who run a baseline can pass
        both usages to ``calculate_token_savings`` and report it themselves.
        """
        telemetry = getattr(self._gantry, "telemetry", None)
        if telemetry is None:
            return

        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return

        if not isinstance(usage, Mapping):
            # SDK usage objects are plain attribute holders; pull the fields
            # the normalizer knows about rather than guessing at a dump method.
            usage = {
                field: value
                for field in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                    "prompt_token_count",
                    "candidates_token_count",
                    "total_token_count",
                )
                if isinstance(value := getattr(usage, field, None), (int, float))
            }
        if not usage:
            return

        try:
            from agent_gantry.metrics.token_usage import ProviderUsage

            provider_usage = ProviderUsage.from_usage(usage)
            # Dict-shaped responses (replayed fixtures, SDK dumps) carry the
            # model under a key, not an attribute; falling straight through to
            # the dialect recorded "openai" in place of the real model name.
            model_name = getattr(response, "model", None)
            if model_name is None and isinstance(response, Mapping):
                model_name = response.get("model")
            model_name = model_name or self._dialect
            await telemetry.record_token_usage(provider_usage, model_name=str(model_name))
        except Exception as exc:  # never fail a user's call over accounting
            _logger.debug("Token usage recording skipped: %s", exc)

    def wrap_async(self, func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
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
                    _logger.warning("Tool retrieval failed, proceeding without tools: %s", e)

            response = await func(*args, **kwargs)
            await self._record_usage(response)
            return response

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
                            future = executor.submit(asyncio.run, self._retrieve_tools(prompt))
                            tools = future.result()
                    except RuntimeError:
                        # No running loop, safe to use asyncio.run()
                        tools = asyncio.run(self._retrieve_tools(prompt))

                    if tools:
                        kwargs[self._tools_param] = tools
                except Exception as e:
                    # If tool retrieval fails, call function without tools
                    _logger.warning("Tool retrieval failed, proceeding without tools: %s", e)

            response = func(*args, **kwargs)
            self._record_usage_sync(response)
            return response

        return wrapper

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
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
    dialect_options: dict[str, Any] | None = ...,
) -> SemanticToolSelector: ...


@overload
def with_semantic_tools(
    gantry_or_func: Callable[P, R],
) -> Callable[P, R]: ...


def with_semantic_tools(
    gantry_or_func: AgentGantry | Callable[..., Any] | None = None,
    *,
    prompt_param: str = "prompt",
    tools_param: str = "tools",
    limit: int = 5,
    dialect: str = "openai",
    auto_sync: bool = True,
    score_threshold: float = 0.0,
    dialect_options: dict[str, Any] | None = None,
) -> SemanticToolSelector | Callable[..., Any]:
    """
    Decorator for automatic semantic tool selection in LLM generate functions.

    This decorator intercepts calls to the wrapped function, extracts the prompt,
    uses Agent Gantry to semantically select relevant tools, and injects them
    into the function call via the specified tools parameter.

    Can be used in three ways:

    1. With explicit gantry instance:
        @with_semantic_tools(gantry, limit=3)
        async def generate(prompt: str, *, tools: list | None = None) -> Response:
            ...

    2. With default gantry (set via set_default_gantry):
        set_default_gantry(gantry)

        @with_semantic_tools(limit=3)
        async def generate(prompt: str, *, tools: list | None = None) -> Response:
            ...

    3. As a factory that returns a selector:
        selector = with_semantic_tools(gantry, dialect="anthropic")

        @selector
        async def call_claude(messages: list, *, tools: list | None = None):
            ...

    Args:
        gantry_or_func: AgentGantry instance, function to wrap, or None for default.
        prompt_param: Parameter name for the prompt (default: "prompt").
                     Also supports OpenAI/Anthropic "messages" format.
        tools_param: Parameter name for injecting tools (default: "tools").
        limit: Maximum tools to retrieve (default: 5).
        dialect: Tool schema format - "openai", "anthropic", "gemini" (default: "openai").
        auto_sync: Deprecated and ignored — the gantry always ensures an
            incremental sync before retrieval.
        dialect_options: Options forwarded to the dialect adapter on every
            call, e.g. ``{"strict": True}`` for OpenAI structured outputs.
        score_threshold: Minimum relevance score for tools (default: 0.0 — no
            filtering, matching every framework adapter in
            ``agent_gantry.integrations.frameworks``; the raw ``ToolQuery``
            schema default is 0.5, but that default is a silent-drop trap for
            convenience APIs like this one — see
            ``agent_gantry.schema.query.ToolQuery.score_threshold``).

    Returns:
        SemanticToolSelector instance or wrapped function.

    Example:
        from agent_gantry import AgentGantry, with_semantic_tools, set_default_gantry
        from openai import OpenAI

        gantry = AgentGantry()

        @gantry.register
        def get_weather(city: str) -> str:
            '''Get current weather for a city.'''
            return f"Weather in {city}: Sunny"

        # Option 1: Explicit gantry
        @with_semantic_tools(gantry, limit=3)
        async def generate1(prompt: str, *, tools: list | None = None):
            ...

        # Option 2: Use default gantry
        set_default_gantry(gantry)

        @with_semantic_tools(limit=3)
        async def generate2(prompt: str, *, tools: list | None = None):
            ...

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

    # If gantry_or_func is None, use the default gantry
    if gantry_or_func is None:
        default_gantry = _gantry_context.get()
        if default_gantry is None:
            raise ValueError(
                "No gantry provided and no default set. Use one of:\n"
                "  1. @with_semantic_tools(gantry, ...)\n"
                "  2. set_default_gantry(gantry) then @with_semantic_tools(...)"
            )
        return SemanticToolSelector(
            default_gantry,
            prompt_param=prompt_param,
            tools_param=tools_param,
            limit=limit,
            dialect=dialect,
            auto_sync=auto_sync,
            score_threshold=score_threshold,
            dialect_options=dialect_options,
        )

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
            dialect_options=dialect_options,
        )

    # Otherwise, assume it's a function and use default gantry
    if callable(gantry_or_func):
        default_gantry = _gantry_context.get()
        if default_gantry is None:
            raise ValueError(
                "No default gantry set. Use set_default_gantry(gantry) first "
                "or pass gantry explicitly: @with_semantic_tools(gantry)"
            )
        selector = SemanticToolSelector(
            default_gantry,
            prompt_param=prompt_param,
            tools_param=tools_param,
            limit=limit,
            dialect=dialect,
            auto_sync=auto_sync,
            score_threshold=score_threshold,
            dialect_options=dialect_options,
        )
        return selector(gantry_or_func)

    # Invalid argument
    raise TypeError(
        f"Invalid argument type: {type(gantry_or_func).__name__}. "
        "Expected AgentGantry instance, callable, or None."
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
        score_threshold: float = 0.0,
        dialect_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the decorator factory.

        Args:
            gantry: The AgentGantry instance for tool retrieval.
            prompt_param: Default parameter name for the prompt.
            tools_param: Default parameter name for tools.
            limit: Default maximum tools to retrieve.
            dialect: Default schema dialect.
            auto_sync: Deprecated and ignored (see ``with_semantic_tools``).
            dialect_options: Default options forwarded to the dialect adapter.
            score_threshold: Default score threshold (0.0 — no filtering,
                matching every framework adapter; see the module-level note on
                ``with_semantic_tools`` for why this differs from the raw
                ``ToolQuery`` schema default of 0.5).
        """
        self._gantry = gantry
        self._prompt_param = prompt_param
        self._tools_param = tools_param
        self._limit = limit
        self._dialect = dialect
        self._auto_sync = auto_sync
        self._score_threshold = score_threshold
        self._dialect_options = dialect_options or {}

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
            dialect_options=self._dialect_options,
        )

        if func is not None:
            return selector(func)

        def decorator(fn: Callable[P, R]) -> Any:
            return selector(fn)

        return decorator
