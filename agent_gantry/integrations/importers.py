"""Reverse-direction importers: register existing framework-native tools into Gantry.

Every other module in ``agent_gantry.integrations`` **exports** Gantry tools to
a framework (:class:`~agent_gantry.integrations.frameworks.langchain.LangChainAdapter`,
``CrewAIAdapter``, ``LlamaIndexAdapter``, ...). This module is the missing
other half: it **imports** tools a user already built with those frameworks
into Gantry's own registry, so they gain semantic routing, security policies,
rate limiting, retries, circuit breakers, and telemetry — and become
re-exportable to any *other* framework via the existing export adapters.
"Register once, use anywhere" only holds if tools can enter Gantry from
outside it, not just leave it.

This mirrors how MCP/A2A discovery builds
:class:`~agent_gantry.schema.tool.ToolDefinition` objects (see
``adapters/executors/mcp_client.py``'s ``_convert_tool`` and
``providers/a2a_client.py``): name + description + JSON-Schema parameters
extracted from the native object, ``source``/``source_uri`` set for
provenance. Where this module goes further than the legacy
``add_mcp_server()``/``add_a2a_agent()`` path is execution wiring: each
converted tool also gets an async wrapper around the native tool's own
invocation method, registered via
:meth:`~agent_gantry.core.gantry.AgentGantry.add_tool`'s optional ``handler``
argument — the same mechanism the ``@gantry.register`` decorator uses
internally — so the imported tool runs through the *normal* ``gantry.execute``
path (security policy, retries, circuit breakers, telemetry) exactly like a
tool defined directly against Gantry, rather than needing a parallel
execution route.

Three public coroutines, one per supported framework:

- :func:`register_langchain_tools` — ``langchain_core.tools.BaseTool`` /
  ``StructuredTool``.
- :func:`register_crewai_tools` — ``crewai.tools.BaseTool``.
- :func:`register_llamaindex_tools` — ``llama_index.core.tools.FunctionTool``.

Each lazily imports its framework (``import agent_gantry`` never requires
langchain-core/crewai/llama-index-core to be installed — the ``ImportError``
only fires when a ``register_*_tools`` call is actually made), converts every
native tool to a ``ToolDefinition`` (``source=ToolSource.FRAMEWORK``) plus an
execution wrapper, and registers it. An object that isn't the expected native
type, or that fails to convert for any other reason, is skipped with a logged
warning rather than aborting the whole batch — one malformed tool shouldn't
sink the other nine. The only case that raises is an empty/falsy ``tools``
argument, since that is almost certainly a caller mistake rather than a
tool-level problem.

Fidelity notes (per framework, see each function's docstring for detail):
LangChain's ``return_direct``/``callbacks`` machinery outside of ``ainvoke``
itself has no Gantry equivalent; CrewAI's ``BaseTool.run()`` usage-count
bookkeeping and console print are bypassed since the handler calls ``_run``
directly; LlamaIndex tools that ``require_context`` will fail at execution
time because Gantry does not supply a ``Context``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from agent_gantry.schema.tool import ToolDefinition, ToolSource

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

logger = logging.getLogger(__name__)

__all__ = [
    "register_crewai_tools",
    "register_langchain_tools",
    "register_llamaindex_tools",
]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

_NAME_SANITIZE_RE = re.compile(r"[^a-z0-9_]+")
_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def _normalize_tool_name(raw_name: str | None) -> str:
    """Coerce an arbitrary framework tool name into Gantry's ``^[a-z][a-z0-9_]*$``.

    Framework tool names are free text — CrewAI in particular favors
    human-readable, space-separated names meant for the LLM to read. The
    original name is never discarded: callers stash it in ``source_uri`` and
    ``metadata["native_name"]`` on the resulting :class:`ToolDefinition` for
    traceability and re-export.
    """
    name = (raw_name or "tool").strip().lower()
    name = _NAME_SANITIZE_RE.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "tool"
    if not name[0].isalpha():
        name = f"t_{name}"
    return name


def _require_nonempty(tools: Sequence[Any], framework: str) -> None:
    """Raise on an empty/falsy ``tools`` argument (a caller mistake, not a per-tool issue)."""
    if not tools:
        raise ValueError(f"register_{framework}_tools() requires at least one tool; got {tools!r}.")


async def _register_converted(
    gantry: AgentGantry,
    tool_def: ToolDefinition,
    handler: Any,
    seen: set[str],
) -> bool:
    """Register one converted tool, guarding against in-batch name collisions.

    Two native tools whose names normalize to the same Gantry identifier
    within one call (e.g. "Get Weather" and "get_weather") would otherwise
    silently overwrite each other in the registry. Skip the second with a
    warning instead.
    """
    key = f"{tool_def.namespace}.{tool_def.name}"
    if key in seen:
        logger.warning(
            "Skipping tool import: normalized name '%s' collides with another "
            "tool already imported in this call. Pass distinct `namespace=` "
            "values or rename the source tools to disambiguate.",
            key,
        )
        return False
    seen.add(key)
    await gantry.add_tool(tool_def, handler=handler)
    return True


# --------------------------------------------------------------------------- #
# LangChain
# --------------------------------------------------------------------------- #


def _langchain_parameters_schema(tool: Any) -> dict[str, Any]:
    """Extract a JSON Schema from a LangChain tool's ``args_schema``.

    ``args_schema`` may be a Pydantic model class, a raw dict schema, or
    absent entirely — in which case LangChain can still derive one via
    ``get_input_schema()`` (this is what a ``BaseTool`` subclass with no
    explicit ``args_schema`` falls back to internally). Falls back to an
    empty object schema only if none of those succeed.
    """
    schema_source = getattr(tool, "args_schema", None)
    if isinstance(schema_source, dict):
        return schema_source
    model_json_schema = getattr(schema_source, "model_json_schema", None)
    if callable(model_json_schema):
        try:
            return model_json_schema()
        except Exception:  # noqa: BLE001 - schema extraction is best-effort
            logger.debug(
                "langchain tool %r: args_schema.model_json_schema() failed",
                tool,
                exc_info=True,
            )

    get_input_schema = getattr(tool, "get_input_schema", None)
    if callable(get_input_schema):
        try:
            return get_input_schema().model_json_schema()
        except Exception:  # noqa: BLE001 - schema extraction is best-effort
            logger.debug("langchain tool %r: get_input_schema() failed", tool, exc_info=True)

    return dict(_EMPTY_SCHEMA)


def _convert_langchain_tool(
    tool: Any, *, namespace: str, tags: list[str] | None
) -> tuple[ToolDefinition, Any]:
    """Convert one LangChain ``BaseTool`` into a ``(ToolDefinition, handler)`` pair."""
    native_name = getattr(tool, "name", None) or type(tool).__name__
    name = _normalize_tool_name(native_name)
    description = (getattr(tool, "description", None) or "").strip() or f"Tool: {name}"
    parameters_schema = _langchain_parameters_schema(tool)

    async def _handler(**kwargs: Any) -> Any:
        # ainvoke routes through LangChain's own Runnable pipeline (callback
        # manager, handle_tool_error/handle_validation_error) and works for
        # both sync- and async-defined tools — LangChain offloads a sync
        # `_run` to a thread internally, so this never blocks the event loop.
        return await tool.ainvoke(kwargs)

    _handler.__name__ = name
    _handler.__doc__ = description

    tool_def = ToolDefinition(
        name=name,
        namespace=namespace,
        description=description,
        parameters_schema=parameters_schema,
        tags=list(tags or []),
        source=ToolSource.FRAMEWORK,
        source_uri=f"langchain://{native_name}",
        metadata={
            "framework": "langchain",
            "native_class": type(tool).__name__,
            "native_name": native_name,
            "return_direct": bool(getattr(tool, "return_direct", False)),
        },
    )
    return tool_def, _handler


async def register_langchain_tools(
    gantry: AgentGantry,
    tools: Sequence[Any],
    *,
    namespace: str = "langchain",
    tags: list[str] | None = None,
) -> int:
    """Register existing LangChain ``BaseTool``/``StructuredTool`` objects into Gantry.

    Extracts ``name`` / ``description`` / ``args_schema`` (transcoded to JSON
    Schema via ``model_json_schema()``; tools without an explicit
    ``args_schema`` fall back to LangChain's own inferred
    ``get_input_schema()``) and wraps each tool's ``ainvoke`` as the Gantry
    execution handler, so calls run through the tool's full native invocation
    path while still being gated by Gantry's security policy, retries, and
    telemetry.

    Lossy: ``return_direct`` (LangChain's agent-loop short-circuit hint) and
    any ``callbacks``/``tags``/``verbose`` configured directly on the tool
    instance are preserved verbatim in ``ToolDefinition.metadata`` for
    inspection but have no equivalent behavior in Gantry's own execution
    model — Gantry always returns the raw result and always applies its own
    retry/telemetry pipeline regardless of what the LangChain tool declares.

    Args:
        gantry: The ``AgentGantry`` instance to register into.
        tools: LangChain ``BaseTool`` (or ``StructuredTool``) instances.
        namespace: Gantry namespace to register the tools under.
        tags: Tags applied to every imported tool.

    Returns:
        Number of tools successfully registered. Tools that aren't a
        ``BaseTool`` instance, or that fail to convert, are skipped (logged
        at WARNING) and not counted.

    Raises:
        ValueError: If ``tools`` is empty.
        ImportError: If ``langchain-core`` is not installed.

    Example:
        >>> from langchain_core.tools import tool
        >>> @tool
        ... def get_weather(city: str) -> str:
        ...     '''Get the current weather for a city.'''
        ...     return f"Sunny in {city}"
        >>> await register_langchain_tools(gantry, [get_weather])
        1
    """
    _require_nonempty(tools, "langchain")
    try:
        from langchain_core.tools import BaseTool
    except ImportError as exc:  # pragma: no cover - exercised via stub in tests
        raise ImportError(
            "register_langchain_tools() requires `langchain-core`. "
            "Install it with `pip install langchain-core`."
        ) from exc

    registered = 0
    seen: set[str] = set()
    for tool in tools:
        if not isinstance(tool, BaseTool):
            logger.warning(
                "Skipping LangChain tool import: %r is not a langchain_core BaseTool instance.",
                tool,
            )
            continue
        try:
            tool_def, handler = _convert_langchain_tool(tool, namespace=namespace, tags=tags)
        except Exception:  # noqa: BLE001 - one bad tool must not abort the batch
            logger.warning(
                "Skipping LangChain tool %r: conversion failed.",
                getattr(tool, "name", tool),
                exc_info=True,
            )
            continue
        if await _register_converted(gantry, tool_def, handler, seen):
            registered += 1
    return registered


# --------------------------------------------------------------------------- #
# CrewAI
# --------------------------------------------------------------------------- #

_CREWAI_DESC_MARKER = "Tool Description: "


def _crewai_raw_description(tool: Any) -> str:
    """Recover the user-authored description from CrewAI's composite string.

    ``crewai.tools.BaseTool.model_post_init`` overwrites its own
    ``description`` field with ``"Tool Name: ...\\nTool Arguments:
    ...\\nTool Description: <original>"`` (see
    ``BaseTool._generate_description``), so the raw text is only recoverable
    by peeling off that prefix. Falls back to the full string when the
    marker isn't present (e.g. a subclass that sets ``description`` outside
    the standard construction flow).
    """
    desc = getattr(tool, "description", None) or ""
    idx = desc.rfind(_CREWAI_DESC_MARKER)
    if idx != -1:
        return desc[idx + len(_CREWAI_DESC_MARKER) :].strip()
    return desc.strip()


def _crewai_parameters_schema(tool: Any) -> dict[str, Any]:
    """Extract a JSON Schema from a CrewAI tool's ``args_schema``.

    Unlike LangChain, CrewAI's ``BaseTool.args_schema`` is never ``None`` in
    practice — it defaults to a schema inferred from ``_run``'s annotations
    when the tool author doesn't supply one — but the fallback below still
    guards against a hand-built duck-typed tool that leaves it unset.
    """
    args_schema = getattr(tool, "args_schema", None)
    model_json_schema = getattr(args_schema, "model_json_schema", None)
    if callable(model_json_schema):
        try:
            return model_json_schema()
        except Exception:  # noqa: BLE001 - schema extraction is best-effort
            logger.debug(
                "crewai tool %r: args_schema.model_json_schema() failed", tool, exc_info=True
            )
    return dict(_EMPTY_SCHEMA)


def _convert_crewai_tool(
    tool: Any, *, namespace: str, tags: list[str] | None
) -> tuple[ToolDefinition, Any]:
    """Convert one CrewAI ``BaseTool`` into a ``(ToolDefinition, handler)`` pair."""
    native_name = getattr(tool, "name", None) or type(tool).__name__
    name = _normalize_tool_name(native_name)
    description = _crewai_raw_description(tool) or f"Tool: {name}"
    parameters_schema = _crewai_parameters_schema(tool)

    run = tool._run  # the abstract method every crewai BaseTool implements

    async def _handler(**kwargs: Any) -> Any:
        # CrewAI's own BaseTool.run() would call asyncio.run() internally if
        # _run returns a coroutine, which raises when invoked from inside
        # Gantry's already-running event loop. Call _run directly instead and
        # handle both cases ourselves: await a coroutine function, or offload
        # a plain sync _run to a worker thread so it doesn't block the loop.
        if inspect.iscoroutinefunction(run):
            return await run(**kwargs)
        return await asyncio.to_thread(run, **kwargs)

    _handler.__name__ = name
    _handler.__doc__ = description

    tool_def = ToolDefinition(
        name=name,
        namespace=namespace,
        description=description,
        parameters_schema=parameters_schema,
        tags=list(tags or []),
        source=ToolSource.FRAMEWORK,
        source_uri=f"crewai://{native_name}",
        metadata={
            "framework": "crewai",
            "native_class": type(tool).__name__,
            "native_name": native_name,
            "result_as_answer": bool(getattr(tool, "result_as_answer", False)),
        },
    )
    return tool_def, _handler


async def register_crewai_tools(
    gantry: AgentGantry,
    tools: Sequence[Any],
    *,
    namespace: str = "crewai",
    tags: list[str] | None = None,
) -> int:
    """Register existing CrewAI ``crewai.tools.BaseTool`` objects into Gantry.

    Extracts ``name`` / ``description`` / ``args_schema`` and wraps ``_run``
    as the Gantry execution handler (awaited directly if ``_run`` is itself a
    coroutine function, otherwise offloaded to a worker thread so it never
    blocks the event loop). ``description`` is recovered from CrewAI's
    composite ``"Tool Name: ...\\nTool Arguments: ...\\nTool Description:
    ..."`` string that ``BaseTool`` generates at construction time — see
    :func:`_crewai_raw_description`.

    Lossy: CrewAI's ``BaseTool.run()`` wrapper (the console "Using Tool:
    ..." print and ``current_usage_count`` bookkeeping) is bypassed since the
    handler calls ``_run`` directly. ``result_as_answer`` (CrewAI's "treat
    this tool's output as the agent's final answer" flag) is preserved in
    ``ToolDefinition.metadata`` but has no equivalent in Gantry's own
    execution model.

    Args:
        gantry: The ``AgentGantry`` instance to register into.
        tools: CrewAI ``BaseTool`` instances.
        namespace: Gantry namespace to register the tools under.
        tags: Tags applied to every imported tool.

    Returns:
        Number of tools successfully registered. Tools that aren't a
        ``BaseTool`` instance, or that fail to convert, are skipped (logged
        at WARNING) and not counted.

    Raises:
        ValueError: If ``tools`` is empty.
        ImportError: If ``crewai`` is not installed.
    """
    _require_nonempty(tools, "crewai")
    try:
        from crewai.tools import BaseTool
    except ImportError as exc:  # pragma: no cover - exercised via stub in tests
        raise ImportError(
            "register_crewai_tools() requires `crewai`. Install it with `pip install crewai`."
        ) from exc

    registered = 0
    seen: set[str] = set()
    for tool in tools:
        if not isinstance(tool, BaseTool):
            logger.warning(
                "Skipping CrewAI tool import: %r is not a crewai.tools.BaseTool instance.",
                tool,
            )
            continue
        try:
            tool_def, handler = _convert_crewai_tool(tool, namespace=namespace, tags=tags)
        except Exception:  # noqa: BLE001 - one bad tool must not abort the batch
            logger.warning(
                "Skipping CrewAI tool %r: conversion failed.",
                getattr(tool, "name", tool),
                exc_info=True,
            )
            continue
        if await _register_converted(gantry, tool_def, handler, seen):
            registered += 1
    return registered


# --------------------------------------------------------------------------- #
# LlamaIndex
# --------------------------------------------------------------------------- #


def _llamaindex_parameters_schema(metadata: Any) -> dict[str, Any]:
    """Extract a JSON Schema from a LlamaIndex ``FunctionTool``'s ``ToolMetadata``.

    ``ToolMetadata.get_parameters_dict()`` is preferred over reading
    ``fn_schema`` directly: it also covers LlamaIndex's own fallback for
    tools built with ``fn_schema=None`` (a single generic ``input: str``
    schema), so this stays correct even for tools that skip typed arguments
    entirely.
    """
    get_parameters_dict = getattr(metadata, "get_parameters_dict", None)
    if callable(get_parameters_dict):
        try:
            schema = get_parameters_dict()
            if isinstance(schema, dict):
                return schema
        except Exception:  # noqa: BLE001 - schema extraction is best-effort
            logger.debug(
                "llamaindex tool metadata %r: get_parameters_dict() failed",
                metadata,
                exc_info=True,
            )

    fn_schema = getattr(metadata, "fn_schema", None)
    model_json_schema = getattr(fn_schema, "model_json_schema", None)
    if callable(model_json_schema):
        try:
            return model_json_schema()
        except Exception:  # noqa: BLE001 - schema extraction is best-effort
            logger.debug(
                "llamaindex tool metadata %r: fn_schema.model_json_schema() failed",
                metadata,
                exc_info=True,
            )

    return dict(_EMPTY_SCHEMA)


def _convert_llamaindex_tool(
    tool: Any, *, namespace: str, tags: list[str] | None
) -> tuple[ToolDefinition, Any]:
    """Convert one LlamaIndex ``FunctionTool`` into a ``(ToolDefinition, handler)`` pair."""
    metadata = tool.metadata
    native_name = getattr(metadata, "name", None) or type(tool).__name__
    name = _normalize_tool_name(native_name)
    description = (getattr(metadata, "description", None) or "").strip() or f"Tool: {name}"
    parameters_schema = _llamaindex_parameters_schema(metadata)

    async def _handler(**kwargs: Any) -> Any:
        # acall() is the tool's full native async call path (respects
        # partial_params/field_defaults/callback overrides); unwrap the
        # ToolOutput down to raw_output so the original Python value (not the
        # stringified `.content`) flows back through ToolResult.result.
        result = await tool.acall(**kwargs)
        return getattr(result, "raw_output", result)

    _handler.__name__ = name
    _handler.__doc__ = description

    tool_def = ToolDefinition(
        name=name,
        namespace=namespace,
        description=description,
        parameters_schema=parameters_schema,
        tags=list(tags or []),
        source=ToolSource.FRAMEWORK,
        source_uri=f"llamaindex://{native_name}",
        metadata={
            "framework": "llamaindex",
            "native_class": type(tool).__name__,
            "native_name": native_name,
            "return_direct": bool(getattr(metadata, "return_direct", False)),
        },
    )
    return tool_def, _handler


async def register_llamaindex_tools(
    gantry: AgentGantry,
    tools: Sequence[Any],
    *,
    namespace: str = "llamaindex",
    tags: list[str] | None = None,
) -> int:
    """Register existing LlamaIndex ``FunctionTool`` objects into Gantry.

    Extracts ``.metadata.name`` / ``.metadata.description`` / parameters (via
    ``metadata.get_parameters_dict()``) and wraps ``.acall()`` as the Gantry
    execution handler, unwrapping the returned ``ToolOutput`` down to its
    ``raw_output`` so the original Python value flows back through
    ``ToolResult.result`` rather than a stringified ``content``.

    Lossy: LlamaIndex tools that ``require_context`` (stateful workflow tools
    expecting a ``Context`` argument) will raise ``ValueError`` at execution
    time — Gantry's execution model has no equivalent of LlamaIndex's
    per-workflow ``Context`` to supply one.

    Args:
        gantry: The ``AgentGantry`` instance to register into.
        tools: LlamaIndex ``FunctionTool`` instances.
        namespace: Gantry namespace to register the tools under.
        tags: Tags applied to every imported tool.

    Returns:
        Number of tools successfully registered. Tools that aren't a
        ``FunctionTool`` instance, or that fail to convert, are skipped
        (logged at WARNING) and not counted.

    Raises:
        ValueError: If ``tools`` is empty.
        ImportError: If ``llama-index-core`` is not installed.
    """
    _require_nonempty(tools, "llamaindex")
    try:
        from llama_index.core.tools import FunctionTool
    except ImportError as exc:  # pragma: no cover - exercised via stub in tests
        raise ImportError(
            "register_llamaindex_tools() requires `llama-index-core`. "
            "Install it with `pip install llama-index-core`."
        ) from exc

    registered = 0
    seen: set[str] = set()
    for tool in tools:
        if not isinstance(tool, FunctionTool):
            logger.warning(
                "Skipping LlamaIndex tool import: %r is not a "
                "llama_index.core.tools.FunctionTool instance.",
                tool,
            )
            continue
        try:
            tool_def, handler = _convert_llamaindex_tool(tool, namespace=namespace, tags=tags)
        except Exception:  # noqa: BLE001 - one bad tool must not abort the batch
            logger.warning(
                "Skipping LlamaIndex tool %r: conversion failed.",
                getattr(getattr(tool, "metadata", None), "name", tool),
                exc_info=True,
            )
            continue
        if await _register_converted(gantry, tool_def, handler, seen):
            registered += 1
    return registered
