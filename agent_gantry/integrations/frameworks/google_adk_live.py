"""Per-turn dynamic-tool provider for Google ADK (the *deep* integration).

Where :mod:`agent_gantry.integrations.frameworks.google_adk` exposes the
*static* path — ``for_google_adk`` selects a fixed slice of tools once and you
hand them to ``Agent(tools=[...])`` — this module wires Agent-Gantry into ADK's
native per-request hook so the tool surface is re-selected **before every model
request**. Each reasoning turn, the latest user content is matched against
Gantry's semantic router and only the top-k relevant tools are injected into the
``LlmRequest`` the model sees. This minimises the tool-definition payload sent to
the LLM — the original design goal of Agent-Gantry — while keeping the agent's
*registered* tool list empty.

The native hook is ``google.adk.agents.Agent(before_model_callback=...)``. ADK
invokes the callback with ``(callback_context, llm_request)`` before each model
request and lets the callback mutate ``llm_request`` in place. The canonical way
to add tools to a request is :meth:`LlmRequest.append_tools` — it builds the
``FunctionDeclaration`` for each ``BaseTool`` (the new experimental JSON-schema
path), appends them to ``llm_request.config.tools``, **and** registers each tool
in ``llm_request.tools_dict`` so ADK's flow can execute the resulting function
calls. That dual responsibility is exactly why the recommended setup is
``tools=[]`` on the agent plus callback injection: the callback both shows the
model the relevant tools *and* wires their execution for that turn — no static
registration needed.

.. code-block:: python

    from agent_gantry import AgentGantry
    from agent_gantry.integrations.frameworks.google_adk_live import (
        gantry_adk_agent,
    )

    gantry = AgentGantry()
    # ... register tools, await gantry.sync() ...

    agent = gantry_adk_agent(
        gantry,
        model="gemini-2.0-flash",
        name="assistant",
        instruction="You are a helpful assistant.",
        limit=5,
    )

Or attach the callback to a hand-built agent:

.. code-block:: python

    from google.adk.agents import Agent
    from agent_gantry.integrations.frameworks.google_adk_live import (
        gantry_before_model_callback,
    )

    agent = Agent(
        model="gemini-2.0-flash",
        name="assistant",
        tools=[],
        before_model_callback=gantry_before_model_callback(gantry, limit=5),
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset
from agent_gantry.integrations.frameworks.google_adk import spec_to_google_adk

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Google ADK support requires `google-adk`. "
    "Install it with `pip install google-adk`."
)


def _import_adk() -> Any:
    """Lazily import the ADK ``Agent`` class with a helpful error.

    Deferred so ``import agent_gantry`` never requires ``google-adk``.
    """
    try:
        from google.adk.agents import Agent
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError(_INSTALL_HINT) from exc
    return Agent


def _content_text(content: Any) -> str:
    """Extract the concatenated text of a ``google.genai.types.Content``.

    Tolerates plain strings (and string parts) as well as the structured
    ``Content`` / ``Part`` objects, so simplified test payloads don't raise.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    parts = getattr(content, "parts", None) or []
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, str):
            if part.strip():
                chunks.append(part)
            continue
        text = getattr(part, "text", None)
        if text:
            chunks.append(text)
    return " ".join(chunks).strip()


def _query_from_callback_context(callback_context: Any) -> str:
    """Derive this turn's retrieval query from the ADK callback context.

    Preference order, most-recent-user-text first:

    1. The last user-role message in the session's event history — this is the
       per-turn signal that makes re-selection actually adapt across turns.
    2. ``callback_context.user_content`` — the content that started the
       invocation (the reliable fallback ADK always populates).
    """
    # 1. Walk the session events backwards for the latest user text. ADK
    #    appends an event per turn, so this reflects the *current* turn even in
    #    a multi-step conversation.
    session = getattr(callback_context, "session", None)
    events = getattr(session, "events", None) or []
    for event in reversed(list(events)):
        if getattr(event, "author", None) != "user":
            continue
        text = _content_text(getattr(event, "content", None))
        if text:
            return text

    # 2. Fall back to the invocation's user content.
    return _content_text(getattr(callback_context, "user_content", None))


async def _inject_selected_tools(
    gantry: AgentGantry,
    query: str,
    llm_request: Any,
    *,
    limit: int,
    score_threshold: float,
) -> list[str]:
    """Select tools for ``query`` and inject them into ``llm_request``.

    Uses :meth:`LlmRequest.append_tools`, the native ADK mechanism: it builds a
    ``FunctionDeclaration`` per tool, appends them to
    ``llm_request.config.tools``, and registers each tool in
    ``llm_request.tools_dict`` so ADK can execute the resulting calls this turn.

    Returns the names of the tools whose declarations were injected (useful for
    logging and tests). Never raises on selection failure — retrieval must not
    break the agent run.
    """
    if not query:
        return []
    try:
        specs = await GantryToolset(gantry).select(
            query, limit=limit, score_threshold=score_threshold
        )
    except Exception:
        logger.exception(
            "gantry_before_model_callback: semantic retrieval failed; "
            "continuing without dynamic tools."
        )
        return []

    tools = [spec_to_google_adk(spec) for spec in specs]
    if not tools:
        return []

    llm_request.append_tools(tools)
    injected = [getattr(t, "name", "") for t in tools]
    logger.debug(
        "gantry_before_model_callback: injected %d tools for query=%r: %s",
        len(injected),
        query,
        injected,
    )
    return injected


def gantry_before_model_callback(
    gantry: AgentGantry,
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
) -> Any:
    """Build an ADK ``before_model_callback`` that injects Gantry tools per turn.

    The returned async callback is suitable for
    ``Agent(before_model_callback=...)``. ADK calls it before **each** model
    request with ``(callback_context, llm_request)``. The callback:

    1. Derives this turn's query from ``callback_context`` (latest user message,
       falling back to the invocation's ``user_content``).
    2. Re-selects the top-``limit`` tools via Gantry's semantic router.
    3. Injects their declarations into ``llm_request`` via
       :meth:`LlmRequest.append_tools`, which also registers the tools in
       ``llm_request.tools_dict`` so ADK can execute them this turn.

    It returns ``None``, so ADK proceeds with the (now tool-augmented) request.
    Recommended setup: ``Agent(tools=[], before_model_callback=...)`` — the
    callback supplies both the tool declarations *and* their execution wiring,
    so no static tool registration on the agent is required.

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` providing
            semantic retrieval and execution.
        limit: Maximum number of tools to select and inject per turn.
        score_threshold: Minimum semantic relevance score (``0.0`` = no
            filtering).

    Returns:
        An ``async`` callback ``(callback_context, llm_request) -> None``.
    """

    async def _callback(callback_context: Any, llm_request: Any) -> None:
        query = _query_from_callback_context(callback_context)
        await _inject_selected_tools(
            gantry,
            query,
            llm_request,
            limit=limit,
            score_threshold=score_threshold,
        )
        # Return None: ADK proceeds with the mutated llm_request.
        return None

    return _callback


def gantry_adk_agent(
    gantry: AgentGantry,
    *,
    model: Any,
    name: str,
    instruction: str = "",
    limit: int = 5,
    score_threshold: float = 0.0,
    **agent_kwargs: Any,
) -> Any:
    """Build an ADK ``Agent`` wired for per-turn dynamic tool selection.

    Constructs ``Agent(model=model, name=name, instruction=instruction,
    tools=[], before_model_callback=gantry_before_model_callback(...))``. The
    agent ships with *no* statically registered tools — the callback injects the
    relevant slice before every model request (see
    :func:`gantry_before_model_callback`).

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` instance.
        model: The ADK model spec (e.g. ``"gemini-2.0-flash"`` or a model
            object) passed straight to ``Agent(model=...)``.
        name: The agent name.
        instruction: The system instruction for the agent.
        limit: Maximum number of tools selected/injected per turn.
        score_threshold: Minimum semantic relevance score for selection.
        **agent_kwargs: Extra keyword arguments forwarded to ``Agent(...)``.

    Returns:
        A configured ``google.adk.agents.Agent``.

    Raises:
        ImportError: If ``google-adk`` is not installed.
    """
    agent_cls = _import_adk()
    return agent_cls(
        model=model,
        name=name,
        instruction=instruction,
        tools=[],
        before_model_callback=gantry_before_model_callback(
            gantry, limit=limit, score_threshold=score_threshold
        ),
        **agent_kwargs,
    )


__all__ = ["gantry_before_model_callback", "gantry_adk_agent"]
