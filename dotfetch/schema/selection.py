"""
Models for non-semantic selection.

Semantic routing embeds a query and searches a vector store. A *selector* is
the alternative: it hands a catalogue and a query to a model that answers
directly which entries are relevant, with no embeddings and no vector store.

These models are deliberately neutral about what is being selected — tools,
Agent Skills and MCP servers all reduce to a :class:`SelectionCandidate` — so a
selector adapter never imports the registry and can be tested on its own.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


def _origin(tool: Any) -> str | None:
    """The MCP server or A2A agent ``tool`` came from, if it came from one."""
    metadata = getattr(tool, "metadata", None) or {}
    origin = metadata.get("mcp_server") or metadata.get("a2a_agent")
    return origin if isinstance(origin, str) and origin else None


class SelectionCandidate(BaseModel):
    """One entry a selector may choose.

    Attributes:
        id: The caller's identifier, echoed back in results. Never sent to a
            provider, so it is free to be a qualified name, a URL or a UUID.
        name: Short human-readable name, sent to the model.
        description: What the entry does, sent to the model. Adapters may
            truncate it to stay within a request budget.
        group: Optional grouping key — a namespace, or the MCP server a tool
            came from. Used by two-stage selection, which narrows to groups
            before narrowing to members.
        tags: Optional extra signals.
        examples: Example requests this entry handles. Carried because the
            embedding path already embeds them via ``to_searchable_text()``,
            and they are the single strongest signal a catalogue entry has:
            withholding them from a selector while giving them to the vector
            store is not a fair comparison, it is a handicap.
    """

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = ""
    group: str | None = None
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)

    @classmethod
    def from_tool(cls, tool: Any) -> SelectionCandidate:
        """Build a candidate from a :class:`~dotfetch.schema.tool.ToolDefinition`.

        The id is the fully qualified name, which is exact and unique but ends
        in a version. The *name* the model reads drops that version: it is
        noise for a relevance judgement, and decision models generally read
        semantic text better than numbers.
        """
        return cls(
            id=tool.qualified_name,
            name=f"{tool.namespace}.{tool.name}",
            description=tool.description,
            # An MCP tool groups by the server it came from, and a tool
            # converted from an A2A skill by its agent. Their namespace is
            # usually ``"default"`` for every server and agent at once, which
            # gave the two-stage path a single group and so nothing to narrow.
            group=_origin(tool) or tool.namespace,
            tags=list(tool.tags),
            examples=list(tool.examples),
        )

    @classmethod
    def from_skill(cls, skill: Any) -> SelectionCandidate:
        """Build a candidate from a :class:`~dotfetch.schema.skill.Skill`.

        The skill's body is deliberately left out: it runs to 50,000
        characters, and what a skill is *for* lives in its description.
        Everything else the embedding path puts in ``to_embedding_text()``
        comes across — the author's own tags as well as the category. Sending
        the category alone would have given the selector strictly less to go on
        than the vector store gets, which is a handicap rather than a choice.
        """
        category = getattr(skill, "category", None)
        tags = list(getattr(skill, "tags", []) or [])
        if category:
            tags.append(str(getattr(category, "value", category)))
        return cls(
            id=f"{skill.namespace}.{skill.name}",
            name=f"{skill.namespace}.{skill.name}",
            description=skill.description,
            group=skill.namespace,
            tags=tags,
        )

    @classmethod
    def from_mcp_server(cls, server: Any) -> SelectionCandidate:
        """Build a candidate from an MCP server definition.

        Servers group by namespace like tools do, but a namespace usually holds
        few servers, so the two-stage path rarely engages here.
        """
        return cls(
            id=f"{server.namespace}.{server.name}",
            name=f"{server.namespace}.{server.name}",
            description=server.description,
            group=server.namespace,
            tags=list(getattr(server, "tags", []) or []),
            examples=list(getattr(server, "examples", []) or []),
        )


    @classmethod
    def from_a2a_agent(cls, agent: Any) -> SelectionCandidate:
        """Build a candidate from an A2A agent definition.

        The description is followed by the agent's skills when its card has
        been read, because the skills are what a request is matched to.
        """
        skills = list(getattr(agent, "skills", []) or [])
        description = agent.description
        if skills:
            description = f"{description} Skills: {'; '.join(skills)}"
        return cls(
            id=f"{agent.namespace}.{agent.name}",
            name=f"{agent.namespace}.{agent.name}",
            description=description[:4000],
            group=agent.namespace,
            tags=list(getattr(agent, "tags", []) or []),
            examples=list(getattr(agent, "examples", []) or []),
        )


class SelectionResult(BaseModel):
    """What a selector returned.

    Attributes:
        selected: Candidate ids that cleared the threshold, best first.
        scores: Every id the model scored, mapped to its probability in
            ``[0, 1]``. Includes ids that did not clear the threshold, so a
            caller can inspect near-misses or apply its own cut.
        fallback: ``True`` when the model was not consulted or did not answer,
            and the caller must fall back to its own ordering. A selector never
            raises for a provider failure: failing closed hands the agent an
            empty tool list, which is worse than an unselected one.
        reason: Why the fallback happened, for logs and tests.
        input_tokens: Provider-reported input tokens.
        output_tokens: Provider-reported output tokens.
        requests: How many provider requests the pass took.
    """

    selected: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    fallback: bool = False
    reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
