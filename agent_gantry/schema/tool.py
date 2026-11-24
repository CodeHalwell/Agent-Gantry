"""
Tool definition models for Agent-Gantry.

Universal representation of tools, regardless of source (Python function, MCP server,
OpenAPI operation, or A2A agent skill).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_gantry.schema.base import HealthMetrics, reject_newlines


class SchemaDialect(str, Enum):
    """Schema dialects for different LLM providers."""

    OPENAI = "openai"
    OPENAI_RESPONSES = "openai_responses"  # OpenAI Responses API (newer API)
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    GROQ = "groq"
    AGENT_FRAMEWORK = "agent_framework"
    AUTO = "auto"


class ToolSource(str, Enum):
    """Source of the tool definition."""

    PYTHON_FUNCTION = "python_function"
    MCP_SERVER = "mcp_server"
    OPENAPI = "openapi"
    A2A_AGENT = "a2a_agent"
    MANUAL = "manual"
    # Imported from another agent framework's native tool object (LangChain
    # BaseTool, CrewAI BaseTool, LlamaIndex FunctionTool, ...) via
    # agent_gantry.integrations.importers. Distinct from PYTHON_FUNCTION
    # (@gantry.register-ed or bare Python callables) so telemetry/governance
    # can tell tools that originated outside Gantry's own registry apart from
    # those authored directly against it. Additive: existing serialized
    # ToolDefinition data using any other source value is unaffected.
    FRAMEWORK = "framework"


class ToolCapability(str, Enum):
    """Capabilities that a tool may have, used for permission checks."""

    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"
    EXECUTE_CODE = "execute_code"
    NETWORK_ACCESS = "network_access"
    FILE_SYSTEM = "file_system"
    FINANCIAL = "financial"
    PII_ACCESS = "pii_access"
    EXTERNAL_API = "external_api"


class ToolCost(BaseModel):
    """Cost model for tool execution."""

    estimated_latency_ms: int = Field(default=100)
    monetary_cost: float | None = Field(default=None)
    rate_limit: int | None = Field(default=None, description="Max calls per minute")
    context_tokens: int = Field(default=0, description="Tokens added when selected")


class ToolHealth(HealthMetrics):
    """Runtime health metrics for a tool."""

    avg_latency_ms: float = Field(default=0.0)
    total_calls: int = Field(default=0)
    circuit_breaker_open: bool = Field(default=False)


#: Bounds on the parameter names :meth:`ToolDefinition.to_searchable_text`
#: indexes, so a tool with a very wide schema cannot crowd out its own
#: description.
_MAX_INDEXED_PARAMETERS = 16
_MAX_INDEXED_PARAMETER_CHARS = 240

#: Parameter names too common to tell one tool from another. They are left out
#: of the searchable text for the reason stop words are left out of an index:
#: on the retrieval eval, indexing them cost paraphrase accuracy (a tool with a
#: ``text`` parameter is no likelier to be about text) and bought nothing.
_GENERIC_PARAMETER_NAMES = frozenset(
    {
        "a", "args", "b", "body", "changes", "content", "count", "cursor", "data",
        "details", "end", "fields", "filter", "filters", "format", "from", "id", "ids",
        "input", "key", "kwargs", "limit", "name", "names", "offset", "options", "order",
        "page", "page_size", "params", "path", "per_page", "q", "query", "since", "size",
        "sort", "start", "text", "to", "type", "until", "url", "value", "values", "x", "y",
    }
)  # fmt: skip

#: Namespace of the pseudo-tools that stand in for MCP servers in the tools
#: index, so that one vector store can route to servers as well as to tools.
MCP_SERVER_NAMESPACE = "__mcp_servers__"

#: Namespace of the pieces a long tool description is indexed in, beside the
#: tool's own entry. See :mod:`agent_gantry.core.chunks`.
CHUNK_NAMESPACE = "__chunks__"

#: Namespace of the pseudo-tools that stand in for A2A agents in the tools
#: index, as :data:`MCP_SERVER_NAMESPACE` does for MCP servers.
A2A_AGENT_NAMESPACE = "__a2a_agents__"

#: Namespaces whose entries are index bookkeeping, not tools. Tool retrieval
#: never returns them.
RESERVED_NAMESPACES = frozenset({MCP_SERVER_NAMESPACE, A2A_AGENT_NAMESPACE, CHUNK_NAMESPACE})

class ToolDefinition(BaseModel):
    """
    Universal representation of a tool.

    Canonical internal format, regardless of original source:
    - Python function
    - MCP server tool
    - OpenAPI operation
    - A2A agent skill
    """

    # Identity
    name: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_]*$")
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    namespace: str = Field(default="default")

    # Discovery
    description: str = Field(..., min_length=10, max_length=2000)
    extended_description: str | None = Field(default=None, max_length=10000)
    examples: list[str] = Field(default_factory=list, max_length=10)
    tags: list[str] = Field(default_factory=list)

    # Schema (canonical JSON Schema for input/output)
    parameters_schema: dict[str, Any] = Field(
        ..., description="JSON Schema for input parameters (OpenAI-style function calling)"
    )
    returns_schema: dict[str, Any] | None = Field(default=None)

    # Provenance
    source: ToolSource = Field(default=ToolSource.PYTHON_FUNCTION)
    source_uri: str | None = Field(default=None)

    # Capabilities & permissions
    capabilities: list[ToolCapability] = Field(default_factory=list)
    requires_confirmation: bool = Field(default=False)

    # Cost model
    cost: ToolCost = Field(default_factory=ToolCost)

    # Runtime (non-persisted)
    health: ToolHealth = Field(default_factory=ToolHealth, exclude=True)

    # Metadata / lifecycle
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    deprecated: bool = Field(default=False)
    deprecation_message: str | None = Field(default=None)
    superseded_by: str | None = Field(default=None)

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    _reject_newline_identifiers = field_validator("name", "version", "namespace")(
        reject_newlines
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that the tool name is not reserved."""
        reserved = {"register", "retrieve", "execute", "list", "delete"}
        if v in reserved:
            raise ValueError(f"Tool name '{v}' is reserved")
        return v

    @property
    def qualified_name(self) -> str:
        """Return namespace.name:version."""
        return f"{self.namespace}.{self.name}:{self.version}"

    def to_dialect(self, dialect: SchemaDialect | str, **options: Any) -> dict[str, Any]:
        """
        Just-in-Time transcoding for specific LLMs or protocols.

        Uses the dialect registry for extensible provider support.

        Args:
            dialect: Target dialect (SchemaDialect enum or string name)
            **options: Provider-specific options (e.g., strict mode for OpenAI)

        Returns:
            Provider-specific tool schema dictionary
        """
        from agent_gantry.adapters.tool_spec.registry import get_adapter

        # Convert enum to string if needed
        dialect_str = dialect.value if isinstance(dialect, SchemaDialect) else dialect
        adapter = get_adapter(dialect_str)
        return adapter.to_provider_schema(self, **options)

    def to_searchable_text(self) -> str:
        """
        Convert tool metadata to searchable text for embedding.

        Combines name, namespace, tags, parameter names, description and
        examples into a single text representation for semantic search.

        The short fields come first on purpose. An embedder reads a bounded
        number of tokens (256 word-pieces for the default
        ``all-MiniLM-L6-v2``, which a 2000-character description alone
        overruns), and whatever sits after a long description is cut off. Tags
        and parameter names are a few tokens each and often the most exact
        signal a tool has, so they go where they cannot be truncated away.

        Returns:
            Concatenated string of tool metadata for embedding

        Example:
            >>> tool = ToolDefinition(
            ...     name="calculate_tax",
            ...     namespace="finance",
            ...     description="Calculate sales tax",
            ...     tags=["math", "money"],
            ...     examples=["tax on $100"],
            ...     parameters_schema={"type": "object", "properties": {}}
            ... )
            >>> text = tool.to_searchable_text()
            >>> "calculate_tax" in text and "finance" in text
            True
        """
        tags = " ".join(self.tags)
        examples = " ".join(self.examples)
        parts = (
            self.name,
            self.namespace,
            tags,
            self._parameter_names(),
            self.description,
            examples,
        )
        return " ".join(part for part in parts if part)

    def _parameter_names(self) -> str:
        """Top-level parameter names, bounded, for :meth:`to_searchable_text`.

        Names only: a parameter's description is prose that competes with the
        tool's own for the embedder's token budget, while its name is often the
        one exact token a query and a tool share (``facet_query``,
        ``resource_group``). Names every API has (``query``, ``limit``,
        ``page``) are skipped; see ``_GENERIC_PARAMETER_NAMES``.
        """
        properties = self.parameters_schema.get("properties")
        if not isinstance(properties, dict):
            return ""
        names: list[str] = []
        length = 0
        for name in properties:
            if not isinstance(name, str) or name.lower() in _GENERIC_PARAMETER_NAMES:
                continue
            length += len(name) + 1
            if len(names) >= _MAX_INDEXED_PARAMETERS or length > _MAX_INDEXED_PARAMETER_CHARS:
                break
            names.append(name)
        return " ".join(names)
