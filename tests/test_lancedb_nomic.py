"""
Tests for LanceDB vector store and Nomic embedder.

Tests the unified local vector store with on-device persistence,
Nomic embeddings with Matryoshka truncation, and skills collection.
"""

from __future__ import annotations

import pytest

from agent_gantry.schema.skill import Skill, SkillCategory
from agent_gantry.schema.tool import ToolDefinition


@pytest.fixture
def sample_skill() -> Skill:
    """Create a sample skill for testing."""
    return Skill(
        name="api_pagination",
        description="How to implement cursor-based pagination for REST API endpoints",
        content="""
When implementing pagination for REST APIs, use cursor-based pagination:

1. Include a 'cursor' parameter in the request
2. Return 'next_cursor' in the response
3. Use stable, opaque cursor values (e.g., base64-encoded IDs)
4. Always include 'has_more' boolean flag

Example response:
{
    "data": [...],
    "next_cursor": "abc123",
    "has_more": true
}
""",
        category=SkillCategory.HOW_TO,
        tags=["api", "pagination", "rest"],
        related_tools=["query_database", "fetch_api"],
    )


class TestSkillSchema:
    """Tests for the Skill schema model."""

    def test_skill_creation(self, sample_skill: Skill) -> None:
        """Test basic skill creation."""
        assert sample_skill.name == "api_pagination"
        assert sample_skill.category == SkillCategory.HOW_TO
        assert "api" in sample_skill.tags

    def test_skill_qualified_name(self, sample_skill: Skill) -> None:
        """Test qualified name generation."""
        assert sample_skill.qualified_name == "default.api_pagination"

    def test_skill_content_hash(self, sample_skill: Skill) -> None:
        """Test content hash is deterministic."""
        hash1 = sample_skill.content_hash
        hash2 = sample_skill.content_hash
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_skill_to_prompt_text(self, sample_skill: Skill) -> None:
        """Test formatting for system prompt."""
        text = sample_skill.to_prompt_text()
        assert "Api Pagination" in text
        assert "how_to" in text
        assert "cursor-based pagination" in text.lower()
        assert "query_database" in text

    def test_skill_to_embedding_text(self, sample_skill: Skill) -> None:
        """Test text generation for embedding."""
        text = sample_skill.to_embedding_text()
        assert "api_pagination" in text
        assert "pagination" in text
        assert "how_to" in text

    def test_skill_all_categories(self) -> None:
        """Test all skill categories are valid."""
        categories = [
            SkillCategory.HOW_TO,
            SkillCategory.PATTERN,
            SkillCategory.PROCEDURE,
            SkillCategory.BEST_PRACTICE,
            SkillCategory.TEMPLATE,
            SkillCategory.EXAMPLE,
            SkillCategory.GUIDELINE,
            SkillCategory.WORKFLOW,
        ]
        for cat in categories:
            skill = Skill(
                name="test_skill",
                description="A test skill for category validation",
                content="Test content",
                category=cat,
            )
            assert skill.category == cat


class TestNomicEmbedder:
    """Tests for the Nomic embedder (with mocked model)."""

    def test_nomic_embedder_initialization(self) -> None:
        """Test Nomic embedder can be imported and configured."""
        from agent_gantry.adapters.embedders.nomic import NomicEmbedder

        embedder = NomicEmbedder(dimension=256)
        assert embedder.dimension == 256
        assert embedder.model_name == "nomic-ai/nomic-embed-text-v1.5"

    def test_nomic_embedder_task_prefixes(self) -> None:
        """Test task type configuration via the public API."""
        from agent_gantry.adapters.embedders.nomic import NomicEmbedder

        # Ensure the embedder accepts a task_type at initialization
        embedder = NomicEmbedder(task_type="search_query")

        # Ensure the embedder exposes a public method to change the task type
        assert hasattr(embedder, "set_task_type")
        assert callable(embedder.set_task_type)

        # Setting the same task type should not raise
        embedder.set_task_type("search_query")

    def test_nomic_embedder_matryoshka_dims(self) -> None:
        """Test Matryoshka dimension constants."""
        from agent_gantry.adapters.embedders.nomic import NomicEmbedder

        assert 768 in NomicEmbedder.MATRYOSHKA_DIMS
        assert 256 in NomicEmbedder.MATRYOSHKA_DIMS
        assert 64 in NomicEmbedder.MATRYOSHKA_DIMS

    def test_nomic_embedder_invalid_dimension_raises(self) -> None:
        """Test that invalid dimensions raise ValueError."""
        from agent_gantry.adapters.embedders.nomic import NomicEmbedder

        with pytest.raises(ValueError, match="dimension must be between"):
            NomicEmbedder(dimension=0)

        with pytest.raises(ValueError, match="dimension must be between"):
            NomicEmbedder(dimension=1000)

    def test_nomic_embedder_invalid_task_type_raises(self) -> None:
        """Test that invalid task types raise ValueError."""
        from agent_gantry.adapters.embedders.nomic import NomicEmbedder

        with pytest.raises(ValueError, match="Unsupported task_type"):
            NomicEmbedder(task_type="invalid_type")


class TestLanceDBVectorStore:
    """Tests for the LanceDB vector store."""

    def test_lancedb_initialization(self, tmp_path) -> None:
        """Test LanceDB store can be initialized."""
        from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

        db_path = str(tmp_path / "test_db")
        store = LanceDBVectorStore(db_path=db_path, dimension=64)

        assert store.dimension == 64
        assert store.db_path == db_path

    def test_lancedb_default_path_resolution(self) -> None:
        """Test default database path resolution."""
        from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

        store = LanceDBVectorStore()
        # Should resolve to some valid path containing the expected suffix
        # Use os.path normalization to handle Windows/Unix path differences
        import os
        expected_suffix = os.path.join(".agent_gantry", "lancedb")
        assert store.db_path is not None
        assert expected_suffix in store.db_path or ".agent_gantry/lancedb" in store.db_path

    @pytest.mark.asyncio
    async def test_lancedb_initialize_creates_tables(self, tmp_path) -> None:
        """Test that initialization creates required tables."""
        pytest.importorskip("lancedb")
        pytest.importorskip("pyarrow")

        from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

        db_path = str(tmp_path / "test_db")
        store = LanceDBVectorStore(db_path=db_path, dimension=64)

        await store.initialize()

        # Verify the store is usable via the public API after initialization
        results = await store.search([0.0] * 64, limit=1)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_lancedb_add_and_search_tools(
        self, tmp_path, sample_tools: list[ToolDefinition]
    ) -> None:
        """Test adding tools and searching."""
        pytest.importorskip("lancedb")
        pytest.importorskip("pyarrow")

        from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

        db_path = str(tmp_path / "test_db")
        store = LanceDBVectorStore(db_path=db_path, dimension=64)

        await store.initialize()

        # Create simple embeddings
        embeddings = [[float(i) / 64 for i in range(64)] for _ in sample_tools]

        # Add tools
        count = await store.add_tools(sample_tools, embeddings)
        assert count == len(sample_tools)

        # Search
        query_vector = [0.5] * 64
        results = await store.search(query_vector, limit=3)
        assert len(results) <= 3
        for tool, score in results:
            assert isinstance(tool, ToolDefinition)
            assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_lancedb_add_and_search_skills(
        self, tmp_path, sample_skill: Skill
    ) -> None:
        """Test adding skills and searching."""
        pytest.importorskip("lancedb")
        pytest.importorskip("pyarrow")

        from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

        db_path = str(tmp_path / "test_db")
        store = LanceDBVectorStore(db_path=db_path, dimension=64)

        await store.initialize()

        # Add skill
        embeddings = [[float(i) / 64 for i in range(64)]]
        count = await store.add_skills([sample_skill], embeddings)
        assert count == 1

        # Search skills
        query_vector = [0.5] * 64
        results = await store.search_skills(query_vector, limit=3)
        assert len(results) >= 1
        skill, score = results[0]
        assert skill.name == "api_pagination"

    @pytest.mark.asyncio
    async def test_lancedb_get_by_name(
        self, tmp_path, sample_tools: list[ToolDefinition]
    ) -> None:
        """Test retrieving tools by name."""
        pytest.importorskip("lancedb")
        pytest.importorskip("pyarrow")

        from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

        db_path = str(tmp_path / "test_db")
        store = LanceDBVectorStore(db_path=db_path, dimension=64)

        await store.initialize()

        embeddings = [[float(i) / 64 for i in range(64)] for _ in sample_tools]
        await store.add_tools(sample_tools, embeddings)

        # Get by name
        tool = await store.get_by_name("query_database")
        assert tool is not None
        assert tool.name == "query_database"

        # Non-existent tool
        tool = await store.get_by_name("nonexistent_tool")
        assert tool is None

    @pytest.mark.asyncio
    async def test_lancedb_delete(
        self, tmp_path, sample_tools: list[ToolDefinition]
    ) -> None:
        """Test deleting tools."""
        pytest.importorskip("lancedb")
        pytest.importorskip("pyarrow")

        from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

        db_path = str(tmp_path / "test_db")
        store = LanceDBVectorStore(db_path=db_path, dimension=64)

        await store.initialize()

        embeddings = [[float(i) / 64 for i in range(64)] for _ in sample_tools]
        await store.add_tools(sample_tools, embeddings)

        # Delete
        result = await store.delete("query_database")
        assert result is True

        # Verify deleted
        tool = await store.get_by_name("query_database")
        assert tool is None

    @pytest.mark.asyncio
    async def test_lancedb_list_and_count(
        self, tmp_path, sample_tools: list[ToolDefinition]
    ) -> None:
        """Test listing and counting tools."""
        pytest.importorskip("lancedb")
        pytest.importorskip("pyarrow")

        from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

        db_path = str(tmp_path / "test_db")
        store = LanceDBVectorStore(db_path=db_path, dimension=64)

        await store.initialize()

        embeddings = [[float(i) / 64 for i in range(64)] for _ in sample_tools]
        await store.add_tools(sample_tools, embeddings)

        # Count
        count = await store.count()
        assert count == len(sample_tools)

        # List all
        tools = await store.list_all()
        assert len(tools) == len(sample_tools)

    @pytest.mark.asyncio
    async def test_lancedb_health_check(self, tmp_path) -> None:
        """Test health check."""
        pytest.importorskip("lancedb")
        pytest.importorskip("pyarrow")

        from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

        db_path = str(tmp_path / "test_db")
        store = LanceDBVectorStore(db_path=db_path, dimension=64)

        healthy = await store.health_check()
        assert healthy is True

    @pytest.mark.asyncio
    async def test_lancedb_upsert_behavior(
        self, tmp_path, sample_tools: list[ToolDefinition]
    ) -> None:
        """Test upsert updates existing tools."""
        pytest.importorskip("lancedb")
        pytest.importorskip("pyarrow")

        from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

        db_path = str(tmp_path / "test_db")
        store = LanceDBVectorStore(db_path=db_path, dimension=64)

        await store.initialize()

        # Add initial tools
        embeddings = [[0.1] * 64 for _ in sample_tools]
        await store.add_tools(sample_tools, embeddings, upsert=True)

        count1 = await store.count()

        # Add same tools with upsert
        embeddings2 = [[0.2] * 64 for _ in sample_tools]
        await store.add_tools(sample_tools, embeddings2, upsert=True)

        count2 = await store.count()

        # Count should remain the same (tools were updated, not duplicated)
        assert count1 == count2


class TestSQLInjectionPrevention:
    """Tests for SQL injection prevention in LanceDB adapter."""

    @pytest.mark.asyncio
    async def test_special_characters_in_tool_name(self, tmp_path) -> None:
        """Test that special characters in names don't cause SQL injection."""
        pytest.importorskip("lancedb")
        pytest.importorskip("pyarrow")

        from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

        db_path = str(tmp_path / "test_db")
        store = LanceDBVectorStore(db_path=db_path, dimension=64)
        await store.initialize()

        # Create tool with potentially dangerous name
        tool = ToolDefinition(
            name="test_tool",
            namespace="namespace'with'quotes",  # Contains SQL injection attempt
            description="A test tool with quotes in namespace",
            parameters_schema={"type": "object", "properties": {}},
        )

        embeddings = [[0.5] * 64]
        count = await store.add_tools([tool], embeddings)
        assert count == 1

        # Should be able to retrieve it
        result = await store.get_by_name("test_tool", "namespace'with'quotes")
        assert result is not None
        assert result.namespace == "namespace'with'quotes"

        # Should be able to delete it
        deleted = await store.delete("test_tool", "namespace'with'quotes")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_special_characters_in_namespace_filter(self, tmp_path) -> None:
        """Test that namespace filters with special characters work safely."""
        pytest.importorskip("lancedb")
        pytest.importorskip("pyarrow")

        from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

        db_path = str(tmp_path / "test_db")
        store = LanceDBVectorStore(db_path=db_path, dimension=64)
        await store.initialize()

        # Create tool in namespace with special characters
        tool = ToolDefinition(
            name="my_tool",
            namespace="test'; DROP TABLE tools; --",  # SQL injection attempt
            description="A test tool for SQL injection testing",
            parameters_schema={"type": "object", "properties": {}},
        )

        embeddings = [[0.5] * 64]
        await store.add_tools([tool], embeddings)

        # Search with namespace filter should work safely
        query_vector = [0.5] * 64
        results = await store.search(
            query_vector, limit=5, filters={"namespace": "test'; DROP TABLE tools; --"}
        )

        # Should find the tool (table wasn't dropped)
        assert len(results) >= 1
        assert results[0][0].name == "my_tool"


class TestConfigIntegration:
    """Tests for configuration integration with LanceDB and Nomic."""

    def test_vector_store_config_lancedb_type(self) -> None:
        """Test that LanceDB is a valid vector store type."""
        from agent_gantry.schema.config import VectorStoreConfig

        config = VectorStoreConfig(
            type="lancedb",
            db_path="/tmp/test_db",
            dimension=256,
        )
        assert config.type == "lancedb"
        assert config.db_path == "/tmp/test_db"
        assert config.dimension == 256

    def test_embedder_config_nomic_type(self) -> None:
        """Test that Nomic is a valid embedder type."""
        from agent_gantry.schema.config import EmbedderConfig

        config = EmbedderConfig(
            type="nomic",
            model="nomic-ai/nomic-embed-text-v1.5",
            dimension=256,
            task_type="search_document",
        )
        assert config.type == "nomic"
        assert config.dimension == 256
        assert config.task_type == "search_document"

    def test_gantry_builds_lancedb_store(self, tmp_path) -> None:
        """Test that AgentGantry can build a LanceDB vector store."""
        pytest.importorskip("lancedb")
        pytest.importorskip("pyarrow")

        from agent_gantry import AgentGantry
        from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore
        from agent_gantry.schema.config import AgentGantryConfig, VectorStoreConfig

        config = AgentGantryConfig(
            vector_store=VectorStoreConfig(
                type="lancedb",
                db_path=str(tmp_path / "test_db"),
                dimension=64,
            )
        )

        gantry = AgentGantry(config=config)
        assert isinstance(gantry._vector_store, LanceDBVectorStore)

    def test_gantry_builds_nomic_embedder(self) -> None:
        """Test that AgentGantry can build a Nomic embedder."""
        from agent_gantry import AgentGantry
        from agent_gantry.adapters.embedders.nomic import NomicEmbedder
        from agent_gantry.schema.config import AgentGantryConfig, EmbedderConfig

        config = AgentGantryConfig(
            embedder=EmbedderConfig(
                type="nomic",
                model="nomic-ai/nomic-embed-text-v1.5",
                dimension=256,
            )
        )

        gantry = AgentGantry(config=config)
        assert isinstance(gantry._embedder, NomicEmbedder)
        assert gantry._embedder.dimension == 256
