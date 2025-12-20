# Local persistence, Matryoshka embeddings, and skills

This guide covers the on-device stack for Agent-Gantry: LanceDB for persistent vector storage,
Nomic Matryoshka embeddings for efficient local retrieval, and the `Skill` schema for storing
how-tos and patterns alongside tools.

## Install the local extras

```bash
pip install agent-gantry[lancedb,nomic]
```

This pulls in:
- `lancedb` and `pyarrow` for local persistence
- `sentence-transformers` for Nomic `nomic-embed-text-v1.5` embeddings

## Configure Agent-Gantry to use LanceDB + Nomic

```python
import asyncio
from agent_gantry import AgentGantry
from agent_gantry.schema.config import AgentGantryConfig, VectorStoreConfig, EmbedderConfig

config = AgentGantryConfig(
    vector_store=VectorStoreConfig(
        type="lancedb",
        db_path=".agent_gantry/lancedb",
        dimension=256,
    ),
    embedder=EmbedderConfig(
        type="nomic",
        model="nomic-ai/nomic-embed-text-v1.5",
        dimension=256,  # Matryoshka truncation (64/128/256/512/768 supported)
        task_type="search_document",
    ),
)

gantry = AgentGantry(config=config)

@gantry.register(tags=["finance"])
def compute_tax(amount: float) -> float:
    """Compute sales tax for a given amount."""
    return amount * 0.08

asyncio.run(gantry.sync())
```

LanceDB stores embeddings on disk under `.agent_gantry/lancedb` by default and reuses the same
database across runs. The configured `dimension` must match both your embedder output and the
LanceDB table dimension.

## Working with skills

`agent_gantry.schema.skill.Skill` models reusable procedural knowledge that you can retrieve and
inject into prompts. LanceDB provides first-class support for storing and searching skills.

```python
import asyncio
from agent_gantry.adapters.embedders.nomic import NomicEmbedder
from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore
from agent_gantry.schema.skill import Skill, SkillCategory

# Initialize embedder and store with matching dimensions
embedder = NomicEmbedder(dimension=256)
store = LanceDBVectorStore(db_path=".agent_gantry/lancedb", dimension=256)
await store.initialize()

# Define skills
skills = [
    Skill(
        name="api_pagination",
        description="How to implement cursor-based pagination for REST APIs",
        content="Use a stable cursor, return next_cursor and has_more.",
        category=SkillCategory.HOW_TO,
        tags=["api", "pagination", "rest"],
        related_tools=["query_database", "fetch_api"],
    ),
    Skill(
        name="refund_safety",
        description="Checklist before executing a refund",
        content="Verify user identity, amount, currency, and approval flag.",
        category=SkillCategory.PROCEDURE,
        tags=["finance", "risk"],
    ),
]

# Embed and add skills
skill_embeddings = [
    await embedder.embed_text(skill.to_embedding_text()) for skill in skills
]
await store.add_skills(skills, skill_embeddings)

# Retrieve skills for a query
query_vector = await embedder.embed_text("handle cursor pagination safely")
results = await store.search_skills(query_vector, limit=3)

for skill, score in results:
    print(f"{skill.name} ({score:.2f})")
    print(skill.to_prompt_text())
```

### Using skills in prompts

- `Skill.to_prompt_text()` formats a skill as markdown for system prompts.
- `Skill.to_embedding_text()` flattens metadata for consistent embeddings.
- `Skill.content_hash` helps detect changes when deciding whether to re-embed.

You can compose the top results into a system prompt section:

```python
prompt_blocks = [skill.to_prompt_text() for skill, _ in results[:2]]
system_prompt = "\n\n".join(["# Relevant Knowledge"] + prompt_blocks)
```

## Tips and constraints

- Keep skill names and namespaces stable; LanceDB uses `namespace.name` as the primary key.
- Ensure the embedding `dimension` matches both LanceDB and the embedder configuration.
- Filters on `namespace` and `category` are supported in `search_skills`; tags are not yet indexed.
- `Skill` retrieval is currently a lower-level API; the main `AgentGantry` facade handles tools,
  while LanceDB exposes skills for prompt injection workflows.
