# Agent-Gantry Copilot Instructions

## Project Overview

Agent-Gantry is a **Universal Tool Orchestration Platform** for LLM-based agent systems. It provides intelligent, secure tool orchestration with semantic routing, multi-protocol support, and zero-trust security.

**Core Philosophy**: *Context is precious. Execution is sacred. Trust is earned.*

### Key Problems We Solve

1. **Context Window Tax**: Reduce token costs by ~90% through semantic routing instead of sending 100+ tools in every prompt
2. **Tool/Protocol Fragmentation**: Write Once, Run Anywhere - support OpenAI, Claude, Gemini, A2A agents, and MCP clients
3. **Operational Blindness**: Zero-trust security with policies, capabilities, and circuit breakers

## Repository Structure

```
agent_gantry/
├── core/                 # Main facade, registry, router, executor
├── schema/               # Pydantic data models (tools, queries, events, config)
├── adapters/             # Protocol adapters
│   ├── vector_stores/    # Qdrant, Chroma, In-Memory, etc.
│   ├── embedders/        # OpenAI, SentenceTransformers, etc.
│   ├── rerankers/        # Cohere, CrossEncoder, etc.
│   └── executors/        # Direct, Sandbox, MCP, HTTP, A2A
├── providers/            # Tool import from various sources
├── servers/              # MCP and A2A server implementations
├── integrations/         # LangChain, AutoGen, LlamaIndex, CrewAI
├── observability/        # Telemetry, metrics, logging
└── cli/                  # Command-line interface

tests/                    # pytest-based test suite
```

## Development Workflow

### Setup

```bash
# Preferred: uv for reproducible environments
pip install uv
uv sync --extra dev

# Or use pip directly
pip install -e ".[dev]"

# Install all optional dependencies
pip install -e ".[all]"
```

### Testing

- **Framework**: pytest with async support (`pytest-asyncio`)
- **Command**: `pytest` to run all tests
- **Coverage**: `pytest --cov=agent_gantry` for coverage reports
- **Location**: All tests in `tests/` directory
- **Fixtures**: Defined in `tests/conftest.py`

**Important**: Always run existing tests before and after changes. We use fixtures like `gantry` and `sample_tools` for consistent test setup.

**Async Tests**: All core functionality is async. Always use `@pytest.mark.asyncio` decorator:
```python
@pytest.mark.asyncio
async def test_retrieve_tools_returns_relevant_results(gantry, sample_tools):
    tools = await gantry.retrieve_tools("calculate sum", limit=5)
    assert len(tools) > 0
```

### Linting and Code Quality

- **Linter**: `ruff` (configured in `pyproject.toml`)
- **Type Checker**: `mypy` with strict mode enabled
- **Line Length**: 100 characters max
- **Python Version**: Python 3.10+ required

Run before committing:
```bash
ruff check agent_gantry/
mypy agent_gantry/
```

**Auto-fix**: Use `ruff check --fix agent_gantry/` to automatically fix linting issues.

**Format**: Use `ruff format agent_gantry/` to format code.

### Building

```bash
# Build package
pip install build
python -m build
```

## Code Style and Conventions

### Python Style

1. **Type Hints**: Use type hints everywhere. We use strict mypy settings.
   ```python
   from typing import Any
   
   def my_function(param: str) -> dict[str, Any]:
       ...
   ```

2. **Async/Await**: Most core functionality is async. Use `async def` and `await` appropriately.
   ```python
   async def retrieve_tools(self, query: str) -> list[ToolDefinition]:
       ...
   ```

3. **Pydantic Models**: Use Pydantic v2 for all data models in `schema/`
   ```python
   from pydantic import BaseModel, Field
   
   class ToolDefinition(BaseModel):
       name: str
       description: str
       ...
   ```

4. **Docstrings**: Use clear, concise docstrings (Google style preferred)
   ```python
   """
   Brief description.
   
   Args:
        param: Description of parameter
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something goes wrong
   """
   ```

5. **Imports**: Follow ruff ordering (enforced by linter):
   - Standard library
   - Third-party packages
   - Local imports
   ```python
   from __future__ import annotations
   
   import asyncio
   from typing import Any
   
   from pydantic import BaseModel
   
   from agent_gantry.schema.tool import ToolDefinition
   ```

### Naming Conventions

- **Classes**: PascalCase (`AgentGantry`, `ToolDefinition`)
- **Functions/Methods**: snake_case (`retrieve_tools`, `execute_tool`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_LIMIT`, `MAX_RETRIES`)
- **Private**: Prefix with underscore (`_internal_method`)
5. **ContextVars**: Use `contextvars` for thread-safe and async-safe state management (see `core/context.py`)

### Tool Registration Pattern

Always use the `@gantry.register()` decorator with tags for semantic search:
```python
@gantry.register(tags=["weather", "api"], examples=["What's the weather in Tokyo?"])
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 72°F and sunny"
 (>80%)
2. **Async Tests**: Use `pytest-asyncio` for async test functions (auto mode enabled)
3. **Fixtures**: Leverage shared fixtures from `conftest.py`
4. **Test Structure**: Mirror the source structure in tests
5. **Test Naming**: Use descriptive names: `test_<function>_<scenario>_<expected>`

**Standard Test Pattern**:
```python
import pytest
from agent_gantry import AgentGantry

@pytest.mark.asyncio
async def test_retrieve_tools_returns_relevant_results(gantry, sample_tools):
    """Test that retrieve_tools returns semantically relevant tools."""
    # Setup
    await gantry.sync()
    
    # Execute
    tools = await gantry.retrieve_tools("calculate sum", limit=5)
    
    # Assert
    assert len(tools) > 0
    assert any("sum" in tool.name.lower() for tool in tools)
```

**Test Fixtures Available**:
- `gantry`: Fresh `AgentGantry` instance
- `sample_tools`: List of `ToolDefinition` objects for testing

**Running Tests**:
```bash
# All tests
pytest

# Specific file
pytest tests/test_tool.py

# Specific test
pytest tests/test_tool.py::TestToolDefinition::test_create_minimal_tool

# With coverage
pytest --cov=agent_gantry --cov-report=htmlurn await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        tools=tools  # Automatically injected
    )
```

### Architecture Patterns

1. **Adapters**: Follow the adapter pattern for extensibility (vector stores, embedders, executors)
2. **Schema-First**: Define Pydantic schemas before implementation
3. **Dependency Injection**: Pass dependencies explicitly rather than using globals
4. **Telemetry**: Emit events for important operations (tool retrieval, execution)

## Testing Requirements

1. **Coverage**: Aim for high test coverage on core functionality
2. **Async Tests**: Use `pytest-asyncio` for async test functions
3. **Fixtures**: Leverage shared fixtures from `conftest.py`
4. **Test Structure**: Mirror the source structure in tests
5. **Test Naming**: Use descriptive names: `test_<function>_<scenario>_<expected>`

Example:
```python
import pytest
from agent_gantry import AgentGantry

@pytest.mark.asyncio
async def test_retrieve_tools_returns_relevant_results(gantry, sample_tools):
    """Test that retrieve_tools returns semantically relevant tools."""
    # Test implementation
    ...
```

## Common Tasks

### Adding a New Tool Schema Field

1. Update the Pydantic model in `agent_gantry/schema/tool.py`
2. Add migration logic if needed (for backward compatibility)
3. Update tests in `tests/test_tool.py`
4. Update documentation

### Adding a New Adapter (e.g., Vector Store)

1. Create new file in appropriate `adapters/` subdirectory
2. Implement the adapter interface/protocol (see `adapters/vector_stores/base.py` for example)
3. Add tests in `tests/` (mirror existing adapter tests)
4. Update optional dependencies in `pyproject.toml` under `[project.optional-dependencies]`
5. Document usage in `adapters/{type}/README.md`

**Adapter Pattern Example**:
```python
# In adapters/vector_stores/my_store.py
from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter

class MyVectorStore(VectorStoreAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        # Initialize your store
        pass
    
    async def add(self, tools: list[ToolDefinition]) -> None:
        # Add tools to vector store
        pass
    
    async def search(self, query: str, limit: int = 10) -> list[ScoredTool]:
        # Perform semantic search
        pass
```

### Adding Framework Integration

1. Create new module in `integrations/`
2. Follow the pattern of wrapping AgentGantry for the framework
3. Add integration tests in `tests/test_framework_adapters.py` or new test file
4. Create example in `examples/agent_frameworks/`
5. Update `integrations/README.md` with usage example

### Adding LLM Provider Support

1. Add transcoding logic in `adapters/tool_spec/providers.py`
2. Update `with_semantic_tools` decorator in `integrations/semantic_tools.py`
3. Add tests in `tests/test_llm_sdk_compatibility.py`
4. Create example in `examples/llm_integration/`
5. Update `docs/llm_sdk_compatibility.md`

## Key Principles

1. **Minimal Changes**: Make small, focused changes
2. **Test First**: Write or update tests before implementation when possible
3. **Type Safety**: Always use type hints and pass mypy checks
4. **Async by Default**: Core operations should be async
5. **Backward Compatibility**: Don't break existing APIs without good reason
6. **Documentation**: Update docstrings and README for user-facing changes
7. **Security**: Be mindful of tool execution - this is a security-critical library

## Dependencies

- **Core**: Pydantic 2.0+
- **Optional**: OpenAI, sentence-transformers, qdrant-client, chromadb
- **Dev**: pytest, pytest-asyncio, pytest-cov, ruff, mypy

## Roadmap Context

The project follows a phased roadmap (see `plan.md`):
- **Phase 1**: Core Foundation (data models, basic routing)
- **Phase 2**: Robustness (execution engine, retries, circuit breakers)
- **Phase 3**: Context-Aware Routing (intent classification, diversity)
- **Phase 4**: Production Adapters (Qdrant, Chroma, OpenAI embeddings)
- **Phase 5**: MCP Integration
- **Phase 6**: A2A Integration
- **Phase 7**: Framework Integrations

When working on features, consider which phase they belong to and follow the architectural patterns established in earlier phases.

## Git Workflow

1. Work on feature branches
2. Write descriptive commit messages
3. Keep commits focused and atomic
4. Run tests and linting before committing
5. Update documentation as needed

## Security Considerations

- **Tool Execution**: Tools may execute arbitrary code - be careful with input validation
- **Capability Checks**: Respect `ToolCapability` and permission systems
- **Circuit Breakers**: Honor circuit breaker states for tool health
- **Secrets**: Never commit API keys or secrets to the repository
- **Input Validation**: Always validate tool arguments before execution

## Getting Help

- Check existing code patterns in the codebase
- Review tests for usage examples
- See `README.md` for high-level architecture
- See `plan.md` for detailed roadmap and design decisions
