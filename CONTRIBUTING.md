# Contributing to Agent-Gantry

Thank you for your interest in contributing to Agent-Gantry! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Publishing](#publishing)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Agent-Gantry.git
   cd Agent-Gantry
   ```
3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Setup

```bash
# Install for development with all optional dependencies
pip install -e ".[all]"

# Or just dev dependencies
pip install -e ".[dev]"
```

### Making Changes

1. Make your changes in your feature branch
2. Add tests for new functionality
3. Ensure all tests pass
4. Run linters and type checkers
5. Update documentation as needed
6. Commit your changes with clear commit messages

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agent_gantry

# Run specific test file
pytest tests/test_tool.py

# Run tests in parallel (faster)
pytest -n auto
```

### Code Quality Checks

Before submitting a pull request, ensure your code passes all quality checks:

```bash
# Run linter
ruff check agent_gantry/

# Auto-fix linting issues
ruff check --fix agent_gantry/

# Run type checker
mypy agent_gantry/

# Format code (if you have ruff format)
ruff format agent_gantry/
```

## Code Style

### Python Style Guidelines

- **Python Version**: Python 3.10+ required
- **Line Length**: Maximum 100 characters
- **Type Hints**: Use type hints for all function signatures
- **Docstrings**: Use Google-style docstrings
- **Async/Await**: Use async/await for I/O operations

### Naming Conventions

- **Classes**: PascalCase (`AgentGantry`, `ToolDefinition`)
- **Functions/Methods**: snake_case (`retrieve_tools`, `execute_tool`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_LIMIT`, `MAX_RETRIES`)
- **Private**: Prefix with underscore (`_internal_method`)

### Example Code

```python
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class MyModel(BaseModel):
    """Brief description of the model.
    
    More detailed explanation if needed.
    
    Attributes:
        field_name: Description of the field
    """
    field_name: str


async def my_function(param: str) -> dict[str, Any]:
    """Brief description of the function.
    
    Args:
        param: Description of parameter
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: Description of when this is raised
    """
    # Implementation
    return {}
```

## Testing

### Test Structure

- Tests are located in the `tests/` directory
- Test files should mirror the source structure
- Use pytest fixtures from `conftest.py`
- Use descriptive test names: `test_<function>_<scenario>_<expected>`

### Writing Tests

```python
import pytest
from agent_gantry import AgentGantry


@pytest.mark.asyncio
async def test_retrieve_tools_returns_relevant_results(gantry, sample_tools):
    """Test that retrieve_tools returns semantically relevant tools."""
    # Given
    query = "calculate sum of numbers"
    
    # When
    tools = await gantry.retrieve_tools(query, limit=5)
    
    # Then
    assert len(tools) > 0
    assert any("sum" in tool.name.lower() for tool in tools)
```

### Test Coverage

- Aim for high test coverage on core functionality (>80%)
- All new features should include tests
- Bug fixes should include regression tests

## Documentation

### Code Documentation

- All public APIs must have docstrings
- Use Google-style docstrings
- Include type hints in signatures
- Document exceptions that can be raised
- Provide usage examples for complex features

### README and Docs

- Update README.md if adding user-facing features
- Add examples to `examples/` directory for new integrations
- Update relevant documentation in `docs/` directory
- Keep CHANGELOG.md updated with your changes

## Pull Request Process

### Before Submitting

1. ✅ All tests pass
2. ✅ Code passes ruff linting
3. ✅ Code passes mypy type checking
4. ✅ Documentation is updated
5. ✅ CHANGELOG.md is updated (if applicable)
6. ✅ Commit messages are clear and descriptive

### PR Guidelines

1. **Title**: Use a clear, descriptive title
   - Good: "Add support for Cohere embeddings"
   - Bad: "Update code"

2. **Description**: Include:
   - What changes were made
   - Why the changes were necessary
   - Any breaking changes
   - Related issues (if any)

3. **Small PRs**: Keep PRs focused and reasonably sized
   - Easier to review
   - Faster to merge
   - Lower risk of conflicts

4. **Tests**: Include tests for new functionality

5. **Documentation**: Update docs for user-facing changes

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added (if applicable)
- [ ] Linting passes
- [ ] Type checking passes

## Documentation
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Examples added/updated (if applicable)

## Related Issues
Closes #123
```

### Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your PR will be merged
4. Your contribution will be included in the next release

## Development Tips

### Useful Commands

```bash
# Install package in development mode
pip install -e ".[dev]"

# Run tests with output
pytest -v

# Run specific test
pytest tests/test_tool.py::TestToolDefinition::test_create_minimal_tool

# Check coverage
pytest --cov=agent_gantry --cov-report=html

# Run linter and auto-fix
ruff check --fix .

# Run type checker
mypy agent_gantry/
```

### Common Issues

**Import Errors**: Make sure you've installed the package in editable mode:
```bash
pip install -e ".[dev]"
```

**Test Failures**: Ensure you're using Python 3.10+:
```bash
python --version
```

**Type Errors**: Check that all dependencies are installed:
```bash
pip install -e ".[all]"
```

## Architecture Guidelines

### Adding New Features

1. **Adapters**: Follow the adapter pattern for extensibility
2. **Schema-First**: Define Pydantic schemas before implementation
3. **Dependency Injection**: Pass dependencies explicitly
4. **Telemetry**: Emit events for important operations
5. **Tests**: Write tests first (TDD) when possible

### Project Structure

```
agent_gantry/
├── core/                 # Main facade, registry, router, executor
├── schema/               # Pydantic data models
├── adapters/             # Protocol adapters
│   ├── vector_stores/
│   ├── embedders/
│   ├── rerankers/
│   └── executors/
├── providers/            # Tool import from various sources
├── servers/              # MCP and A2A server implementations
├── integrations/         # Framework integrations
├── observability/        # Telemetry, metrics, logging
└── cli/                  # Command-line interface
```

## Publishing

For instructions on how to build and publish releases to PyPI, please see [PUBLISHING.md](PUBLISHING.md).

## Questions?

- Check existing issues and pull requests
- Open a new issue for questions or discussions
- Tag issues appropriately (bug, feature, question, etc.)

## License

By contributing to Agent-Gantry, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🚀
