# Publishing Agent-Gantry to PyPI

This guide explains how to build and publish Agent-Gantry to PyPI using `uv`.

## Prerequisites

1.  **PyPI Account**: You need an account on [PyPI](https://pypi.org/).
2.  **API Token**: It is highly recommended to use an [API token](https://pypi.org/help/#apitoken) for publishing.
3.  **uv**: Ensure you have `uv` installed.

## Step 1: Prepare the Release

1.  **Update Version**: Ensure the version in `pyproject.toml` and `agent_gantry/__init__.py` is correct.
2.  **Update Changelog**: Ensure `CHANGELOG.md` has an entry for the new version with the correct date.
3.  **Run Tests**: Ensure all tests pass.
    ```bash
    uv run pytest
    ```
4.  **Check Linting**: Ensure code passes linting and type checks.
    ```bash
    uv run ruff check agent_gantry/
    uv run mypy agent_gantry/
    ```

## Step 2: Build the Distribution

Use `uv build` to create the source distribution and wheel.

```bash
uv build
```

This will create a `dist/` directory containing:
- `agent_gantry-<version>.tar.gz` (source distribution)
- `agent_gantry-<version>-py3-none-any.whl` (wheel)

## Step 3: Publish to PyPI

Use `uv publish` to upload the distribution files to PyPI.

```bash
uv publish
```

`uv` will prompt you for your PyPI credentials (username and password/token).

### Using an API Token (Recommended)

When prompted for a username, enter `__token__`.
When prompted for a password, enter your PyPI API token (including the `pypi-` prefix).

Alternatively, you can set environment variables:

```bash
# On Windows (PowerShell)
$env:UV_PUBLISH_TOKEN = "your-pypi-token"

# On Linux/macOS
export UV_PUBLISH_TOKEN="your-pypi-token"
```

Then run:

```bash
uv publish
```

## Step 4: Verify the Release

1.  Check the project page on PyPI: `https://pypi.org/project/agent-gantry/`
2.  Try installing the new version in a fresh environment:
    ```bash
    uv venv test-env
    . test-env/bin/activate  # or test-env\Scripts\activate on Windows
    uv pip install agent-gantry
    ```

## Troubleshooting

### Build Failures
If `uv build` fails, ensure your `pyproject.toml` is valid and all dependencies are correctly specified.

### Authentication Errors
If `uv publish` fails with authentication errors, double-check your API token and ensure it has the necessary permissions for the project.

### Version Conflicts
If you try to publish a version that already exists on PyPI, the upload will fail. You must increment the version number for every new release.
