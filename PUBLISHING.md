# Publishing Agent-Gantry

This guide describes how to build and publish `agent-gantry` to PyPI using `uv`.

## Prerequisites

- `uv` installed (see [installation guide](https://github.com/astral-sh/uv))
- Access to the PyPI project `agent-gantry` (or ability to create it)
- A PyPI API token (if publishing manually)

## Automated Publishing (Recommended)

The project is configured to automatically publish to PyPI when a GitHub Release is published.

1. Update the version in `pyproject.toml`:
   ```toml
   [project]
   version = "0.1.1"  # Update this
   ```

2. Commit and push the change:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.1.1"
   git push
   ```

3. Create a new Release on GitHub:
   - Tag version: `v0.1.1`
   - Title: `v0.1.1`
   - Description: Release notes...
   - Click "Publish release"

4. The GitHub Action `.github/workflows/publish.yml` will trigger, build the package using `uv build`, and publish it using `uv publish`.

### Trusted Publishing Configuration

To use the automated workflow without managing secrets, configure **Trusted Publishing** on PyPI:

1. Go to your project on PyPI > Settings > Publishing.
2. Add a new publisher:
   - Owner: `CodeHalwell`
   - Repository: `Agent-Gantry`
   - Workflow name: `publish.yml`
   - Environment: `pypi`

## Manual Publishing

You can also build and publish locally using `uv`.

1. **Build the package**:
   ```bash
   uv build
   ```
   This creates the distribution files in `dist/`.

2. **Publish to PyPI**:
   ```bash
   uv publish
   ```
   
   You will be prompted for your PyPI API token. You can also set it via environment variable:
   ```bash
   export UV_PUBLISH_TOKEN=pypi-AgEI...
   uv publish
   ```

## Testing the Build

To verify the build artifacts without publishing:

```bash
uv build
ls -l dist/
```

You should see a `.whl` file and a `.tar.gz` file.
