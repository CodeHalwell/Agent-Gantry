# Publishing Agent-Gantry to PyPI

**Releases are made by the "Publish to PyPI" workflow, not by hand.** It builds and checks the
distribution, smoke-installs the wheel on Python 3.10-3.13, publishes with trusted publishing (no
long-lived token), and then tags the commit and creates the GitHub Release. The steps, the one-time
trusted-publisher setup and the TestPyPI option are in [RELEASING.md](RELEASING.md).

This page covers only the manual fallback, for the case where the workflow itself is unavailable.
Prefer the workflow: a manual upload creates **no tag and no GitHub Release**, and skips the
smoke-install.

## Manual fallback

1.  Check out the commit to release, with the version bumped everywhere `RELEASING.md` lists and a
    `CHANGELOG.md` entry for it.
2.  Run the checks the workflow would:
    ```bash
    uv run pytest
    uv run ruff check agent_gantry/
    ```
3.  Build and check the distribution:
    ```bash
    uv build
    uv tool run twine check dist/*
    ```
4.  Upload with an [API token](https://pypi.org/help/#apitoken). `uv publish` prompts for
    credentials; enter `__token__` as the username and the token (with its `pypi-` prefix) as the
    password, or set the token in the environment:
    ```bash
    export UV_PUBLISH_TOKEN="pypi-..."   # PowerShell: $env:UV_PUBLISH_TOKEN = "pypi-..."
    uv publish
    ```
5.  **Record the release**, which the workflow would have done for you:
    ```bash
    git tag -a "v<version>" -m "Release v<version>"
    git push origin "v<version>"
    gh release create "v<version>" dist/* --title "v<version>" --generate-notes
    ```
6.  Verify in a fresh environment:
    ```bash
    uv venv test-env && . test-env/bin/activate   # test-env\Scripts\activate on Windows
    uv pip install agent-gantry==<version>
    python -c "import agent_gantry; print(agent_gantry.__version__)"
    ```

## Troubleshooting

- **Build fails:** check `pyproject.toml` is valid and its dependencies resolve.
- **Authentication error:** check the token and that it is scoped to the project.
- **"File already exists":** PyPI never accepts the same version twice; bump it. (The workflow's
  `skip-existing` setting hides this as a successful no-op, so confirm the new version really
  appears on https://pypi.org/p/agent-gantry.)
