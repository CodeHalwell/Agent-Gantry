# Releasing Agent-Gantry

Releases are published by **manually running the "Publish to PyPI" workflow**
(`.github/workflows/publish.yml`). The same run that publishes to PyPI also tags
the commit `v<version>` and creates the GitHub Release — so tagging is part of
the publish, not a separate step.

## How a release happens

1. Bump the version in **both** places (they must match):
   - `pyproject.toml` → `[project] version`
   - `agent_gantry/__init__.py` → `__version__`
2. Add a `CHANGELOG.md` entry for the new version.
3. Open a PR and merge it to `main` (CI must pass: `test`, `lint`,
   `Framework adapter smoke tests`, `Verify package builds`).
4. Go to **Actions → Publish to PyPI → Run workflow**, pick the `main` branch and
   set **target = `pypi`**, then run it. The workflow:
   - builds the sdist + wheel and runs `twine check`;
   - smoke-installs the wheel on Python 3.10–3.13 and imports it;
   - **publishes to PyPI** via trusted publishing;
   - **tags `v<version>`** (created only if missing) and **creates the GitHub
     Release** `v<version>` with generated notes.

Both the tag and the GitHub Release are created only when they don't already
exist, so re-running after a partial failure is safe.

### Test publishes (TestPyPI)

Run the same workflow with **target = `testpypi`** to publish to TestPyPI without
touching PyPI, tags, or releases (the tag/release step only runs for `pypi`).

### Publishing from an existing GitHub Release

The workflow also triggers automatically when a GitHub Release is *published*
(`on: release: [published]`). In that case the tag and release already exist, so
the run publishes to PyPI only and skips the tag/release step.

## One-time setup

Trusted publishing (no long-lived tokens) must be configured once:

1. **PyPI → Trusted Publishers** (https://docs.pypi.org/trusted-publishers/) for
   the `agent-gantry` project, with:
   - Owner / repository: `CodeHalwell/Agent-Gantry`
   - Workflow filename: `publish.yml`
   - Environment: `pypi`
2. **GitHub → Settings → Environments → `pypi`** (and `testpypi`): referenced by
   the publish jobs. Leave without required reviewers for hands-off publishes, or
   add required reviewers for a manual approval gate before each publish.

> If you prefer API tokens over trusted publishing, add a `PYPI_API_TOKEN` secret
> and set `password: ${{ secrets.PYPI_API_TOKEN }}` on the publish step.

## Verifying

After the run, the new version appears on https://pypi.org/p/agent-gantry and a
`v<version>` GitHub Release is created. The workflow `twine check`s and
smoke-installs the wheel before publishing as a final safety net.
