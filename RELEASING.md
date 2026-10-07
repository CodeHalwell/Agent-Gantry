# Releasing Agent-Gantry

Releases are published by **manually running the "Publish to PyPI" workflow**
(`.github/workflows/publish.yml`). The same run that publishes to PyPI also tags
the commit `v<version>` and creates the GitHub Release — so tagging is part of
the publish, not a separate step.

## How a release happens

1. Bump the version in **all four** places (they must match; `tests/test_version_consistency.py`
   fails if they do not):
   - `pyproject.toml` → `[project] version`
   - `agent_gantry/__init__.py` → `__version__`
   - `package.json` → `version` (the documentation site's "vX docs" label and footer read it)
   - `README.md` → the `**vX.Y.Z**` in the opening paragraph

   Then run `uv lock` and commit the result: the project's own entry in `uv.lock` carries the
   version too, so a bump without it leaves the lockfile stale.
2. Add a `CHANGELOG.md` entry for the new version.
3. Open a PR and merge it to `main` (CI must pass: `test`, `lint`,
   `Framework adapter smoke tests`, `Verify package builds`).
4. Go to **Actions → Publish to PyPI → Run workflow**, pick the `main` branch and
   set **target = `pypi`**, then run it. The workflow:
   - checks the version before building: `pyproject.toml` and `agent_gantry/__init__.py`
     must agree, and for a published GitHub Release the tag must be `v<version>`;
   - builds the sdist + wheel and runs `twine check`;
   - smoke-installs the wheel on Python 3.10–3.13 and imports it;
   - **publishes to PyPI** via trusted publishing (a version PyPI already has is skipped,
     with a warning in the run summary, so a re-run can still reach the tag and release);
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
the run publishes to PyPI only and skips the tag/release step. The tag must read
`v<version>` for the version in the tree, or the build step fails before anything is uploaded.

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
