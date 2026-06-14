# Releasing Agent-Gantry

Releases are **automated**: merging to `main` with a bumped version builds the
package, publishes it to PyPI, tags the commit, and creates a GitHub Release —
no manual `uv publish` step. The workflow is `.github/workflows/release.yml`.

## How a release happens

1. Bump the version in **both** places (they must match — CI enforces it):
   - `pyproject.toml` → `[project] version`
   - `agent_gantry/__init__.py` → `__version__`
2. Add a `CHANGELOG.md` entry for the new version.
3. Open a PR and merge it to `main`.
4. On the push to `main`, `release.yml`:
   - checks the version and **only proceeds if the tag `v<version>` doesn't
     already exist** (so ordinary merges that don't bump the version are a no-op);
   - builds the sdist + wheel and runs `twine check`;
   - smoke-installs the wheel and imports it;
   - **publishes to PyPI** via trusted publishing (`skip-existing` keeps re-runs safe);
   - pushes the `v<version>` tag and creates a GitHub Release with generated notes.

Bumping the version is the single trigger — nothing else needs to be tagged or
clicked.

## One-time setup

Trusted publishing (no long-lived tokens) must be configured once:

1. **PyPI → Trusted Publishers** (https://docs.pypi.org/trusted-publishers/) for
   the `agent-gantry` project, with:
   - Owner / repository: `CodeHalwell/Agent-Gantry`
   - Workflow filename: `release.yml`
   - Environment: `pypi`
2. **GitHub → Settings → Environments → `pypi`**: this environment is referenced
   by the release job. Leave it without required reviewers for fully hands-off
   releases, or add required reviewers if you want a manual approval gate before
   each publish.

> The existing `publish.yml` (manual `workflow_dispatch` to TestPyPI/PyPI) is
> retained for ad-hoc/test publishes and is unaffected. If you prefer API tokens
> over trusted publishing, add a `PYPI_API_TOKEN` secret and set
> `password: ${{ secrets.PYPI_API_TOKEN }}` on the publish step in `release.yml`.

## Verifying

After the merge, watch the **Release** workflow in the Actions tab. On success
the new version appears on https://pypi.org/p/agent-gantry and a `v<version>`
GitHub Release is created.
