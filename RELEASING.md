# Releasing Agent-Gantry

The [Publish to PyPI workflow](.github/workflows/publish.yml) builds one source
archive and wheel, validates them, publishes with PyPI trusted publishing, and
attaches the distributions and SHA-256 checksums to a GitHub Release.

## Prepare the release

1. Update `pyproject.toml`, `agent_gantry/__init__.py`, the editable project
   version in `uv.lock`, `package.json`, the root versions in `package-lock.json`,
   and the version in `README.md`.
2. Move the accumulated `CHANGELOG.md` notes into a dated release entry, keeping
   an empty `Unreleased` section for subsequent work. Include compatibility and
   migration notes, especially for pre-1.0 minor releases.
3. Check the metadata and review the exact notes that GitHub will receive:

   ```bash
   python scripts/check_release.py --tag v0.20.0 --notes-file /tmp/release-notes.md
   uv lock --check
   ```

4. Run the relevant tests, including `tests/examples/` and
   `tests/frameworks/test_real_packages.py` with the optional SDKs installed.
   Check `ruff check agent_gantry examples scripts tests`, `uv build`,
   `uv tool run twine check dist/*`, and `npm run build` for the docs.
5. Merge the release PR to `main` after CI passes, including the native SDK
   matrix and package build job. Preparing a release does not publish it.

The metadata check runs in CI and again before the publish build. It rejects
version drift, a mismatched GitHub Release tag, missing dated changelog notes,
and an existing version tag that points to a different source commit.

## Publish from GitHub Actions

Open **Actions → Publish to PyPI → Run workflow**, choose `main`, and choose a
publish target:

- `testpypi` (the default) publishes to TestPyPI only. It creates no tag or GitHub Release.
- `pypi` tags the checked-out source `v<version>`, publishes to PyPI, and creates
  the GitHub Release using the dated changelog entry. The release includes both
  distributions and `SHA256SUMS`.

Manual production runs must select `main`; other refs fail before building.
The source tag is created after wheel checks and before the PyPI upload. If an
upload fails, retry the original workflow to publish the same tagged source.

Before either upload, the workflow checks distribution metadata and installs
the built wheel on Python 3.10, 3.11, 3.12, 3.13 and 3.14. Those clean installations
verify the version, typing marker, tool selection and execution, and lazy
construction of all twelve framework adapters without their SDKs.

Publishing an existing GitHub Release also triggers this workflow. Its tag
must match the package version. The workflow publishes that tagged source to
PyPI and attaches the artifacts to the existing release.

Runs are serialized. A retry skips distributions already on the package index
and repairs missing GitHub release assets; it never moves an existing tag.
Uploading to an existing release replaces same-named distribution and checksum
assets, including files attached by hand; other assets are left alone. Re-run
the original failed workflow to retain its source commit. Starting a new run
from an advanced branch for an already-tagged version is rejected; use the
original workflow or prepare a new version. Release-triggered runs build their
published tag rather than `main`.

## One-time setup

Configure a [PyPI trusted publisher](https://docs.pypi.org/trusted-publishers/)
for `CodeHalwell/Agent-Gantry`, workflow `publish.yml`, environment `pypi`.
Configure the corresponding TestPyPI publisher for environment `testpypi`.
Create both environments in **GitHub → Settings → Environments**; any configured
required reviewers must approve that environment before publication.

No long-lived PyPI token is required. Only the publish jobs receive OIDC
permission; only the production job can write tags and release assets.

## Verify the published version

Check [PyPI](https://pypi.org/project/agent-gantry/) and
[GitHub Releases](https://github.com/CodeHalwell/Agent-Gantry/releases). Download
`SHA256SUMS` and the artifacts to verify their hashes, then install the version
in a fresh environment:

```bash
uv venv /tmp/gantry-release-check
uv pip install --python /tmp/gantry-release-check/bin/python agent-gantry==0.20.0
/tmp/gantry-release-check/bin/python -I -c 'import agent_gantry; print(agent_gantry.__version__)'
```
