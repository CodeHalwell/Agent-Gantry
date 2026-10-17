# Publishing Agent-Gantry

The GitHub [Publish to PyPI workflow](.github/workflows/publish.yml) is the
supported publishing path; local `uv publish` bypasses the release checks and
GitHub release creation and is not part of this process.
[RELEASING.md](RELEASING.md) is the release checklist and documents version
updates, changelog notes, validation, TestPyPI, and production publication.

The workflow uses PyPI trusted publishing through GitHub OIDC. It builds and
checks the distributions, smoke-installs the wheel on all supported Python
versions, and only then uploads. A production run also creates the matching
GitHub tag and Release with changelog notes and checksums.

If trusted publishing fails, check the PyPI publisher's repository, workflow and
environment names against [RELEASING.md](RELEASING.md), then retry the original
run. An emergency token-based upload must use the validated workflow artifacts
and checksums; it also requires completing the tag and GitHub release steps.

To inspect artifacts locally without publishing:

```bash
python scripts/check_release.py
uv build
uv tool run twine check dist/*
```

To publish after the release PR has merged and CI has passed, open
**Actions → Publish to PyPI → Run workflow**, select `main`, and choose
`testpypi` or `pypi`. TestPyPI is selected by default. A published GitHub Release
also triggers the workflow for that tag.
