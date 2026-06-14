"""Keep the framework-integration examples green.

These run the offline example/verification scripts in-process and assert their
observable outcomes, so the examples can't silently rot as the library evolves.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.asyncio
async def test_verify_all_core_checks_pass() -> None:
    mod = importlib.import_module("examples.frameworks.verify_all")
    summary = await mod.run()
    assert summary["all_core_passed"], f"core checks failed: {summary['core_checks']}"
    assert not summary["adapters_failed"], "a framework adapter raised an unexpected error"
    # autogen needs no third-party framework, so at least one adapter always builds.
    assert summary["adapters_built"] >= 1


@pytest.mark.asyncio
async def test_universal_adapters_example_runs() -> None:
    mod = importlib.import_module("examples.frameworks.universal_adapters_example")
    # Should complete without raising (frameworks are skipped if absent).
    await mod.main()


@pytest.mark.asyncio
async def test_multi_turn_refresher_example_runs() -> None:
    mod = importlib.import_module("examples.frameworks.multi_turn_refresher_example")
    await mod.main()
