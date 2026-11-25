"""Cross-framework conformance matrix for the native tool adapters.

Every ``<Framework>Adapter`` in ``dotfetch.integrations.frameworks`` must
honor the same contract regardless of which third-party framework it targets:

1. It exposes ``select`` (async select+convert) and ``convert`` (single convert
   staticmethod), both importable without the framework installed.
2. ``select`` runs semantic selection against a real fetcher and returns one
   converted tool per selected tool — verified end-to-end, per framework,
   through a lightweight ``sys.modules`` stub (mirroring how the real package
   would be structured) so the check runs everywhere without needing every
   framework installed.
3. ``convert`` raises a clean ``ImportError`` carrying a ``pip install`` hint
   when the framework is absent — not ``AttributeError`` / ``KeyError``.

This locks the uniform surface so a new adapter can't silently drift.

``AgentFrameworkAdapter`` shares the selection base and also exposes provider,
bridge and middleware factories. Its specialized AF factory tests remain below;
the uniform select/convert surface is additionally covered by
``test_agent_framework_adapter.py`` and the twelve-framework real-SDK matrix.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from dotfetch import DotFetch
from dotfetch.adapters.embedders.simple import SimpleEmbedder
from dotfetch.agent_framework import AgentFrameworkAdapter
from dotfetch.core.security import SecurityPolicy
from dotfetch.integrations import frameworks as F  # noqa: N812


@pytest.fixture
async def fetcher():
    g = DotFetch(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["email"])
    def send_email(to: str, body: str = "") -> str:
        "Send an email message to a recipient."
        return f"sent:{to}"

    @g.register(tags=["math"])
    def add(a: int, b: int) -> int:
        "Add two integers together."
        return a + b

    await g.sync()
    return g


@pytest.fixture
async def fetcher_with_failing_tool(fetcher: DotFetch) -> DotFetch:
    """The shared conformance ``fetcher`` plus one tool that always raises.

    Shared by the tool-failure conformance test (below) and reused verbatim
    by the pattern already proven in ``tests/frameworks/test_dspy.py`` /
    ``test_strands.py``. A separate fixture (rather than extending ``fetcher``
    itself) keeps this tool out of every other conformance test's tool count.
    """

    @fetcher.register(tags=["danger"])
    def explode() -> str:
        "Always raises when invoked."
        raise RuntimeError("boom: this tool always fails")

    await fetcher.sync()
    return fetcher


# --------------------------------------------------------------------------- #
# Per-framework stub factories for the end-to-end select→convert smoke below.
#
# Each factory returns a fresh ``{attr_name: value}`` mapping installed onto
# the *leaf* module of the adapter's import chain (see ``_install_stub``). The
# stub is intentionally permissive — just enough structure for the adapter's
# ``convert`` to succeed — real-API drift (Pydantic v2 model fields, signature
# validation, …) is the job of ``test_real_packages.py`` against the actual
# installed package.
# --------------------------------------------------------------------------- #


def _echo(**kwargs: Any) -> Any:
    """Return the kwargs as an object, standing in for a real constructor."""
    return types.SimpleNamespace(**kwargs)


class _StubToolException(Exception):  # noqa: N818  (mirrors the real name)
    """Stand-in for ``langchain_core.tools.ToolException``."""


def _stub_langchain_attrs() -> dict[str, object]:
    structured_tool = types.SimpleNamespace()
    structured_tool.from_function = staticmethod(_echo)
    return {"StructuredTool": structured_tool, "ToolException": _StubToolException}


def _stub_llamaindex_attrs() -> dict[str, object]:
    function_tool = types.SimpleNamespace()
    function_tool.from_defaults = staticmethod(_echo)
    return {"FunctionTool": function_tool}


class _StubCrewAIBaseTool:
    """Plain stand-in for ``crewai.tools.BaseTool`` (a real Pydantic v2 model).

    ``_spec_to_crewai`` dynamically subclasses ``BaseTool`` and instantiates it
    with ``name``/``description``/``args_schema`` kwargs; a plain class that
    stores whatever kwargs it's given is sufficient to exercise that shape
    without pulling in CrewAI's Pydantic machinery.
    """

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _stub_crewai_attrs() -> dict[str, object]:
    return {"BaseTool": _StubCrewAIBaseTool}


class _StubPydanticAITool:
    """Stand-in for ``pydantic_ai.tools.Tool`` exposing the preferred ``from_schema`` path."""

    @classmethod
    def from_schema(cls, **kwargs: Any) -> Any:
        return types.SimpleNamespace(**kwargs)


def _stub_pydantic_ai_attrs() -> dict[str, object]:
    return {"Tool": _StubPydanticAITool}


class _StubModelRetryError(Exception):
    """Stand-in for ``pydantic_ai.exceptions.ModelRetry``."""


class _StubToolFailedError(Exception):
    """Stand-in for ``pydantic_ai.exceptions.ToolFailed``."""


class _StubApprovalRequiredError(Exception):
    """Stand-in for ``pydantic_ai.exceptions.ApprovalRequired``."""

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        self.metadata = metadata
        super().__init__()


def _stub_pydantic_ai_extra_modules() -> dict[str, dict[str, object]]:
    return {
        "pydantic_ai.exceptions": {
            "ModelRetry": _StubModelRetryError,
            "ToolFailed": _StubToolFailedError,
            "ApprovalRequired": _StubApprovalRequiredError,
        }
    }


def _stub_openai_agents_attrs() -> dict[str, object]:
    return {"FunctionTool": _echo}


def _stub_haystack_attrs() -> dict[str, object]:
    return {"Tool": _echo}


def _stub_agno_attrs() -> dict[str, object]:
    return {"Function": _echo}


def _stub_google_adk_attrs() -> dict[str, object]:
    return {"FunctionTool": _echo}


def _stub_strands_attrs() -> dict[str, object]:
    def _tool(**kwargs: Any):
        def _decorator(fn: Any) -> Any:
            return types.SimpleNamespace(fn=fn, **kwargs)

        return _decorator

    return {"tool": _tool}


def _stub_dspy_attrs() -> dict[str, object]:
    class _StubDSPyTool:
        def __init__(
            self, func, name=None, desc=None, args=None, arg_types=None, arg_desc=None
        ):
            self.func, self.name, self.desc = func, name, desc
            self.args, self.arg_types, self.arg_desc = args, arg_types, arg_desc

    def _convert(schema):
        props = (schema or {}).get("properties") or {}
        return dict(props), {k: "Any" for k in props}, {k: "" for k in props}

    return {"Tool": _StubDSPyTool, "convert_input_schema_to_tool_args": _convert}


@dataclass(frozen=True)
class AdapterCase:
    """One row of the cross-framework conformance matrix.

    Attributes:
        name: Short framework key, also the pytest parametrize id.
        adapter_cls: The ``<Framework>Adapter`` class under test.
        modules: ``sys.modules`` entries ``convert`` needs to import the
            native framework. Stubbing these to ``None`` simulates "framework
            not installed"; installing real stub modules there (via
            ``stub_attrs``) simulates "framework installed" cheaply.
        live_tier: The adapter's documented ``live_tier`` — ``"per-turn"`` or
            ``"per-run"``. Locked against ``<Framework>Adapter.live_tier``
            and cross-referenced with ``integrations/frameworks/README.md``'s
            table (see ``test_adapter_live_tier_matches_capability_table``).
        live_delegate: Name of the bespoke method ``adapter.live()`` delegates
            to (e.g. ``"react_agent"``, ``"toolset"``, ``"agent_builder"``).
            ``"select"`` is a sentinel for LangChain, whose ``live()`` returns
            a *bound alias* of ``select`` rather than delegating to a
            separate bespoke method — it is exercised by its own dedicated
            test, not the generic delegation-smoke test.
        stub_attrs: Zero-arg factory building the ``{attr: value}`` mapping
            installed on the leaf module in ``modules`` for the end-to-end
            select→convert smoke.
        live_extra_kwargs: Zero-arg factory building the extra
            ``framework_kwargs`` a call to ``adapter.live()`` requires (e.g.
            ``{"model": ...}`` for LangGraph, whose native hook is bound to a
            specific chat model). ``None`` when ``live()`` needs nothing
            beyond ``limit``/``score_threshold``/``namespaces``.
        error_kind: How a failing ``fetcher.execute()`` surfaces through the
            native tool object this adapter's ``convert()`` produces.
            ``"raises"`` (every adapter, currently) — the native object's own
            call convention (``.func``/``._run``/``.forward``/``.method``/…)
            propagates :class:`~dotfetch.integrations.frameworks.base.ToolExecutionError`
            uncaught, matching the documented default contract (see
            "Error-handling policy" in ``integrations/frameworks/README.md``).
            ``"native_tool_exception"`` (LangChain, LangGraph) -- the failure
            is re-raised as LangChain's ``ToolException`` chained from the
            :class:`~dotfetch.integrations.frameworks.base.ToolExecutionError`;
            the tool sets ``handle_tool_error=True``, so the model receives an
            error ``ToolMessage`` instead of the run ending.
            ``error_mode="raise"`` restores ``"raises"``.
            ``"native_tool_result"`` (Strands) -- the tool returns an error
            ``ToolResult`` (``{"status": "error", "content": [...]}``);
            ``error_mode="raise"`` restores ``"raises"``.
            ``"native_tool_output"`` (OpenAI Agents SDK; CrewAI, whose real
            ``_run`` returns a ``ToolFailure`` and falls back to ``Error: ...``
            text when that class is missing, as with this matrix's stub) -- the failure is
            returned as the tool's output text (``Error: ...``) for the model
            to read, since the SDK has no retry exception and would otherwise
            end the run with ``UserError``; ``error_mode="raise"`` restores
            ``"raises"``.
            ``"native_tool_failed"`` (Pydantic AI) -- the failure is
            re-raised as the framework's own ``ToolFailed`` (``ModelRetry``
            for argument errors, ``ApprovalRequired`` for confirmation-gated
            calls), chained from the
            :class:`~dotfetch.integrations.frameworks.base.ToolExecutionError`,
            so the model sees it instead of the run ending; ``error_mode="raise"``
            restores ``"raises"``.
            The three deliberate "framework absorbs the error" deviations
            (Microsoft Agent Framework's JSON error string, the *live*
            ``Workbench.call_tool``, Strands' real ``Agent`` tool-execution
            loop) live one layer deeper than ``convert()``/``select()`` — see
            their own dedicated tests, not this matrix.
        invoke_failure: Async callable that takes the native object returned
            by ``adapter_cls.convert(spec)`` and invokes it the way that
            framework's own runtime would (its ``.func``/``._run``/
            ``.forward``/``.on_invoke_tool``/… entry point), awaiting through
            if that entry point is itself a coroutine function. Used by
            :func:`test_adapter_tool_failure_matches_documented_error_kind`
            to prove ``error_kind`` end-to-end for every adapter, including
            the sync wrappers (CrewAI/Agno/Haystack/DSPy/
            LangChain/LlamaIndex/Google ADK), which this exercises through
            the real ``_run_coroutine_sync`` worker-thread bridge since the
            test itself runs inside a running event loop.
    """

    name: str
    adapter_cls: type
    modules: list[str]
    live_tier: str
    live_delegate: str
    invoke_failure: Callable[[Any], Any]
    stub_attrs: Callable[[], dict[str, object]] | None = None
    live_extra_kwargs: Callable[[], dict[str, object]] | None = None
    error_kind: str = "raises"
    extra_stub_modules: Callable[[], dict[str, dict[str, object]]] | None = None


# --------------------------------------------------------------------------- #
# Per-framework "invoke the native tool object the way the real framework
# would" callables, used by the tool-failure conformance test below. Each
# takes the object `adapter_cls.convert(spec)` returns and calls through its
# real invocation entry point — the same attribute a genuine LangChain
# StructuredTool / CrewAI BaseTool / etc. would call. All are ``async`` so the
# test can ``await`` uniformly regardless of whether the underlying call is
# itself a coroutine function (openai_agents, pydantic_ai, google_adk,
# strands) or a synchronous bridge through
# ``ToolSpec.invoke`` (langchain, langgraph, llamaindex, crewai,
# haystack, agno, dspy) — the latter exercises the `_run_coroutine_sync`
# worker-thread bridge since these tests run inside a running event loop.
# --------------------------------------------------------------------------- #


async def _invoke_langchain_failure(native: Any) -> Any:
    return native.func()


async def _invoke_llamaindex_failure(native: Any) -> Any:
    return native.fn()


async def _invoke_crewai_failure(native: Any) -> Any:
    return native._run()


async def _invoke_pydantic_ai_failure(native: Any) -> Any:
    # ``Tool.from_schema(takes_ctx=True)``: Pydantic AI passes the RunContext
    # positionally. ``None`` stands in for a context-free direct call.
    return await native.function(None)


async def _invoke_openai_agents_failure(native: Any) -> Any:
    return await native.on_invoke_tool(None, "{}")



async def _invoke_haystack_failure(native: Any) -> Any:
    return native.function()


async def _invoke_agno_failure(native: Any) -> Any:
    return native.entrypoint()



async def _invoke_google_adk_failure(native: Any) -> Any:
    return await native.func()



async def _invoke_strands_failure(native: Any) -> Any:
    return await native.fn()


async def _invoke_dspy_failure(native: Any) -> Any:
    return native.func()


ADAPTERS: list[AdapterCase] = [
    AdapterCase(
        "langchain",
        F.LangChainAdapter,
        ["langchain_core", "langchain_core.tools"],
        live_tier="per-run",
        live_delegate="select",
        stub_attrs=_stub_langchain_attrs,
        invoke_failure=_invoke_langchain_failure,
        error_kind="native_tool_exception",
    ),
    AdapterCase(
        "langgraph",
        F.LangGraphAdapter,
        ["langchain_core", "langchain_core.tools"],
        live_tier="per-turn",
        live_delegate="react_agent",
        stub_attrs=_stub_langchain_attrs,
        live_extra_kwargs=lambda: {"model": object()},
        invoke_failure=_invoke_langchain_failure,
        error_kind="native_tool_exception",
    ),
    AdapterCase(
        "llamaindex",
        F.LlamaIndexAdapter,
        ["llama_index", "llama_index.core", "llama_index.core.tools"],
        live_tier="per-turn",
        live_delegate="tool_retriever",
        stub_attrs=_stub_llamaindex_attrs,
        invoke_failure=_invoke_llamaindex_failure,
    ),
    AdapterCase(
        "crewai",
        F.CrewAIAdapter,
        ["crewai", "crewai.tools"],
        live_tier="per-run",
        live_delegate="agent_builder",
        stub_attrs=_stub_crewai_attrs,
        invoke_failure=_invoke_crewai_failure,
        error_kind="native_tool_output",
    ),
    AdapterCase(
        "pydantic_ai",
        F.PydanticAIAdapter,
        ["pydantic_ai", "pydantic_ai.tools"],
        live_tier="per-turn",
        live_delegate="toolset",
        stub_attrs=_stub_pydantic_ai_attrs,
        invoke_failure=_invoke_pydantic_ai_failure,
        error_kind="native_tool_failed",
        extra_stub_modules=_stub_pydantic_ai_extra_modules,
    ),
    AdapterCase(
        "openai_agents",
        F.OpenAIAgentsAdapter,
        ["agents"],
        live_tier="per-turn",
        live_delegate="session",
        stub_attrs=_stub_openai_agents_attrs,
        live_extra_kwargs=lambda: {"agent": object()},
        invoke_failure=_invoke_openai_agents_failure,
        error_kind="native_tool_output",
    ),
    AdapterCase(
        "haystack",
        F.HaystackAdapter,
        ["haystack", "haystack.tools"],
        live_tier="per-run",
        live_delegate="tool_invoker_builder",
        stub_attrs=_stub_haystack_attrs,
        invoke_failure=_invoke_haystack_failure,
    ),
    AdapterCase(
        "agno",
        F.AgnoAdapter,
        ["agno", "agno.tools", "agno.tools.function"],
        live_tier="per-run",
        live_delegate="agent_builder",
        stub_attrs=_stub_agno_attrs,
        invoke_failure=_invoke_agno_failure,
    ),
    AdapterCase(
        "google_adk",
        F.GoogleADKAdapter,
        ["google.adk", "google.adk.tools"],
        live_tier="per-turn",
        live_delegate="before_model_callback",
        stub_attrs=_stub_google_adk_attrs,
        invoke_failure=_invoke_google_adk_failure,
    ),
    AdapterCase(
        "strands",
        F.StrandsAdapter,
        ["strands"],
        live_tier="per-turn",
        live_delegate="tool_hook",
        stub_attrs=_stub_strands_attrs,
        invoke_failure=_invoke_strands_failure,
        error_kind="native_tool_result",
    ),
    AdapterCase(
        "dspy",
        F.DSPyAdapter,
        ["dspy", "dspy.adapters", "dspy.adapters.types", "dspy.adapters.types.tool"],
        live_tier="per-run",
        live_delegate="agent_builder",
        live_extra_kwargs=lambda: {"signature": "question -> answer"},
        stub_attrs=_stub_dspy_attrs,
        invoke_failure=_invoke_dspy_failure,
    ),
]


@pytest.mark.parametrize("case", ADAPTERS, ids=[c.name for c in ADAPTERS])
def test_adapter_exposes_uniform_surface(case: AdapterCase) -> None:
    assert callable(getattr(case.adapter_cls, "select", None)), (
        f"{case.name}: {case.adapter_cls.__name__}.select missing"
    )
    assert callable(getattr(case.adapter_cls, "convert", None)), (
        f"{case.name}: {case.adapter_cls.__name__}.convert missing"
    )


# --------------------------------------------------------------------------- #
# Uniform `live_tier` / `live()` entry point (BaseFrameworkAdapter)
#
# Every adapter's dynamic ("live") re-selection tier is named differently per
# framework (`react_agent`, `toolset`, `tool_hook`, `function_provider`,
# `agent_builder`, …). `live_tier` and `live()` give callers a single,
# framework-agnostic way to discover how deep an adapter's dynamic tier goes
# and get the live object for it, without knowing the bespoke method name.
# These checks lock that uniform surface the same way the matrix above locks
# select/convert; the bespoke methods themselves stay untouched and are still
# the documented framework-idiomatic path.
# --------------------------------------------------------------------------- #


def test_per_run_tier_still_equals_its_old_spelling() -> None:
    """``"per-run"`` was ``"per-call"`` before 0.24.0, when
    ``query_strategy="per_call"`` named the opposite granularity. Code that
    branches on the old value keeps working and is told what to write."""
    import copy
    import json
    import pickle
    import warnings

    from dotfetch.integrations.frameworks.base import PER_RUN_TIER

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the new spelling never warns
        assert PER_RUN_TIER == "per-run" and "per-run" == PER_RUN_TIER
        assert not (PER_RUN_TIER != "per-run") and PER_RUN_TIER != "per-turn"
        assert PER_RUN_TIER in ("per-turn", "per-run")
        assert str(PER_RUN_TIER) == "per-run" and repr(PER_RUN_TIER) == "'per-run'"
        assert json.dumps(PER_RUN_TIER) == '"per-run"'
        assert {"per-run": 1}[PER_RUN_TIER] == 1
        assert PER_RUN_TIER != 3 and PER_RUN_TIER is not None
        assert type(copy.deepcopy(PER_RUN_TIER)) is str
        assert pickle.loads(pickle.dumps(PER_RUN_TIER)) == "per-run"

    for compare in (
        lambda: PER_RUN_TIER == "per-call",
        lambda: "per-call" == PER_RUN_TIER,
        lambda: not (PER_RUN_TIER != "per-call"),
        lambda: PER_RUN_TIER in ("per-turn", "per-call"),
    ):
        with pytest.warns(DeprecationWarning, match="'per-call' is now 'per-run'"):
            assert compare()


def test_the_old_spelling_warning_points_at_the_comparison() -> None:
    """Every comparison form attributes its DeprecationWarning to the caller's
    line. ``!=`` used to go through ``__eq__`` and land one frame short, inside
    the library, where the default filter also hides it."""
    import warnings

    from dotfetch.integrations.frameworks.base import PER_RUN_TIER

    forms = {
        "==": lambda: PER_RUN_TIER == "per-call",
        "reflected ==": lambda: "per-call" == PER_RUN_TIER,
        "!=": lambda: PER_RUN_TIER != "per-call",
        "in": lambda: PER_RUN_TIER in ("per-turn", "per-call"),
    }
    for form, compare in forms.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compare()
        assert [w.category for w in caught] == [DeprecationWarning], form
        assert caught[0].filename == __file__, f"{form}: attributed to {caught[0].filename}"


@pytest.mark.parametrize("case", ADAPTERS, ids=[c.name for c in ADAPTERS])
def test_adapter_tier_still_matches_the_old_per_call_spelling(case: AdapterCase) -> None:
    if case.live_tier != "per-run":
        pytest.skip("only the renamed tier has an old spelling")
    with pytest.warns(DeprecationWarning):
        assert case.adapter_cls.live_tier == "per-call"


@pytest.mark.parametrize("case", ADAPTERS, ids=[c.name for c in ADAPTERS])
def test_adapter_exposes_live_tier_and_live(case: AdapterCase) -> None:
    """Every adapter declares a valid ``live_tier`` and a callable ``live``."""
    assert getattr(case.adapter_cls, "live_tier", None) in ("per-turn", "per-run"), (
        f"{case.name}: {case.adapter_cls.__name__}.live_tier missing or invalid "
        f"(got {getattr(case.adapter_cls, 'live_tier', None)!r})"
    )
    assert callable(getattr(case.adapter_cls, "live", None)), (
        f"{case.name}: {case.adapter_cls.__name__}.live missing"
    )


@pytest.mark.parametrize("case", ADAPTERS, ids=[c.name for c in ADAPTERS])
def test_adapter_live_tier_matches_capability_table(case: AdapterCase) -> None:
    """``live_tier`` matches the documented per-framework capability table.

    Mirrors ``integrations/frameworks/README.md``'s uniform-tier table and the
    audit findings in the task brief: LangGraph, LlamaIndex, Pydantic AI,
    OpenAI Agents SDK, Google ADK, and Strands
    genuinely re-select tools on every model turn (``"per-turn"``); LangChain,
    CrewAI, Agno, and Haystack fix their tool list at agent
    construction with no native mid-run hook, so the deepest dotfetch can do is
    rebuild before each new top-level call (``"per-run"``).
    """
    assert case.adapter_cls.live_tier == case.live_tier, (
        f"{case.name}: expected live_tier={case.live_tier!r}, got {case.adapter_cls.live_tier!r}"
    )


@pytest.mark.parametrize(
    "case", [c for c in ADAPTERS if c.live_delegate != "select"], ids=lambda c: c.name
)
def test_adapter_live_delegates_to_bespoke_method(case: AdapterCase, monkeypatch) -> None:
    """``adapter.live(...)`` calls the documented bespoke method with the same kwargs.

    Cheaply proves delegation via a stub: monkeypatches the bespoke method
    (``case.live_delegate``) on the adapter class with a recorder, calls
    ``live()`` with distinctive ``limit``/``score_threshold``/``namespaces``,
    and asserts the recorder saw them and that ``live()`` returned the
    recorder's sentinel unchanged. No real fetcher or framework install is
    needed — this only proves the wiring, not the bespoke method's own
    behaviour (that's covered by each framework's dedicated live tests).
    """
    sentinel = object()
    calls: list[tuple[tuple, dict]] = []

    def _recorder(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(case.adapter_cls, case.live_delegate, _recorder)

    adapter = case.adapter_cls(fetcher=None)  # type: ignore[arg-type]
    extra = case.live_extra_kwargs() if case.live_extra_kwargs is not None else {}
    result = adapter.live(limit=7, score_threshold=0.3, namespaces=["ns1"], **extra)

    assert result is sentinel, f"{case.name}: live() did not return {case.live_delegate}()'s result"
    assert len(calls) == 1, f"{case.name}: expected exactly one {case.live_delegate}() call"
    _, kwargs = calls[0]
    assert kwargs.get("limit") == 7, f"{case.name}: limit not forwarded to {case.live_delegate}"
    assert kwargs.get("score_threshold") == 0.3, (
        f"{case.name}: score_threshold not forwarded to {case.live_delegate}"
    )
    assert kwargs.get("namespaces") == ["ns1"], (
        f"{case.name}: namespaces not forwarded to {case.live_delegate}"
    )


def test_langchain_live_is_a_bound_select_alias(monkeypatch) -> None:
    """LangChain has no framework-native live hook, so ``live()`` is a thin,
    uniform-signature alias of :meth:`~dotfetch.integrations.frameworks.langchain.LangChainAdapter.select`
    (see that method's docstring for why: the mid-run hook lives one layer up,
    in ``LangGraphAdapter``). ``live()`` itself must stay synchronous (it
    returns an object, matching every other adapter) — the returned callable
    is what's async.
    """
    calls: list[tuple[str, dict]] = []

    async def _fake_select(self: Any, query: str, **kwargs: Any) -> list[Any]:
        calls.append((query, kwargs))
        return ["tool"]

    monkeypatch.setattr(F.LangChainAdapter, "select", _fake_select)

    adapter = F.LangChainAdapter(fetcher=None)  # type: ignore[arg-type]
    live_select = adapter.live(limit=7, score_threshold=0.3, namespaces=["ns1"])
    assert not isinstance(live_select, list), (
        "live() must return an object, not run selection eagerly"
    )
    assert not calls, "live() must not call select() before the returned callable is invoked"

    import asyncio

    result = asyncio.run(live_select("a query"))
    assert result == ["tool"]
    assert calls == [
        (
            "a query",
            {
                "limit": 7,
                "score_threshold": 0.3,
                "namespaces": ["ns1"],
                "required": None,
                "always_include": None,
            },
        )
    ]


@pytest.mark.parametrize("case", ADAPTERS, ids=[c.name for c in ADAPTERS])
def test_missing_framework_raises_clean_importerror(case: AdapterCase, monkeypatch) -> None:
    # Ensure the framework's import resolves to "not installed".
    for mod in case.modules:
        monkeypatch.setitem(sys.modules, mod, None)

    from dotfetch.integrations.frameworks.base import ToolSpec

    dummy = ToolSpec(
        name="t",
        qualified_name="default.t",
        description="d",
        parameters={"type": "object", "properties": {}},
        requires_confirmation=False,
        score=0.0,
        _fetcher=None,  # type: ignore[arg-type]
        _namespace="default",
    )

    with pytest.raises(ImportError):
        case.adapter_cls.convert(dummy)


def _install_stub(monkeypatch, modules: list[str], attrs: dict[str, object]) -> None:
    """Create stub modules; set ``attrs`` on the *last* (leaf) module."""
    for i, mod in enumerate(modules):
        m = types.ModuleType(mod)
        if i == len(modules) - 1:
            for k, v in attrs.items():
                setattr(m, k, v)
        monkeypatch.setitem(sys.modules, mod, m)


@pytest.mark.parametrize("case", ADAPTERS, ids=[c.name for c in ADAPTERS])
async def test_for_each_framework_selects_and_converts(
    case: AdapterCase, fetcher, monkeypatch
) -> None:
    """Smoke every adapter's ``select`` end-to-end against a real fetcher.

    Each adapter gets a lightweight ``sys.modules`` stub (see the factories
    above) just structured enough for ``convert`` to build something, so this
    runs in any environment regardless of which frameworks are actually
    installed.
    """
    if case.stub_attrs is not None:
        _install_stub(monkeypatch, case.modules, case.stub_attrs())

    tools = await case.adapter_cls(fetcher).select("send an email to my boss", limit=2)
    assert tools, f"{case.name}: expected at least one converted tool"


@pytest.mark.parametrize("case", ADAPTERS, ids=[c.name for c in ADAPTERS])
async def test_for_each_framework_select_with_required(
    case: AdapterCase, fetcher, monkeypatch
) -> None:
    """``select(required=...)`` pins a tool outside the semantic slice, uniformly.

    Regression guard for porting ``required``/``always_include`` from the
    Microsoft Agent Framework provider (``DotFetchContextProvider``) into the
    shared ``BaseFrameworkAdapter.select`` — every adapter shares the same
    ``DotFetchToolset.select`` under the hood (see
    ``dotfetch/integrations/frameworks/base.py``), so a ``required`` tool
    the semantic slice (bounded by ``limit``) wouldn't otherwise have picked
    must still surface, uncounted against ``limit``, regardless of which
    framework is converting it.
    """
    if case.stub_attrs is not None:
        _install_stub(monkeypatch, case.modules, case.stub_attrs())

    # limit=1 bounds the semantic slice to the top match for the email query
    # (send_email); "add" is unrelated and would not otherwise be selected,
    # so its presence proves `required` was threaded through, and the total
    # count proves it wasn't counted against `limit`.
    tools = await case.adapter_cls(fetcher).select(
        "send an email to my boss", limit=1, required=["add"]
    )
    assert len(tools) == 2, (
        f"{case.name}: expected 1 semantic tool + 1 pinned required tool, got {len(tools)}"
    )


# --------------------------------------------------------------------------- #
# Error-handling policy (see "Error-handling policy" in
# integrations/frameworks/README.md for the full write-up):
#
#   Default contract: a failing `fetcher.execute()` raises `ToolExecutionError`
#   out of `ToolSpec.ainvoke`/`invoke`, and every adapter's native wrapper
#   (`.func`/`._run`/`.forward`/…) lets it propagate uncaught so the
#   framework's own error handling takes over. That is `error_kind="raises"`
#   below, for all 14 adapters.
#
#   Three deliberate deviations exist one layer *below* convert()/select()
#   (i.e. not exercised by this matrix — see their own tests): MAF's
#   `_build_callable_for_tool` returns a JSON `{"error": ...}` string to the model;
#   The *live* workbench-style providers return an error
#   `ToolResult(is_error=True)`; Strands' real `Agent` tool-execution loop
#   (`DecoratedFunctionTool.stream`) converts any exception into an error
#   `ToolResult` — that one is Strands' own native contract, not dotfetch code.
#
#   Every *_live.py per-turn selection path (a *different* failure mode --
#   `fetcher.retrieve()`/selection failing, not tool execution) must not raise
#   at all: it logs a WARNING and degrades gracefully, either to "no tools
#   this turn" (stateless per-turn recomputation: Google ADK, LangGraph,
#   LlamaIndex, Pydantic AI) or "leave the previous turn's tools in place"
#   (stateful in-place mutation: OpenAI Agents SDK,
#   Strands) — see the second block of tests below.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", ADAPTERS, ids=[c.name for c in ADAPTERS])
async def test_adapter_tool_failure_matches_documented_error_kind(
    case: AdapterCase, fetcher_with_failing_tool: DotFetch, monkeypatch
) -> None:
    """A failing ``fetcher.execute()`` surfaces per the documented ``error_kind``.

    Selects the ``explode`` tool (registered by ``fetcher_with_failing_tool``,
    always raises), converts it via ``adapter_cls.convert``, then invokes the
    *native* object the way that framework's own runtime would
    (``case.invoke_failure``) — not ``spec.ainvoke`` directly, so this proves
    the contract survives each adapter's own wrapper, not just ``base.py``.
    Locks in that no ``_spec_to_*`` conversion accidentally swallows the
    error for any of the 14 adapters.
    """
    if case.stub_attrs is not None:
        _install_stub(monkeypatch, case.modules, case.stub_attrs())
        for module, attrs in (case.extra_stub_modules or dict)().items():
            _install_stub(monkeypatch, [module], attrs)

    from dotfetch.integrations.frameworks.base import DotFetchToolset, ToolExecutionError

    specs = await DotFetchToolset(fetcher_with_failing_tool).select("always raises", limit=1)
    assert specs, f"{case.name}: failing tool not found by selection"
    native = case.adapter_cls.convert(specs[0])

    if case.error_kind == "native_tool_failed":
        with pytest.raises(Exception) as native_info:
            await case.invoke_failure(native)
        tool_failed = sys.modules["pydantic_ai.exceptions"].ToolFailed
        assert isinstance(native_info.value, tool_failed)
        assert "boom" in str(native_info.value)
        cause = native_info.value.__cause__
        assert isinstance(cause, ToolExecutionError)
        assert cause.tool_name == "explode"
        return

    if case.error_kind == "native_tool_exception":
        # LangChain's ToolException, which a tool with handle_tool_error=True
        # turns into an error ToolMessage for the model.
        with pytest.raises(Exception) as native_info:
            await case.invoke_failure(native)
        assert type(native_info.value).__name__ in {"ToolException", "_StubToolException"}
        assert "boom" in str(native_info.value)
        assert isinstance(native_info.value.__cause__, ToolExecutionError)
        return

    if case.error_kind == "native_tool_result":
        # Strands: an error ToolResult dict for the model, not an exception.
        output = await case.invoke_failure(native)
        assert output["status"] == "error"
        assert "boom" in output["content"][0]["text"]
        return

    if case.error_kind == "native_tool_output":
        # Returned to the model as the tool's output instead of ending the run.
        output = await case.invoke_failure(native)
        assert isinstance(output, str) and output.startswith("Error: ")
        assert "boom" in output
        return

    assert case.error_kind == "raises", f"{case.name}: unknown error_kind {case.error_kind!r}"
    with pytest.raises(ToolExecutionError) as exc_info:
        await case.invoke_failure(native)
    assert exc_info.value.tool_name == "explode"
    assert exc_info.value.error is not None and "boom" in exc_info.value.error


def test_tool_execution_error_message_format() -> None:
    """Locks ``ToolExecutionError``'s message format so callers can pattern-match it.

    Every one of the 14 adapters' native tools raises this exact type with
    this exact shape on a failed execution (see the parametrized test above),
    so downstream users (a LangChain ``try/except``, a CrewAI callback, …) can
    rely on ``.tool_name`` / ``.status`` / ``.error`` rather than string-parsing.
    """
    from dotfetch.integrations.frameworks.base import ToolExecutionError

    err = ToolExecutionError("send_email", "failure", "SMTP connection refused")

    assert err.tool_name == "send_email"
    assert err.status == "failure"
    assert err.error == "SMTP connection refused"
    assert str(err) == "Tool 'send_email' failed (status=failure): SMTP connection refused"

    # A ``None`` error still produces a stable, non-empty message.
    err_no_detail = ToolExecutionError("add", "failure", None)
    assert str(err_no_detail) == "Tool 'add' failed (status=failure): no detail"


# --------------------------------------------------------------------------- #
# Per-turn live selection-failure policy
#
# A *different* failure mode from tool execution above: `fetcher.retrieve()`
# (semantic selection) itself raising mid-conversation -- e.g. the vector
# store is briefly unavailable. Every per-turn live provider must not let
# that kill the agent's turn. `DotFetchToolset.select` is the single choke
# point every *_live.py per-turn provider's selection call routes through
# (directly, or via `select_or_empty`), so patching it here exercises all
# eight uniformly.
# --------------------------------------------------------------------------- #


async def _raise_on_select(self: Any, query: str, **kwargs: Any) -> list[Any]:
    raise RuntimeError("boom: vector store unavailable")


@pytest.fixture
def broken_selection(monkeypatch) -> None:
    """Make every ``DotFetchToolset.select`` call raise (simulates a broken retrieval)."""
    from dotfetch.integrations.frameworks.base import DotFetchToolset

    monkeypatch.setattr(DotFetchToolset, "select", _raise_on_select)


async def test_google_adk_live_selection_failure_degrades_gracefully(
    fetcher: DotFetch, broken_selection, caplog
) -> None:
    from dotfetch.integrations.frameworks.google_adk_live import _inject_selected_tools

    llm_request = types.SimpleNamespace(config=types.SimpleNamespace(tools=[]), tools_dict={})
    with caplog.at_level("WARNING"):
        injected = await _inject_selected_tools(
            fetcher, "weather", llm_request, limit=3, score_threshold=0.0
        )

    assert injected == []  # stateless per-turn: degrades to "no tools this turn"
    assert any("semantic retrieval failed" in r.message for r in caplog.records)


async def test_langgraph_live_selection_failure_degrades_gracefully(
    fetcher: DotFetch, broken_selection, caplog
) -> None:
    from dotfetch.integrations.frameworks.langgraph_live import _select_tools_for_state

    state = {"messages": [{"role": "user", "content": "weather forecast"}]}
    with caplog.at_level("WARNING"):
        tools = await _select_tools_for_state(fetcher, state, limit=3, score_threshold=0.0)

    assert tools == []  # stateless per-turn: degrades to "no tools this turn"
    assert any("semantic retrieval failed" in r.message for r in caplog.records)


async def test_llamaindex_live_selection_failure_degrades_gracefully(
    fetcher: DotFetch, broken_selection, caplog, monkeypatch
) -> None:
    import dotfetch.integrations.frameworks.llamaindex_live as li_live

    class _FakeObjectRetriever:
        pass

    stub_objects = types.ModuleType("llama_index.core.objects")
    stub_objects.ObjectRetriever = _FakeObjectRetriever
    monkeypatch.setitem(sys.modules, "llama_index", types.ModuleType("llama_index"))
    monkeypatch.setitem(sys.modules, "llama_index.core", types.ModuleType("llama_index.core"))
    monkeypatch.setitem(sys.modules, "llama_index.core.objects", stub_objects)

    # The subclass is cached per base class, so the stub gets its own build.
    retriever = li_live._fetcher_tool_retriever(fetcher, limit=3)
    with caplog.at_level("WARNING"):
        tools = await retriever.aretrieve("weather forecast")

    assert tools == []  # stateless per-turn: degrades to "no tools this step"
    assert any("semantic retrieval failed" in r.message for r in caplog.records)


async def test_pydantic_ai_live_selection_failure_degrades_gracefully(
    fetcher: DotFetch, broken_selection, caplog, monkeypatch
) -> None:
    import dotfetch.integrations.frameworks.pydantic_ai_live as pai_live

    class _FakeToolDefinition:
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _FakeAbstractToolset:
        pass

    class _FakeToolsetTool:
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    pa_tools = types.ModuleType("pydantic_ai.tools")
    pa_tools.ToolDefinition = _FakeToolDefinition
    pa_toolsets = types.ModuleType("pydantic_ai.toolsets")
    pa_toolsets.AbstractToolset = _FakeAbstractToolset
    pa_toolsets_abstract = types.ModuleType("pydantic_ai.toolsets.abstract")
    pa_toolsets_abstract.ToolsetTool = _FakeToolsetTool
    monkeypatch.setitem(sys.modules, "pydantic_ai", types.ModuleType("pydantic_ai"))
    monkeypatch.setitem(sys.modules, "pydantic_ai.tools", pa_tools)
    monkeypatch.setitem(sys.modules, "pydantic_ai.toolsets", pa_toolsets)
    monkeypatch.setitem(sys.modules, "pydantic_ai.toolsets.abstract", pa_toolsets_abstract)

    toolset = pai_live._fetcher_toolset(fetcher, limit=3)
    toolset.set_query("weather forecast")
    with caplog.at_level("WARNING"):
        tools = await toolset.get_tools(types.SimpleNamespace(max_retries=1))

    assert tools == {}  # stateless per-step: degrades to "no tools this step"
    assert any("semantic retrieval failed" in r.message for r in caplog.records)


async def test_openai_agents_live_selection_failure_degrades_gracefully(
    fetcher: DotFetch, broken_selection, caplog, monkeypatch
) -> None:
    monkeypatch.setitem(sys.modules, "agents", types.ModuleType("agents"))
    from dotfetch.integrations.frameworks.openai_agents_live import _refresh_agent_tools

    agent = types.SimpleNamespace(tools=["existing_tool"])
    with caplog.at_level("WARNING"):
        tools = await _refresh_agent_tools(agent, fetcher, "weather forecast", limit=3)

    # stateful (`agent.tools` persists across turns): the mutation is skipped,
    # so both the return value and `agent.tools` keep the previous turn's tools.
    assert tools == ["existing_tool"]
    assert agent.tools == ["existing_tool"]
    assert any("semantic retrieval failed" in r.message for r in caplog.records)


async def test_strands_live_selection_failure_degrades_gracefully(
    fetcher: DotFetch, broken_selection, caplog
) -> None:
    from dotfetch.integrations.frameworks.strands_live import DotFetchStrandsToolHook

    hook = DotFetchStrandsToolHook(fetcher, limit=3)
    sentinel = object()
    tool_registry = types.SimpleNamespace(registry={"stale": sentinel}, dynamic_tools={})
    event = types.SimpleNamespace(
        agent=types.SimpleNamespace(
            messages=[{"role": "user", "content": [{"text": "weather forecast"}]}],
            tool_registry=tool_registry,
        )
    )

    with caplog.at_level("WARNING"):
        await hook._on_before_model_call(event)

    # stateful (`agent.tool_registry` persists across turns): the registry is
    # left completely untouched rather than retracting the previous tools.
    assert tool_registry.registry == {"stale": sentinel}
    assert any("semantic retrieval failed" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Microsoft Agent Framework (AgentFrameworkAdapter)
#
# AF-specific factory methods supplement the shared select/convert surface. Its
# surface is a set of factory methods that build distinct AF primitives.
# These checks assert the same three properties as the ADAPTERS matrix above
# (uniform surface / clean ImportError / end-to-end smoke), adapted to that
# shape:
#   - `context_provider`, `approval_middleware`, `observability_middleware`,
#     `tool_choice_middleware` all require `agent-framework` and raise a
#     clean ImportError when it's absent.
#   - `tool_bridge` is the deliberate exception: `DotFetchToolBridge` degrades
#     to bare callables when `agent-framework` is missing (see
#     `agent_framework_bridge._maybe_wrap_as_function_tool`), so it never
#     raises — that graceful-degradation contract is asserted explicitly
#     rather than lumped in with the "raises ImportError" cases.
#   - `tool_bridge().get_tools(...)` is the `select`-equivalent: it always
#     returns callables, whether or not `agent-framework` is installed, so
#     the end-to-end smoke needs no stub.
# --------------------------------------------------------------------------- #


def test_agent_framework_adapter_exposes_uniform_surface() -> None:
    adapter = AgentFrameworkAdapter(object())
    for method_name in (
        "context_provider",
        "tool_bridge",
        "approval_middleware",
        "observability_middleware",
        "tool_choice_middleware",
    ):
        assert callable(getattr(adapter, method_name, None)), (
            f"AgentFrameworkAdapter.{method_name} missing"
        )


def test_agent_framework_adapter_exposes_live_tier_and_live() -> None:
    """AgentFrameworkAdapter also participates in the uniform live_tier/live()
    facade (see AdapterCase docstring and ADAPTERS above for the 13
    BaseFrameworkAdapter subclasses) — AF genuinely supports per-round
    (``query_strategy="per_turn"``) dynamic tool re-selection, so its tier is
    ``"per-turn"``, matching LangGraph/LlamaIndex/Pydantic AI/etc.
    """
    assert AgentFrameworkAdapter.live_tier == "per-turn"
    assert callable(getattr(AgentFrameworkAdapter, "live", None))


def test_agent_framework_adapter_live_delegates_to_context_provider(monkeypatch) -> None:
    """``live()`` delegates to :meth:`AgentFrameworkAdapter.context_provider`,
    forwarding ``limit`` as ``top_k`` and defaulting ``query_strategy`` to
    ``"per_turn"`` (the deepest tier) rather than :meth:`context_provider`'s
    own back-compatible ``"per_run"`` default.
    """
    sentinel = object()
    calls: list[dict] = []

    def _recorder(self: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(AgentFrameworkAdapter, "context_provider", _recorder)

    adapter = AgentFrameworkAdapter(object())
    result = adapter.live(limit=7, score_threshold=0.3, namespaces=["ns1"])

    assert result is sentinel
    assert len(calls) == 1
    assert calls[0]["top_k"] == 7
    assert calls[0]["score_threshold"] == 0.3
    assert calls[0]["query_strategy"] == "per_turn"
    assert calls[0]["namespaces"] == ["ns1"]

    # An explicit query_strategy in framework_kwargs is respected, not overridden.
    calls.clear()
    adapter.live(limit=3, query_strategy="per_run")
    assert calls[0]["query_strategy"] == "per_run"


_AF_IMPORT_GATED_CASES: list[tuple[str, Callable[[AgentFrameworkAdapter], Any]]] = [
    ("context_provider", lambda a: a.context_provider()),
    ("approval_middleware", lambda a: a.approval_middleware(SecurityPolicy())),
    ("observability_middleware", lambda a: a.observability_middleware()),
    ("tool_choice_middleware", lambda a: a.tool_choice_middleware(lambda ctx: "auto")),
]


@pytest.mark.parametrize(
    "method_name,call",
    _AF_IMPORT_GATED_CASES,
    ids=[c[0] for c in _AF_IMPORT_GATED_CASES],
)
def test_agent_framework_adapter_missing_framework_raises_clean_importerror(
    method_name: str, call: Callable[[AgentFrameworkAdapter], Any], monkeypatch
) -> None:
    # `approval_middleware` / `observability_middleware` share a process-wide
    # `functools.lru_cache` (`_build_middleware_classes`) that's a real,
    # intentional optimisation — once `agent-framework` has been imported
    # successfully anywhere in the process, the built middleware classes are
    # cached and reused. Clear it here so this test's "framework absent"
    # simulation can't be short-circuited by a previous test (in this file or
    # elsewhere in the suite) that already built the middleware for real.
    from dotfetch.integrations.agent_framework_middleware import _build_middleware_classes

    _build_middleware_classes.cache_clear()

    monkeypatch.setitem(sys.modules, "agent_framework", None)
    adapter = AgentFrameworkAdapter(object())
    with pytest.raises(ImportError):
        call(adapter)


def test_agent_framework_adapter_tool_bridge_degrades_gracefully_without_framework(
    monkeypatch,
) -> None:
    """Unlike the other four entry points, `tool_bridge()` never requires
    agent-framework: `DotFetchToolBridge` falls back to bare callables when AF
    is absent, so building it (and using it) must NOT raise."""
    monkeypatch.setitem(sys.modules, "agent_framework", None)
    adapter = AgentFrameworkAdapter(object())
    bridge = adapter.tool_bridge()
    assert bridge is not None


async def test_agent_framework_adapter_select_and_convert_smoke(fetcher) -> None:
    """`tool_bridge().get_tools(...)` is AgentFrameworkAdapter's select→convert
    equivalent. Exercised with no stub: `DotFetchToolBridge` degrades to bare
    callables when `agent-framework` is absent and upgrades to real
    `FunctionTool`s when it's installed (it is, in this venv, via the
    `agent-frameworks` extra) — either way the result is a list of callables.
    """
    bridge = AgentFrameworkAdapter(fetcher).tool_bridge()
    tools = await bridge.get_tools("send an email to my boss", limit=2)
    assert tools, "agent_framework: tool_bridge().get_tools() returned no tools"
    assert all(callable(t) for t in tools)


# --------------------------------------------------------------------------- #
# Configuration errors never degrade to a tool-less agent
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", ADAPTERS, ids=[c.name for c in ADAPTERS])
def test_adapter_live_rejects_out_of_bounds_limit(case: AdapterCase) -> None:
    """``live(limit=60)`` raises at once instead of building an agent that
    silently runs with no tools (the per-turn hooks used to swallow it)."""
    from dotfetch.integrations.frameworks.base import QueryBoundsError

    adapter = case.adapter_cls(fetcher=None)  # type: ignore[arg-type]
    extra = case.live_extra_kwargs() if case.live_extra_kwargs is not None else {}
    with pytest.raises(QueryBoundsError, match="limit=60"):
        adapter.live(limit=60, **extra)


@pytest.mark.parametrize(
    "case", [c for c in ADAPTERS if c.live_extra_kwargs is not None], ids=lambda c: c.name
)
def test_adapter_live_missing_required_kwarg_is_a_type_error(case: AdapterCase) -> None:
    adapter = case.adapter_cls(fetcher=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="requires"):
        adapter.live()


async def test_live_hooks_and_builders_propagate_bounds_errors(fetcher: DotFetch) -> None:
    """Per-turn hooks re-raise ``QueryBoundsError`` (unlike a transient
    retrieval failure) and per-run builders reject it at construction."""
    from dotfetch.integrations.frameworks.base import QueryBoundsError
    from dotfetch.integrations.frameworks.google_adk_live import _inject_selected_tools
    from dotfetch.integrations.frameworks.langgraph_live import _select_tools_for_state
    from dotfetch.integrations.frameworks.live_wrappers import DotFetchLiveCrewAgent
    from dotfetch.integrations.frameworks.strands_live import DotFetchStrandsToolHook

    state = {"messages": [{"role": "user", "content": "send an email"}]}
    with pytest.raises(QueryBoundsError):
        await _select_tools_for_state(fetcher, state, limit=60)
    with pytest.raises(QueryBoundsError):
        await _inject_selected_tools(
            fetcher, "send an email", types.SimpleNamespace(), limit=60, score_threshold=0.0
        )
    with pytest.raises(QueryBoundsError):
        DotFetchStrandsToolHook(fetcher, limit=60)
    with pytest.raises(QueryBoundsError):
        DotFetchLiveCrewAgent(fetcher, limit=60)


async def test_openai_agents_hook_leaves_other_agents_alone(
    fetcher: DotFetch, monkeypatch
) -> None:
    """Run hooks fire for every agent in a run (handoff targets included);
    only the bound agent's ``tools`` are dotfetch's to rewrite."""
    stub = types.ModuleType("agents")
    stub.RunHooks = type("RunHooks", (), {})
    stub.FunctionTool = _echo
    monkeypatch.setitem(sys.modules, "agents", stub)
    from dotfetch.integrations.frameworks.openai_agents_live import _fetcher_run_hooks

    bound = types.SimpleNamespace(tools=[])
    other = types.SimpleNamespace(tools=["theirs"])
    hooks = _fetcher_run_hooks(fetcher, bound, limit=1)
    turn = [{"role": "user", "content": "send an email message to a recipient"}]

    await hooks.on_llm_start(None, other, None, turn)
    assert other.tools == ["theirs"]
    await hooks.on_llm_start(None, bound, None, turn)
    assert bound.tools and other.tools == ["theirs"]
