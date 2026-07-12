"""Cross-framework conformance matrix for the native tool adapters.

Every ``<Framework>Adapter`` in ``agent_gantry.integrations.frameworks`` must
honor the same contract regardless of which third-party framework it targets:

1. It exposes ``select`` (async select+convert) and ``convert`` (single convert
   staticmethod), both importable without the framework installed.
2. ``select`` runs semantic selection against a real gantry and returns one
   converted tool per selected tool — verified end-to-end, per framework,
   through a lightweight ``sys.modules`` stub (mirroring how the real package
   would be structured) so the check runs everywhere without needing every
   framework installed. ``AutoGenAdapter.convert`` is the one exception: it
   returns a plain ``{name, description, callable}`` mapping rather than an
   opaque framework object (see ``AdapterCase.convert_kind`` below), so it
   needs no stub at all.
3. ``convert`` raises a clean ``ImportError`` carrying a ``pip install`` hint
   when the framework is absent — not ``AttributeError`` / ``KeyError``. This
   does not apply to ``AutoGenAdapter.convert`` (import-free by design; see
   above) — its contract is asserted explicitly instead of being skipped.

This locks the uniform surface so a new adapter can't silently drift.

``AgentFrameworkAdapter`` (Microsoft Agent Framework, ``agent_gantry.agent_framework``)
is deliberately **not** part of ``ADAPTERS`` below: it has no ``select``/``convert``
staticmethods at all. It is a small factory whose methods
(``context_provider``, ``tool_bridge``, ``approval_middleware``,
``observability_middleware``, ``tool_choice_middleware``) build distinct AF
primitives (a ``ContextProvider``, a ``GantryToolBridge``, AF middleware), and
its ``select``-equivalent is ``tool_bridge().get_tools(...)``. It gets its own
parametrized contract checks in the "Microsoft Agent Framework" section below,
mirroring the same three properties (uniform surface / clean ImportError /
end-to-end smoke) adapted to its actual shape.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.agent_framework import AgentFrameworkAdapter
from agent_gantry.core.security import SecurityPolicy
from agent_gantry.integrations import frameworks as F  # noqa: N812


@pytest.fixture
async def gantry():
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

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


def _stub_langchain_attrs() -> dict[str, object]:
    structured_tool = types.SimpleNamespace()
    structured_tool.from_function = staticmethod(_echo)
    return {"StructuredTool": structured_tool}


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


def _stub_openai_agents_attrs() -> dict[str, object]:
    return {"FunctionTool": _echo}


class _StubSmolagentsTool:
    """Stand-in base for ``smolagents.Tool``.

    ``_spec_to_smolagents`` builds a dynamic subclass via ``type(...)`` (class
    attributes for ``name``/``description``/``inputs``/``output_type``/
    ``forward``) and instantiates it with no arguments — a no-op ``__init__``
    is enough to stand in for smolagents' real validation logic.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def _stub_smolagents_attrs() -> dict[str, object]:
    return {"Tool": _StubSmolagentsTool}


def _stub_haystack_attrs() -> dict[str, object]:
    return {"Tool": _echo}


def _stub_agno_attrs() -> dict[str, object]:
    return {"Function": _echo}


def _stub_semantic_kernel_attrs() -> dict[str, object]:
    def _kernel_function(*, name: str | None = None, description: str | None = None):
        def _decorator(fn: Any) -> Any:
            return fn

        return _decorator

    def _kernel_function_from_method(*, method: Any, plugin_name: str) -> Any:
        return types.SimpleNamespace(
            method=method, plugin_name=plugin_name, name=getattr(method, "__name__", None)
        )

    return {
        "KernelFunctionFromMethod": _kernel_function_from_method,
        "kernel_function": _kernel_function,
    }


def _stub_google_adk_attrs() -> dict[str, object]:
    return {"FunctionTool": _echo}


def _stub_strands_attrs() -> dict[str, object]:
    def _tool(**kwargs: Any):
        def _decorator(fn: Any) -> Any:
            return types.SimpleNamespace(fn=fn, **kwargs)

        return _decorator

    return {"tool": _tool}


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
            ``"per-call"``. Locked against ``<Framework>Adapter.live_tier``
            and cross-referenced with ``integrations/frameworks/README.md``'s
            table (see ``test_adapter_live_tier_matches_capability_table``).
        live_delegate: Name of the bespoke method ``adapter.live()`` delegates
            to (e.g. ``"react_agent"``, ``"toolset"``, ``"agent_builder"``).
            ``"select"`` is a sentinel for LangChain, whose ``live()`` returns
            a *bound alias* of ``select`` rather than delegating to a
            separate bespoke method — it is exercised by its own dedicated
            test, not the generic delegation-smoke test.
        convert_kind: ``"native"`` (default) — ``convert`` requires the
            framework and returns an opaque native tool object. ``"dict"`` —
            ``convert`` is import-free and returns a plain registrable
            mapping (currently only ``AutoGenAdapter``).
        stub_attrs: Zero-arg factory building the ``{attr: value}`` mapping
            installed on the leaf module in ``modules`` for the end-to-end
            select→convert smoke. ``None`` for import-free adapters.
        live_extra_kwargs: Zero-arg factory building the extra
            ``framework_kwargs`` a call to ``adapter.live()`` requires (e.g.
            ``{"model": ...}`` for LangGraph, whose native hook is bound to a
            specific chat model). ``None`` when ``live()`` needs nothing
            beyond ``limit``/``score_threshold``/``namespaces``.
    """

    name: str
    adapter_cls: type
    modules: list[str]
    live_tier: str
    live_delegate: str
    convert_kind: str = "native"
    stub_attrs: Callable[[], dict[str, object]] | None = None
    live_extra_kwargs: Callable[[], dict[str, object]] | None = None


# NOTE for the DSPy adapter (added separately, see CLAUDE.md task boundaries):
# add its row here with `live_tier`/`live_delegate` set once its native live
# hook (if any) is decided — if DSPy has no native per-turn/per-call hook,
# follow the LangChain precedent (`live_delegate="select"`, tier "per-call",
# `live()` returns a bound alias of `select`) rather than skip the field.
ADAPTERS: list[AdapterCase] = [
    AdapterCase(
        "langchain",
        F.LangChainAdapter,
        ["langchain_core", "langchain_core.tools"],
        live_tier="per-call",
        live_delegate="select",
        stub_attrs=_stub_langchain_attrs,
    ),
    AdapterCase(
        "langgraph",
        F.LangGraphAdapter,
        ["langchain_core", "langchain_core.tools"],
        live_tier="per-turn",
        live_delegate="react_agent",
        stub_attrs=_stub_langchain_attrs,
        live_extra_kwargs=lambda: {"model": object()},
    ),
    AdapterCase(
        "llamaindex",
        F.LlamaIndexAdapter,
        ["llama_index", "llama_index.core", "llama_index.core.tools"],
        live_tier="per-turn",
        live_delegate="tool_retriever",
        stub_attrs=_stub_llamaindex_attrs,
    ),
    AdapterCase(
        "crewai",
        F.CrewAIAdapter,
        ["crewai", "crewai.tools"],
        live_tier="per-call",
        live_delegate="agent_builder",
        stub_attrs=_stub_crewai_attrs,
    ),
    AdapterCase(
        "pydantic_ai",
        F.PydanticAIAdapter,
        ["pydantic_ai", "pydantic_ai.tools"],
        live_tier="per-turn",
        live_delegate="toolset",
        stub_attrs=_stub_pydantic_ai_attrs,
    ),
    AdapterCase(
        "openai_agents",
        F.OpenAIAgentsAdapter,
        ["agents"],
        live_tier="per-turn",
        live_delegate="session",
        stub_attrs=_stub_openai_agents_attrs,
        live_extra_kwargs=lambda: {"agent": object()},
    ),
    AdapterCase(
        "smolagents",
        F.SmolagentsAdapter,
        ["smolagents"],
        live_tier="per-call",
        live_delegate="agent_builder",
        stub_attrs=_stub_smolagents_attrs,
    ),
    AdapterCase(
        "haystack",
        F.HaystackAdapter,
        ["haystack", "haystack.tools"],
        live_tier="per-call",
        live_delegate="tool_invoker_builder",
        stub_attrs=_stub_haystack_attrs,
    ),
    AdapterCase(
        "agno",
        F.AgnoAdapter,
        ["agno", "agno.tools", "agno.tools.function"],
        live_tier="per-call",
        live_delegate="agent_builder",
        stub_attrs=_stub_agno_attrs,
    ),
    AdapterCase(
        "semantic_kernel",
        F.SemanticKernelAdapter,
        ["semantic_kernel", "semantic_kernel.functions"],
        live_tier="per-turn",
        live_delegate="function_provider",
        stub_attrs=_stub_semantic_kernel_attrs,
        live_extra_kwargs=lambda: {"kernel": object()},
    ),
    AdapterCase(
        "google_adk",
        F.GoogleADKAdapter,
        ["google.adk", "google.adk.tools"],
        live_tier="per-turn",
        live_delegate="before_model_callback",
        stub_attrs=_stub_google_adk_attrs,
    ),
    AdapterCase(
        "autogen",
        F.AutoGenAdapter,
        [],
        live_tier="per-turn",
        live_delegate="workbench",
        convert_kind="dict",
    ),
    AdapterCase(
        "strands",
        F.StrandsAdapter,
        ["strands"],
        live_tier="per-turn",
        live_delegate="tool_hook",
        stub_attrs=_stub_strands_attrs,
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


@pytest.mark.parametrize("case", ADAPTERS, ids=[c.name for c in ADAPTERS])
def test_adapter_exposes_live_tier_and_live(case: AdapterCase) -> None:
    """Every adapter declares a valid ``live_tier`` and a callable ``live``."""
    assert getattr(case.adapter_cls, "live_tier", None) in ("per-turn", "per-call"), (
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
    OpenAI Agents SDK, Semantic Kernel, Google ADK, AutoGen, and Strands
    genuinely re-select tools on every model turn (``"per-turn"``); LangChain,
    CrewAI, Agno, Haystack, and Smolagents fix their tool list at agent
    construction with no native mid-run hook, so the deepest Gantry can do is
    rebuild before each new top-level call (``"per-call"``).
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
    recorder's sentinel unchanged. No real gantry or framework install is
    needed — this only proves the wiring, not the bespoke method's own
    behaviour (that's covered by each framework's dedicated live tests).
    """
    sentinel = object()
    calls: list[tuple[tuple, dict]] = []

    def _recorder(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(case.adapter_cls, case.live_delegate, _recorder)

    adapter = case.adapter_cls(gantry=None)  # type: ignore[arg-type]
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
    uniform-signature alias of :meth:`~agent_gantry.integrations.frameworks.langchain.LangChainAdapter.select`
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

    adapter = F.LangChainAdapter(gantry=None)  # type: ignore[arg-type]
    live_select = adapter.live(limit=7, score_threshold=0.3, namespaces=["ns1"])
    assert not isinstance(live_select, list), (
        "live() must return an object, not run selection eagerly"
    )
    assert not calls, "live() must not call select() before the returned callable is invoked"

    import asyncio

    result = asyncio.run(live_select("a query"))
    assert result == ["tool"]
    assert calls == [("a query", {"limit": 7, "score_threshold": 0.3, "namespaces": ["ns1"]})]


@pytest.mark.parametrize("case", ADAPTERS, ids=[c.name for c in ADAPTERS])
def test_missing_framework_raises_clean_importerror(case: AdapterCase, monkeypatch) -> None:
    # Ensure the framework's import resolves to "not installed".
    for mod in case.modules:
        monkeypatch.setitem(sys.modules, mod, None)

    from agent_gantry.integrations.frameworks.base import ToolSpec

    dummy = ToolSpec(
        name="t",
        qualified_name="default.t",
        description="d",
        parameters={"type": "object", "properties": {}},
        requires_confirmation=False,
        score=0.0,
        _gantry=None,  # type: ignore[arg-type]
        _namespace="default",
    )

    if case.convert_kind == "dict":
        # Import-free adapters (currently only AutoGen) must NOT require the
        # framework to build their registrable mapping — convert() succeeds
        # even with the framework fully absent, and callers only hit an
        # ImportError later, when they actually register the tool.
        result = case.adapter_cls.convert(dummy)
        assert isinstance(result, dict) and callable(result.get("callable")), (
            f"{case.name}: expected an import-free {{name, description, callable}} "
            f"mapping even without the framework installed, got {result!r}"
        )
        return

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
    case: AdapterCase, gantry, monkeypatch
) -> None:
    """Smoke every adapter's ``select`` end-to-end against a real gantry.

    Import-free adapters (``convert_kind == "dict"``) need no stub at all —
    everything else gets a lightweight ``sys.modules`` stub (see the factories
    above) just structured enough for ``convert`` to build something, so this
    runs in any environment regardless of which frameworks are actually
    installed.
    """
    if case.stub_attrs is not None:
        _install_stub(monkeypatch, case.modules, case.stub_attrs())

    tools = await case.adapter_cls(gantry).select("send an email to my boss", limit=2)
    assert tools, f"{case.name}: expected at least one converted tool"

    if case.convert_kind == "dict":
        assert all("callable" in t and callable(t["callable"]) for t in tools), (
            f"{case.name}: expected registrable mappings with a callable"
        )


# --------------------------------------------------------------------------- #
# Microsoft Agent Framework (AgentFrameworkAdapter)
#
# Not a `BaseFrameworkAdapter` — no `select`/`convert` staticmethods. Its
# surface is a set of factory methods that build distinct AF primitives.
# These checks assert the same three properties as the ADAPTERS matrix above
# (uniform surface / clean ImportError / end-to-end smoke), adapted to that
# shape:
#   - `context_provider`, `approval_middleware`, `observability_middleware`,
#     `tool_choice_middleware` all require `agent-framework` and raise a
#     clean ImportError when it's absent.
#   - `tool_bridge` is the deliberate exception: `GantryToolBridge` degrades
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
    (``query_strategy="per_call"``) dynamic tool re-selection, so its tier is
    ``"per-turn"``, matching LangGraph/LlamaIndex/Pydantic AI/etc.
    """
    assert AgentFrameworkAdapter.live_tier == "per-turn"
    assert callable(getattr(AgentFrameworkAdapter, "live", None))


def test_agent_framework_adapter_live_delegates_to_context_provider(monkeypatch) -> None:
    """``live()`` delegates to :meth:`AgentFrameworkAdapter.context_provider`,
    forwarding ``limit`` as ``top_k`` and defaulting ``query_strategy`` to
    ``"per_call"`` (the deepest tier) rather than :meth:`context_provider`'s
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
    assert calls[0]["query_strategy"] == "per_call"
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
    from agent_gantry.integrations.agent_framework_middleware import _build_middleware_classes

    _build_middleware_classes.cache_clear()

    monkeypatch.setitem(sys.modules, "agent_framework", None)
    adapter = AgentFrameworkAdapter(object())
    with pytest.raises(ImportError):
        call(adapter)


def test_agent_framework_adapter_tool_bridge_degrades_gracefully_without_framework(
    monkeypatch,
) -> None:
    """Unlike the other four entry points, `tool_bridge()` never requires
    agent-framework: `GantryToolBridge` falls back to bare callables when AF
    is absent, so building it (and using it) must NOT raise."""
    monkeypatch.setitem(sys.modules, "agent_framework", None)
    adapter = AgentFrameworkAdapter(object())
    bridge = adapter.tool_bridge()
    assert bridge is not None


async def test_agent_framework_adapter_select_and_convert_smoke(gantry) -> None:
    """`tool_bridge().get_tools(...)` is AgentFrameworkAdapter's select→convert
    equivalent. Exercised with no stub: `GantryToolBridge` degrades to bare
    callables when `agent-framework` is absent and upgrades to real
    `FunctionTool`s when it's installed (it is, in this venv, via the
    `agent-frameworks` extra) — either way the result is a list of callables.
    """
    bridge = AgentFrameworkAdapter(gantry).tool_bridge()
    tools = await bridge.get_tools("send an email to my boss", limit=2)
    assert tools, "agent_framework: tool_bridge().get_tools() returned no tools"
    assert all(callable(t) for t in tools)
