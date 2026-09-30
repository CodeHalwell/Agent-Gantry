"""Google ADK native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a Google Agent
Development Kit ``FunctionTool`` — the native tool object an ADK agent
introspects (from the callable's name, docstring and signature) and invokes.
The ``google.adk`` import is lazy so ``import agent_gantry`` never requires ADK
to be installed.

Public entry point: :class:`GoogleADKAdapter`.
"""

from __future__ import annotations

import copy
from typing import Any

from agent_gantry.integrations.frameworks.base import BaseFrameworkAdapter, ToolSpec


def _spec_to_google_adk(spec: ToolSpec) -> Any:
    """Wrap a :class:`ToolSpec` as a Google ADK ``FunctionTool``.

    The declaration the model sees comes from the Gantry tool's own JSON
    schema whenever the installed ``google-genai`` supports
    ``FunctionDeclaration.parameters_json_schema`` (genai 2.x, which both
    current google-adk lines require): a ``FunctionTool`` subclass overrides
    ``_get_declaration`` to pass the schema through verbatim, so per-parameter
    descriptions, enums, typed array items, nested objects and true
    required/optional split all survive. On older stacks it falls back to
    ADK's signature introspection of :meth:`ToolSpec.callable_for_signature`
    — an async function that carries a real ``__signature__`` derived from
    the tool's JSON schema (not a bare ``**kwargs``). Execution routes
    through ``gantry.execute`` either way, and argument handling still uses
    the callable's signature (ADK filters model args to declared names).

    Raises:
        ImportError: If ``google-adk`` is not installed.
    """
    try:
        from google.adk.tools import FunctionTool
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(
            "Google ADK support requires `google-adk`. Install it with `pip install google-adk` (or `uv add google-adk`)."
        ) from exc

    # ADK's automatic function calling rejects `T | None` and `None`-typed
    # defaults, so opt into type-matched empty defaults for optional params.
    fn = spec.callable_for_signature(type_matched_defaults=True)

    try:

        class _GantryADKFunctionTool(FunctionTool):  # type: ignore[misc, valid-type]
            def _get_declaration(self) -> Any:
                declaration = _declaration_from_schema(spec)
                if declaration is not None:
                    return declaration
                return super()._get_declaration()

    except TypeError:
        # ``FunctionTool`` isn't subclassable (test doubles, exotic shims) —
        # fall back to the plain wrapper; ADK then derives the declaration
        # from the callable's signature as before.
        return FunctionTool(func=fn)
    return _GantryADKFunctionTool(func=fn)


def _declaration_from_schema(spec: ToolSpec) -> Any:
    """Build a genai ``FunctionDeclaration`` carrying the schema verbatim.

    Returns ``None`` (caller falls back to ADK's signature introspection)
    when the installed ``google-genai`` predates ``parameters_json_schema``
    or the declaration can't be constructed.
    """
    try:
        from google.genai import types as genai_types
    except ImportError:  # pragma: no cover - genai always ships with adk
        return None
    declaration_cls = getattr(genai_types, "FunctionDeclaration", None)
    if declaration_cls is None or "parameters_json_schema" not in getattr(
        declaration_cls, "model_fields", {}
    ):
        return None
    # Deep, not shallow: a ``dict(...)`` leaves every nested ``properties`` /
    # ``items`` subschema aliased to the registry's canonical
    # ``ToolDefinition.parameters_schema``, so anything that adjusts the
    # declaration corrupts the tool for every other consumer. Property names
    # are the callable's aliases: ADK filters the model's arguments to the
    # signature's parameter names, so the declaration must use the same ones.
    schema = copy.deepcopy(spec.aliased_parameters()) if spec.parameters else {}
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    try:
        return declaration_cls(
            name=spec.name,
            description=spec.description,
            parameters_json_schema=schema,
        )
    except Exception:  # noqa: BLE001 - fall back to signature introspection
        return None


class GoogleADKAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into Google ADK.

    Static slice (``google.adk.tools.FunctionTool`` objects) plus deep per-turn
    live wiring (re-selects tools before every model request). Every call routes
    through ``gantry.execute``. :meth:`live` returns the
    :meth:`before_model_callback` — plug it into
    ``Agent(tools=[], before_model_callback=<result>)``.
    """

    live_tier = "per-turn"
    _live_delegate = "before_model_callback"

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as a Google ADK ``FunctionTool``."""
        return _spec_to_google_adk(spec)

    def before_model_callback(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
    ) -> Any:
        """Build an ADK ``before_model_callback`` that injects Gantry tools per turn."""
        from agent_gantry.integrations.frameworks.google_adk_live import (
            _gantry_before_model_callback,
        )

        return _gantry_before_model_callback(
            self._gantry,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
        )

    def agent(
        self,
        *,
        model: Any,
        name: str,
        instruction: str = "",
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **agent_kwargs: Any,
    ) -> Any:
        """Build an ADK ``Agent`` wired for per-turn dynamic tool selection (tools=[] + callback)."""
        from agent_gantry.integrations.frameworks.google_adk_live import (
            _gantry_adk_agent,
        )

        return _gantry_adk_agent(
            self._gantry,
            model=model,
            name=name,
            instruction=instruction,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            **agent_kwargs,
        )
