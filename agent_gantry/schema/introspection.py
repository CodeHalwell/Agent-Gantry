"""
Tool introspection utilities for Agent-Gantry.

Provides schema building from Python function signatures and type hints.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable


def build_parameters_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """
    Build JSON Schema for function parameters from Python type hints.

    Handles basic Python types (int, float, bool, str) with automatic
    detection of required vs optional parameters based on defaults.

    Args:
        func: The function to introspect

    Returns:
        JSON Schema dict with type, properties, and required fields

    Example:
        >>> def my_func(x: int, y: str = "default") -> str:
        ...     return f"{x}: {y}"
        >>> schema = build_parameters_schema(my_func)
        >>> schema["required"]
        ['x']
        >>> schema["properties"]["y"]["type"]
        'string'
    """
    sig = inspect.signature(func)

    # Try to get type hints, fall back to empty dict
    try:
        type_hints = func.__annotations__
    except AttributeError:
        type_hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        # Skip self and cls parameters
        if param_name in ("self", "cls"):
            continue

        param_type = type_hints.get(param_name, str)
        param_schema = _type_to_json_schema(param_type)
        properties[param_name] = param_schema

        # Mark as required if no default value
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _type_to_json_schema(param_type: Any) -> dict[str, str]:
    """
    Map Python type to JSON Schema type.

    Args:
        param_type: Python type annotation

    Returns:
        Dict with "type" field set to appropriate JSON Schema type
    """
    # Map Python types to JSON Schema types
    type_map = {
        int: "integer",
        float: "number",
        bool: "boolean",
        str: "string",
    }

    # Direct type match (most reliable)
    if param_type in type_map:
        return {"type": type_map[param_type]}

    # Try to get the origin type for generic types (e.g., Optional[int])
    try:
        import typing
        origin = typing.get_origin(param_type)
        if origin is not None:
            # For Optional[T], get T
            args = typing.get_args(param_type)
            if args:
                # Recursively check the first argument
                return _type_to_json_schema(args[0])
    except (AttributeError, ImportError):
        pass

    # Check string representations as fallback (less reliable)
    type_str = str(param_type)
    # Use word boundaries to avoid false positives
    if type_str in ("int", "<class 'int'>"):
        return {"type": "integer"}
    elif type_str in ("float", "<class 'float'>"):
        return {"type": "number"}
    elif type_str in ("bool", "<class 'bool'>"):
        return {"type": "boolean"}
    elif type_str in ("str", "<class 'str'>"):
        return {"type": "string"}

    # Default to string for unknown types
    return {"type": "string"}
