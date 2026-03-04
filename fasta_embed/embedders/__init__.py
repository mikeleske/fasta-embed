"""Embedder registry and factory.

Every concrete embedder decorates itself with ``@register("name")`` so that
the CLI / config can look it up by string key at runtime.
"""

from __future__ import annotations

from typing import Any

from .base import Embedder

_REGISTRY: dict[str, type[Embedder]] = {}


def register(name: str):
    """Class decorator that adds an :class:`Embedder` subclass to the global
    registry under *name*."""

    def decorator(cls: type[Embedder]) -> type[Embedder]:
        if name in _REGISTRY:
            raise ValueError(
                f"Embedder '{name}' is already registered "
                f"({_REGISTRY[name].__qualname__})"
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


def create_embedder(name: str, **kwargs: Any) -> Embedder:
    """Instantiate a registered embedder by *name*, forwarding *kwargs* to its
    constructor."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown embedder '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_embedders() -> list[str]:
    """Return sorted list of registered embedder names."""
    return sorted(_REGISTRY)


# Import concrete embedders so their @register decorators execute.
from . import dnabert  # noqa: E402, F401
from . import ntv2  # noqa: E402, F401
from . import ntv3  # noqa: E402, F401
