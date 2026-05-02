"""Provider package exposing registry helpers and built-in providers."""

from __future__ import annotations

from typing import Dict, Type

from .base import Provider, ModelParams

PROVIDER_REGISTRY: Dict[str, Type[Provider]] = {}

def register_provider(name: str):
    """Decorator to register provider classes by name."""

    def decorator(cls: Type[Provider]) -> Type[Provider]:
        PROVIDER_REGISTRY[name] = cls
        return cls

    return decorator

def make_provider(name: str, params: ModelParams) -> Provider:
    """Instantiate a provider by registered name."""
    provider_cls = PROVIDER_REGISTRY[name]
    return provider_cls(params)

# Import built-in providers so registration side effects run on import.
from . import openai as _openai  # noqa: F401

__all__ = [
    "ModelParams",
    "Provider",
    "PROVIDER_REGISTRY",
    "register_provider",
    "make_provider",
]
