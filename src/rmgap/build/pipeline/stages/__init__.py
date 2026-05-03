"""Stages package exposing the registry helpers and built-in implementations."""

from __future__ import annotations

import importlib
from typing import Callable, Dict, TYPE_CHECKING

from .base import Stage

if TYPE_CHECKING:  # pragma: no cover - only imported for typing
    from ..core.config import Config

StageFactory = Callable[["Config"], Stage]
_STAGE_REGISTRY: Dict[str, StageFactory] = {}
_BUILTIN_STAGE_MODULES: Dict[str, str] = {
    "pro_eval": ".pro_eval",
    "pro_gen": ".pro_gen",
    "res_eval": ".res_eval",
    "res_gen": ".res_gen",
    "rw_eval": ".rw_eval",
    "rw_gen": ".rw_gen",
    "write_test": ".write_test",
}


def register_stage(name: str) -> Callable[[StageFactory], StageFactory]:
    """Decorator to register stage factories by name."""

    def decorator(factory: StageFactory) -> StageFactory:
        _STAGE_REGISTRY[name] = factory
        return factory

    return decorator


def get_stage_factory(name: str) -> StageFactory:
    """Return the factory previously registered under the supplied name."""
    if name not in _STAGE_REGISTRY and name in _BUILTIN_STAGE_MODULES:
        importlib.import_module(_BUILTIN_STAGE_MODULES[name], __name__)
    return _STAGE_REGISTRY[name]


__all__ = [
    "StageFactory",
    "register_stage",
    "get_stage_factory",
]
