"""Stages package exposing the registry helpers and built-in implementations."""

from __future__ import annotations

from typing import Callable, Dict, TYPE_CHECKING

from .base import Stage

if TYPE_CHECKING:  # pragma: no cover - only imported for typing
    from ..core.config import Config

StageFactory = Callable[["Config"], Stage]
_STAGE_REGISTRY: Dict[str, StageFactory] = {}


def register_stage(name: str) -> Callable[[StageFactory], StageFactory]:
    """Decorator to register stage factories by name."""

    def decorator(factory: StageFactory) -> StageFactory:
        _STAGE_REGISTRY[name] = factory
        return factory

    return decorator


def get_stage_factory(name: str) -> StageFactory:
    """Return the factory previously registered under the supplied name."""
    return _STAGE_REGISTRY[name]


# Import built-in stages so that they register themselves with the registry.
from . import pro_eval as _pro_eval  # noqa: F401
from . import pro_gen as _pro_gen  # noqa: F401
from . import res_eval as _res_eval  # noqa: F401
from . import res_gen as _res_gen  # noqa: F401
from . import rw_eval as _rw_eval  # noqa: F401
from . import rw_gen as _rw_gen  # noqa: F401
from . import write_test as _write_test  # noqa: F401


__all__ = [
    "StageFactory",
    "register_stage",
    "get_stage_factory",
]