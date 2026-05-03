from __future__ import annotations

import importlib
from typing import Any

from rmgap.rm.base import BaseRM, GenerativeRM

RM_REGISTRY = {}
_RM_MODULES = {
    "EndoRM": ".endorm",
    "ScalarRM": ".scalar",
    "GenerativeVerifierRM": ".generative_verifier",
    "GenRMPointwise": ".genrm_pointwise",
    "GenRMPairwise": ".genrm_pairwise",
    "DpoImplicitRM": ".dpo_implicit",
}


def register_rm(cls):
    """Register decorator for RM classes."""
    RM_REGISTRY[cls.__name__] = cls
    return cls


def _load_rm(name: str) -> None:
    if name in RM_REGISTRY:
        return
    module_name = _RM_MODULES[name]
    importlib.import_module(module_name, __name__)


def make_rm(name: str, *args: Any, **kwargs: Any) -> BaseRM:
    """Create RM instance by name."""
    if name not in RM_REGISTRY:
        _load_rm(name)
    return RM_REGISTRY[name](*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name in _RM_MODULES:
        _load_rm(name)
        return RM_REGISTRY[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EndoRM",
    "ScalarRM",
    "GenerativeVerifierRM",
    "GenRMPointwise",
    "GenRMPairwise",
    "DpoImplicitRM",
    "GenerativeRM",
    "register_rm",
    "make_rm",
]
