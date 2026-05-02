from rmgap.rm.base import BaseRM, GenerativeRM

RM_REGISTRY = {}


def register_rm(cls):
    """Register decorator for RM classes."""
    RM_REGISTRY[cls.__name__] = cls
    return cls


from .endorm import EndoRM  # noqa: E402
from .scalar import ScalarRM  # noqa: E402
from .generative_verifier import GenerativeVerifierRM  # noqa: E402
from .genrm_pointwise import GenRMPointwise  # noqa: E402
from .genrm_pairwise import GenRMPairwise  # noqa: E402
from .dpo_implicit import DpoImplicitRM  # noqa: E402


def make_rm(name: str, *args, **kwargs) -> BaseRM:
    """Create RM instance by name."""
    return RM_REGISTRY[name](*args, **kwargs)


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
