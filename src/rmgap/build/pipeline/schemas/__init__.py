"""Data module containing data protocol definitions and loading."""

from .loader import LoadResult, load_from_protocols, load_from_raw
from .protocol import Protocol, RawRecord

__all__ = [
    "Protocol",
    "RawRecord",
    "load_from_raw",
    "load_from_protocols",
    "LoadResult",
]
