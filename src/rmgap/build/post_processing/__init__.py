"""Post-processing utilities for dataset curation."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = (
    "CodeAnswerValidator",
    "MathAnswerValidator",
    "ValidationStats",
    "Validator",
    "filter_reasoning_protocols",
)


def __getattr__(name: str):
    if name in {"CodeAnswerValidator"}:
        from .code import CodeAnswerValidator

        return CodeAnswerValidator
    if name in {"MathAnswerValidator"}:
        from .math import MathAnswerValidator

        return MathAnswerValidator
    if name in {"ValidationStats", "Validator", "filter_reasoning_protocols"}:
        from .validator import ValidationStats, Validator, filter_reasoning_protocols

        return {
            "ValidationStats": ValidationStats,
            "Validator": Validator,
            "filter_reasoning_protocols": filter_reasoning_protocols,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover
    from .code import CodeAnswerValidator
    from .math import MathAnswerValidator
    from .validator import ValidationStats, Validator, filter_reasoning_protocols
