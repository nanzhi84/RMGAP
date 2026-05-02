"""Provider abstraction for pluggable LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelParams:
    """Model invocation parameters passed to providers as a single payload."""

    name: str
    temp: float = 0.0
    max_tokens: int | None = None
    extras: dict[str, Any] | None = None


class Provider(ABC):
    """Provider interface for text generation models."""

    @abstractmethod
    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        context: str,
        expect_json: bool = True,
    ) -> Any:
        """Generate a response using the provider's configured model."""
        raise NotImplementedError


