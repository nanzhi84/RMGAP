"""Core data protocol definitions for pipeline communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Protocol:
    """
    Unified data protocol for passing information between pipeline stages.
    
    Intermediate JSONL files store JSON-serialized Protocol objects line by line.
    Missing data is represented by empty strings, lists, or dictionaries.
    """

    record_id: str = ""
    original_prompt: str = ""
    source: str = ""
    domain: str = ""
    # Pipeline state table, e.g. {"res": "res_not_generated", "pro": "pro_not_generated"}.
    # Keys represent logical phases, values are phase-specific state labels.
    state: Dict[str, str] = field(
        default_factory=lambda: {
            "res": "res_not_generated",
            "pro": "pro_not_generated",
            "rw": "rw_not_generated",
        }
    )
    responses: Dict[str, Any] = field(default_factory=dict)
    res_eval: Dict[str, Any] = field(default_factory=dict)
    pro_gen: Dict[str, Any] = field(default_factory=dict)
    pro_eval: Dict[str, Any] = field(default_factory=dict)
    rw_eval: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "original_prompt": self.original_prompt,
            "source": self.source,
            "domain": self.domain,
            "state": self.state,
            "responses": self.responses,
            "res_eval": self.res_eval,
            "pro_gen": self.pro_gen,
            "pro_eval": self.pro_eval,
            "rw_eval": self.rw_eval,
            "models": self.models,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Protocol:
        """Create Protocol from dictionary."""
        return cls(
            record_id=data.get("record_id", ""),
            original_prompt=data.get("original_prompt", ""),
            source=data.get("source", ""),
            domain=data.get("domain", ""),
            state=data.get(
                "state",
                {
                    "res": "res_not_generated",
                    "pro": "pro_not_generated",
                    "rw": "rw_not_generated",
                },
            ),
            responses=data.get("responses", {}),
            res_eval=data.get("res_eval", {}),
            pro_gen=data.get("pro_gen", {}),
            pro_eval=data.get("pro_eval", {}),
            rw_eval=data.get("rw_eval", {}),
            models=data.get("models", {}),
            meta=data.get("meta", {}),
        )


@dataclass
class RawRecord:
    """Input record representation."""

    record_id: str
    prompt: str
    source: str = ""
    domain: str = ""