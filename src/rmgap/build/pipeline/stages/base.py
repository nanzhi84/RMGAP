"""Stage base implementing a simple template method."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from ..schemas.protocol import Protocol


logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    protocols: List[Protocol]
    metrics: Dict[str, Any]


class Stage(ABC):
    def execute(self, protocols: List[Protocol]) -> StageResult:
        eligible = self.filter_eligible(protocols)
        processed = self.process(eligible)
        merged = self.merge_results(protocols, processed)
        metrics = {
            "total": len(eligible),
            "success": len(processed),
            "failed": max(0, len(eligible) - len(processed)),
        }

        extra_metrics = getattr(self, "_latest_metrics", None)
        if isinstance(extra_metrics, dict):
            metrics.update(extra_metrics)
        if hasattr(self, "_latest_metrics"):
            delattr(self, "_latest_metrics")

        return StageResult(merged, metrics)

    @abstractmethod
    def filter_eligible(self, protocols: List[Protocol]) -> List[Protocol]:  # noqa: D401
        return []

    @abstractmethod
    def process(self, protocols: List[Protocol]) -> List[Protocol]:  # noqa: D401
        return []

    def merge_results(self, original: List[Protocol], processed: List[Protocol]) -> List[Protocol]:
        by_id = {p.record_id: p for p in processed}
        return [by_id.get(p.record_id, p) for p in original]