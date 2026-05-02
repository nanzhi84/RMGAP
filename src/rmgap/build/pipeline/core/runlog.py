from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from ..schemas.protocol import Protocol
from ..stages.base import Stage, StageResult


logger = logging.getLogger(__name__)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


@dataclass
class StageRunRecord:
    timestamp: str
    input: int
    eligible: int
    success: int
    failed: int
    passed: int


class RunLogger:
    def __init__(self, out_dir: Path, filename: str = "runs.jsonl") -> None:
        self.out_dir = out_dir
        self.filename = filename

    def append(self, stage: str, record: StageRunRecord) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {stage: asdict(record)}
        log_path = self.out_dir / self.filename
        try:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False))
                handle.write("\n")
        except OSError as exc:
            logger.error("Failed to write stage log for %s to %s: %s", stage, log_path, exc)


class StageRunner:
    def __init__(self, logger: RunLogger) -> None:
        self.logger = logger

    def run(self, stage_name: str, stage_obj: Stage, protocols: List[Protocol]) -> StageResult:
        input_count = len(protocols)
        result = stage_obj.execute(protocols)
        metrics = dict(result.metrics or {})
        record = StageRunRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            input=input_count,
            eligible=_safe_int(metrics.get("total")),
            success=_safe_int(metrics.get("success")),
            failed=_safe_int(metrics.get("failed")),
            passed=_safe_int(metrics.get("passed")),
        )
        self.logger.append(stage_name, record)
        return result