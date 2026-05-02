"""Reasoning domain validator orchestrating math/code checks."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from ..pipeline.schemas.protocol import Protocol
from .code import CodeAnswerValidator
from .math import MathAnswerValidator
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _normalize_eval_types(meta: dict | None) -> List[str]:
    if not isinstance(meta, dict):
        return []
    raw = meta.get("type")
    if isinstance(raw, str):
        value = raw.strip().lower()
        return [value] if value else []
    if isinstance(raw, Sequence):
        normalized: List[str] = []
        for item in raw:
            text = str(item).strip().lower()
            if text:
                normalized.append(text)
        return normalized
    return []


@dataclass
class ValidationStats:
    total: int = 0
    reasoning: int = 0
    accepted: int = 0
    rejected: int = 0

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "reasoning": self.reasoning,
            "accepted": self.accepted,
            "rejected": self.rejected,
        }


class Validator:
    """Dispatch math/code validators based on eval metadata."""

    def __init__(self) -> None:
        self._math = MathAnswerValidator()
        self._code = CodeAnswerValidator()

    def validate(self, protocol: Protocol) -> bool:
        eval_meta = protocol.meta.get("eval") if isinstance(protocol.meta, dict) else None
        eval_types = _normalize_eval_types(eval_meta)
        if not eval_types:
            logger.debug("record_id=%s missing eval type", protocol.record_id)
            return False

        responses = list((protocol.responses or {}).values())
        checks: List[bool] = []
        if "math" in eval_types and isinstance(eval_meta, dict):
            reference = eval_meta.get("answer", "")
            checks.append(self._math.is_correct(responses, reference))
        if "code" in eval_types and isinstance(eval_meta, dict):
            checks.append(self._code.is_correct(responses, eval_meta))

        return bool(checks) and all(checks)


def filter_reasoning_protocols(
    *,
    input_path: Path,
    output_path: Path,
    rejects_path: Path | None = None,
) -> ValidationStats:
    validator = Validator()
    stats = ValidationStats()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reject_handle = None
    if rejects_path:
        rejects_path.parent.mkdir(parents=True, exist_ok=True)
        reject_handle = rejects_path.open("w", encoding="utf-8")

    progress = tqdm(unit="record", desc="Validating protocols", leave=False)

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as accepted_handle:
        for line_number, line in enumerate(source, 1):
            stats.total += 1
            payload = line.strip()
            if not payload:
                progress.update(1)
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse JSON on line %d: %s", line_number, exc)
                progress.update(1)
                continue

            protocol = Protocol.from_dict(record)
            if protocol.domain.strip().lower() != "reasoning":
                progress.update(1)
                continue
            stats.reasoning += 1
            serialized = protocol.to_dict()
            if validator.validate(protocol):
                stats.accepted += 1
                accepted_handle.write(json.dumps(serialized, ensure_ascii=False))
                accepted_handle.write("\n")
            else:
                stats.rejected += 1
                if reject_handle:
                    reject_handle.write(json.dumps(serialized, ensure_ascii=False))
                    reject_handle.write("\n")
            progress.update(1)

    if reject_handle:
        reject_handle.close()
    progress.close()

    return stats


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter reasoning protocols by correctness.")
    parser.add_argument("--input", required=True, type=Path, help="Path to protocols.jsonl file")
    parser.add_argument("--output", required=True, type=Path, help="Path for accepted rows")
    parser.add_argument("--rejects", type=Path, help="Optional path for rejected rows")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    stats = filter_reasoning_protocols(
        input_path=args.input,
        output_path=args.output,
        rejects_path=args.rejects,
    )
    logger.info("Validation summary: %s", stats.as_dict())


if __name__ == "__main__":
    main()
