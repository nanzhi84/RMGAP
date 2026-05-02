"""Minimal orchestrator using new provider-based stages."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from .config import Config
from .runlog import RunLogger, StageRunner
from ..schemas.loader import load_from_protocols, load_from_raw
from ..schemas.protocol import Protocol
from ..stages import get_stage_factory
from ..stages.base import StageResult
from ..utils.io import write_jsonl


logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run(cfg: Config, stage: str) -> List[Protocol]:
    logger.info("Starting pipeline execution - stage: %s", stage)

    # Load protocols: resume from checkpoint or create from input file
    if cfg.resume_dir:
        result = load_from_protocols(Path(cfg.resume_dir))
        protocols = result.items 
    else:
        result = load_from_raw(
            Path(cfg.input_dir),
            prompt_field=cfg.prompt_field,
            limit=cfg.limit,
        )
        protocols = result.items 

    # Resolve output directory
    # res_gen:
    #   - if --resume provided: overwrite in that directory;
    #   - else: use the specified output_dir directly.
    # other stages:
    #   - require --resume and overwrite in that directory.
    if stage == "res_gen":
        if cfg.resume_dir:
            resume_path = Path(cfg.resume_dir)
            out_dir = resume_path if resume_path.is_dir() else resume_path.parent
            _ensure_dir(out_dir)
        else:
            out_dir = Path(cfg.output_dir)
            _ensure_dir(out_dir)
    else:
        if not cfg.resume_dir:
            raise ValueError("For non-res_gen stages, --resume is required.")
        resume_path = Path(cfg.resume_dir)
        out_dir = resume_path if resume_path.is_dir() else resume_path.parent
        _ensure_dir(out_dir)

    # Resolve stage via registry; orchestrator itself does not know concrete stage classes.
    factory = get_stage_factory(stage)
    stage_obj = factory(cfg)
    runner = StageRunner(RunLogger(out_dir))
    res: StageResult = runner.run(stage, stage_obj, protocols)
    protocols = res.protocols
    
    # Robust metrics logging across stages
    total = int(res.metrics.get("total", 0))
    if "success" in res.metrics and "failed" in res.metrics:
        success = int(res.metrics["success"])
        failed = int(res.metrics["failed"])
        logger.info(
            "Stage %s completed - eligible: %d, success: %d, failed: %d",
            stage, total, success, failed
        )
    elif "eligible" in res.metrics:
        eligible = int(res.metrics["eligible"])
        failed = max(0, total - eligible)
        logger.info(
            "Stage %s completed - total: %d, eligible: %d, failed: %d",
            stage, total, eligible, failed
        )
    else:
        logger.info("Stage %s completed - metrics: %s", stage, res.metrics)

    # Persist protocols.jsonl
    write_jsonl(out_dir / "protocols.jsonl", [p.to_dict() for p in protocols])
    return protocols