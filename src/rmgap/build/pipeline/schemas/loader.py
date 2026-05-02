"""Data loading from input file."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List

from .protocol import Protocol, RawRecord

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Result of loading data from a JSONL file."""
    items: List  # Can be List[RawRecord] or List[Protocol]
    total_read: int
    limit: int 


def _read_jsonl_dicts(path: Path) -> Iterator[Dict]:
    """Read JSONL file and yield dict objects, skipping invalid lines."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    text = path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        if not raw.strip():
            continue
        
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                yield data
        except json.JSONDecodeError as exc:
            logger.warning("JSON parsing failed for a line: %s", exc)


def _build_raw_record(data: Dict, prompt_field: str) -> RawRecord | None:
    """Build RawRecord assuming dictionary input; return None if invalid."""
    prompt = str(data.get(prompt_field, "")).strip()
    if not prompt or prompt == "nan":
        return None

    normalized = " ".join(prompt.split())
    hash_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    domain = str(data.get("domain", "")).strip()
    source = str(data.get("source", "")).strip()
    return RawRecord(record_id=hash_id, prompt=prompt, source=source, domain=domain)


def load_from_protocols(path: Path) -> LoadResult:
    """Load Protocol objects from a protocols JSONL file (for resume functionality)."""
    protocols = []
    total_read = 0
    
    for data in _read_jsonl_dicts(path):
        protocol = Protocol.from_dict(data)
        if not protocol.record_id or not protocol.original_prompt:
            continue
        
        protocols.append(protocol)
        total_read += 1
    
    logger.info("Loaded %d protocols from resume file", len(protocols))
    return LoadResult(items=protocols, total_read=total_read, limit=0)


def load_from_raw(
    input_file: Path,
    *,
    prompt_field: str,
    limit: int,
) -> LoadResult:
    """Load Protocol objects from raw input JSONL file."""
    protocols = []
    total_read = 0
    
    for data in _read_jsonl_dicts(input_file):
        record = _build_raw_record(data, prompt_field)
        if not record:
            continue
        
        protocol = Protocol(
            record_id=record.record_id,
            original_prompt=record.prompt,
            source=record.source,
            domain=record.domain,
            meta={
                "eval": data.get("eval"),
            },
        )
        protocols.append(protocol)
        total_read += 1
        
        if limit and len(protocols) >= limit:
            logger.info("Reached limit=%d, stopping (raw: %d)", limit, total_read)
            break
    
    logger.info("Loading complete - loaded: %d", len(protocols))
    return LoadResult(items=protocols, total_read=total_read, limit=limit)