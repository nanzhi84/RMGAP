"""Write-test stage (writes test.jsonl)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, TYPE_CHECKING

from tqdm import tqdm

from .base import Stage, StageResult
from . import register_stage
from .routing.state_router import (
    PHASE_PRO,
    PHASE_RES,
    PHASE_RW,
    PRO_PASSED,
    RES_PASSED,
    RW_PASSED,
)
from ..schemas.protocol import Protocol
from ..utils.io import write_jsonl

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from ..core.config import Config


logger = logging.getLogger(__name__)


REQUIRED_RESPONSES = ["response1", "response2", "response3", "response4"]
STYLE_ASSIGNMENTS_META_KEY = "style_assignments"
EXPECTED_STYLE_ASSIGNMENT_KEYS = {f"r{i}" for i in range(1, len(REQUIRED_RESPONSES) + 1)}
def _is_eligible_for_test(protocol: Protocol) -> bool:
    """Check if a protocol is eligible for test set writing."""
    res = protocol.responses
    if not all(key in res and res[key] is not None for key in REQUIRED_RESPONSES):
        return False
    
    groups = protocol.pro_gen.get("prompt_groups") or []
    if len(groups) != 4:
        return False

    # Require style assignments metadata to be present and well-formed.
    style_assignments = protocol.meta.get(STYLE_ASSIGNMENTS_META_KEY)
    if not isinstance(style_assignments, dict):
        return False
    if set(style_assignments.keys()) != EXPECTED_STYLE_ASSIGNMENT_KEYS:
        return False

    # Require that each group has a valid winner, base, and two rewrite variants.
    valid_winner_keys = set(REQUIRED_RESPONSES)
    for group in groups:
        winner_key = group.get("winner")
        if winner_key not in valid_winner_keys:
            return False

        base_prompt = (group.get("base_prompt") or "").strip()
        variants = group.get("variants")
        if not isinstance(variants, list):
            return False
        if not base_prompt or len(variants) < 2:
            return False

    # Require that res, pro, and rw phases have all passed their evaluations.
    if protocol.state.get(PHASE_RES) != RES_PASSED:
        return False
    if protocol.state.get(PHASE_PRO) != PRO_PASSED:
        return False
    if protocol.state.get(PHASE_RW) != RW_PASSED:
        return False

    return True


class WriteTestStage(Stage):
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    def filter_eligible(self, protocols: List[Protocol]) -> List[Protocol]:
        # Return only protocols that meet eligibility criteria
        return [p for p in protocols if _is_eligible_for_test(p)]

    def process(self, protocols: List[Protocol]) -> List[Protocol]:
        if not protocols:
            return protocols
        
        rows = []
        for p in tqdm(protocols, desc="Writing test set", unit="item"):
            raw_groups = p.pro_gen.get("prompt_groups") or []

            # Convert internal prompt_groups structure to test.jsonl-compatible format.
            # Internal:
            #   {
            #     "group": int,
            #     "winner": "responseX",
            #     "base_prompt": str,
            #     "variants": [{"id": int, "text": str}, ...],
            #   }
            # Exported:
            #   {
            #     "group": int,
            #     "winner": "responseX",
            #     "prompts": [
            #       {"id": 1, "role": "base", "text": "..."},
            #       {"id": 2, "role": "rewrite", "text": "..."},
            #       {"id": 3, "role": "rewrite", "text": "..."},
            #     ],
            #   }
            groups = []
            for group in raw_groups:
                base_prompt = (group.get("base_prompt") or "").strip()
                variants = group.get("variants") or []

                prompts = []
                if base_prompt:
                    prompts.append(
                        {
                            "role": "base",
                            "text": base_prompt,
                        }
                    )

                for variant in variants[:2]:
                    text = (variant.get("text") or "").strip()
                    prompts.append(
                        {
                            "role": "rewrite",
                            "text": text,
                        }
                    )

                groups.append(
                    {
                        "group": group.get("group"),
                        "winner": group.get("winner"),
                        "prompts": prompts,
                    }
                )

            def _get_text(key: str) -> str:
                return p.responses.get(key)

            responses_list = [
                {"key": key, "text": _get_text(key)}
                for key in REQUIRED_RESPONSES
            ]
            style_assignments = p.meta.get(STYLE_ASSIGNMENTS_META_KEY)

            domain = p.domain
            rows.append({
                "id": p.record_id,
                "domain": domain,
                "source": p.source,
                "models": dict(p.models),
                "responses": responses_list,
                "prompt_groups": groups,
                "style_assignments": style_assignments,
            })
            logger.debug("Added record_id=%s to test set", p.record_id)
        
        # Ensure output directory exists before writing any artifacts
        self.out_dir.mkdir(parents=True, exist_ok=True)
        test_path = self.out_dir / "test.jsonl"
        write_jsonl(test_path, rows)

        return protocols

    def execute(self, protocols: List[Protocol]) -> StageResult:
        eligible_list = self.filter_eligible(protocols)
        # Write test set using only eligible protocols; do not mutate the input list
        self.process(eligible_list)
        return StageResult(protocols, {"total": len(protocols), "eligible": len(eligible_list)})

@register_stage("write_test")
def build_write_test_stage(cfg: "Config") -> Stage:
    """Factory for write_test stage, used by the orchestrator via the registry."""
    from pathlib import Path

    # Mirror orchestrator output-dir resolution so that test.jsonl
    # is written alongside protocols.jsonl.
    if cfg.resume_dir:
        resume_path = Path(cfg.resume_dir)
        out_dir = resume_path if resume_path.is_dir() else resume_path.parent
    else:
        out_dir = Path(cfg.output_dir)
    return WriteTestStage(out_dir)