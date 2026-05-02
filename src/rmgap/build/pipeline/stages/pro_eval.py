"""Reverse prompt evaluation stage (provider-based, multi-judge)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, TYPE_CHECKING

from tqdm import tqdm

from .base import Stage
from . import register_stage
from .llm_config import JUDGE_CONFIGS
from .routing.state_router import (
    STAGE_PRO_EVAL,
    apply_transition,
    filter_eligible_for_stage,
)
from ..providers import make_provider
from ..providers.base import Provider, ModelParams
from ..schemas.protocol import Protocol
from ..prompts.renderer import load_default_templates

if TYPE_CHECKING:
    from ..core.config import Config


logger = logging.getLogger(__name__)

# Evaluation thresholds
QUALITY_RANGE: tuple[int | None, int | None] = (7, None)
BIAS_RANGE: tuple[int | None, int | None] = (7, None)
SEMANTIC_RANGE: tuple[int | None, int | None] = (7, None)


def _in_range(value: float, rng: tuple[int | float | None, int | float | None]) -> bool:
    """Check if value is within the given range [lo, hi] (inclusive)."""
    lo, hi = rng
    return (lo is None or value >= lo) and (hi is None or value <= hi)


class EvaluateReverseStage(Stage):
    def __init__(self, *, judges: Dict[str, tuple[Provider, ModelParams]], workers: int):
        """Initialize with multiple judge providers.

        Args:
            judges: Dict mapping judge name to (provider, params) tuple.
            workers: Number of parallel workers for protocol processing.
        """
        self.judges = judges
        self.workers = max(1, workers)
        self._latest_metrics: dict[str, Any] = {}

    def filter_eligible(self, protocols: List[Protocol]) -> List[Protocol]:
        """Select protocols eligible for pro_eval based on declarative state rules."""
        eligible = filter_eligible_for_stage(protocols, STAGE_PRO_EVAL)
        logger.info(
            "Filtered %d eligible protocols from %d total for pro_eval",
            len(eligible),
            len(protocols),
        )
        return eligible

    def _call_judge(self, protocol: Protocol, tpl, judge_name: str) -> dict:
        """Call a single judge and return the evaluation result.

        Raises exception on API error or JSON parsing failure.
        """
        provider, params = self.judges[judge_name]
        groups = protocol.pro_gen.get("prompt_groups", [])
        prompts_by_group = {g.get("group"): g.get("base_prompt", "") for g in groups}

        def _get_response(key: str) -> str:
            return protocol.responses.get(key) or ""

        messages = tpl.render(
            R1=_get_response("response1"),
            R2=_get_response("response2"),
            R3=_get_response("response3"),
            R4=_get_response("response4"),
            P1=prompts_by_group.get(1, ""),
            P2=prompts_by_group.get(2, ""),
            P3=prompts_by_group.get(3, ""),
            P4=prompts_by_group.get(4, ""),
        )
        return provider.generate(
            messages=messages,
            context=f"pro_eval:{protocol.record_id}:{judge_name}",
        )

    def _process_one(self, protocol: Protocol, tpl) -> bool:
        """Wrapper that calls _one and handles state transition."""
        passed = self._one(protocol, tpl)
        outcome = "success" if passed else "failure"
        apply_transition(protocol, STAGE_PRO_EVAL, outcome)
        return passed

    def _one(self, protocol: Protocol, tpl) -> bool:
        """Core multi-judge evaluation logic for a single protocol.

        Decision logic:
            - If both reviewers pass: final_decision = True
            - If both reviewers fail: final_decision = False
            - If reviewers disagree: AC makes the final decision

        Returns True if the protocol passes evaluation, False otherwise.
        Raises exception if any judge fails (API error, JSON parsing), leaving state unchanged.
        """
        result: Dict[str, Any] = {}

        # Identify reviewers and AC from judge configs
        reviewer_names = [
            name for name, cfg in JUDGE_CONFIGS.items() if cfg["role"] == "reviewer"
        ]
        ac_name = next(name for name, cfg in JUDGE_CONFIGS.items() if cfg["role"] == "ac")

        # Step 1: Call both reviewers (exception propagates up, aborting evaluation)
        reviewer_passed: List[bool] = []
        for judge_name in reviewer_names:
            res = self._call_judge(protocol, tpl, judge_name)
            passed = self._check_pass_criteria(res)
            result[judge_name] = res
            reviewer_passed.append(passed)

        # Step 2: Determine consensus or disagreement
        if all(reviewer_passed):
            result["ac"] = None
            result["final_decision"] = True
            result["decision_path"] = "unanimous_pass"
        elif not any(reviewer_passed):
            result["ac"] = None
            result["final_decision"] = False
            result["decision_path"] = "unanimous_fail"
        else:
            # Disagreement: call AC for final decision
            ac_res = self._call_judge(protocol, tpl, ac_name)
            ac_passed = self._check_pass_criteria(ac_res)
            result["ac"] = ac_res
            result["final_decision"] = ac_passed
            result["decision_path"] = "ac_decision"

        protocol.pro_eval = result
        return result["final_decision"]

    def process(self, protocols: List[Protocol]) -> List[Protocol]:
        if not protocols:
            self._latest_metrics = {"passed": 0, "passed_ids": []}
            return []

        tpl = load_default_templates()["reverse_evaluation"]
        out: List[Protocol] = []
        passed = 0
        success = 0
        failed = 0
        total = len(protocols)
        passed_ids: set[str] = set()
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            fut = {ex.submit(self._process_one, p, tpl): p for p in protocols}
            with tqdm(total=total, desc="Evaluating reverse prompts", unit="item") as pbar:
                for f in as_completed(fut):
                    proto = fut[f]
                    try:
                        passed_criteria = f.result()
                        if passed_criteria:
                            passed += 1
                            passed_ids.add(proto.record_id)
                        success += 1
                        out.append(proto)
                    except Exception as exc:
                        failed += 1
                        logger.warning("Reverse eval failed for record_id=%s: %s", proto.record_id, exc)
                    pbar.update(1)
        logger.info(
            "pro_eval completed - total=%d, success=%d, failed=%d, passed=%d",
            total,
            success,
            failed,
            passed,
        )
        self._latest_metrics = {"passed": passed, "passed_ids": sorted(passed_ids)}
        return out

    def _check_pass_criteria(self, res: dict) -> bool:
        """Check if evaluation result passes all criteria."""
        quality_block = res.get("quality_scores") or {}
        bias_block = res.get("bias_scores") or {}
        semantic_block = res.get("semantic_scores") or {}

        keys = ("p1", "p2", "p3", "p4")
        quality_values = [int(quality_block.get(key) or 0) for key in keys]
        bias_values = [int(bias_block.get(key) or 0) for key in keys]
        semantic_values = [int(semantic_block.get(key) or 0) for key in keys]

        avg_quality = sum(quality_values) / len(quality_values)
        avg_bias = sum(bias_values) / len(bias_values)
        avg_semantic = sum(semantic_values) / len(semantic_values)

        quality_is_acceptable = _in_range(avg_quality, QUALITY_RANGE)
        bias_is_acceptable = _in_range(avg_bias, BIAS_RANGE)
        semantic_is_acceptable = _in_range(avg_semantic, SEMANTIC_RANGE)

        return quality_is_acceptable and bias_is_acceptable and semantic_is_acceptable


@register_stage(STAGE_PRO_EVAL)
def build_pro_eval_stage(cfg: "Config") -> Stage:
    """Factory for pro_eval stage, used by the orchestrator via the registry."""
    judges: Dict[str, tuple[Provider, ModelParams]] = {}
    for judge_name, judge_cfg in JUDGE_CONFIGS.items():
        params = judge_cfg["params"]
        provider = make_provider(judge_cfg["provider_name"], params)
        judges[judge_name] = (provider, params)

    return EvaluateReverseStage(
        judges=judges,
        workers=cfg.max_workers,
    )
