"""Centralized state model and routing rules for pipeline stages.

This module defines:
    - phase-specific state labels (res/pro);
    - a declarative mapping from states to eligible stages;
    - a declarative transition table from (state, stage, outcome) to new state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from ...schemas.protocol import Protocol


# Phase keys in Protocol.state
PHASE_RES = "res"
PHASE_PRO = "pro"
PHASE_RW = "rw"

# res-phase states
RES_NOT_GENERATED = "res_not_generated"
RES_GENERATED_NOT_EVALUATED = "res_generated_not_evaluated"
RES_PASSED = "res_passed"
RES_FAILED = "res_failed"

# pro-phase states
PRO_NOT_GENERATED = "pro_not_generated"
PRO_GENERATED_NOT_EVALUATED = "pro_generated_not_evaluated"
PRO_PASSED = "pro_passed"
PRO_FAILED = "pro_failed"

# rw-phase states
RW_NOT_GENERATED = "rw_not_generated"
RW_GENERATED_NOT_FILTERED = "rw_generated_not_filtered"
RW_PASSED = "rw_passed"
RW_FAILED = "rw_failed"

# Stage identifiers
STAGE_RES_GEN = "res_gen"
STAGE_RES_EVAL = "res_eval"
STAGE_PRO_GEN = "pro_gen"
STAGE_PRO_EVAL = "pro_eval"
STAGE_RW_GEN = "rw_gen"
STAGE_RW_EVAL = "rw_eval"


@dataclass(frozen=True)
class StageStateRule:
    """Declarative rule describing which states a stage accepts as input."""

    stage: str
    allowed_res_statuses: Tuple[str, ...] = ()
    allowed_pro_statuses: Tuple[str, ...] = ()
    allowed_rw_statuses: Tuple[str, ...] = ()


# Requirement 5.2: mapping from states to eligible stages.
STAGE_STATE_RULES: Dict[str, StageStateRule] = {
    STAGE_RES_GEN: StageStateRule(
        stage=STAGE_RES_GEN,
        allowed_res_statuses=(
            RES_NOT_GENERATED,
            RES_FAILED,
        ),
    ),
    STAGE_RES_EVAL: StageStateRule(
        stage=STAGE_RES_EVAL,
        allowed_res_statuses=(
            RES_GENERATED_NOT_EVALUATED,
        ),
    ),
    STAGE_PRO_GEN: StageStateRule(
        stage=STAGE_PRO_GEN,
        allowed_res_statuses=(RES_PASSED,),
        allowed_pro_statuses=(
            PRO_NOT_GENERATED,
            PRO_FAILED,
        ),
    ),
    STAGE_PRO_EVAL: StageStateRule(
        stage=STAGE_PRO_EVAL,
        allowed_pro_statuses=(PRO_GENERATED_NOT_EVALUATED,),
    ),
    STAGE_RW_GEN: StageStateRule(
        stage=STAGE_RW_GEN,
        allowed_res_statuses=(RES_PASSED,),
        allowed_pro_statuses=(PRO_PASSED,),
        allowed_rw_statuses=(
            RW_NOT_GENERATED,
            RW_FAILED,
        ),
    ),
    STAGE_RW_EVAL: StageStateRule(
        stage=STAGE_RW_EVAL,
        allowed_res_statuses=(RES_PASSED,),
        allowed_pro_statuses=(PRO_PASSED,),
        allowed_rw_statuses=(
            RW_GENERATED_NOT_FILTERED,
            RW_FAILED,
        ),
    ),
}


# Transition table:
#   STAGE_TRANSITIONS[stage][phase][outcome] = new_state_label
STAGE_TRANSITIONS: Dict[str, Dict[str, Dict[str, str]]] = {
    STAGE_RES_GEN: {
        PHASE_RES: {
            "success": RES_GENERATED_NOT_EVALUATED,
        },
    },
    STAGE_RES_EVAL: {
        PHASE_RES: {
            "success": RES_PASSED,
            "failure": RES_FAILED,
        },
    },
    STAGE_PRO_GEN: {
        PHASE_PRO: {
            "success": PRO_GENERATED_NOT_EVALUATED,
        },
    },
    STAGE_PRO_EVAL: {
        PHASE_PRO: {
            "success": PRO_PASSED,
            "failure": PRO_FAILED,
        },
    },
    STAGE_RW_GEN: {
        PHASE_RW: {
            "success": RW_GENERATED_NOT_FILTERED,
        },
    },
    STAGE_RW_EVAL: {
        PHASE_RW: {
            "success": RW_PASSED,
            "failure": RW_FAILED,
        },
    },
}


def filter_eligible_for_stage(
    protocols: Iterable[Protocol],
    stage: str,
) -> List[Protocol]:
    """Filter protocols that are eligible to be processed by the given stage."""
    rule = STAGE_STATE_RULES[stage]
    eligible: List[Protocol] = []
    for protocol in protocols:
        res_status = protocol.state.get(PHASE_RES)
        pro_status = protocol.state.get(PHASE_PRO)
        rw_status = protocol.state.get(PHASE_RW)

        if rule.allowed_res_statuses and res_status not in rule.allowed_res_statuses:
            continue
        if rule.allowed_pro_statuses and pro_status not in rule.allowed_pro_statuses:
            continue
        if rule.allowed_rw_statuses and rw_status not in rule.allowed_rw_statuses:
            continue
        eligible.append(protocol)
    return eligible


def apply_transition(protocol: Protocol, stage: str, outcome: str) -> None:
    """Apply a state transition for the given stage and outcome.

    This function assumes stage and outcome are configured in STAGE_TRANSITIONS.
    """
    if stage not in STAGE_TRANSITIONS:
        raise ValueError(f"Unknown stage '{stage}' in STAGE_TRANSITIONS")

    stage_rules = STAGE_TRANSITIONS[stage]
    current = dict(protocol.state or {})
    for phase_key, outcomes in stage_rules.items():
        if outcome not in outcomes:
            raise ValueError(
                f"Invalid outcome '{outcome}' for stage '{stage}', phase '{phase_key}'. "
                f"Valid outcomes: {list(outcomes.keys())}"
            )
        current[phase_key] = outcomes[outcome]

    protocol.state = current


__all__ = [
    "PHASE_RES",
    "PHASE_PRO",
    "PHASE_RW",
    "RES_NOT_GENERATED",
    "RES_GENERATED_NOT_EVALUATED",
    "RES_PASSED",
    "RES_FAILED",
    "PRO_NOT_GENERATED",
    "PRO_GENERATED_NOT_EVALUATED",
    "PRO_PASSED",
    "PRO_FAILED",
    "RW_NOT_GENERATED",
    "RW_GENERATED_NOT_FILTERED",
    "RW_PASSED",
    "RW_FAILED",
    "STAGE_RES_GEN",
    "STAGE_RES_EVAL",
    "STAGE_PRO_GEN",
    "STAGE_PRO_EVAL",
    "STAGE_RW_GEN",
    "STAGE_RW_EVAL",
    "StageStateRule",
    "STAGE_STATE_RULES",
    "STAGE_TRANSITIONS",
    "filter_eligible_for_stage",
    "apply_transition",
]


