from __future__ import annotations

from copy import deepcopy

from rmgap.build.pipeline.schemas.protocol import Protocol
from rmgap.build.pipeline.stages.routing.state_router import (
    PHASE_PRO,
    PHASE_RES,
    PHASE_RW,
    PRO_PASSED,
    RES_PASSED,
    RW_PASSED,
)
from rmgap.build.pipeline.stages.write_test import _is_eligible_for_test
from rmgap.build.pipeline.prompts.variables import assign_styles


def _eligible_protocol() -> Protocol:
    return Protocol(
        record_id="record-1",
        domain="Chat",
        state={
            PHASE_RES: RES_PASSED,
            PHASE_PRO: PRO_PASSED,
            PHASE_RW: RW_PASSED,
        },
        responses={f"response{i}": f"response {i}" for i in range(1, 5)},
        pro_gen={
            "prompt_groups": [
                {
                    "group": group_id,
                    "winner": f"response{group_id}",
                    "base_prompt": f"base prompt {group_id}",
                    "variants": [
                        {"id": 1, "text": f"rewrite a {group_id}"},
                        {"id": 2, "text": f"rewrite b {group_id}"},
                    ],
                }
                for group_id in range(1, 5)
            ]
        },
        meta={"style_assignments": assign_styles(123, "record-1")},
    )


def test_write_test_rejects_duplicate_prompt_texts() -> None:
    protocol = _eligible_protocol()
    protocol.pro_gen = deepcopy(protocol.pro_gen)
    protocol.pro_gen["prompt_groups"][0]["variants"][0]["text"] = "base prompt 1"

    assert not _is_eligible_for_test(protocol)


def test_write_test_rejects_low_distance_style_assignments() -> None:
    protocol = _eligible_protocol()
    style = deepcopy(protocol.meta["style_assignments"]["r1"])
    protocol.meta["style_assignments"] = {
        "r1": style,
        "r2": deepcopy(style),
        "r3": deepcopy(protocol.meta["style_assignments"]["r3"]),
        "r4": deepcopy(protocol.meta["style_assignments"]["r4"]),
    }

    assert not _is_eligible_for_test(protocol)


def test_write_test_accepts_unique_prompts_and_distant_styles() -> None:
    assert _is_eligible_for_test(_eligible_protocol())
