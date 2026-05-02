from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rmgap.build.pipeline.schemas.protocol import Protocol
from rmgap.build.post_processing.validator import Validator, filter_reasoning_protocols

MATH_PROMPT = (
    "If $f(x) = \\frac{3x-2}{x-2}$, what is the value of $f(-2) +f(-1)+f(0)$? "
    "Express your answer as a common fraction."
)
MATH_EVAL = {"type": "math", "answer": "\\frac{14}{3}"}

CODE_PROMPT = (
    "from typing import List\n\n"
    "\n"
    "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
    "    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n"
    "    given threshold.\n"
    "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
    "    False\n"
    "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
    "    True\n"
    "    \"\"\"\n"
)
CODE_EVAL = {
    "type": "code",
    "entry_point": "has_close_elements",
    "tests": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
}


GOOD_CODE = """
import math

def has_close_elements(numbers, threshold):
    n = len(numbers)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
"""

BAD_CODE = """
def has_close_elements(numbers, threshold):
    return False
"""

TIMEOUT_CODE = """
def has_close_elements(numbers, threshold):
    while True:
        pass
"""

IMPORT_CODE = """
import os

def has_close_elements(numbers, threshold):
    return len(os.listdir('.')) == 0
"""


def _build_math_protocol(
    record_id: str, response: str, *, answer: str | None = None
) -> Protocol:
    return Protocol(
        record_id=record_id,
        original_prompt=MATH_PROMPT,
        domain="Reasoning",
        responses={"response1": response},
        meta={
            "eval": {
                **MATH_EVAL,
                **({"answer": answer} if answer is not None else {}),
            }
        },
    )


def _build_code_protocol(record_id: str, response: str) -> Protocol:
    return Protocol(
        record_id=record_id,
        original_prompt=CODE_PROMPT,
        domain="Reasoning",
        responses={"response1": response},
        meta={"eval": dict(CODE_EVAL)},
    )


class ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = Validator()

    def test_math_validator_accepts_boxed_answer(self) -> None:
        proto = _build_math_protocol("math-pass", "Solution: \\boxed{\\frac{14}{3}}")
        self.assertTrue(self.validator.validate(proto))

    def test_math_validator_rejects_incorrect_value(self) -> None:
        proto = _build_math_protocol("math-fail", "Final: \\boxed{5}")
        self.assertFalse(self.validator.validate(proto))

    def test_math_validator_handles_nested_wrappers(self) -> None:
        proto = _build_math_protocol(
            "math-nested",
            "Answer: \\boxed{\\left(\\frac{14}{3}\\right)}",
        )
        self.assertTrue(self.validator.validate(proto))

    def test_math_validator_handles_degree_suffix(self) -> None:
        proto = _build_math_protocol(
            "math-degree",
            "Angle \\boxed{90}",
            answer="90^{\\circ}",
        )
        self.assertTrue(self.validator.validate(proto))

    def test_math_validator_accepts_equation_format(self) -> None:
        proto = _build_math_protocol(
            "math-equation",
            "Result \\boxed{5}",
            answer="x = 5",
        )
        self.assertTrue(self.validator.validate(proto))

    def test_math_validator_requires_boxed_notation(self) -> None:
        proto = _build_math_protocol("math-no-box", "Final answer 14/3")
        self.assertFalse(self.validator.validate(proto))

    def test_math_validator_passes_when_any_box_matches(self) -> None:
        proto = _build_math_protocol(
            "math-multi-box",
            "Try values \\boxed{0} and \\boxed{\\frac{14}{3}}",
        )
        self.assertTrue(self.validator.validate(proto))

    def test_math_validator_matches_unordered_sets(self) -> None:
        proto = _build_math_protocol(
            "math-set",
            "Solutions \\boxed{-2, 1}",
            answer="1,-2",
        )
        self.assertTrue(self.validator.validate(proto))

    def test_math_validator_expands_pm_notation(self) -> None:
        proto = _build_math_protocol(
            "math-pm",
            "Roots \\boxed{1+\\sqrt{19}, 1-\\sqrt{19}}",
            answer="1 \\pm \\sqrt{19}",
        )
        self.assertTrue(self.validator.validate(proto))

    def test_code_validator_accepts_passing_solution(self) -> None:
        proto = _build_code_protocol(
            "code-pass",
            f"Working implementation:\n```python\n{GOOD_CODE}\n```",
        )
        self.assertTrue(self.validator.validate(proto))

    def test_code_validator_rejects_failing_solution(self) -> None:
        proto = _build_code_protocol(
            "code-fail",
            f"Always false:\n```python\n{BAD_CODE}\n```",
        )
        self.assertFalse(self.validator.validate(proto))

    def test_code_validator_requires_code_block(self) -> None:
        proto = _build_code_protocol("code-no-block", GOOD_CODE)
        self.assertFalse(self.validator.validate(proto))

    def test_code_validator_rejects_timeout(self) -> None:
        proto = _build_code_protocol(
            "code-timeout",
            f"Infinite loop:\n```python\n{TIMEOUT_CODE}\n```",
        )
        self.assertFalse(self.validator.validate(proto))

    def test_code_validator_blocks_imports(self) -> None:
        proto = _build_code_protocol(
            "code-import",
            f"Imports disallowed:\n```python\n{IMPORT_CODE}\n```",
        )
        self.assertFalse(self.validator.validate(proto))

    def test_filter_reasoning_protocols_outputs_only_valid_records(self) -> None:
        valid_proto = _build_math_protocol("math-pass", "Result \\boxed{\\frac{14}{3}}")
        invalid_proto = _build_code_protocol(
            "code-fail",
            f"Broken:\n```python\n{BAD_CODE}\n```",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "protocols.jsonl"
            output_path = tmp_path / "accepted.jsonl"
            rejects_path = tmp_path / "rejected.jsonl"

            with input_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(valid_proto.to_dict(), ensure_ascii=False) + "\n")
                handle.write(json.dumps(invalid_proto.to_dict(), ensure_ascii=False) + "\n")

            stats = filter_reasoning_protocols(
                input_path=input_path,
                output_path=output_path,
                rejects_path=rejects_path,
            )

            self.assertEqual(stats.reasoning, 2)
            self.assertEqual(stats.accepted, 1)
            self.assertEqual(stats.rejected, 1)

            accepted_rows = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(accepted_rows), 1)
            accepted = json.loads(accepted_rows[0])
            self.assertEqual(accepted["record_id"], "math-pass")

            rejected_rows = rejects_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rejected_rows), 1)
            rejected = json.loads(rejected_rows[0])
            self.assertEqual(rejected["record_id"], "code-fail")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
