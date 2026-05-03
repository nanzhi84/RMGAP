"""Code answer validation entry point."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, List

logger = logging.getLogger(__name__)

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def run_in_sandbox(
    *,
    code_snippet: str,
    tests_snippet: str,
    entry_point: str,
    timeout: float = 5.0,
) -> bool:
    """Fail closed until a real isolated code runner is configured."""
    del code_snippet, tests_snippet, entry_point, timeout
    logger.warning(
        "Code validation is disabled because this repository does not provide an "
        "OS-isolated execution sandbox."
    )
    return False


def _extract_code_blocks(text: str) -> List[str]:
    return [match.strip() for match in _CODE_BLOCK_RE.findall(text or "") if match.strip()]


@dataclass
class CodeAnswerValidator:
    """Reject code answers until an OS-isolated runner is available."""

    def is_response_correct(self, response: str, eval_meta: dict) -> bool:
        entry_point = eval_meta.get("entry_point")
        tests_snippet = eval_meta.get("tests")
        if not isinstance(entry_point, str) or not isinstance(tests_snippet, str):
            logger.debug("Missing code eval metadata: entry_point/tests")
            return False

        for snippet in _extract_code_blocks(response):
            if run_in_sandbox(
                code_snippet=snippet,
                tests_snippet=tests_snippet,
                entry_point=entry_point,
            ):
                return True
        return False

    def is_correct(self, responses: Iterable[str], eval_meta: dict) -> bool:
        response_list = list(responses)
        if not response_list:
            return False
        return all(
            bool(response) and self.is_response_correct(response, eval_meta)
            for response in response_list
        )
