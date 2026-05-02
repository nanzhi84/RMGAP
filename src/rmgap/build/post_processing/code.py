"""Code answer validation inspired by OpenAI's HumanEval harness."""

from __future__ import annotations

import builtins
import logging
import multiprocessing as mp
import queue
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

SAFE_BUILTIN_NAMES = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "Exception",
    "False",
    "filter",
    "float",
    "int",
    "isinstance",
    "issubclass",
    "len",
    "list",
    "map",
    "max",
    "min",
    "object",
    "pow",
    "print",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "True",
    "tuple",
    "type",
    "ValueError",
    "zip",
}

ALLOWED_MODULES = {"math"}

_SAFE_BUILTINS: Dict[str, Any] = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
_ORIGINAL_IMPORT = builtins.__import__


def _safe_import(name: str, *args: Any, **kwargs: Any):
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(f"Import of module '{name}' is not allowed in sandbox")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


def _build_safe_globals() -> Dict[str, Any]:
    safe_builtins = dict(_SAFE_BUILTINS)
    safe_builtins["__import__"] = _safe_import
    return {"__builtins__": safe_builtins}


def _sandbox_worker(code: str, tests: str, entry_point: str, result_queue: mp.Queue) -> None:
    try:
        candidate_globals: Dict[str, Any] = _build_safe_globals()
        exec(textwrap.dedent(code), candidate_globals)
        candidate = candidate_globals.get(entry_point)
        if not callable(candidate):
            raise ValueError(f"Entry point '{entry_point}' is not defined or not callable")

        tests_globals: Dict[str, Any] = _build_safe_globals()
        exec(textwrap.dedent(tests), tests_globals)
        check_fn = tests_globals.get("check")
        if not callable(check_fn):
            raise ValueError("Test harness missing 'check' callable")

        check_fn(candidate)
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"ok": False, "error": repr(exc)})
        return

    result_queue.put({"ok": True})


def run_in_sandbox(
    *,
    code_snippet: str,
    tests_snippet: str,
    entry_point: str,
    timeout: float = 5.0,
) -> bool:
    """Execute candidate + tests in a child process with limited builtins."""
    result_queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=_sandbox_worker,
        args=(code_snippet, tests_snippet, entry_point, result_queue),
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return False

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        return False
    return bool(result.get("ok"))


def _extract_code_blocks(text: str) -> List[str]:
    return [match.strip() for match in _CODE_BLOCK_RE.findall(text or "") if match.strip()]


@dataclass
class CodeAnswerValidator:
    """Validate code answers using HumanEval-style execution."""

    def is_correct(self, responses: Iterable[str], eval_meta: dict) -> bool:
        entry_point = eval_meta.get("entry_point")
        tests_snippet = eval_meta.get("tests")
        if not isinstance(entry_point, str) or not isinstance(tests_snippet, str):
            logger.debug("Missing code eval metadata: entry_point/tests")
            return False

        for response in responses:
            if not response:
                continue
            for snippet in _extract_code_blocks(response):
                if run_in_sandbox(
                    code_snippet=snippet,
                    tests_snippet=tests_snippet,
                    entry_point=entry_point,
                ):
                    return True
        return False
