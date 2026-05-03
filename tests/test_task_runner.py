from __future__ import annotations

import importlib
import json
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec

import pytest


OPTIONAL_BACKEND_MODULES = {
    "sentence_transformers",
    "sglang",
    "torch",
    "transformers",
}
RMGAP_MODULES_TO_REFRESH = (
    "rmgap.rm.genrm_pointwise",
    "rmgap.rm.base",
    "rmgap.rm",
    "rmgap.provider",
    "rmgap.task_runner",
)


class BlockOptionalBackendImports(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: object | None = None,
    ) -> ModuleSpec | None:
        top_level_name = fullname.split(".", 1)[0]
        if top_level_name in OPTIONAL_BACKEND_MODULES:
            raise ModuleNotFoundError(f"No module named '{top_level_name}'")
        return None


def _clear_modules(
    monkeypatch: pytest.MonkeyPatch,
    module_names: tuple[str, ...],
) -> None:
    for module_name in module_names:
        monkeypatch.delitem(sys.modules, module_name, raising=False)


def test_task_runner_and_genrm_import_without_optional_backend_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_modules(
        monkeypatch,
        (*RMGAP_MODULES_TO_REFRESH, *OPTIONAL_BACKEND_MODULES),
    )
    monkeypatch.setattr(
        sys,
        "meta_path",
        [BlockOptionalBackendImports(), *sys.meta_path],
    )

    task_runner_module = importlib.import_module("rmgap.task_runner")
    rm_module = importlib.import_module("rmgap.rm")

    assert task_runner_module.TaskRunner is not None
    rm = rm_module.make_rm("GenRMPointwise")
    assert type(rm).__name__ == "GenRMPointwise"


def test_load_dataset_rejects_noncanonical_domain(tmp_path) -> None:  # noqa: ANN001
    from rmgap.task_runner import TaskRunner

    dataset_path = tmp_path / "test.jsonl"
    record = {
        "id": "bad-domain",
        "domain": "Reasonning",
        "responses": [
            {"key": f"response{i}", "text": f"response {i}"}
            for i in range(1, 5)
        ],
        "prompt_groups": [
            {
                "group": group_id,
                "winner": "response1",
                "prompts": [
                    {"text": f"prompt {group_id}-{prompt_id}"}
                    for prompt_id in range(1, 4)
                ],
            }
            for group_id in range(1, 5)
        ],
    }
    dataset_path.write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="noncanonical domain 'Reasonning'"):
        TaskRunner()._load_dataset(str(dataset_path))
