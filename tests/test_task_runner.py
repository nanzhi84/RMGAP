from __future__ import annotations

import json

import pytest

from rmgap.task_runner import TaskRunner


def test_load_dataset_rejects_noncanonical_domain(tmp_path) -> None:  # noqa: ANN001
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
