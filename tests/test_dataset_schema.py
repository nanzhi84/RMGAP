from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def test_released_dataset_schema() -> None:
    dataset_path = Path("data/test.jsonl")
    records = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(records) == 1097
    assert Counter(record["domain"] for record in records) == {
        "Chat": 397,
        "Reasoning": 225,
        "Safety": 222,
        "Writing": 253,
    }

    for record in records:
        responses = record["responses"]
        prompt_groups = record["prompt_groups"]
        response_keys = {response["key"] for response in responses}

        assert len(responses) == 4
        assert len(response_keys) == 4
        assert len(prompt_groups) == 4

        for response in responses:
            assert response["text"].strip()

        for group in prompt_groups:
            assert group["winner"] in response_keys
            assert len(group["prompts"]) == 3
            for prompt in group["prompts"]:
                assert prompt["text"].strip()
