from __future__ import annotations

import torch

from rmgap.build.pipeline.stages.rw_eval import _min_base_to_variant_similarity


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):  # noqa: ANN001, ANN003
        del kwargs
        vectors = {
            "base": [1.0, 0.0],
            "rewrite-a": [0.8, 0.6],
            "rewrite-b": [0.8, -0.6],
        }
        return torch.tensor([vectors[text] for text in texts], dtype=torch.float32)


def test_similarity_checks_base_to_each_variant_only() -> None:
    score = _min_base_to_variant_similarity(
        FakeEmbeddingModel(),
        torch.device("cpu"),
        "base",
        ["rewrite-a", "rewrite-b"],
    )

    assert score == torch.tensor(0.8).item()
