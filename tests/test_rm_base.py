from __future__ import annotations

from typing import Any, List

import pytest

from rmgap.rm.base import ChatMessage, GenerativeRM


class DummyGenerativeRM(GenerativeRM):
    DEFAULT_RESULT = 0.0

    def _prepare(self, data: List[dict[str, Any]]) -> tuple[List[List[ChatMessage]], list[int]]:
        return [], []

    def _parse_generated_text(self, text: str) -> float | None:
        return None

    def _run_with_sglang(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("Unknown backends must not fall through to sglang.")


def test_unknown_backend_raises_before_sglang() -> None:
    rm = DummyGenerativeRM()

    with pytest.raises(ValueError, match="Unsupported RM backend 'open_ai'"):
        rm(
            model_path="model",
            data=[],
            sglang_cfg={"backend": "open_ai"},
        )

