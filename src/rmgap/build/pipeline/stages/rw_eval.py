"""Rewrite prompts semantic similarity evaluation stage."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, TYPE_CHECKING

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .base import Stage
from . import register_stage
from .routing.state_router import (
    STAGE_RW_EVAL,
    apply_transition,
    filter_eligible_for_stage,
)
from ..schemas.protocol import Protocol

if TYPE_CHECKING:  # imported only for type checking
    from ..core.config import Config


logger = logging.getLogger(__name__)


EMBEDDING_MODEL_ENV = "RMGAP_EMBEDDING_MODEL_PATH"
SIMILARITY_THRESHOLD: float = 0.7


def _resolve_embedding_model_path(configured_path: str | None) -> str:
    """Resolve the embedding model path from config or environment."""
    if configured_path:
        return configured_path
    env_path = os.getenv(EMBEDDING_MODEL_ENV)
    if env_path:
        return env_path
    raise ValueError(
        "rw_eval requires `embedding_model_path` in generate/config.yaml "
        f"or the {EMBEDDING_MODEL_ENV} environment variable."
    )


def _load_embedding_model(model_path: str) -> SentenceTransformer:
    """Load embedding model from local directory."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding model directory not found at {model_path}. "
            f"Please ensure the embedding model is downloaded to this location."
        )
    return SentenceTransformer(str(path))


def _min_pairwise_similarity(
    model: SentenceTransformer,
    device: torch.device,
    texts: list[str],
) -> float:
    """Compute the minimum pairwise cosine similarity among a list of texts."""
    if len(texts) < 2:
        return 0.0

    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=False,
    )
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.T)

    # Ignore self-similarity on the diagonal by setting them to +1.
    num = sim_matrix.size(0)
    sim_matrix = sim_matrix.clone()
    sim_matrix[torch.arange(num), torch.arange(num)] = 1.0

    min_sim = sim_matrix.min().item()
    return float(min_sim)


class RewriteSimilarityEvalStage(Stage):
    """Stage that filters rewrite variants based on semantic similarity to the base prompt."""

    def __init__(self, model_path: str | None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _load_embedding_model(
            _resolve_embedding_model_path(model_path)
        )
        self.model.to(self.device)
        self._latest_metrics: dict[str, Any] = {}

    def filter_eligible(self, protocols: List[Protocol]) -> List[Protocol]:
        """Select protocols eligible for rw_eval based on declarative state rules."""
        return filter_eligible_for_stage(protocols, STAGE_RW_EVAL)

    def _process_one(self, protocol: Protocol) -> bool:
        """Wrapper that calls _one and handles state transition."""
        passed = self._one(protocol)
        outcome = "success" if passed else "failure"
        apply_transition(protocol, STAGE_RW_EVAL, outcome)
        return passed

    def _one(self, protocol: Protocol) -> bool:
        """Core logic for evaluating rewrite variants.

        Returns:
            True if every group has three prompts (base + two variants)
            whose pairwise similarities all exceed the threshold.
        """
        raw_groups = protocol.pro_gen.get("prompt_groups", [])
        if not isinstance(raw_groups, list) or not raw_groups:
            return False

        any_group_failed = False
        updated_groups: list[dict] = []
        per_group_similarity: list[dict[str, Any]] = []

        for group in raw_groups:
            if not isinstance(group, dict):
                any_group_failed = True
                continue

            base_prompt = str(group.get("base_prompt") or "").strip()
            variants_raw = group.get("variants") or []
            if not isinstance(variants_raw, list):
                any_group_failed = True
                updated_groups.append(group)
                continue

            # We expect exactly three prompts per group: base + two variants.
            if not base_prompt or len(variants_raw) < 2:
                any_group_failed = True
                updated_groups.append(group)
                continue

            # Take the first two variants (rw_gen is expected to produce exactly two).
            variant_texts: list[str] = []
            for variant in variants_raw[:2]:
                if not isinstance(variant, dict):
                    any_group_failed = True
                    break
                text = str(variant.get("text") or "").strip()
                if not text:
                    any_group_failed = True
                    break
                variant_texts.append(text)

            if len(variant_texts) != 2:
                updated_groups.append(group)
                continue

            all_texts = [base_prompt] + variant_texts
            min_sim = _min_pairwise_similarity(self.model, self.device, all_texts)

            updated_groups.append(dict(group))
            per_group_similarity.append(
                {
                    "group": group.get("group"),
                    "min_pairwise_similarity": min_sim,
                }
            )

            if min_sim < SIMILARITY_THRESHOLD:
                any_group_failed = True

        protocol.pro_gen["prompt_groups"] = updated_groups
        overall_min = min(
            (entry["min_pairwise_similarity"] for entry in per_group_similarity),
            default=None,
        )
        # Flatten per-group similarity into top-level keys for simpler downstream use.
        rw_eval_payload: dict[str, Any] = {
            "overall_min_pairwise_similarity": overall_min,
        }
        for entry in per_group_similarity:
            group_id = entry.get("group")
            key = f"group_{group_id}_min_pairwise_similarity"
            rw_eval_payload[key] = entry.get("min_pairwise_similarity")

        protocol.rw_eval = rw_eval_payload
        return not any_group_failed

    def process(self, protocols: List[Protocol]) -> List[Protocol]:
        if not protocols:
            self._latest_metrics = {"passed": 0, "passed_ids": []}
            return []

        out: List[Protocol] = []
        passed = 0
        success = 0
        failed = 0
        total = len(protocols)
        passed_ids: set[str] = set()

        for protocol in tqdm(protocols, desc="Evaluating rewrite similarity", unit="item"):
            try:
                passed_criteria = self._process_one(protocol)
                if passed_criteria:
                    passed += 1
                    passed_ids.add(protocol.record_id)
                success += 1
                out.append(protocol)
            except Exception as exc:  # defensive logging; state transition may be incomplete
                failed += 1
                logger.warning("rw_eval failed for record_id=%s: %s", protocol.record_id, exc)

        logger.info(
            "rw_eval completed - total=%d, success=%d, failed=%d, passed=%d",
            total,
            success,
            failed,
            passed,
        )
        self._latest_metrics = {"passed": passed, "passed_ids": sorted(passed_ids)}
        return out


@register_stage(STAGE_RW_EVAL)
def build_rw_eval_stage(cfg: "Config") -> Stage:
    """Factory for rw_eval stage, used by the orchestrator via the registry."""
    return RewriteSimilarityEvalStage(model_path=cfg.embedding_model_path)
