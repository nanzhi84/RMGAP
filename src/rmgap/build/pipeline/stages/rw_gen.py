"""Rewrite prompt generation stage (provider-based)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List

from typing import TYPE_CHECKING

from tqdm import tqdm


from .base import Stage
from . import register_stage
from .llm_config import (
    GenerationEntry,
    GENERATION_MODEL_POOL,
    select_generation_model,
)
from .routing.state_router import (
    STAGE_RW_GEN,
    apply_transition,
    filter_eligible_for_stage,
)
from ..providers import make_provider
from ..providers.base import Provider
from ..schemas.protocol import Protocol
from ..prompts.renderer import load_default_templates

if TYPE_CHECKING:  # imported only for type checking
    from ..core.config import Config


logger = logging.getLogger(__name__)


class RewriteGenerateStage(Stage):
    def __init__(
        self,
        *,
        workers: int,
        stage_key: str,
        model_selector: Callable[..., GenerationEntry],
    ):
        self.workers = max(1, workers)
        self.stage_key = stage_key
        self._model_selector = model_selector
        self._provider_cache: Dict[str, Provider] = {}

    def _get_provider(self, entry: GenerationEntry) -> Provider:
        cache_key = f"{entry['provider_name']}::{entry['params'].name}"
        if cache_key not in self._provider_cache:
            self._provider_cache[cache_key] = make_provider(
                entry["provider_name"], entry["params"]
            )
        return self._provider_cache[cache_key]

    def filter_eligible(self, protocols: List[Protocol]) -> List[Protocol]:
        """Select protocols eligible for rw_gen based on declarative state rules."""
        return filter_eligible_for_stage(protocols, STAGE_RW_GEN)

    def _process_one(self, protocol: Protocol, tpl) -> None:
        """Wrapper that calls _one and handles state transition."""
        self._one(protocol, tpl)
        apply_transition(protocol, STAGE_RW_GEN, "success")

    def _one(self, protocol: Protocol, tpl) -> None:
        """Generate rewrite variants for each base prompt in prompt_groups."""
        groups = protocol.pro_gen.get("prompt_groups", [])
        if not groups:
            raise ValueError(
                f"rw_gen requires non-empty prompt_groups (record_id={protocol.record_id})"
            )

        model_entry = self._model_selector(stage_key=self.stage_key, record_id=protocol.record_id)

        def _generate_with_fallback(*, messages: list[dict[str, str]], context: str) -> str:
            candidates = [model_entry] + [
                entry for entry in GENERATION_MODEL_POOL
                if entry["params"].name != model_entry["params"].name
            ]
            for entry in candidates:
                provider = self._get_provider(entry)
                result = provider.generate(
                    messages=messages,
                    context=context,
                    expect_json=False,
                )
                if isinstance(result, str) and result.strip():
                    protocol.models[self.stage_key] = entry["params"].name
                    return result.strip()
            raise ValueError(
                f"rw_gen received empty content from all generation models "
                f"(record_id={protocol.record_id}, context={context})."
            )

        updated_groups: list[dict] = []
        for group in groups:
            base_prompt = group.get("base_prompt", "")
            group_id = group.get("group")
            if not isinstance(base_prompt, str) or not base_prompt.strip():
                raise ValueError(
                    f"rw_gen requires non-empty base_prompt for group={group_id} "
                    f"(record_id={protocol.record_id})."
                )

            existing_variants = group.get("variants") or []
            valid_existing = [
                v for v in existing_variants
                if (v or {}).get("text") and str((v or {}).get("text")).strip()
            ]
            need = max(0, 2 - len(valid_existing))

            variants: list[dict] = list(valid_existing)
            for idx in range(need):
                messages = tpl.render(
                    BASE_PROMPT=base_prompt,
                )
                generated = _generate_with_fallback(
                    messages=messages,
                    context=f"rw_gen:{protocol.record_id}:{group_id}:{len(variants) + 1}",
                )
                if not isinstance(generated, str):
                    raise ValueError(
                        f"rw_gen expected string variant for group={group_id}, idx={len(variants) + 1} "
                        f"(record_id={protocol.record_id})."
                    )
                text = generated.strip()
                if not text:
                    raise ValueError(
                        f"rw_gen returned empty variant for group={group_id}, idx={len(variants) + 1} "
                        f"(record_id={protocol.record_id})."
                    )
                variants.append({"id": len(variants) + 1, "text": text})

            if len(variants) != 2:
                updated_groups.append(group)
                continue

            group = dict(group)
            group["variants"] = variants[:2]
            updated_groups.append(group)

        # Final validation: each group must have two non-empty variants
        for g in updated_groups:
            vts = g.get("variants") or []
            if len(vts) != 2 or any(not (vt.get("text") or "").strip() for vt in vts):
                protocol.pro_gen["prompt_groups"] = updated_groups
                raise ValueError(
                    f"rw_gen incomplete variants for group={g.get('group')} "
                    f"(record_id={protocol.record_id})."
                )

        protocol.pro_gen["prompt_groups"] = updated_groups
    
    def process(self, protocols: List[Protocol]) -> List[Protocol]:
        if not protocols:
            return []

        tpl = load_default_templates()["rewrite_generation"]
        out: List[Protocol] = []
        success = 0
        failed = 0
        total = len(protocols)
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            fut = {ex.submit(self._process_one, p, tpl): p for p in protocols}
            with tqdm(total=total, desc="Generating rewrite prompts", unit="item") as pbar:
                for f in as_completed(fut):
                    proto = fut[f]
                    try:
                        f.result()
                        out.append(proto)
                    except Exception as exc:
                        failed += 1
                        logger.warning(
                            "Rewrite gen failed for record_id=%s: %s",
                            proto.record_id,
                            exc,
                        )
                    else:
                        success += 1
                    pbar.update(1)
        logger.info(
            "rw_gen completed - total=%d, success=%d, failed=%d",
            total,
            success,
            failed,
        )
        return out


@register_stage(STAGE_RW_GEN)
def build_rw_gen_stage(cfg: "Config") -> Stage:
    """Factory for rw_gen stage, used by the orchestrator via the registry."""
    return RewriteGenerateStage(
        workers=cfg.max_workers,
        stage_key="rw",
        model_selector=select_generation_model,
    )