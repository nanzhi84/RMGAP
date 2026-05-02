"""Reverse prompt generation stage (provider-based)."""

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
    STAGE_PRO_GEN,
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


class ReverseGenerateStage(Stage):
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
        """Select protocols eligible for pro_gen based on declarative state rules."""
        return filter_eligible_for_stage(protocols, STAGE_PRO_GEN)

    def _process_one(self, protocol: Protocol, tpl) -> None:
        """Wrapper that calls _one and handles state transition."""
        self._one(protocol, tpl)
        apply_transition(protocol, STAGE_PRO_GEN, "success")

    def process(self, protocols: List[Protocol]) -> List[Protocol]:
        if not protocols:
            return []
        
        tpl = load_default_templates()["reverse_generation"]
        out: List[Protocol] = []
        success = 0
        failed = 0
        total = len(protocols)
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            fut = {ex.submit(self._process_one, p, tpl): p for p in protocols}
            with tqdm(total=total, desc="Generating reverse prompts", unit="item") as pbar:
                for f in as_completed(fut):
                    proto = fut[f]
                    try:
                        f.result()
                        out.append(proto)
                    except Exception as exc:
                        failed += 1
                        logger.warning("Reverse gen failed for record_id=%s: %s", proto.record_id, exc)
                    else:
                        success += 1
                    pbar.update(1)
        logger.info(
            "pro_gen completed - total=%d, success=%d, failed=%d",
            total,
            success,
            failed,
        )
        return out

    def _one(self, protocol: Protocol, tpl) -> None:
        """Generate one biased prompt per response (single-call polling)."""
        model_entry = self._model_selector(stage_key=self.stage_key, record_id=protocol.record_id)

        responses = protocol.responses or {}
        response_keys = [f"response{i}" for i in range(1, 5)]
        missing_resp = [key for key in response_keys if not (responses.get(key) or "").strip()]
        if missing_resp:
            raise ValueError(
                f"pro_gen requires complete responses for keys {missing_resp!r} "
                f"(record_id={protocol.record_id})."
            )

        existing_groups = protocol.pro_gen.get("prompt_groups") or []
        existing_by_winner = {
            g.get("winner"): g for g in existing_groups if (g or {}).get("base_prompt")
        }

        def _get_response(key: str) -> str:
            return str(responses.get(key) or "")

        context_prefix = f"pro_gen:{protocol.record_id}"
        prompt_groups: dict[str, dict] = dict(existing_by_winner)

        pending = [
            key for key in response_keys
            if not (existing_by_winner.get(key, {}).get("base_prompt") or "").strip()
        ]

        def _generate_with_fallback(*, messages: list[dict[str, str]], context: str) -> str:
            """Call provider; if empty string, try other models in pool."""
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
                f"pro_gen received empty content from all generation models "
                f"(record_id={protocol.record_id}, context={context})."
            )

        if pending:
            with ThreadPoolExecutor(max_workers=len(pending)) as executor:
                futures = {}
                for key in pending:
                    group_index = int(key.removeprefix("response"))
                    variables = {
                        "ORIGINAL_PROMPT": protocol.original_prompt or "",
                        "TARGET_ID": f"R{group_index}",
                        "R1": _get_response("response1"),
                        "R2": _get_response("response2"),
                        "R3": _get_response("response3"),
                        "R4": _get_response("response4"),
                    }
                    messages = tpl.render(**variables)
                    future = executor.submit(
                        _generate_with_fallback,
                        messages=messages,
                        context=f"{context_prefix}:{key}",
                    )
                    futures[future] = (group_index, key)

                for future in as_completed(futures):
                    group_index, key = futures[future]
                    generated = future.result()
                    if not isinstance(generated, str):
                        raise ValueError(
                            f"reverse_generation returned non-string for {key} "
                            f"(record_id={protocol.record_id})."
                        )
                    base_prompt = generated.strip()
                    if not base_prompt:
                        raise ValueError(
                            f"reverse_generation returned empty prompt for {key} "
                            f"(record_id={protocol.record_id})."
                        )
                    prompt_groups[key] = {
                        "group": group_index,
                        "winner": key,
                        "base_prompt": base_prompt,
                    }

        missing_after = [
            key for key in response_keys
            if not (prompt_groups.get(key, {}).get("base_prompt") or "").strip()
        ]
        if missing_after:
            protocol.pro_gen["prompt_groups"] = sorted(
                prompt_groups.values(), key=lambda item: item["group"]
            ) if prompt_groups else existing_groups
            raise ValueError(
                f"reverse_generation missing prompts {missing_after!r} "
                f"(record_id={protocol.record_id})."
            )

        protocol.pro_gen = {
            "prompt_groups": sorted(prompt_groups.values(), key=lambda item: item["group"])
        }


@register_stage(STAGE_PRO_GEN)
def build_pro_gen_stage(cfg: "Config") -> Stage:
    """Factory for pro_gen stage, used by the orchestrator via the registry."""
    return ReverseGenerateStage(
        workers=cfg.max_workers,
        stage_key="pro",
        model_selector=select_generation_model,
    )