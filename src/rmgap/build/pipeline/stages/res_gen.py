"""Generate stage (provider-based)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List

from typing import TYPE_CHECKING

from tqdm import tqdm

from .base import Stage
from . import register_stage
from .llm_config import (
    GLOBAL_GENERATION_SEED,
    GENERATION_MODEL_POOL,
    select_generation_model,
    GenerationEntry,
)
from .routing.state_router import (
    STAGE_RES_GEN,
    apply_transition,
    filter_eligible_for_stage,
)
from ..providers import make_provider
from ..providers.base import Provider
from ..schemas.protocol import Protocol
from ..prompts.renderer import load_default_templates
from ..prompts.variables import build_prompt_variables, assign_styles, render_style_profile

if TYPE_CHECKING:  # imported only for type checking
    from ..core.config import Config


logger = logging.getLogger(__name__)


class GenerateStage(Stage):
    def __init__(
        self,
        *,
        workers: int,
        style_seed: int,
        stage_key: str,
        model_selector: Callable[..., GenerationEntry],
    ):
        self.workers = max(1, workers)
        self.style_seed = int(style_seed)
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
        """Select protocols eligible for res_gen based on declarative state rules."""
        return filter_eligible_for_stage(protocols, STAGE_RES_GEN)

    def _process_one(self, protocol: Protocol, tpl) -> None:
        """Wrapper that calls _one and handles state transition."""
        self._one(protocol, tpl)
        apply_transition(protocol, STAGE_RES_GEN, "success")

    def _one(self, protocol: Protocol, tpl) -> None:
        """Core logic for generating responses."""
        model_entry = self._model_selector(stage_key=self.stage_key, record_id=protocol.record_id)

        base_variables = build_prompt_variables(protocol)
        style_meta = assign_styles(self.style_seed, protocol.record_id)
        protocol.meta["style_assignments"] = style_meta

        # Reuse existing responses and only fill missing responseX entries
        existing = protocol.responses or {}
        responses: dict[str, str] = dict(existing)
        context_prefix = f"res_gen:{protocol.record_id}"
        pending = {
            f"response{style_key.lstrip('r')}": render_style_profile(assignment)
            for style_key, assignment in style_meta.items()
            if not (existing.get(f"response{style_key.lstrip('r')}") or "").strip()
        }

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
                f"res_gen received empty content from all generation models "
                f"(record_id={protocol.record_id}, context={context})."
            )

        if pending:
            with ThreadPoolExecutor(max_workers=len(pending)) as executor:
                futures = {}
                for response_key, style_profile in pending.items():
                    variables = dict(base_variables)
                    variables["STYLE_PROFILE"] = style_profile
                    messages = tpl.render(**variables)
                    context = f"{context_prefix}:{response_key}"
                    future = executor.submit(
                        _generate_with_fallback,
                        messages=messages,
                        context=context,
                    )
                    futures[future] = response_key

                for future in as_completed(futures):
                    response_key = futures[future]
                    responses[response_key] = future.result()

        # Validate that all four responses are now filled
        missing = [
            key for key in (f"response{i}" for i in range(1, 5))
            if not (responses.get(key) or "").strip()
        ]
        if missing:
            protocol.responses = dict(sorted(responses.items()))
            raise ValueError(
                f"res_gen missing responses {missing!r} (record_id={protocol.record_id})."
            )

        protocol.responses = dict(sorted(responses.items()))

    def process(self, protocols: List[Protocol]) -> List[Protocol]:
        if not protocols:
            return []
        
        tpl = load_default_templates()["pairs_generation"]
        out: List[Protocol] = []
        success = 0
        failed = 0
        total = len(protocols)
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            fut = {ex.submit(self._process_one, p, tpl): p for p in protocols}
            with tqdm(total=total, desc="Generating responses", unit="item") as pbar:
                for f in as_completed(fut):
                    proto = fut[f]
                    try:
                        f.result()
                        out.append(proto)
                    except Exception as exc:
                        failed += 1
                        logger.warning("Generate failed for record_id=%s: %s", proto.record_id, exc)
                    else:
                        success += 1
                    pbar.update(1)
        logger.info(
            "res_gen completed - total=%d, success=%d, failed=%d",
            total,
            success,
            failed,
        )
        return out


@register_stage(STAGE_RES_GEN)
def build_res_gen_stage(cfg: "Config") -> Stage:
    """Factory for res_gen stage, used by the orchestrator via the registry."""
    return GenerateStage(
        workers=cfg.max_workers,
        style_seed=GLOBAL_GENERATION_SEED,
        stage_key="res",
        model_selector=select_generation_model,
    )