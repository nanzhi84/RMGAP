from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)

ChatMessage = Dict[str, str]

DEFAULT_MAX_WORKERS = 8


@dataclass
class ProviderParams:
    model: str
    extras: Dict[str, Any] | None = None


class OpenAIChatProvider:
    """Provider backed by OpenAI's chat.completions API."""

    def __init__(self, params: ProviderParams) -> None:
        self.params = params
        extras = params.extras or {}

        api_key = extras.get("api_key")
        base_url = extras.get("base_url")
        if not api_key:
            raise ValueError("OpenAIChatProvider requires 'api_key' in extras.")

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

        raw_max_workers = extras.get("max_workers")
        self.max_workers: int = (
            int(raw_max_workers)
            if raw_max_workers is not None
            else DEFAULT_MAX_WORKERS
        )
        if self.max_workers <= 0:
            raise ValueError(
                "OpenAIChatProvider extras['max_workers'] must be positive."
            )

    def _generate_single(
        self,
        chat: List[ChatMessage],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.params.model,
            messages=chat,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        message = response.choices[0].message
        return (message.content or "").strip()

    def generate_batch(
        self,
        chats: List[List[ChatMessage]],
        *,
        max_tokens: int,
        sampling_params: Dict[str, Any],
        progress_bar: bool,
        failure_threshold: float = 0.5,
    ) -> List[str]:
        if not chats:
            return []

        temperature = float(sampling_params.get("temperature", 0.0))
        top_p = float(sampling_params.get("top_p", 1.0))

        total = len(chats)
        results: List[str] = [""] * total
        failure_count = 0

        pbar = tqdm(total=total, desc="Generating") if progress_bar else None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(
                    self._generate_single,
                    chat,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                ): index
                for index, chat in enumerate(chats)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:  # noqa: BLE001
                    failure_count += 1
                    logger.warning(
                        "API request failed for sample %d: %s: %s",
                        index, type(exc).__name__, exc,
                    )
                finally:
                    if pbar is not None:
                        pbar.update(1)

        if pbar is not None:
            pbar.close()

        if failure_count > 0:
            failure_rate = failure_count / total
            logger.warning(
                "Batch completed with %d/%d failures (%.1f%%).",
                failure_count, total, failure_rate * 100,
            )
            if failure_rate >= failure_threshold:
                raise RuntimeError(
                    f"OpenAIChatProvider: {failure_count}/{total} requests "
                    f"failed ({failure_rate:.0%}), exceeding threshold "
                    f"{failure_threshold:.0%}. Likely a systemic issue "
                    f"(wrong API key, service down, etc.)."
                )

        return results
