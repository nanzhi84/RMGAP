"""Centralized configuration for provider-backed pipeline stages."""

from __future__ import annotations

import hashlib
from typing import Literal, TypedDict

from ...providers.base import ModelParams


class JudgeEntry(TypedDict):
    role: Literal["reviewer", "ac"]
    provider_name: str
    params: ModelParams


class GenerationEntry(TypedDict):
    provider_name: str
    params: ModelParams


StageKey = Literal["res", "pro", "rw"]


_OPENROUTER_EXTRAS = {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_env": "OPENROUTER_API_KEY",
}

_DEEPSEEK_EXTRAS = {
    "base_url": "https://api.deepseek.com/v1",
    "api_key_env": "DEEPSEEK_API_KEY",
}

_POE_EXTRAS = {
    "base_url": "https://api.poe.com/v1",
    "api_key_env": "POE_API_KEY",
}


GLOBAL_GENERATION_SEED = 42


def _hash_index(*, seed_text: str, modulo: int) -> int:
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value % max(1, modulo)


def select_generation_model(*, stage_key: StageKey, record_id: str) -> GenerationEntry:
    normalized_stage = stage_key.strip().lower()
    if normalized_stage not in {"res", "pro", "rw"}:
        raise ValueError(f"Unknown stage_key '{stage_key}' for model selection.")
    if not GENERATION_MODEL_POOL:
        raise ValueError("GENERATION_MODEL_POOL must not be empty.")
    seed_text = f"{GLOBAL_GENERATION_SEED}:{normalized_stage}:{record_id}"
    index = _hash_index(seed_text=seed_text, modulo=len(GENERATION_MODEL_POOL))
    return GENERATION_MODEL_POOL[index]


GENERATION_MODEL_POOL: tuple[GenerationEntry, ...] = (
    {
        "provider_name": "openai",
        "params": ModelParams(
            name="google/gemini-2.5-flash-lite-preview-09-2025",
            temp=1.0,
            max_tokens=None,
            extras=_OPENROUTER_EXTRAS,
        ),
    },
    {
        "provider_name": "openai",
        "params": ModelParams(
            name="anthropic/claude-sonnet-4.5",
            temp=1.0,
            max_tokens=None,
            extras=_OPENROUTER_EXTRAS,
        ),
    },
    {
        "provider_name": "openai",
        "params": ModelParams(
            name="openai/gpt-5-mini",
            temp=1.0,
            max_tokens=None,
            extras=_OPENROUTER_EXTRAS,
        ),
    },
    {
        "provider_name": "openai",
        "params": ModelParams(
            name="qwen/qwen3-30b-a3b-instruct-2507",
            temp=1.0,
            max_tokens=None,
            extras=_OPENROUTER_EXTRAS,
        ),
    },
    {
        "provider_name": "openai",
        "params": ModelParams(
            name="meta-llama/llama-4-scout",
            temp=1.0,
            max_tokens=None,
            extras=_OPENROUTER_EXTRAS,
        ),
    },
    {
        "provider_name": "openai",
        "params": ModelParams(
            name="x-ai/grok-4.1-fast",
            temp=1.0,
            max_tokens=None,
            extras={
                **_OPENROUTER_EXTRAS,
                "extra_body": {
                    "reasoning": {
                        "enabled": False,
                    },
                },
            },
        ),
    },
    {
        "provider_name": "openai",
        "params": ModelParams(
            name="z-ai/glm-4.5-air",
            temp=1.0,
            max_tokens=None,
            extras=_OPENROUTER_EXTRAS,
        ),
    }
)


JUDGE_CONFIGS: dict[str, JudgeEntry] = {
    "reviewer1": {
        "role": "reviewer",
        "provider_name": "openai",
        "params": ModelParams(
            name="deepseek-chat",
            temp=0.0,
            max_tokens=256,
            extras=_DEEPSEEK_EXTRAS,
        ),
    },
    "reviewer2": {
        "role": "reviewer",
        "provider_name": "openai",
        "params": ModelParams(
            name="qwen/qwen3-next-80b-a3b-instruct",
            temp=0.0,
            max_tokens=256,
            extras=_OPENROUTER_EXTRAS,
        ),
    },
    "ac": {
        "role": "ac",
        "provider_name": "openai",
        "params": ModelParams(
            name="openai/gpt-5.1",
            temp=0.0,
            max_tokens=256,
            extras=_OPENROUTER_EXTRAS,
        ),
    },
}


__all__ = [
    "JUDGE_CONFIGS",
    "GLOBAL_GENERATION_SEED",
    "GENERATION_MODEL_POOL",
    "select_generation_model",
    "GenerationEntry",
]
