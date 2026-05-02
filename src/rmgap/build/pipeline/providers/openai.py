"""OpenAI Provider implementation using the chat.completions API."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict
from threading import Lock

from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

from . import register_provider
from .base import Provider, ModelParams


logger = logging.getLogger(__name__)


@register_provider("openai")
class OpenAIProvider(Provider):
    """Registered provider for OpenAI-compatible endpoints."""

    def __init__(self, params: ModelParams):
        self.params = params
        self._client: OpenAI | None = None
        self._lock: Lock = Lock()

    def _parse_json_response(self, content: str, *, context: str) -> Any:
        """Parse JSON response with strict validation and error handling."""
        cleaned = content.strip()
        if not cleaned:
            logger.error("%s JSON parsing failed: empty response", context)
            raise ValueError(f"{context} JSON parsing failed: empty payload")
        
        # Remove markdown code block markers if present (e.g., ```json ... ```)
        if cleaned.startswith("```"):
            # Match ```json or ``` followed by newline, then capture content until closing ```
            match = re.match(r"^```(?:json)?\n?(.*?)\n?```$", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
            else:
                # Fallback: remove leading ```json or ``` and trailing ```
                cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned).strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            snippet = cleaned if len(cleaned) <= 200 else f"{cleaned[:200]}..."
            logger.error("%s JSON parsing failed: %s | snippet: %s", context, exc, snippet)
            raise ValueError(f"{context} JSON parsing failed") from exc

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        context: str,
        expect_json: bool = True,
    ) -> Any:
        client = self._ensure_client()
        request_params: Dict[str, Any] = {
            "model": self.params.name,
            "messages": messages,
            "temperature": self.params.temp,
        }
        if expect_json:
            request_params["response_format"] = {"type": "json_object"}
        if self.params.max_tokens is not None:
            request_params["max_tokens"] = self.params.max_tokens
        extras = self.params.extras or {}
        extra_body = extras.get("extra_body")
        if extra_body is not None:
            request_params["extra_body"] = extra_body

        try:
            response = client.chat.completions.create(**request_params)
        except (APITimeoutError, RateLimitError, APIConnectionError, APIStatusError) as exc:
            logger.error("%s API error: %s", context, exc)
            raise

        content = (response.choices[0].message.content or "").strip()
        if expect_json:
            return self._parse_json_response(content, context=context)
        return content

    def _ensure_client(self) -> OpenAI:
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            extras = self.params.extras or {}
            api_key = extras.get("api_key")
            api_key_env = extras.get("api_key_env")
            if api_key is None and api_key_env:
                api_key = os.getenv(str(api_key_env))
            if not api_key:
                env_hint = (
                    f" or set {api_key_env}" if api_key_env else ""
                )
                raise ValueError(
                    f"Missing API key for model '{self.params.name}'. "
                    f"Provide extras['api_key']{env_hint}."
                )
            kwargs = {k: v for k, v in {
                "api_key": api_key,
                "base_url": extras.get("base_url"),
            }.items() if v is not None}
            self._client = OpenAI(**kwargs)
            logger.info("OpenAI client initialized (base_url=%s)", kwargs.get("base_url", "default"))
            return self._client
