import atexit
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterator, List, Tuple, Union

import numpy as np
import sglang as sgl
from sglang.lang.interpreter import ProgramState
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from rmeval.provider import ChatMessage, OpenAIChatProvider, ProviderParams


def tokenizer_supports_enable_thinking(tokenizer: PreTrainedTokenizer) -> bool:
    """Check if the tokenizer's chat template accepts ``enable_thinking``."""
    tpl = getattr(tokenizer, "chat_template", "") or ""
    return "enable_thinking" in tpl


class BaseRM(ABC):
    @abstractmethod
    def __call__(
        self,
        model_path: str,
        data: List[Dict[str, str]],
        sglang_cfg: Dict[str, Any],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _post_process(
        self,
        reward_scores: List[float],
        response_counts: List[int],
    ) -> List[Dict[str, np.ndarray]]:
        outputs = []
        pointer = 0
        for count in response_counts:
            scores = np.array(reward_scores[pointer : pointer + count])
            pointer += count
            outputs.append({"scores": scores})
        return outputs

    @staticmethod
    def get_text_from_sglang_state(
        output: Union[Dict, Iterator[Dict]], state: Any
    ) -> str:
        """Extract generated text from an sglang ProgramState."""
        if isinstance(output, dict):
            text = output.get("text") or output.get("output_text")
            if text:
                return str(text)
        try:
            return str(state["answer"])
        except Exception:
            raise RuntimeError(
                f"Failed to extract generated text from sglang state. "
                f"output={output!r}"
            )

    @staticmethod
    def strip_thinking_block(text: str) -> str:
        """Remove ``<think>...</think>`` block (e.g. Qwen3 thinking mode)."""
        marker = "</think>"
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker) :].strip()
        return text.strip()


class GenerativeRM(BaseRM, ABC):
    """Template base for generative RMs that score via LLM text generation.

    Required (abstract):
      _prepare              – build chat messages and scoring context from data
      _parse_generated_text – extract a structured result from generated text

    Optional:
      DEFAULT_RESULT – class attr for fallback when parsing fails
      _finalize      – convert results + context into scores
                       (default: ``_post_process(results, context)``)
    """

    @abstractmethod
    def _prepare(
        self, data: List[Dict[str, Any]]
    ) -> Tuple[List[List[ChatMessage]], Any]:
        raise NotImplementedError

    @abstractmethod
    def _parse_generated_text(self, text: str) -> Any:
        raise NotImplementedError

    def _finalize(
        self, results: List[Any], context: Any
    ) -> List[Dict[str, np.ndarray]]:
        return self._post_process(results, context)

    def __call__(
        self,
        model_path: str,
        data: List[Dict[str, Any]],
        sglang_cfg: Dict[str, Any],
        **kwargs,
    ) -> List[Dict[str, np.ndarray]]:
        backend = sglang_cfg.get("backend", "sglang")
        if backend == "openai":
            return self._run_with_provider(model_path, data, sglang_cfg)
        return self._run_with_sglang(model_path, data, sglang_cfg)

    def _generate_and_parse(
        self,
        num_items: int,
        generate_fn: Callable[[List[int], int], List[str]],
        max_attempts: int,
    ) -> List[Any]:
        """Retry loop: call *generate_fn(pending, attempt)* → texts,
        preprocess, parse, and fill defaults for failures."""
        results: List[Any] = [None] * num_items
        pending = list(range(num_items))
        attempt = 1

        while pending and attempt <= max_attempts:
            texts = generate_fn(pending, attempt)
            for local_idx, text in enumerate(texts):
                global_idx = pending[local_idx]
                result = self._parse_generated_text(text)
                if result is not None:
                    results[global_idx] = result
            pending = [i for i in pending if results[i] is None]
            attempt += 1

        if pending:
            warnings.warn(
                f"{type(self).__name__}: failed to parse {len(pending)} "
                f"samples after {max_attempts} attempts. Using defaults.",
                UserWarning,
            )
            for idx in pending:
                results[idx] = self.DEFAULT_RESULT

        return results

    def _run_with_sglang(
        self,
        model_path: str,
        data: List[Dict[str, Any]],
        sglang_cfg: Dict[str, Any],
    ) -> List[Dict[str, np.ndarray]]:
        runtime = sgl.Runtime(model_path=model_path, **sglang_cfg["engine"])
        sgl.set_default_backend(runtime)
        atexit.register(runtime.shutdown)

        tokenizer: PreTrainedTokenizer = runtime.get_tokenizer()
        chats, context = self._prepare(data)

        sampling_params = dict(sglang_cfg.get("sampling_params", {}))
        max_tokens = int(sampling_params.pop("max_tokens", 4096))
        batch_size = int(sglang_cfg.get("batch_size", 256))

        @sgl.function
        def gen_fn(s: ProgramState, sequence: str):
            s += sequence
            s += sgl.gen("answer", max_tokens=max_tokens, **sampling_params)

        template_kwargs: Dict[str, Any] = dict(
            tokenize=False, add_generation_prompt=True,
        )
        if tokenizer_supports_enable_thinking(tokenizer):
            template_kwargs["enable_thinking"] = bool(
                sglang_cfg.get("enable_thinking", False)
            )

        def generate(pending_indices: List[int], attempt: int) -> List[str]:
            all_texts: List[str] = []
            for chunk_start in tqdm(
                range(0, len(pending_indices), batch_size),
                desc=f"{type(self).__name__} attempt {attempt}",
            ):
                chunk = pending_indices[
                    chunk_start : chunk_start + batch_size
                ]
                sequences: List[str] = tokenizer.apply_chat_template(
                    [chats[i] for i in chunk], **template_kwargs,
                )
                states = gen_fn.run_batch(
                    [{"sequence": seq} for seq in sequences],
                    progress_bar=True,
                )
                for state in states:
                    all_texts.append(
                        self.get_text_from_sglang_state(
                            state.get_meta_info("answer"), state,
                        )
                    )
            return all_texts

        max_attempts = int(sglang_cfg.get("max_attempts", 3))
        results = self._generate_and_parse(len(chats), generate, max_attempts)

        try:
            sgl.flush_cache()
            runtime.shutdown()
        except Exception:
            pass

        return self._finalize(results, context)

    def _run_with_provider(
        self,
        model_path: str,
        data: List[Dict[str, Any]],
        sglang_cfg: Dict[str, Any],
    ) -> List[Dict[str, np.ndarray]]:
        chats, context = self._prepare(data)

        sampling_params = dict(sglang_cfg.get("sampling_params", {}))
        max_tokens = int(sampling_params.pop("max_tokens", 4096))

        provider_cfg = sglang_cfg.get("provider", {})
        provider_model = provider_cfg.get("model") or model_path
        provider = OpenAIChatProvider(
            ProviderParams(
                model=provider_model,
                extras=provider_cfg.get("extras", {}),
            ),
        )

        def generate(pending_indices: List[int], _attempt: int) -> List[str]:
            return provider.generate_batch(
                [chats[i] for i in pending_indices],
                max_tokens=max_tokens,
                sampling_params=sampling_params,
                progress_bar=True,
            )

        max_attempts = int(sglang_cfg.get("max_attempts", 3))
        results = self._generate_and_parse(len(chats), generate, max_attempts)

        return self._finalize(results, context)
