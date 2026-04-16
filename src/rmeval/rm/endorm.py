"""Generalist Reward Model: Found inside Large Language Models.

arXiv: https://arxiv.org/abs/2506.23235
"""

import atexit
from typing import Any, Dict, Iterator, List, Union

import numpy as np
import sglang as sgl
from sglang.lang.interpreter import ProgramState
from transformers import PreTrainedTokenizer

from rmeval.rm.base import BaseRM
from rmeval.rm import register_rm


@register_rm
class EndoRM(BaseRM):
    def __call__(
        self,
        model_path: str,
        data: List[Dict[str, str]],
        sglang_cfg: Dict[str, Any],
        aggregate: Dict[str, Any],
        **kwargs,
    ) -> List[Dict[str, np.ndarray]]:
        self.aggregate_mode = aggregate["mode"]
        if self.aggregate_mode == "discounted_sum":
            self.gamma = aggregate["gamma"]
            self.clip_min = aggregate["clip_min"]

        sequences, prompts, response_counts = [], [], []
        for item in data:
            prompt = item["prompt"]
            responses = item["responses"]
            response_counts.append(len(responses))
            for resp in responses:
                sequences.append(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": resp},
                    ]
                )
                prompts.append([{"role": "user", "content": prompt}])

        runtime = sgl.Runtime(model_path=model_path, **sglang_cfg["engine"])
        sgl.set_default_backend(runtime)
        atexit.register(runtime.shutdown)

        tokenizer: PreTrainedTokenizer = runtime.get_tokenizer()
        sequences = tokenizer.apply_chat_template(
            sequences, tokenize=False, add_generation_prompt=False,
        )
        prompts = tokenizer.apply_chat_template(
            prompts, tokenize=True, add_generation_prompt=True,
        )
        response_start_idx_lst = [len(item) for item in prompts]

        @sgl.function
        def text_qa(s: ProgramState, sequence: str, response_start_idx: int):
            s += sequence
            s += sgl.gen(
                "answer",
                max_tokens=1,
                return_logprob=True,
                logprob_start_len=response_start_idx - 1,
                **sglang_cfg["sampling_params"],
            )

        states = text_qa.run_batch(
            [
                dict(sequence=seq, response_start_idx=rsi)
                for seq, rsi in zip(sequences, response_start_idx_lst)
            ],
            progress_bar=True,
        )

        reward_scores: List[float] = []
        for state in states:
            response_logprobs = self._extract_response_logprobs(
                state.get_meta_info("answer")
            )
            reward_scores.append(self._aggregate_logprobs(response_logprobs))

        return self._post_process(reward_scores, response_counts)

    def _aggregate_logprobs(self, response_logprobs: List[float]) -> float:
        arr = np.array(response_logprobs, dtype=float)
        if self.aggregate_mode == "mean":
            return np.mean(arr).item()
        if self.aggregate_mode == "sum":
            return np.sum(arr).item()
        if self.aggregate_mode == "discounted_sum":
            gammas = self.gamma ** np.arange(len(arr))
            gammas = np.clip(gammas, a_min=self.clip_min, a_max=None)
            return np.sum(arr * gammas).item()
        raise ValueError(f"Invalid aggregate mode: {self.aggregate_mode}")

    def _extract_response_logprobs(
        self, output: Union[Dict, Iterator[Dict]]
    ) -> List[float]:
        # index 0 is None (sglang convention), last token is `\n` from Qwen tokenizer
        response_token_logprobs = output["input_token_logprobs"][1:-1]
        return [item[0] for item in response_token_logprobs]