"""DPO-style implicit reward model based on log-probability differences.

Score(y) = sum_i [ log pi_theta(t_i | x, t_<i) - log pi_ref(t_i | x, t_<i) ]
"""

import atexit
import warnings
from typing import Any, Dict, Iterator, List, Union

import numpy as np
import sglang as sgl
from sglang.lang.interpreter import ProgramState
from transformers import PreTrainedTokenizer

from rmeval.rm.base import BaseRM
from rmeval.rm import register_rm


@register_rm
class DpoImplicitRM(BaseRM):
    def __call__(
        self,
        model_path: str,
        data: List[Dict[str, str]],
        sglang_cfg: Dict[str, Any],
        reference_model_path: str,
        aggregate: Dict[str, Any],
        **kwargs,
    ) -> List[Dict[str, np.ndarray]]:
        if not reference_model_path:
            raise ValueError(
                "DpoImplicitRM requires a non-empty `reference_model_path`."
            )

        self.aggregate_mode = aggregate["mode"]
        if self.aggregate_mode == "discounted_sum":
            self.gamma = aggregate["gamma"]
            self.clip_min = aggregate["clip_min"]

        sequences: List[List[Dict[str, str]]] = []
        prompts: List[List[Dict[str, str]]] = []
        response_counts: List[int] = []

        for item in data:
            prompt_text = item["prompt"]
            responses = item["responses"]
            response_counts.append(len(responses))

            for response_text in responses:
                sequences.append(
                    [
                        {"role": "user", "content": prompt_text},
                        {"role": "assistant", "content": response_text},
                    ]
                )
                prompts.append([{"role": "user", "content": prompt_text}])

        policy_logprobs = self._get_response_logprobs(
            model_path, sequences, prompts, sglang_cfg,
        )
        reference_logprobs = self._get_response_logprobs(
            reference_model_path, sequences, prompts, sglang_cfg,
        )

        if len(policy_logprobs) != len(reference_logprobs):
            raise RuntimeError(
                f"Mismatch between policy and reference log-likelihood lengths: "
                f"{len(policy_logprobs)} vs {len(reference_logprobs)}."
            )

        reward_scores: List[float] = []
        for seq_idx, (policy_seq, reference_seq) in enumerate(
            zip(policy_logprobs, reference_logprobs)
        ):
            if policy_seq.shape != reference_seq.shape:
                warnings.warn(
                    f"Shape mismatch for sample {seq_idx}: "
                    f"{policy_seq.shape} vs {reference_seq.shape}. "
                    f"Marking reward as NaN."
                )
                reward_scores.append(float("nan"))
                continue
            reward_scores.append(
                self._aggregate_log_ratio(policy_seq - reference_seq)
            )

        return self._post_process(reward_scores, response_counts)

    def _get_response_logprobs(
        self,
        model_path: str,
        sequences: List[List[Dict[str, str]]],
        prompts: List[List[Dict[str, str]]],
        sglang_cfg: Dict[str, Any],
    ) -> List[np.ndarray]:
        runtime = sgl.Runtime(model_path=model_path, **sglang_cfg["engine"])
        sgl.set_default_backend(runtime)
        atexit.register(runtime.shutdown)

        tokenizer: PreTrainedTokenizer = runtime.get_tokenizer()
        sequence_texts: List[str] = tokenizer.apply_chat_template(
            sequences, tokenize=False, add_generation_prompt=False,
        )
        prompt_token_ids: List[List[int]] = tokenizer.apply_chat_template(
            prompts, tokenize=True, add_generation_prompt=True,
        )
        response_start_indices = [len(ids) for ids in prompt_token_ids]

        sampling_params = dict(sglang_cfg["sampling_params"])

        @sgl.function
        def text_qa(s: ProgramState, sequence: str, response_start_idx: int):
            s += sequence
            s += sgl.gen(
                "answer",
                max_tokens=1,
                return_logprob=True,
                logprob_start_len=response_start_idx - 1,
                **sampling_params,
            )

        states = text_qa.run_batch(
            [
                {"sequence": seq, "response_start_idx": rsi}
                for seq, rsi in zip(sequence_texts, response_start_indices)
            ],
            progress_bar=True,
        )

        result: List[np.ndarray] = []
        for state_idx, state in enumerate(states):
            output = state.get_meta_info("answer")
            try:
                logprobs = self._extract_response_logprobs(output)
            except Exception as exc:
                warnings.warn(
                    f"Failed to extract logprobs for sample {state_idx}: {exc}. "
                    f"Marking as NaN."
                )
                logprobs = np.array([np.nan], dtype=float)
            result.append(logprobs)

        try:
            sgl.flush_cache()
            runtime.shutdown()
        except Exception:
            pass

        return result

    def _extract_response_logprobs(
        self, output: Union[Dict, Iterator[Dict[str, Any]]]
    ) -> np.ndarray:
        token_logprobs = output["input_token_logprobs"][1:-1]
        return np.array([item[0] for item in token_logprobs], dtype=float)

    def _aggregate_log_ratio(self, log_ratio: np.ndarray) -> float:
        if log_ratio.shape[0] == 0:
            return 0.0
        if self.aggregate_mode == "mean":
            return float(np.mean(log_ratio))
        if self.aggregate_mode == "sum":
            return float(np.sum(log_ratio))
        if self.aggregate_mode == "discounted_sum":
            gammas = self.gamma ** np.arange(log_ratio.shape[0])
            gammas = np.clip(gammas, a_min=self.clip_min, a_max=None)
            return float(np.sum(log_ratio * gammas))
        raise ValueError(f"Invalid aggregate mode: {self.aggregate_mode}")
