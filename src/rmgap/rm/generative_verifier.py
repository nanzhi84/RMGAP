"""Generative Verifiers: Reward Modeling as Next-Token Prediction."""

import atexit
from typing import Any, Dict, Iterator, List, Union

import numpy as np
import sglang as sgl
from sglang.lang.interpreter import ProgramState
from transformers import PreTrainedTokenizer

from rmgap.rm.base import BaseRM
from rmgap.rm import register_rm

SYSTEM_PROMPT_GV = (
    "You are an AI evaluator. Your role is to assess AI-generated text "
    "for its quality and adherence to instructions."
)


def _build_gv_user_prompt(query: str, response: str) -> str:
    return (
        "You need to determine if the 'Response' is a good response to the 'Query'. "
        "A 'good' response MUST satisfy the following conditions:\n\n"
        "1. **Relevance to Query**: The answer must directly and comprehensively "
        "address the user's query without any irrelevant information.\n"
        "2. **Factuality and Helpfulness**: The information in the answer should "
        "be accurate and useful.\n\n"
        "Please carefully review the following materials.\n\n"
        f"### Query\n{query}\n\n"
        f"### Response\n{response}\n\n"
        "Based on your evaluation, is the answer a good response? "
        "Answer with only 'YES' or 'NO'."
    )


@register_rm
class GenerativeVerifierRM(BaseRM):
    """YES/NO generative verifier via constrained logprob scoring."""

    def __call__(
        self,
        model_path: str,
        data: List[Dict[str, str]],
        sglang_cfg: Dict[str, Any],
        **kwargs,
    ) -> List[Dict[str, np.ndarray]]:
        runtime = sgl.Runtime(model_path=model_path, **sglang_cfg["engine"])
        sgl.set_default_backend(runtime)
        atexit.register(runtime.shutdown)

        sequences, prompts, response_counts = [], [], []
        for item in data:
            prompt = item["prompt"]
            responses = item["responses"]
            response_counts.append(len(responses))
            for resp in responses:
                sequences.append(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT_GV},
                        {"role": "user", "content": _build_gv_user_prompt(prompt, resp)},
                        {"role": "assistant", "content": "YES"},
                    ]
                )
                prompts.append(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT_GV},
                        {"role": "user", "content": _build_gv_user_prompt(prompt, resp)},
                    ]
                )

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

        reward_scores = [
            self._parse_reward_score(state.get_meta_info("answer"))
            for state in states
        ]
        return self._post_process(reward_scores, response_counts)

    def _parse_reward_score(self, output: Union[Dict, Iterator[Dict]]) -> float:
        response_token_logprobs = output["input_token_logprobs"]
        response_logprobs = np.array(
            [
                float(item[0])
                for item in response_token_logprobs
                if item and item[0] is not None
            ]
        )
        if response_logprobs.size == 0:
            raise ValueError("No verifier token logprobs found in sglang output.")
        return np.mean(response_logprobs).item()
