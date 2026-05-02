import atexit
from typing import Any, Dict, Iterator, List, Union

import numpy as np
import sglang as sgl
from transformers import AutoTokenizer, PreTrainedTokenizer

from rmgap.rm.base import BaseRM
from rmgap.rm import register_rm


@register_rm
class ScalarRM(BaseRM):
    def __call__(
        self,
        model_path: str,
        data: List[Dict[str, str]],
        sglang_cfg: Dict[str, Any],
        **kwargs,
    ) -> List[Dict[str, np.ndarray]]:
        llm = sgl.Engine(
            model_path=model_path,
            is_embedding=True,
            **sglang_cfg["engine"],
        )
        atexit.register(llm.shutdown)

        sequences, response_counts = [], []
        for item in data:
            prompt = item["prompt"]
            responses = item["responses"]
            response_counts.append(len(responses))
            sequences.extend(
                [
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": resp},
                    ]
                    for resp in responses
                ]
            )

        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path)
        sequences = tokenizer.apply_chat_template(
            sequences,
            tokenize=False,
            add_generation_prompt=False,
        )

        outputs = llm.encode(prompt=sequences)

        reward_scores = [self._parse_output(output) for output in outputs]
        return self._post_process(reward_scores, response_counts)

    def _parse_output(self, output: Union[Dict, Iterator[Dict]]) -> float:
        return output["embedding"][0]
