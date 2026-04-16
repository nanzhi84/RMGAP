import random
import re
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from rmeval.rm.base import ChatMessage, GenerativeRM
from rmeval.rm import register_rm

PAIRWISE_SYSTEM = (
    "You are an AI evaluator. Your role is to assess AI-generated text "
    "for its quality and adherence to instructions."
)


def _pairwise_prompt(query: str, responses: List[str]) -> str:
    """Build a pairwise ranking prompt over exactly four responses."""
    if len(responses) != 4:
        raise ValueError("GenRMPairwise expects exactly 4 responses per sample.")

    labeled = [f"{chr(65 + i)}) {responses[i]}" for i in range(4)]

    return (
        "You need to evaluate and rank the 'Responses' for the given 'Query'.\n\n"
        "Evaluation Criteria:\n"
        "1. Relevance to Query: Does the answer directly and comprehensively "
        "address the user's query?\n"
        "2. Factuality and Helpfulness: Is the information accurate and useful?\n\n"
        "Please review the following:\n\n"
        "### Query\n" + query + "\n\n"
        "### Responses\n" + "\n\n".join(labeled) + "\n\n"
        "Output only the ranking from best to worst as a permutation of letters "
        "A>B>C>D, with '>' separators and no extra text.\n\n"
        "IMPORTANT: Respond EXACTLY with a single permutation like A>B>C>D. "
        "Use uppercase letters and '>' separators. No spaces. No extra text."
    )


@register_rm
class GenRMPairwise(GenerativeRM):
    """Pairwise generative RM that ranks four responses (A>B>C>D)
    and maps ranks to scores 4,3,2,1."""

    DEFAULT_RESULT = ["A", "B", "C", "D"]

    def _prepare(
        self, data: List[Dict[str, Any]]
    ) -> Tuple[
        List[List[ChatMessage]],
        Tuple[List[int], List[List[int]]],
    ]:
        chats: List[List[ChatMessage]] = []
        response_counts: List[int] = []
        permutations: List[List[int]] = []

        for item in data:
            responses = item["responses"]
            if len(responses) != 4:
                raise ValueError(
                    f"GenRMPairwise requires exactly 4 responses per item, "
                    f"got {len(responses)}."
                )

            permutation = list(range(4))
            random.shuffle(permutation)
            shuffled = [responses[i] for i in permutation]
            permutations.append(permutation)

            chats.append(
                [
                    {"role": "system", "content": PAIRWISE_SYSTEM},
                    {"role": "user", "content": _pairwise_prompt(item["prompt"], shuffled)},
                ]
            )
            response_counts.append(len(responses))

        return chats, (response_counts, permutations)

    def _parse_generated_text(self, text: str) -> Union[List[str], None]:
        if not text:
            return None
        text = text.upper()
        m = re.search(
            r"\b([ABCD])\s*>\s*([ABCD])\s*>\s*([ABCD])\s*>\s*([ABCD])\b", text,
        )
        if m:
            letters = [m.group(i) for i in range(1, 5)]
            if len(set(letters)) == 4:
                return letters
        tokens = re.findall(r"[ABCD]", text)
        if len(tokens) >= 4:
            for i in range(len(tokens) - 3):
                window = tokens[i : i + 4]
                if len(set(window)) == 4:
                    return window
        return None

    def _finalize(
        self,
        results: List[List[str]],
        context: Tuple[List[int], List[List[int]]],
    ) -> List[Dict[str, np.ndarray]]:
        response_counts, permutations = context
        reward_scores: List[float] = []
        for idx, order in enumerate(results):
            if order is None:
                raise RuntimeError(
                    f"Ranking for sample {idx} is unexpectedly None."
                )
            score_by_letter = {
                letter: float(4 - rank_pos)
                for rank_pos, letter in enumerate(order)
            }
            permutation = permutations[idx]
            score_by_original_pos = [0.0] * 4
            for letter_pos, letter in enumerate(["A", "B", "C", "D"]):
                original_pos = permutation[letter_pos]
                score_by_original_pos[original_pos] = score_by_letter[letter]
            for j in range(response_counts[idx]):
                reward_scores.append(score_by_original_pos[j])

        return self._post_process(reward_scores, response_counts)
