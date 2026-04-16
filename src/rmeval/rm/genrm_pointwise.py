import re
from typing import Any, Dict, List, Tuple, Union

from rmeval.rm.base import ChatMessage, GenerativeRM
from rmeval.rm import register_rm

POINTWISE_SYSTEM = (
    "You are an AI evaluator. Your role is to assess AI-generated text "
    "for its quality and adherence to instructions."
)


def _pointwise_prompt(query: str, response: str) -> str:
    return (
        "You need to evaluate the 'Response' in response to the 'Query'. "
        "Your evaluation should be a score from 1 to 10.\n\n"
        "Evaluation Criteria:\n"
        "1. Relevance to Query: Does the answer directly and comprehensively "
        "address the user's query?\n"
        "2. Factuality and Helpfulness: Is the information accurate and useful?\n\n"
        "Please review the following:\n\n"
        "### Query\n" + query + "\n\n"
        "### Response\n" + response + "\n\n"
        "Based on your assessment, provide a single integer score from 1 to 10.\n\n"
        "Scoring Guide:\n"
        "1-2: Very Poor. Fails on most criteria.\n"
        "3-4: Poor. Significant issues in multiple areas.\n"
        "5-6: Average. Meets basic requirements but has clear flaws.\n"
        "7-8: Good. Solid response with minor issues.\n"
        "9-10: Excellent. A nearly perfect response that excels in all criteria.\n\n"
        "Provide only the numeric score.\n\n"
        "IMPORTANT: Output only a single integer between 1 and 10. "
        "Do not add any extra text."
    )


@register_rm
class GenRMPointwise(GenerativeRM):
    """Pointwise generative RM that extracts a 1-10 integer score."""

    DEFAULT_RESULT = 5.0

    def _prepare(
        self, data: List[Dict[str, Any]]
    ) -> Tuple[List[List[ChatMessage]], List[int]]:
        chats: List[List[ChatMessage]] = []
        response_counts: List[int] = []
        for item in data:
            prompt_text = item["prompt"]
            responses = item["responses"]
            response_counts.append(len(responses))
            for resp in responses:
                chats.append(
                    [
                        {"role": "system", "content": POINTWISE_SYSTEM},
                        {"role": "user", "content": _pointwise_prompt(prompt_text, resp)},
                    ]
                )
        return chats, response_counts

    def _parse_generated_text(self, text: str) -> Union[float, None]:
        match = re.search(r"\d+", text)
        if match:
            return float(max(1, min(10, int(match.group(0)))))
        return None
