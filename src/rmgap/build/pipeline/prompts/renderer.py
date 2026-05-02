"""Prompt template rendering."""

from __future__ import annotations

from typing import Dict


class PromptTemplate:
    """Prompt template with user messages."""
    
    def __init__(self, template: str):
        self.template = template

    def render(self, **variables: str) -> list[dict[str, str]]:
        """Render template by substituting {{VARIABLE}} placeholders."""
        content = self.template
        for key, value in variables.items():
            content = content.replace(f"{{{{{key}}}}}", value)
        return [{"role": "user", "content": content}]


def load_default_templates() -> Dict[str, PromptTemplate]:
    """Load all default prompt templates."""
    from .templates import (
        nuanced_writer_prompt,
        pair_evaluator_prompt,
        reverse_prompt_generator,
        reverse_prompts_evaluator,
        rewrite_prompt_generator,
    )

    return {
        "pairs_generation": PromptTemplate(nuanced_writer_prompt),
        "pairs_evaluation": PromptTemplate(pair_evaluator_prompt),
        "reverse_generation": PromptTemplate(reverse_prompt_generator),
        "reverse_evaluation": PromptTemplate(reverse_prompts_evaluator),
        "rewrite_generation": PromptTemplate(rewrite_prompt_generator),
    }
