"""pipeline - Prompt generation and evaluation pipeline.

Configuration-driven pipeline for generating four listwise responses from prompts,
followed by listwise evaluation and groupwise reverse-prompt filtering.

Module structure:
- core: New config + orchestrator
- models: Data layer (protocols, loading)
- prompts: Prompt layer (templates, variables, styles)
- providers: Provider layer (API wrappers)
- utils: Utility layer (common functions)
- stages: Stage layer (generation, evaluation, writing)
"""

__version__ = "0.3.0"

__all__ = ["__version__"]