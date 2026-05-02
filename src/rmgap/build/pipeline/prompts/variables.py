"""Prompt variable preparation and deterministic style assignment."""

from __future__ import annotations

import hashlib
from itertools import product
from typing import Dict, List, Optional, Tuple

from ..schemas.protocol import Protocol


# Deterministic style dimensions (fixed levels per dimension)
STYLE_DIMENSIONS: Dict[str, List[str]] = {
    "Formality": [
        "casual",
        "informal",
        "neutral",
        "formal",
        "highly_formal",
    ],
    "Conciseness": [
        "terse",
        "concise",
        "moderate",
        "detailed",
        "verbose",
    ],
    "Technicality": [
        "layman",
        "accessible",
        "semi_technical",
        "technical",
        "highly_specialized",
    ],
    "Objectivity": [
        "highly_subjective",
        "subjective",
        "balanced",
        "objective",
        "strictly_objective",
    ],
    "Structural_Coherence": [
        "fragmented",
        "loose",
        "organized",
        "well_structured",
        "rigorous",
    ],
}

MIN_STYLE_DISTANCE: int = 8  # Enforce pairwise Manhattan distance > 8.
DIMENSION_NAMES: List[str] = list(STYLE_DIMENSIONS.keys())
STYLE_LEVEL_INDEX: Dict[str, Dict[str, int]] = {
    dimension: {level: idx for idx, level in enumerate(levels)}
    for dimension, levels in STYLE_DIMENSIONS.items()
}


def _build_style_spaces() -> Tuple[List[Dict[str, str]], List[Tuple[int, ...]]]:
    """Enumerate the full style space once for deterministic reuse."""
    combinations: List[Dict[str, str]] = []
    vectors: List[Tuple[int, ...]] = []
    level_grid = [STYLE_DIMENSIONS[dimension] for dimension in DIMENSION_NAMES]

    for level_tuple in product(*level_grid):
        assignment = {
            dimension: level for dimension, level in zip(DIMENSION_NAMES, level_tuple)
        }
        vector = tuple(
            STYLE_LEVEL_INDEX[dimension][assignment[dimension]]
            for dimension in DIMENSION_NAMES
        )
        combinations.append(assignment)
        vectors.append(vector)

    return combinations, vectors


ALL_STYLE_COMBINATIONS, ALL_STYLE_VECTORS = _build_style_spaces()


def _normalize_eval_types(protocol: Protocol) -> List[str]:
    """Extract eval type(s) from protocol metadata."""
    eval_meta = protocol.meta.get("eval") if isinstance(protocol.meta, dict) else None
    raw_type = eval_meta.get("type") if isinstance(eval_meta, dict) else None
    if isinstance(raw_type, str):
        return [raw_type.strip().lower()] if raw_type.strip() else []
    if isinstance(raw_type, list):
        normalized: List[str] = []
        for item in raw_type:
            text = str(item).strip().lower()
            if text:
                normalized.append(text)
        return normalized
    return []


def _build_format_requirements(protocol: Protocol) -> str:
    """Construct optional domain-specific formatting instructions."""
    if protocol.domain.strip().lower() != "reasoning":
        return ""

    eval_types = set(_normalize_eval_types(protocol))
    requirements: List[str] = []

    if "math" in eval_types:
        requirements.append(
            "Present the final numeric answer enclosed in LaTeX \\boxed{...} notation."
        )
    if "code" in eval_types:
        requirements.append(
            "Wrap the complete Python solution inside fenced Markdown code blocks that "
            "begin with ```python and end with ```."
        )

    if not requirements:
        return ""

    lines = ["Additional formatting requirements:"]
    lines.extend(f"- {req}" for req in requirements)
    return "\n".join(lines)


def build_prompt_variables(protocol: Protocol) -> Dict[str, str]:
    """Prepare variables for template rendering."""
    original_prompt = protocol.original_prompt.strip() or "(Empty prompt)"
    format_requirements = _build_format_requirements(protocol)
    if format_requirements:
        format_requirements = f"{format_requirements}\n"
    return {
        "ORIGINAL_PROMPT": original_prompt,
        "FORMAT_REQUIREMENTS": format_requirements,
    }


def _hash_index(*, seed_text: str, modulo: int) -> int:
    """Compute a deterministic index in [0, modulo) using SHA-256."""
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value % max(1, modulo)


def render_style_profile(assignment: Dict[str, str]) -> str:
    """Render a single style profile line without response index."""
    parts = [f"{dim}={assignment[dim]}" for dim in STYLE_DIMENSIONS]
    return "; ".join(parts)


def _manhattan_distance(vector_a: Tuple[int, ...], vector_b: Tuple[int, ...]) -> int:
    """Compute Manhattan distance between two style vectors."""
    return sum(abs(a - b) for a, b in zip(vector_a, vector_b))


def _select_style_indices(*, base_seed: str) -> List[int]:
    """Pick 4 indices via farthest-point sampling under the distance constraint."""
    total = len(ALL_STYLE_COMBINATIONS)
    if total < 4:
        raise ValueError("Style space must contain at least 4 combinations.")

    first_index = _hash_index(seed_text=f"{base_seed}:start", modulo=total)
    selected: List[int] = [first_index]

    while len(selected) < 4:
        best_index: Optional[int] = None
        best_distance = -1
        best_tiebreaker = -1

        for candidate in range(total):
            if candidate in selected:
                continue

            candidate_vector = ALL_STYLE_VECTORS[candidate]
            min_distance = min(
                _manhattan_distance(candidate_vector, ALL_STYLE_VECTORS[idx])
                for idx in selected
            )
            if min_distance <= MIN_STYLE_DISTANCE:
                continue

            candidate_tiebreaker = _hash_index(
                seed_text=f"{base_seed}:tie:{len(selected)}:{candidate}",
                modulo=2**31,
            )

            if min_distance > best_distance or (
                min_distance == best_distance and candidate_tiebreaker > best_tiebreaker
            ):
                best_distance = min_distance
                best_index = candidate
                best_tiebreaker = candidate_tiebreaker

        if best_index is None:
            raise ValueError(
                "Unable to select four style profiles satisfying the distance constraint."
            )

        selected.append(best_index)

    return selected


def assign_styles(master_seed: int, record_id: str) -> Dict[str, Dict[str, str]]:
    """Assign deterministic style combinations for 4 responses.

    Returns:
        A dict with keys r1..r4 mapping to the chosen levels per dimension.
    """
    base_seed = f"{master_seed}:{record_id}".strip()
    selected_indices = _select_style_indices(base_seed=base_seed)
    meta = {
        f"r{i + 1}": ALL_STYLE_COMBINATIONS[index].copy()
        for i, index in enumerate(selected_indices)
    }
    return meta