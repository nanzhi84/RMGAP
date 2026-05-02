"""Math answer validation mirroring OpenAI's SymPy-based graders."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional

import sympy as sp
from sympy.core.sympify import SympifyError
from sympy.parsing.latex import parse_latex

_LATEX_TEXT_RE = re.compile(r"\\text\{([^}]*)\}")
_WS_RE = re.compile(r"\s+")
_VAR_PREFIX_RE = re.compile(r"^[a-zA-Z]\s*=\s*")
_WORD_UNITS_RE = re.compile(r"(?i)\b(deg|degree|degrees|rad|radian|radians)\b")
_SEPARATOR_RE = re.compile(r"[;,]")
_PLUS_MINUS_TOKEN = "±"


def _is_balanced_parentheses(text: str) -> bool:
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _strip_latex_wrappers(expr: str) -> str:
    """Remove lightweight formatting artifacts before parsing."""
    text = expr
    while True:
        text, replaced = _LATEX_TEXT_RE.subn(lambda match: match.group(1), text)
        if replaced == 0:
            break
    replacements = (
        ("\\left", ""),
        ("\\right", ""),
        ("\\!", ""),
        ("\\,", ""),
        ("\\;", ""),
        ("\\:", ""),
        ("~", ""),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return _WS_RE.sub("", text).strip()


def _clean_common_symbols(value: str) -> str:
    value = value.replace("\\pm", _PLUS_MINUS_TOKEN)
    value = value.replace("−", "-").replace("–", "-")
    return value


def _remove_unit_tokens(value: str) -> str:
    replacements = [
        "^\\circ",
        "^{\\circ}",
        "\\degree",
        "\\textdegree",
        "°",
    ]
    for token in replacements:
        value = value.replace(token, "")
    value = _WORD_UNITS_RE.sub("", value)
    return value


def _prepare_expression_for_parse(value: str) -> str:
    cleaned = _strip_latex_wrappers(value or "")
    cleaned = _clean_common_symbols(cleaned)
    cleaned = _remove_unit_tokens(cleaned)
    cleaned = _VAR_PREFIX_RE.sub("", cleaned, count=1)
    return cleaned


def _normalize_plain_expression(expr: str) -> str:
    text = _prepare_expression_for_parse(expr or "")
    text = text.replace(_PLUS_MINUS_TOKEN, "")
    while (
        text.startswith("(")
        and text.endswith(")")
        and _is_balanced_parentheses(text[1:-1])
    ):
        text = text[1:-1]
    return text.strip().lower()


def _extract_boxed_segments(text: str) -> List[str]:
    segments: List[str] = []
    start = 0
    needle = "\\boxed{"
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            return segments
        cursor = idx + len(needle)
        depth = 1
        buffer: List[str] = []
        while cursor < len(text) and depth > 0:
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    cursor += 1
                    break
            if depth > 0:
                buffer.append(char)
            cursor += 1
        if depth == 0:
            segments.append("".join(buffer).strip())
            start = cursor
        else:
            return segments


def _parse_expression(expr: str) -> Optional[sp.Expr]:
    try:
        return parse_latex(expr)
    except Exception:  # noqa: BLE001
        pass
    try:
        return sp.sympify(expr)
    except (SympifyError, TypeError, ValueError):
        return None


def _expand_pm(expr: str) -> List[str]:
    expr = _clean_common_symbols(expr or "")
    variants = [expr]
    results: List[str] = []
    while variants:
        current = variants.pop()
        if _PLUS_MINUS_TOKEN in current:
            before, after = current.split(_PLUS_MINUS_TOKEN, 1)
            variants.append(before + "+" + after)
            variants.append(before + "-" + after)
        else:
            results.append(current)
    return results


def _split_into_tokens(expr: str) -> List[str]:
    parts = _SEPARATOR_RE.split(expr)
    tokens = [part.strip() for part in parts if part.strip()]
    if not tokens:
        stripped = expr.strip()
        return [stripped] if stripped else []
    return tokens


def _token_signature(raw: str) -> str:
    cleaned_for_parse = _prepare_expression_for_parse(raw)
    expr = _parse_expression(cleaned_for_parse)
    if expr is not None:
        simplified = sp.simplify(expr)
        return f"expr:{sp.srepr(simplified)}"
    normalized = _normalize_plain_expression(raw)
    return f"str:{normalized}"


def _tokenize_answer(raw_answer: str) -> List[str]:
    stripped = _strip_latex_wrappers(raw_answer or "")
    if not stripped:
        return []
    tokens: List[str] = []
    for variant in _expand_pm(stripped):
        pieces = _split_into_tokens(variant)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            signature = _token_signature(piece)
            tokens.append(signature)
    return tokens


@dataclass
class MathAnswerValidator:
    """Validate math answers with SymPy, similar to OpenAI's MATH graders."""

    def extract_candidates(self, responses: Iterable[str]) -> List[str]:
        candidates: List[str] = []
        for response in responses:
            if not response:
                continue
            for segment in _extract_boxed_segments(response):
                cleaned = _strip_latex_wrappers(segment)
                if cleaned:
                    candidates.append(cleaned)
        return candidates

    def is_correct(self, responses: Iterable[str], reference_answer: str) -> bool:
        reference_tokens = _tokenize_answer(reference_answer)
        if not reference_tokens:
            return False

        reference_counter = Counter(reference_tokens)

        for candidate in self.extract_candidates(responses):
            candidate_tokens = _tokenize_answer(candidate)
            if not candidate_tokens:
                continue
            if Counter(candidate_tokens) == reference_counter:
                return True
        return False
