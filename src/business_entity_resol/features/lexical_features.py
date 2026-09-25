"""General lexical similarity features."""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz


def _text(value: Any) -> str:
    """Convert a value to a comparable string."""
    if value is None:
        return ""
    return str(value).strip()


def _tokens(value: str) -> list[str]:
    """Return whitespace-separated tokens."""
    return value.split()


def _token_set(value: str) -> set[str]:
    """Return unique whitespace-separated tokens."""
    return set(_tokens(value))


def _jaccard(left: set[str], right: set[str]) -> float:
    """Compute token Jaccard similarity."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def lexical_features(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    field: str,
) -> dict[str, float | int]:
    """Generate general lexical features for one text field."""

    value1 = _text(left.get(field))
    value2 = _text(right.get(field))

    tokens1 = _token_set(value1)
    tokens2 = _token_set(value2)

    return {
        "lexical_exact": int(
            bool(value1) and bool(value2) and value1 == value2
        ),
        "lexical_ratio": fuzz.ratio(value1, value2) / 100.0,
        "lexical_partial_ratio": (
            fuzz.partial_ratio(value1, value2) / 100.0
        ),
        "lexical_token_sort_ratio": (
            fuzz.token_sort_ratio(value1, value2) / 100.0
        ),
        "lexical_token_set_ratio": (
            fuzz.token_set_ratio(value1, value2) / 100.0
        ),
        "lexical_token_jaccard": _jaccard(tokens1, tokens2),
        "lexical_token_count_1": len(tokens1),
        "lexical_token_count_2": len(tokens2),
        "lexical_token_overlap_count": len(tokens1 & tokens2),
        "lexical_length_1": len(value1),
        "lexical_length_2": len(value2),
        "lexical_length_diff": abs(len(value1) - len(value2)),
    }