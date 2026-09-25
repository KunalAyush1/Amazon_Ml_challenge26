"""Pairwise similarity features for business names."""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


def _text(value: Any) -> str:
    """Convert a value to a comparable string."""
    if value is None:
        return ""
    return str(value).strip()


def _tokenize(value: str) -> set[str]:
    """Return whitespace-separated tokens."""
    return set(value.split())


def _jaccard(left: set[str], right: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def name_features(
    source1: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float | int]:
    """Generate pairwise name similarity features.

    The function expects normalized/compact name fields when available:
    ``name``, ``name_norm`` and ``name_compact``.
    """

    name1 = _text(source1.get("name"))
    name2 = _text(candidate.get("name"))

    name1_norm = _text(source1.get("name_norm"))
    name2_norm = _text(candidate.get("name_norm"))

    name1_compact = _text(source1.get("name_compact"))
    name2_compact = _text(candidate.get("name_compact"))

    # Fall back to the raw name when normalized/compact fields are absent.
    compare1 = name1_norm or name1
    compare2 = name2_norm or name2

    compact1 = name1_compact or compare1.replace(" ", "")
    compact2 = name2_compact or compare2.replace(" ", "")

    tokens1 = _tokenize(compare1)
    tokens2 = _tokenize(compare2)

    return {
        "name_exact": int(bool(name1) and name1 == name2),
        "name_norm_exact": int(
            bool(name1_norm) and bool(name2_norm) and name1_norm == name2_norm
        ),
        "name_compact_exact": int(
            bool(name1_compact)
            and bool(name2_compact)
            and name1_compact == name2_compact
        ),
        "name_ratio": fuzz.ratio(compare1, compare2) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(compare1, compare2) / 100.0,
        "name_jaro_winkler": JaroWinkler.normalized_similarity(
    compare1, compare2
),
        "name_token_sort_ratio": (
            fuzz.token_sort_ratio(compare1, compare2) / 100.0
        ),
        "name_token_set_ratio": (
            fuzz.token_set_ratio(compare1, compare2) / 100.0
        ),
        "name_token_jaccard": _jaccard(tokens1, tokens2),
        "name_length_1": len(compare1),
        "name_length_2": len(compare2),
        "name_length_diff": abs(len(compare1) - len(compare2)),
        "name_compact_length_1": len(compact1),
        "name_compact_length_2": len(compact2),
        "name_compact_length_diff": abs(len(compact1) - len(compact2)),
    }