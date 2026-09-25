"""Pairwise similarity features for business addresses."""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz


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


def address_features(
    source1: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float | int]:
    """Generate pairwise address similarity features."""

    address1 = _text(source1.get("address"))
    address2 = _text(candidate.get("address"))

    address1_norm = _text(source1.get("address_norm"))
    address2_norm = _text(candidate.get("address_norm"))

    address1_compact = _text(source1.get("address_compact"))
    address2_compact = _text(candidate.get("address_compact"))

    compare1 = address1_norm or address1
    compare2 = address2_norm or address2

    compact1 = address1_compact or compare1.replace(" ", "")
    compact2 = address2_compact or compare2.replace(" ", "")

    tokens1 = _tokenize(compare1)
    tokens2 = _tokenize(compare2)

    return {
        "address_exact": int(bool(address1) and address1 == address2),
        "address_norm_exact": int(
            bool(address1_norm)
            and bool(address2_norm)
            and address1_norm == address2_norm
        ),
        "address_compact_exact": int(
            bool(address1_compact)
            and bool(address2_compact)
            and address1_compact == address2_compact
        ),
        "address_ratio": fuzz.ratio(compare1, compare2) / 100.0,
        "address_partial_ratio": fuzz.partial_ratio(
            compare1, compare2
        ) / 100.0,
        "address_token_sort_ratio": (
            fuzz.token_sort_ratio(compare1, compare2) / 100.0
        ),
        "address_token_set_ratio": (
            fuzz.token_set_ratio(compare1, compare2) / 100.0
        ),
        "address_token_jaccard": _jaccard(tokens1, tokens2),
        "address_length_1": len(compare1),
        "address_length_2": len(compare2),
        "address_length_diff": abs(len(compare1) - len(compare2)),
        "address_compact_length_1": len(compact1),
        "address_compact_length_2": len(compact2),
        "address_compact_length_diff": abs(
            len(compact1) - len(compact2)
        ),
    }