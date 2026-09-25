"""Pairwise numeric evidence features."""

from __future__ import annotations

import re
from typing import Any


_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def _text(value: Any) -> str:
    """Convert a value to a string."""
    if value is None:
        return ""
    return str(value).strip()


def _extract_numbers(value: Any) -> list[str]:
    """Extract numeric tokens from a value."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        ]
    
    text = _text(value)
    
    return _NUMBER_PATTERN.findall(text)


def _number_set(value: Any) -> set[str]:
    """Return unique numeric tokens."""
    return set(_extract_numbers(value))


def numeric_features(
    source1: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float | int]:
    """Generate pairwise features from numeric evidence."""

    values1 = _number_set(
        source1.get("address_numbers", source1.get("address"))
    )
    values2 = _number_set(
        candidate.get("address_numbers", candidate.get("address"))
    )

    intersection = values1 & values2
    union = values1 | values2

    return {
        "numeric_count_1": len(values1),
        "numeric_count_2": len(values2),
        "numeric_overlap_count": len(intersection),
        "numeric_jaccard": (
            len(intersection) / len(union) if union else 0.0
        ),
        "numeric_exact_match": int(
            bool(values1) and bool(values2) and values1 == values2
        ),
        "numeric_any_overlap": int(bool(intersection)),
    }