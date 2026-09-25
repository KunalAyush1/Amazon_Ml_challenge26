"""Pairwise country matching features."""

from __future__ import annotations

from typing import Any


def _text(value: Any) -> str:
    """Convert a value to a normalized comparison string."""
    if value is None:
        return ""
    return str(value).strip().casefold()


def country_features(
    source1: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float | int]:
    """Generate pairwise country comparison features."""

    country1 = _text(source1.get("country"))
    country2 = _text(candidate.get("country"))

    country1_norm = _text(
        source1.get("country_norm", source1.get("country"))
    )
    country2_norm = _text(
        candidate.get("country_norm", candidate.get("country"))
    )

    country1_code = _text(
        source1.get("country_code", source1.get("country"))
    )
    country2_code = _text(
        candidate.get("country_code", candidate.get("country"))
    )

    return {
        "country_exact": int(
            bool(country1) and bool(country2) and country1 == country2
        ),
        "country_norm_exact": int(
            bool(country1_norm)
            and bool(country2_norm)
            and country1_norm == country2_norm
        ),
        "country_code_exact": int(
            bool(country1_code)
            and bool(country2_code)
            and country1_code == country2_code
        ),
        "country_both_present": int(bool(country1) and bool(country2)),
        "country_missing_1": int(not bool(country1)),
        "country_missing_2": int(not bool(country2)),
    }