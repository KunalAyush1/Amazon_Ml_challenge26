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
    """Return unique whitespace-separated tokens."""
    return set(value.split())


def _jaccard(
    left: set[str],
    right: set[str],
) -> float:
    """Compute token Jaccard similarity.

    Empty fields represent missing evidence rather than a perfect
    similarity.
    """
    if not left or not right:
        return 0.0

    return len(left & right) / len(left | right)


def address_features(
    source1: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float | int]:
    """Generate pairwise address similarity features.

    The function supports the challenge schema:

        business_address
        address_norm
        address_compact

    Missing addresses are represented explicitly instead of being
    treated as perfectly similar.
    """

    address1 = _text(
        source1.get(
            "business_address",
            source1.get("address"),
        )
    )

    address2 = _text(
        candidate.get(
            "business_address",
            candidate.get("address"),
        )
    )

    address1_norm = _text(
        source1.get("address_norm")
    )

    address2_norm = _text(
        candidate.get("address_norm")
    )

    address1_compact = _text(
        source1.get("address_compact")
    )

    address2_compact = _text(
        candidate.get("address_compact")
    )

    # Prefer normalized representations for similarity calculations.
    compare1 = address1_norm or address1
    compare2 = address2_norm or address2

    compact1 = (
        address1_compact
        or compare1.replace(" ", "")
    )

    compact2 = (
        address2_compact
        or compare2.replace(" ", "")
    )

    tokens1 = _tokenize(compare1)
    tokens2 = _tokenize(compare2)

    return {
        # Raw-address exact match.
        "address_exact": int(
            bool(address1)
            and bool(address2)
            and address1 == address2
        ),

        # Normalized-address exact match.
        "address_norm_exact": int(
            bool(address1_norm)
            and bool(address2_norm)
            and address1_norm == address2_norm
        ),

        # Whitespace-independent exact match.
        "address_compact_exact": int(
            bool(address1_compact)
            and bool(address2_compact)
            and address1_compact == address2_compact
        ),

        # Explicit missingness indicators.
        "address_missing_1": int(
            not bool(compare1)
        ),

        "address_missing_2": int(
            not bool(compare2)
        ),

        "address_both_missing": int(
            not bool(compare1)
            and not bool(compare2)
        ),

        # Character-level similarity.
        "address_ratio": (
            fuzz.ratio(
                compare1,
                compare2,
            )
            / 100.0
        ),

        "address_partial_ratio": (
            fuzz.partial_ratio(
                compare1,
                compare2,
            )
            / 100.0
        ),

        # Token-order-aware similarity.
        "address_token_sort_ratio": (
            fuzz.token_sort_ratio(
                compare1,
                compare2,
            )
            / 100.0
        ),

        # Token-set similarity.
        "address_token_set_ratio": (
            fuzz.token_set_ratio(
                compare1,
                compare2,
            )
            / 100.0
        ),

        # Token overlap.
        "address_token_jaccard": _jaccard(
            tokens1,
            tokens2,
        ),

        # Length information.
        "address_length_1": len(compare1),
        "address_length_2": len(compare2),
        "address_length_diff": abs(
            len(compare1) - len(compare2)
        ),

        # Compact representation length.
        "address_compact_length_1": len(compact1),
        "address_compact_length_2": len(compact2),
        "address_compact_length_diff": abs(
            len(compact1) - len(compact2)
        ),
    }