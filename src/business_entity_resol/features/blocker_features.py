"""Features derived from candidate-blocker provenance."""

from __future__ import annotations

from typing import Any


KNOWN_BLOCKERS = (
    "exact",
    "rare_token",
    "numeric",
    "tfidf",
    "dense",
)


def blocker_features(
    candidate: dict[str, Any],
) -> dict[str, float | int]:
    """Generate numeric features from blocker provenance.

    Expected candidate fields
    -------------------------
    blocker_provenance:
        Iterable containing blocker names that discovered the candidate.

    ranks:
        Mapping from blocker name to retrieval rank.

    num_blockers:
        Optional precomputed number of blockers.

    best_rank:
        Optional precomputed best rank.

    Returns
    -------
    dict
        Numeric blocker-provenance features suitable for LightGBM.
    """

    provenance = candidate.get(
        "blocker_provenance",
        [],
    )

    ranks = candidate.get(
        "ranks",
        {},
    )

    if provenance is None:
        provenance = []

    if ranks is None:
        ranks = {}

    # Normalize blocker names and remove duplicates.
    provenance_set = {
        str(blocker).strip().lower()
        for blocker in provenance
        if blocker is not None
        and str(blocker).strip()
    }

    # Normalize rank values.
    valid_ranks: list[int] = []

    for rank in ranks.values():
        if rank is None:
            continue

        try:
            rank_value = int(rank)
        except (TypeError, ValueError):
            continue

        if rank_value >= 1:
            valid_ranks.append(rank_value)

    # Derive values from the actual provenance rather than trusting
    # duplicated metadata when possible.
    num_blockers = len(provenance_set)

    best_rank = (
        min(valid_ranks)
        if valid_ranks
        else None
    )

    features: dict[str, float | int] = {
        "blocker_num_blockers": num_blockers,

        # 0 means no ranked blocker supplied a rank.
        "blocker_best_rank": (
            best_rank
            if best_rank is not None
            else 0
        ),

        "blocker_has_rank": int(
            bool(valid_ranks)
        ),

        "blocker_rank_count": len(
            valid_ranks
        ),

        "blocker_provenance_count": len(
            provenance_set
        ),
    }

    # One-hot indicator for every known blocker.
    for blocker in KNOWN_BLOCKERS:
        features[
            f"blocker_has_{blocker}"
        ] = int(
            blocker in provenance_set
        )

    # Rank-specific information where available.
    for blocker in KNOWN_BLOCKERS:
        rank = ranks.get(blocker)

        if rank is None:
            features[
                f"blocker_{blocker}_rank"
            ] = 0

            features[
                f"blocker_{blocker}_has_rank"
            ] = 0

            continue

        try:
            rank_value = int(rank)
        except (TypeError, ValueError):
            rank_value = 0

        if rank_value < 0:
            rank_value = 0

        features[
            f"blocker_{blocker}_rank"
        ] = rank_value

        features[
            f"blocker_{blocker}_has_rank"
        ] = int(
            rank_value > 0
        )

    return features