"""Features derived from candidate-blocker provenance."""

from __future__ import annotations

from typing import Any


def blocker_features(
    candidate: dict[str, Any],
) -> dict[str, float | int]:
    """Generate features from candidate blocker provenance.

    Expected fields:

        blocker_provenance:
            Iterable containing the names of blockers that found
            this candidate.

        ranks:
            Mapping from blocker name to retrieval rank.

        num_blockers:
            Optional number of distinct blockers.

        best_rank:
            Optional best rank across blockers.

    The function is intentionally generic because blocker names are
    supplied dynamically by CandidateUnion.
    """

    provenance = candidate.get("blocker_provenance", [])
    ranks = candidate.get("ranks", {})

    if provenance is None:
        provenance = []

    if ranks is None:
        ranks = {}

    provenance_set = {
        str(blocker)
        for blocker in provenance
        if blocker is not None
    }

    valid_ranks = [
        int(rank)
        for rank in ranks.values()
        if rank is not None
    ]

    num_blockers = len(provenance_set)

    best_rank = min(valid_ranks) if valid_ranks else None

    return {
        "blocker_num_blockers": num_blockers,
        "blocker_best_rank": best_rank if best_rank is not None else 0,
        "blocker_has_rank": int(bool(valid_ranks)),
        "blocker_rank_count": len(valid_ranks),
        "blocker_provenance_count": len(provenance_set),
    }