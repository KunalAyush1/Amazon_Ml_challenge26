"""Hard-negative mining from candidate pairs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _hard_negative_score(features: Mapping[str, Any]) -> float:
    """
    Calculate a deterministic similarity score for hard-negative mining.

    The score uses existing pairwise similarity features. It is only
    used to rank negative candidates; it is not the final model score.
    """

    return (
        float(features.get("name_ratio", 0.0))
        + float(features.get("name_jaro_winkler", 0.0))
        + float(features.get("name_token_set_ratio", 0.0))
        + float(features.get("address_ratio", 0.0))
        + float(features.get("address_token_set_ratio", 0.0))
        + float(features.get("numeric_jaccard", 0.0))
        + float(features.get("country_exact", 0))
    )


def mine_hard_negatives(
    candidates: Mapping[str, Iterable[str]],
    ground_truth: Mapping[str, Iterable[str]],
    pair_features: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    hard_negative_ratio: int = 1,
) -> list[dict[str, Any]]:
    """
    Select hard negatives from the actual candidate pool.

    Parameters
    ----------
    candidates:
        Mapping from Source-1 entity ID to candidate IDs.

    ground_truth:
        Mapping from Source-1 entity ID to known positive candidate IDs.

    pair_features:
        Mapping of (source1_entity_id, candidate_entity_id) to the
        corresponding pairwise feature dictionary.

    hard_negative_ratio:
        Number of hard negatives to select per positive match.

    Returns
    -------
    list[dict]
        Hard-negative pairs with label=0 and their similarity score.
    """

    if hard_negative_ratio < 1:
        raise ValueError("hard_negative_ratio must be >= 1")

    results: list[dict[str, Any]] = []

    for source1_id, candidate_ids in candidates.items():
        positive_ids = set(
            ground_truth.get(source1_id, ())
        )

        available = []

        for candidate_id in candidate_ids:
            if candidate_id in positive_ids:
                continue

            key = (source1_id, candidate_id)

            features = pair_features.get(key)

            if features is None:
                continue

            score = _hard_negative_score(features)

            available.append(
                (
                    score,
                    candidate_id,
                )
            )

        if not available:
            continue

        available.sort(
            key=lambda item: (-item[0], item[1])
        )

        positive_count = len(
            set(candidate_ids) & positive_ids
        )

        target_count = (
            positive_count * hard_negative_ratio
            if positive_count > 0
            else hard_negative_ratio
        )

        for score, candidate_id in available[:target_count]:
            results.append(
                {
                    "source1_entity_id": source1_id,
                    "candidate_entity_id": candidate_id,
                    "label": 0,
                    "hard_negative_score": score,
                }
            )

    return results