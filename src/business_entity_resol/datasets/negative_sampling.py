"""Negative pair sampling from candidate pools."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import Any


def sample_negative_pairs(
    candidates: Mapping[str, Iterable[str]],
    ground_truth: Mapping[str, Iterable[str]],
    *,
    negative_ratio: int = 1,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Sample negative pairs from the actual candidate pool.

    Parameters
    ----------
    candidates:
        Mapping from Source-1 entity ID to candidate Source-2/Source-3
        entity IDs.

    ground_truth:
        Mapping from Source-1 entity ID to known matching entity IDs.

    negative_ratio:
        Number of negative pairs to sample relative to the number of
        available positive matches for each Source-1 entity.

    random_seed:
        Seed for deterministic sampling.

    Returns
    -------
    list[dict]
        Negative pairs with label=0.
    """
    if negative_ratio < 1:
        raise ValueError("negative_ratio must be >= 1")

    rng = random.Random(random_seed)
    negatives: list[dict[str, Any]] = []

    for source1_id, candidate_ids in candidates.items():
        candidate_set = set(candidate_ids)

        if not candidate_set:
            continue

        positive_ids = set(ground_truth.get(source1_id, ()))

        available_negatives = sorted(
            candidate_set - positive_ids
        )

        if not available_negatives:
            continue

        positive_count = len(
            candidate_set & positive_ids
        )

        # If no known positive is present in the candidate pool,
        # still sample a small deterministic amount of negatives.
        target_count = (
            positive_count * negative_ratio
            if positive_count > 0
            else negative_ratio
        )

        target_count = min(
            target_count,
            len(available_negatives),
        )

        if target_count == len(available_negatives):
            selected = available_negatives
        else:
            selected = rng.sample(
                available_negatives,
                target_count,
            )

        for candidate_id in selected:
            negatives.append(
                {
                    "source1_entity_id": source1_id,
                    "candidate_entity_id": candidate_id,
                    "label": 0,
                }
            )

    return negatives