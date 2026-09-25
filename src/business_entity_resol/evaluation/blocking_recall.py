"""Evaluation utilities for candidate-generation recall."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from statistics import mean, median


def candidate_link_recall(
    ground_truth: Mapping[str, Iterable[str] | None],
    candidates: Mapping[str, Iterable[str] | None],
) -> float:
    """Calculate recall over true S2/S3 links.

    Example:
        Ground truth:
            S1-001 -> {S2-1, S3-1}

        Candidates:
            S1-001 -> {S2-1, S2-9, S3-1}

        Both true links were retrieved, so recall = 1.0.

    Entities with no true matches do not contribute to link recall.
    """
    total_true_links = 0
    retrieved_true_links = 0

    for entity_id, true_ids in ground_truth.items():
        true_set = set(true_ids or [])
        candidate_set = set(candidates.get(entity_id, []))

        if not true_set:
            continue

        total_true_links += len(true_set)
        retrieved_true_links += len(true_set & candidate_set)

    if total_true_links == 0:
        raise ValueError(
            "Cannot calculate candidate recall: "
            "no positive ground-truth links were provided."
        )

    return retrieved_true_links / total_true_links


def complete_entity_recall(
    ground_truth: Mapping[str, Iterable[str] | None],
    candidates: Mapping[str, Iterable[str] | None],
) -> float:
    """Calculate the fraction of non-singleton S1 entities whose
    complete true match set is contained in the candidate set.

    Example:

        Truth:
            S1-001 -> {S2-1, S3-1}
            S1-002 -> {S2-2}

        Candidates:
            S1-001 -> {S2-1, S3-1, S2-99}
            S1-002 -> {S2-2, S2-88}

        Both entities are completely retrieved -> 1.0.
    """
    eligible_entities = 0
    complete_entities = 0

    for entity_id, true_ids in ground_truth.items():
        true_set = set(true_ids or [])

        # Singleton/no-match entities are excluded from this metric.
        if not true_set:
            continue

        eligible_entities += 1

        candidate_set = set(candidates.get(entity_id, []))

        if true_set.issubset(candidate_set):
            complete_entities += 1

    if eligible_entities == 0:
        raise ValueError(
            "Cannot calculate complete entity recall: "
            "no non-singleton entities were provided."
        )

    return complete_entities / eligible_entities


def per_entity_candidate_recall(
    ground_truth: Mapping[str, Iterable[str] | None],
    candidates: Mapping[str, Iterable[str] | None],
) -> dict[str, float | None]:
    """Return link recall for every Source-1 entity.

    Entities with no true matches receive None because there is no
    positive link whose retrieval can be measured.
    """
    results: dict[str, float | None] = {}

    for entity_id, true_ids in ground_truth.items():
        true_set = set(true_ids or [])

        if not true_set:
            results[entity_id] = None
            continue

        candidate_set = set(candidates.get(entity_id, []))

        results[entity_id] = len(
            true_set & candidate_set
        ) / len(true_set)

    return results
def candidate_count_stats(
    candidates: Mapping[str, Iterable[str] | None],
) -> dict[str, float | int]:
    """Calculate candidate-count statistics per Source-1 entity.

    Returns the mean, median, p95, and maximum number of candidates
    generated for each Source-1 entity.

    Entities with no candidates are included with a count of zero.
    """
    if not candidates:
        raise ValueError(
            "Cannot calculate candidate-count statistics: "
            "no candidate entities were provided."
        )

    counts = [
        len(set(candidate_ids or []))
        for candidate_ids in candidates.values()
    ]

    sorted_counts = sorted(counts)

    p95_index = max(
        0,
        min(
            len(sorted_counts) - 1,
            int(0.95 * len(sorted_counts)),
        ),
    )

    return {
        "mean": mean(counts),
        "median": median(counts),
        "p95": float(sorted_counts[p95_index]),
        "max": max(counts),
    }