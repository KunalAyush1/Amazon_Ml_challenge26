
from __future__ import annotations

from collections.abc import Mapping

from business_entity_resol.blocking.candidate_union import CandidateUnion

from .blocking_recall import (
    candidate_count_stats,
    candidate_link_recall,
    complete_entity_recall,
    per_entity_candidate_recall,
)


def candidate_union_to_mapping(
    candidate_union: CandidateUnion,
) -> dict[str, set[str]]:
    """
    Convert CandidateUnion output into the mapping expected by
    blocking-recall metrics.

    Returns:
        {
            "S1-001": {"S2-001", "S3-001"},
            ...
        }
    """
    result: dict[str, set[str]] = {}

    for candidate in candidate_union.get():
        result.setdefault(
            candidate.source1_entity_id,
            set(),
        ).add(candidate.candidate_entity_id)

    return result


def evaluate_candidate_union(
    ground_truth: Mapping[str, set[str]],
    candidate_union: CandidateUnion,
) -> dict[str, object]:
    """
    Evaluate a generated candidate union against ground truth.

    Returns link recall, complete entity recall, per-entity recall,
    and candidate-count statistics.
    """
    candidates = candidate_union_to_mapping(candidate_union)

    return {
        "link_recall": candidate_link_recall(
            ground_truth,
            candidates,
        ),
        "complete_entity_recall": complete_entity_recall(
            ground_truth,
            candidates,
        ),
        "per_entity_recall": per_entity_candidate_recall(
            ground_truth,
            candidates,
        ),
        "candidate_count_stats": candidate_count_stats(
            candidates,
        ),
    }
