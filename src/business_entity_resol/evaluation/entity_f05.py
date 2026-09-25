"""Entity-level F0.5 evaluation for the competition.

The competition evaluates predictions as a set of matched
Source-2/Source-3 IDs for each Source-1 entity and then
macro-averages F0.5 across Source-1 entities.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class EntityF05Result:
    """Detailed F0.5 result for one Source-1 entity."""

    precision: float
    recall: float
    f05: float
    true_positive: int
    false_positive: int
    false_negative: int


def _as_id_set(entity_ids: Iterable[str] | None) -> set[str]:
    """Convert IDs to a set, treating None as an empty set."""
    if entity_ids is None:
        return set()

    return set(entity_ids)


def score_entity_f05(
    true_ids: Iterable[str] | None,
    predicted_ids: Iterable[str] | None,
) -> EntityF05Result:
    """Calculate F0.5 for one Source-1 entity.

    Competition singleton behavior:
    - true empty + predicted empty -> 1.0
    - true empty + predicted non-empty -> 0.0
    """
    true_set = _as_id_set(true_ids)
    predicted_set = _as_id_set(predicted_ids)

    # Singleton: no true matches.
    if not true_set:
        if not predicted_set:
            return EntityF05Result(
                precision=1.0,
                recall=1.0,
                f05=1.0,
                true_positive=0,
                false_positive=0,
                false_negative=0,
            )

        return EntityF05Result(
            precision=0.0,
            recall=0.0,
            f05=0.0,
            true_positive=0,
            false_positive=len(predicted_set),
            false_negative=0,
        )

    true_positive = len(true_set & predicted_set)
    false_positive = len(predicted_set - true_set)
    false_negative = len(true_set - predicted_set)

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    beta_squared = 0.5**2

    denominator = (
        beta_squared * precision
        + recall
    )

    f05 = (
        (1.0 + beta_squared) * precision * recall / denominator
        if denominator > 0
        else 0.0
    )

    return EntityF05Result(
        precision=precision,
        recall=recall,
        f05=f05,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def macro_f05(
    ground_truth: Mapping[str, Iterable[str] | None],
    predictions: Mapping[str, Iterable[str] | None],
) -> float:
    """Calculate macro-averaged entity-level F0.5.

    Every Source-1 entity must have exactly one prediction entry.
    """
    true_entities = set(ground_truth)
    predicted_entities = set(predictions)

    missing = true_entities - predicted_entities
    extra = predicted_entities - true_entities

    if missing:
        raise ValueError(
            "Predictions are missing Source-1 entities: "
            f"{sorted(missing)[:10]}"
        )

    if extra:
        raise ValueError(
            "Predictions contain unknown Source-1 entities: "
            f"{sorted(extra)[:10]}"
        )

    if not true_entities:
        raise ValueError(
            "Cannot calculate macro F0.5 for zero Source-1 entities."
        )

    scores = [
        score_entity_f05(
            ground_truth[entity_id],
            predictions[entity_id],
        ).f05
        for entity_id in sorted(true_entities)
    ]

    return sum(scores) / len(scores)


def per_entity_f05(
    ground_truth: Mapping[str, Iterable[str] | None],
    predictions: Mapping[str, Iterable[str] | None],
) -> dict[str, EntityF05Result]:
    """Return detailed F0.5 results for every Source-1 entity."""
    true_entities = set(ground_truth)
    predicted_entities = set(predictions)

    missing = true_entities - predicted_entities
    extra = predicted_entities - true_entities

    if missing:
        raise ValueError(
            "Predictions are missing Source-1 entities: "
            f"{sorted(missing)[:10]}"
        )

    if extra:
        raise ValueError(
            "Predictions contain unknown Source-1 entities: "
            f"{sorted(extra)[:10]}"
        )

    return {
        entity_id: score_entity_f05(
            ground_truth[entity_id],
            predictions[entity_id],
        )
        for entity_id in sorted(true_entities)
    }