"""Threshold search utilities for entity-resolution predictions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from business_entity_resol.evaluation.entity_f05 import macro_f05


@dataclass(frozen=True)
class ThresholdSearchResult:
    """Result of threshold optimization."""

    threshold: float
    score: float


def select_matches_by_threshold(
    candidate_scores: Mapping[str, Mapping[str, float]],
    threshold: float,
) -> dict[str, set[str]]:
    """Select candidate IDs whose score is at least the threshold.

    Parameters
    ----------
    candidate_scores:
        Mapping:
            source1_id -> candidate_id -> model_score

    threshold:
        Minimum score required for a candidate to be selected.

    Returns
    -------
    dict
        Mapping:
            source1_id -> selected candidate IDs
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    predictions: dict[str, set[str]] = {}

    for source1_id, scores in candidate_scores.items():
        predictions[source1_id] = {
            candidate_id
            for candidate_id, score in scores.items()
            if score >= threshold
        }

    return predictions


def search_best_threshold(
    ground_truth: Mapping[str, Iterable[str] | None],
    candidate_scores: Mapping[str, Mapping[str, float]],
    thresholds: Iterable[float] | None = None,
) -> ThresholdSearchResult:
    """Find the threshold maximizing macro entity-level F0.5.

    The search uses only the supplied validation/OOF predictions.
    """
    if thresholds is None:
        thresholds = np.linspace(
            0.0,
            1.0,
            101,
        )

    threshold_values = list(thresholds)

    if not threshold_values:
        raise ValueError("At least one threshold is required.")

    for threshold in threshold_values:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Invalid threshold: {threshold}. "
                "Thresholds must be between 0 and 1."
            )

    missing_entities = set(ground_truth) - set(candidate_scores)

    if missing_entities:
        raise ValueError(
            "Candidate scores are missing Source-1 entities: "
            f"{sorted(missing_entities)[:10]}"
        )

    best_threshold: float | None = None
    best_score = -1.0

    for threshold in threshold_values:
        predictions = select_matches_by_threshold(
            candidate_scores,
            threshold,
        )

        score = macro_f05(
            ground_truth,
            predictions,
        )

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    assert best_threshold is not None

    return ThresholdSearchResult(
        threshold=best_threshold,
        score=best_score,
    )