import pytest

from business_entity_resol.evaluation.threshold_search import (
    search_best_threshold,
    select_matches_by_threshold,
)


def test_select_matches_by_threshold():
    scores = {
        "S1-001": {
            "S2-001": 0.95,
            "S2-002": 0.40,
            "S3-001": 0.90,
        },
        "S1-002": {
            "S2-010": 0.30,
        },
    }

    predictions = select_matches_by_threshold(
        scores,
        threshold=0.80,
    )

    assert predictions == {
        "S1-001": {
            "S2-001",
            "S3-001",
        },
        "S1-002": set(),
    }


def test_threshold_zero_keeps_all_candidates():
    scores = {
        "S1-001": {
            "S2-001": 0.10,
            "S2-002": 0.90,
        }
    }

    predictions = select_matches_by_threshold(
        scores,
        threshold=0.0,
    )

    assert predictions["S1-001"] == {
        "S2-001",
        "S2-002",
    }


def test_threshold_one_keeps_only_perfect_scores():
    scores = {
        "S1-001": {
            "S2-001": 1.0,
            "S2-002": 0.99,
        }
    }

    predictions = select_matches_by_threshold(
        scores,
        threshold=1.0,
    )

    assert predictions["S1-001"] == {"S2-001"}


def test_invalid_threshold_is_rejected():
    scores = {
        "S1-001": {
            "S2-001": 0.5,
        }
    }

    with pytest.raises(ValueError):
        select_matches_by_threshold(
            scores,
            threshold=-0.1,
        )

    with pytest.raises(ValueError):
        select_matches_by_threshold(
            scores,
            threshold=1.1,
        )


def test_search_best_threshold():
    ground_truth = {
        "S1-001": {"S2-001"},
        "S1-002": set(),
    }

    candidate_scores = {
        "S1-001": {
            "S2-001": 0.95,
            "S2-002": 0.20,
        },
        "S1-002": {
            "S2-010": 0.10,
        },
    }

    result = search_best_threshold(
        ground_truth,
        candidate_scores,
        thresholds=[
            0.0,
            0.5,
            0.8,
        ],
    )

    assert result.threshold == 0.5
    assert result.score == pytest.approx(1.0)


def test_search_rejects_missing_entity():
    ground_truth = {
        "S1-001": {"S2-001"},
        "S1-002": set(),
    }

    candidate_scores = {
        "S1-001": {
            "S2-001": 0.95,
        }
    }

    with pytest.raises(ValueError):
        search_best_threshold(
            ground_truth,
            candidate_scores,
        )
        