import pytest

from business_entity_resol.evaluation.blocking_recall import (
    candidate_link_recall,
    complete_entity_recall,
    per_entity_candidate_recall,
)


def test_candidate_link_recall_is_one_for_complete_retrieval():
    ground_truth = {
        "S1-001": {"S2-001", "S3-001"},
        "S1-002": {"S2-002"},
    }

    candidates = {
        "S1-001": {
            "S2-001",
            "S3-001",
            "S2-999",
        },
        "S1-002": {
            "S2-002",
            "S2-888",
        },
    }

    assert candidate_link_recall(
        ground_truth,
        candidates,
    ) == 1.0


def test_candidate_link_recall_counts_missing_links():
    ground_truth = {
        "S1-001": {
            "S2-001",
            "S3-001",
        }
    }

    candidates = {
        "S1-001": {
            "S2-001",
            "S2-999",
        }
    }

    assert candidate_link_recall(
        ground_truth,
        candidates,
    ) == pytest.approx(0.5)


def test_complete_entity_recall_requires_all_true_matches():
    ground_truth = {
        "S1-001": {
            "S2-001",
            "S3-001",
        },
        "S1-002": {
            "S2-002",
        },
    }

    candidates = {
        "S1-001": {
            "S2-001",
            "S3-001",
        },
        "S1-002": {
            "S2-002",
        },
    }

    assert complete_entity_recall(
        ground_truth,
        candidates,
    ) == 1.0


def test_complete_entity_recall_fails_if_one_match_is_missing():
    ground_truth = {
        "S1-001": {
            "S2-001",
            "S3-001",
        },
        "S1-002": {
            "S2-002",
        },
    }

    candidates = {
        "S1-001": {
            "S2-001",
        },
        "S1-002": {
            "S2-002",
        },
    }

    assert complete_entity_recall(
        ground_truth,
        candidates,
    ) == pytest.approx(0.5)


def test_singletons_are_excluded_from_link_recall():
    ground_truth = {
        "S1-001": {"S2-001"},
        "S1-002": set(),
    }

    candidates = {
        "S1-001": {"S2-001"},
        "S1-002": set(),
    }

    assert candidate_link_recall(
        ground_truth,
        candidates,
    ) == 1.0


def test_per_entity_candidate_recall():
    ground_truth = {
        "S1-001": {
            "S2-001",
            "S2-002",
        },
        "S1-002": {
            "S3-001",
        },
        "S1-003": set(),
    }

    candidates = {
        "S1-001": {
            "S2-001",
        },
        "S1-002": {
            "S3-001",
            "S3-999",
        },
        "S1-003": set(),
    }

    result = per_entity_candidate_recall(
        ground_truth,
        candidates,
    )

    assert result["S1-001"] == pytest.approx(0.5)
    assert result["S1-002"] == pytest.approx(1.0)
    assert result["S1-003"] is None


def test_missing_candidate_entity_counts_as_empty_candidates():
    ground_truth = {
        "S1-001": {"S2-001"},
    }

    candidates = {}

    assert candidate_link_recall(
        ground_truth,
        candidates,
    ) == 0.0


def test_no_positive_links_raises():
    with pytest.raises(ValueError):
        candidate_link_recall(
            {"S1-001": set()},
            {"S1-001": set()},
        )


def test_no_non_singleton_entities_raises():
    with pytest.raises(ValueError):
        complete_entity_recall(
            {"S1-001": set()},
            {"S1-001": set()},
        )