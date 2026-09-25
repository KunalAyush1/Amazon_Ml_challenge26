from business_entity_resol.datasets.negative_sampling import (
    sample_negative_pairs,
)


def test_negative_pairs_come_from_candidate_pool():
    candidates = {
        "S1-001": {"S2-001", "S2-002", "S3-001"},
    }

    ground_truth = {
        "S1-001": {"S2-001"},
    }

    negatives = sample_negative_pairs(
        candidates,
        ground_truth,
        negative_ratio=2,
        random_seed=42,
    )

    assert len(negatives) == 2

    for pair in negatives:
        assert pair["source1_entity_id"] == "S1-001"
        assert pair["candidate_entity_id"] in candidates["S1-001"]
        assert pair["candidate_entity_id"] != "S2-001"
        assert pair["label"] == 0


def test_positive_candidates_are_never_sampled_as_negative():
    candidates = {
        "S1-001": {"S2-001", "S2-002"},
        "S1-002": {"S3-001", "S3-002"},
    }

    ground_truth = {
        "S1-001": {"S2-001"},
        "S1-002": {"S3-001"},
    }

    negatives = sample_negative_pairs(
        candidates,
        ground_truth,
        negative_ratio=1,
        random_seed=42,
    )

    negative_ids = {
        (pair["source1_entity_id"], pair["candidate_entity_id"])
        for pair in negatives
    }

    assert ("S1-001", "S2-001") not in negative_ids
    assert ("S1-002", "S3-001") not in negative_ids


def test_sampling_is_deterministic():
    candidates = {
        "S1-001": {
            "S2-001",
            "S2-002",
            "S2-003",
            "S2-004",
        }
    }

    ground_truth = {
        "S1-001": {"S2-001"},
    }

    first = sample_negative_pairs(
        candidates,
        ground_truth,
        negative_ratio=2,
        random_seed=42,
    )

    second = sample_negative_pairs(
        candidates,
        ground_truth,
        negative_ratio=2,
        random_seed=42,
    )

    assert first == second


def test_no_candidates_returns_no_negatives():
    candidates = {
        "S1-001": set(),
    }

    ground_truth = {
        "S1-001": {"S2-001"},
    }

    negatives = sample_negative_pairs(
        candidates,
        ground_truth,
    )

    assert negatives == []