from business_entity_resol.datasets.hard_negative_mining import (
    mine_hard_negatives,
)


def test_mines_highest_scoring_non_positive_candidates():
    candidates = {
        "S1-001": {
            "S2-001",
            "S2-002",
            "S2-003",
        }
    }

    ground_truth = {
        "S1-001": {"S2-001"},
    }

    features = {
        ("S1-001", "S2-001"): {
            "name_ratio": 1.0,
            "name_jaro_winkler": 1.0,
            "name_token_set_ratio": 1.0,
            "address_ratio": 1.0,
            "address_token_set_ratio": 1.0,
            "numeric_jaccard": 1.0,
            "country_exact": 1,
        },
        ("S1-001", "S2-002"): {
            "name_ratio": 0.95,
            "name_jaro_winkler": 0.90,
            "name_token_set_ratio": 0.92,
            "address_ratio": 0.85,
            "address_token_set_ratio": 0.80,
            "numeric_jaccard": 0.50,
            "country_exact": 1,
        },
        ("S1-001", "S2-003"): {
            "name_ratio": 0.40,
            "name_jaro_winkler": 0.35,
            "name_token_set_ratio": 0.30,
            "address_ratio": 0.20,
            "address_token_set_ratio": 0.15,
            "numeric_jaccard": 0.0,
            "country_exact": 0,
        },
    }

    negatives = mine_hard_negatives(
        candidates,
        ground_truth,
        features,
        hard_negative_ratio=1,
    )

    assert len(negatives) == 1
    assert negatives[0]["candidate_entity_id"] == "S2-002"
    assert negatives[0]["label"] == 0
    assert negatives[0]["hard_negative_score"] > 0


def test_positive_pairs_are_never_hard_negatives():
    candidates = {
        "S1-001": {
            "S2-001",
            "S2-002",
        }
    }

    ground_truth = {
        "S1-001": {"S2-001"},
    }

    features = {
        ("S1-001", "S2-001"): {
            "name_ratio": 1.0,
        },
        ("S1-001", "S2-002"): {
            "name_ratio": 0.9,
        },
    }

    negatives = mine_hard_negatives(
        candidates,
        ground_truth,
        features,
    )

    assert all(
        pair["candidate_entity_id"] != "S2-001"
        for pair in negatives
    )


def test_missing_features_are_skipped():
    candidates = {
        "S1-001": {
            "S2-001",
            "S2-002",
        }
    }

    ground_truth = {
        "S1-001": set(),
    }

    features = {
        ("S1-001", "S2-001"): {
            "name_ratio": 0.8,
        }
    }

    negatives = mine_hard_negatives(
        candidates,
        ground_truth,
        features,
    )

    assert len(negatives) == 1
    assert negatives[0]["candidate_entity_id"] == "S2-001"


def test_deterministic_tie_breaking():
    candidates = {
        "S1-001": {
            "S2-002",
            "S2-001",
        }
    }

    ground_truth = {
        "S1-001": set(),
    }

    features = {
        ("S1-001", "S2-002"): {
            "name_ratio": 0.8,
        },
        ("S1-001", "S2-001"): {
            "name_ratio": 0.8,
        },
    }

    negatives = mine_hard_negatives(
        candidates,
        ground_truth,
        features,
        hard_negative_ratio=1,
    )

    assert len(negatives) == 1
    assert negatives[0]["candidate_entity_id"] == "S2-001"