import numpy as np
import pandas as pd
import pytest

from business_entity_resol.models.lightgbm.model import (
    LightGBMMatcher,
)
from business_entity_resol.models.lightgbm.predict import (
    score_candidate_pairs,
    scores_to_entity_mapping,
)


@pytest.fixture
def trained_model() -> LightGBMMatcher:
    X = pd.DataFrame(
        {
            "name_similarity": [
                0.95,
                0.90,
                0.20,
                0.10,
                0.85,
                0.30,
            ],
            "address_similarity": [
                0.92,
                0.88,
                0.15,
                0.20,
                0.80,
                0.25,
            ],
            "country_equal": [
                1,
                1,
                0,
                0,
                1,
                0,
            ],
        }
    )

    y = np.array([1, 1, 0, 0, 1, 0])

    model = LightGBMMatcher()
    model.fit(X, y)

    return model


@pytest.fixture
def pair_ids() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source1_entity_id": [
                "S1-001",
                "S1-001",
                "S1-002",
            ],
            "candidate_entity_id": [
                "S2-001",
                "S3-001",
                "S2-010",
            ],
        }
    )


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name_similarity": [
                0.95,
                0.90,
                0.20,
            ],
            "address_similarity": [
                0.92,
                0.88,
                0.15,
            ],
            "country_equal": [
                1,
                1,
                0,
            ],
        }
    )


def test_score_candidate_pairs(
    trained_model,
    pair_ids,
    features,
):
    result = score_candidate_pairs(
        trained_model,
        pair_ids,
        features,
    )

    assert list(result.columns) == [
        "source1_entity_id",
        "candidate_entity_id",
        "match_probability",
    ]

    assert len(result) == 3

    assert result["match_probability"].between(
        0.0,
        1.0,
    ).all()


def test_score_candidate_pairs_preserves_order(
    trained_model,
    pair_ids,
    features,
):
    result = score_candidate_pairs(
        trained_model,
        pair_ids,
        features,
    )

    assert result["source1_entity_id"].tolist() == [
        "S1-001",
        "S1-001",
        "S1-002",
    ]

    assert result["candidate_entity_id"].tolist() == [
        "S2-001",
        "S3-001",
        "S2-010",
    ]


def test_pair_ids_and_features_must_have_same_length(
    trained_model,
    pair_ids,
):
    features = pd.DataFrame(
        {
            "name_similarity": [0.9],
            "address_similarity": [0.9],
            "country_equal": [1],
        }
    )

    with pytest.raises(ValueError):
        score_candidate_pairs(
            trained_model,
            pair_ids,
            features,
        )


def test_required_id_columns_are_checked(
    trained_model,
    features,
):
    pair_ids = pd.DataFrame(
        {
            "wrong_id": ["S2-001"],
            "candidate_entity_id": ["S2-001"],
        }
    )

    with pytest.raises(ValueError):
        score_candidate_pairs(
            trained_model,
            pair_ids,
            features.iloc[[0]],
        )


def test_scores_to_entity_mapping():
    scored_pairs = pd.DataFrame(
        {
            "source1_entity_id": [
                "S1-001",
                "S1-001",
                "S1-002",
            ],
            "candidate_entity_id": [
                "S2-001",
                "S3-001",
                "S2-010",
            ],
            "match_probability": [
                0.97,
                0.91,
                0.12,
            ],
        }
    )

    result = scores_to_entity_mapping(
        scored_pairs
    )

    assert result == {
        "S1-001": {
            "S2-001": 0.97,
            "S3-001": 0.91,
        },
        "S1-002": {
            "S2-010": 0.12,
        },
    }


def test_duplicate_pairs_are_rejected():
    scored_pairs = pd.DataFrame(
        {
            "source1_entity_id": [
                "S1-001",
                "S1-001",
            ],
            "candidate_entity_id": [
                "S2-001",
                "S2-001",
            ],
            "match_probability": [
                0.90,
                0.91,
            ],
        }
    )

    with pytest.raises(ValueError):
        scores_to_entity_mapping(
            scored_pairs
        )


def test_invalid_probability_is_rejected():
    scored_pairs = pd.DataFrame(
        {
            "source1_entity_id": ["S1-001"],
            "candidate_entity_id": ["S2-001"],
            "match_probability": [1.5],
        }
    )

    with pytest.raises(ValueError):
        scores_to_entity_mapping(
            scored_pairs
        )