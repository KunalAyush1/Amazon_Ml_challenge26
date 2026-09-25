import numpy as np
import pandas as pd
import pytest

from business_entity_resol.models.lightgbm.model import (
    LightGBMMatcher,
)


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, np.ndarray]:
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

    return X, y


def test_lightgbm_fit(training_data):
    X, y = training_data

    model = LightGBMMatcher()

    model.fit(X, y)

    assert model.is_fitted_ is True
    assert model.feature_names_ == list(X.columns)


def test_lightgbm_predict_proba(training_data):
    X, y = training_data

    model = LightGBMMatcher()
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X),)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_lightgbm_predict(training_data):
    X, y = training_data

    model = LightGBMMatcher()
    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == (len(X),)
    assert set(predictions).issubset({0, 1})


def test_unfitted_model_cannot_predict(training_data):
    X, _ = training_data

    model = LightGBMMatcher()

    with pytest.raises(RuntimeError):
        model.predict_proba(X)


def test_invalid_target_is_rejected(training_data):
    X, _ = training_data

    model = LightGBMMatcher()

    with pytest.raises(ValueError):
        model.fit(
            X,
            [0, 1, 2, 0, 1, 0],
        )


def test_mismatched_lengths_are_rejected(training_data):
    X, _ = training_data

    model = LightGBMMatcher()

    with pytest.raises(ValueError):
        model.fit(
            X,
            [0, 1],
        )


def test_missing_feature_values_are_rejected():
    X = pd.DataFrame(
        {
            "name_similarity": [0.9, np.nan],
            "address_similarity": [0.9, 0.2],
        }
    )

    y = np.array([1, 0])

    model = LightGBMMatcher()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_feature_importance(training_data):
    X, y = training_data

    model = LightGBMMatcher()
    model.fit(X, y)

    importance = model.feature_importance()

    assert list(importance.columns) == [
        "feature",
        "importance",
    ]

    assert set(importance["feature"]) == set(X.columns)


def test_save_and_load(tmp_path, training_data):
    X, y = training_data

    model = LightGBMMatcher()
    model.fit(X, y)

    path = tmp_path / "lightgbm_matcher.joblib"

    model.save(path)

    loaded = LightGBMMatcher.load(path)

    original_scores = model.predict_proba(X)
    loaded_scores = loaded.predict_proba(X)

    np.testing.assert_allclose(
        original_scores,
        loaded_scores,
    )