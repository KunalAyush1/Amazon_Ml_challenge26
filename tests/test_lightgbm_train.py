import numpy as np
import pandas as pd
import pytest

from business_entity_resol.models.lightgbm.train import (
    train_lightgbm_matcher,
)


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.Series]:
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

    y = pd.Series([1, 1, 0, 0, 1, 0])

    return X, y


@pytest.fixture
def validation_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "name_similarity": [
                0.96,
                0.15,
                0.82,
                0.25,
            ],
            "address_similarity": [
                0.94,
                0.10,
                0.79,
                0.20,
            ],
            "country_equal": [
                1,
                0,
                1,
                0,
            ],
        }
    )

    y = pd.Series([1, 0, 1, 0])

    return X, y


def test_train_lightgbm_matcher(
    training_data,
    validation_data,
):
    X_train, y_train = training_data
    X_valid, y_valid = validation_data

    result = train_lightgbm_matcher(
        X_train,
        y_train,
        X_valid,
        y_valid,
    )

    assert result.train_rows == len(X_train)
    assert result.validation_rows == len(X_valid)
    assert result.train_positive_rate == pytest.approx(
        y_train.mean()
    )
    assert result.validation_positive_rate == pytest.approx(
        y_valid.mean()
    )
    assert result.model.is_fitted_ is True


def test_training_without_validation(
    training_data,
):
    X_train, y_train = training_data

    result = train_lightgbm_matcher(
        X_train,
        y_train,
    )

    assert result.train_rows == len(X_train)
    assert result.validation_rows is None
    assert result.validation_positive_rate is None


def test_validation_arguments_must_be_paired(
    training_data,
):
    X_train, y_train = training_data

    with pytest.raises(ValueError):
        train_lightgbm_matcher(
            X_train,
            y_train,
            X_valid=X_train,
        )


def test_trained_model_can_predict(
    training_data,
    validation_data,
):
    X_train, y_train = training_data
    X_valid, y_valid = validation_data

    result = train_lightgbm_matcher(
        X_train,
        y_train,
        X_valid,
        y_valid,
    )

    predictions = result.model.predict_proba(X_valid)

    assert len(predictions) == len(X_valid)
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)