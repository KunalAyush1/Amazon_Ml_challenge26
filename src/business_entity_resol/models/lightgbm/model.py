"""LightGBM wrapper for binary entity matching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


class LightGBMMatcher:
    """Thin wrapper around LightGBM for entity-pair classification.

    The model expects one row per candidate pair and numerical features
    produced by the feature-engineering layer.

    Target:
        1 -> same entity
        0 -> different entity

    Thresholding and final entity-level selection are intentionally
    handled outside this class.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the matcher.

        Parameters
        ----------
        params:
            Optional LightGBM parameters. The defaults are deliberately
            conservative; final hyperparameter tuning comes later.
        """
        default_params: dict[str, Any] = {
            "objective": "binary",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": 42,
            "n_jobs": -1,
        }

        if params:
            default_params.update(params)

        self.params = default_params
        self.model: lgb.LGBMClassifier | None = None
        self.feature_names_: list[str] | None = None
        self.is_fitted_: bool = False

    @staticmethod
    def _validate_features(
        X: pd.DataFrame | np.ndarray,
    ) -> pd.DataFrame | np.ndarray:
        """Validate feature matrix."""
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                "X must be a pandas DataFrame or NumPy ndarray."
            )

        if len(X) == 0:
            raise ValueError("Feature matrix X is empty.")

        if isinstance(X, pd.DataFrame):
            if X.columns.duplicated().any():
                raise ValueError(
                    "Feature matrix contains duplicate column names."
                )

            if X.isnull().any().any():
                raise ValueError(
                    "Feature matrix contains missing values."
                )

        else:
            if not np.isfinite(X).all():
                raise ValueError(
                    "NumPy feature matrix contains NaN or infinite values."
                )

        return X

    @staticmethod
    def _validate_target(
        y: pd.Series | np.ndarray | list[int],
    ) -> np.ndarray:
        """Validate binary target."""
        y_array = np.asarray(y)

        if y_array.ndim != 1:
            raise ValueError("Target y must be one-dimensional.")

        if len(y_array) == 0:
            raise ValueError("Target y is empty.")

        unique_values = set(np.unique(y_array))

        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"Target y must contain only 0/1 labels. "
                f"Found: {sorted(unique_values)}"
            )

        return y_array.astype(np.int8)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | list[int],
        eval_set: tuple[
            pd.DataFrame | np.ndarray,
            pd.Series | np.ndarray | list[int],
        ]
        | None = None,
    ) -> "LightGBMMatcher":
        """Fit the LightGBM matcher."""
        X = self._validate_features(X)
        y_array = self._validate_target(y)

        if len(X) != len(y_array):
            raise ValueError(
                f"X and y have different lengths: {len(X)} != {len(y_array)}"
            )

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [
                f"feature_{i}"
                for i in range(X.shape[1])
            ]

        self.model = lgb.LGBMClassifier(**self.params)

        fit_kwargs: dict[str, Any] = {}

        if eval_set is not None:
            X_valid, y_valid = eval_set

            X_valid = self._validate_features(X_valid)
            y_valid = self._validate_target(y_valid)

            if len(X_valid) != len(y_valid):
                raise ValueError(
                    "Validation feature matrix and target have "
                    "different lengths."
                )

            fit_kwargs["eval_X"] = X_valid
            fit_kwargs["eval_y"] = y_valid

        self.model.fit(
            X,
            y_array,
            **fit_kwargs,
        )

        self.is_fitted_ = True

        return self

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Return P(match) for each candidate pair."""
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError(
                "LightGBMMatcher must be fitted before prediction."
            )

        X = self._validate_features(X)

        probabilities = self.model.predict_proba(X)

        # Column 1 corresponds to the positive class: match = 1.
        return probabilities[:, 1]

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Return binary predictions using LightGBM's default 0.5 threshold.

        This method is mainly for diagnostics.

        Final competition thresholding must be handled by the
        entity-level evaluation/selection layer.
        """
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError(
                "LightGBMMatcher must be fitted before prediction."
            )

        X = self._validate_features(X)

        return self.model.predict(X).astype(np.int8)

    def feature_importance(
        self,
        importance_type: str = "gain",
    ) -> pd.DataFrame:
        """Return feature importance as a DataFrame."""
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError(
                "LightGBMMatcher must be fitted before requesting "
                "feature importance."
            )

        if self.feature_names_ is None:
            raise RuntimeError("Feature names are not available.")

        if importance_type not in {"gain", "split"}:
            raise ValueError(
                "importance_type must be either 'gain' or 'split'."
            )

        importance = self.model.booster_.feature_importance(
            importance_type=importance_type
        )

        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "importance": importance,
                }
            )
            .sort_values(
                "importance",
                ascending=False,
                ignore_index=True,
            )
        )

    def save(self, path: str | Path) -> None:
        """Save the fitted matcher."""
        if not self.is_fitted_:
            raise RuntimeError(
                "Cannot save an unfitted LightGBMMatcher."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMMatcher":
        """Load a previously saved matcher."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Model file does not exist: {path}"
            )

        model = joblib.load(path)

        if not isinstance(model, cls):
            raise TypeError(
                f"Expected {cls.__name__}, got {type(model).__name__}"
            )

        return model