"""Training utilities for the LightGBM entity matcher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from business_entity_resol.models.lightgbm.model import LightGBMMatcher


@dataclass(frozen=True)
class LightGBMTrainingResult:
    """Artifacts returned after training."""

    model: LightGBMMatcher
    train_rows: int
    validation_rows: int | None
    train_positive_rate: float
    validation_positive_rate: float | None


def train_lightgbm_matcher(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
    params: dict | None = None,
) -> LightGBMTrainingResult:
    """Train a LightGBM matcher.

    This function deliberately does NOT perform:
    - candidate generation
    - feature engineering
    - train/validation splitting
    - threshold selection
    - entity-level prediction selection

    Those responsibilities belong elsewhere in the pipeline.
    """
    if X_valid is None and y_valid is not None:
        raise ValueError(
            "y_valid was provided but X_valid is None."
        )

    if X_valid is not None and y_valid is None:
        raise ValueError(
            "X_valid was provided but y_valid is None."
        )

    model = LightGBMMatcher(params=params)

    eval_set = None

    if X_valid is not None and y_valid is not None:
        eval_set = (X_valid, y_valid)

    model.fit(
        X=X_train,
        y=y_train,
        eval_set=eval_set,
    )

    validation_positive_rate = None

    if y_valid is not None:
        validation_positive_rate = float(y_valid.mean())

    return LightGBMTrainingResult(
        model=model,
        train_rows=len(X_train),
        validation_rows=(
            len(X_valid)
            if X_valid is not None
            else None
        ),
        train_positive_rate=float(y_train.mean()),
        validation_positive_rate=validation_positive_rate,
    )


def save_trained_matcher(
    result: LightGBMTrainingResult,
    path: str | Path,
) -> None:
    """Save the trained LightGBM matcher."""
    result.model.save(path)