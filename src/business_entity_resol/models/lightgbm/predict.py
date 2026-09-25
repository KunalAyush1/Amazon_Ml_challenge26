"""Prediction utilities for the LightGBM entity matcher."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from business_entity_resol.models.lightgbm.model import LightGBMMatcher


REQUIRED_ID_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
)


def score_candidate_pairs(
    model: LightGBMMatcher,
    pair_ids: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Score candidate pairs using a trained LightGBM matcher.

    Parameters
    ----------
    model:
        Fitted LightGBMMatcher.

    pair_ids:
        DataFrame containing:
            source1_entity_id
            candidate_entity_id

    features:
        Numerical feature matrix with one row per candidate pair.

    Returns
    -------
    pd.DataFrame
        Columns:
            source1_entity_id
            candidate_entity_id
            match_probability

    Notes
    -----
    The row ordering of `pair_ids` and `features` must be identical.
    """
    if not model.is_fitted_:
        raise RuntimeError(
            "LightGBMMatcher must be fitted before scoring."
        )

    missing_columns = [
        column
        for column in REQUIRED_ID_COLUMNS
        if column not in pair_ids.columns
    ]

    if missing_columns:
        raise ValueError(
            "pair_ids is missing required columns: "
            f"{missing_columns}"
        )

    if len(pair_ids) != len(features):
        raise ValueError(
            "pair_ids and features must contain the same number "
            f"of rows: {len(pair_ids)} != {len(features)}"
        )

    probabilities = model.predict_proba(features)

    result = pair_ids.loc[
        :,
        list(REQUIRED_ID_COLUMNS),
    ].copy()

    result["match_probability"] = probabilities

    return result.reset_index(drop=True)


def scores_to_entity_mapping(
    scored_pairs: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Convert pair scores into the structure used by threshold search.

    Input:
        source1_entity_id | candidate_entity_id | match_probability

    Output:
        {
            "S1-001": {
                "S2-001": 0.97,
                "S3-001": 0.91,
            },
            "S1-002": {
                "S2-010": 0.22,
            },
        }
    """
    required_columns = {
        "source1_entity_id",
        "candidate_entity_id",
        "match_probability",
    }

    missing = required_columns - set(scored_pairs.columns)

    if missing:
        raise ValueError(
            "scored_pairs is missing required columns: "
            f"{sorted(missing)}"
        )

    if scored_pairs.empty:
        raise ValueError(
            "Cannot create entity score mapping from an empty dataframe."
        )

    if scored_pairs[
        ["source1_entity_id", "candidate_entity_id"]
    ].isnull().any().any():
        raise ValueError(
            "Entity IDs must not contain missing values."
        )

    probabilities = scored_pairs["match_probability"]

    if not np.isfinite(probabilities.to_numpy()).all():
        raise ValueError(
            "match_probability contains NaN or infinite values."
        )

    if (
        (probabilities < 0.0)
        | (probabilities > 1.0)
    ).any():
        raise ValueError(
            "match_probability must be between 0 and 1."
        )

    result: dict[str, dict[str, float]] = {}

    for row in scored_pairs.itertuples(index=False):
        source1_id = str(row.source1_entity_id)
        candidate_id = str(row.candidate_entity_id)
        probability = float(row.match_probability)

        if source1_id not in result:
            result[source1_id] = {}

        if candidate_id in result[source1_id]:
            raise ValueError(
                "Duplicate candidate pair detected: "
                f"{source1_id} -> {candidate_id}"
            )

        result[source1_id][candidate_id] = probability

    return result


def load_model_and_score(
    model_path: str | Path,
    pair_ids: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Load a saved LightGBM model and score candidate pairs."""
    model = LightGBMMatcher.load(model_path)

    return score_candidate_pairs(
        model=model,
        pair_ids=pair_ids,
        features=features,
    )