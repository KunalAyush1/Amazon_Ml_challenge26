"""Leakage-safe validation utilities for entity resolution."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def group_train_val_split(
    df: pd.DataFrame,
    group_col: str = "source1_entity_id",
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a pair-level dataframe without splitting an S1 entity.

    All rows belonging to the same Source-1 entity remain entirely
    in either train or validation.

    Parameters
    ----------
    df:
        Pair-level dataframe.
    group_col:
        Column identifying the Source-1 entity.
    val_size:
        Fraction of unique groups assigned to validation.
    random_state:
        Random seed for reproducibility.
    """
    if group_col not in df.columns:
        raise ValueError(
            f"Missing required group column: {group_col}"
        )

    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")

    if df.empty:
        raise ValueError("Cannot split an empty dataframe.")

    groups = df[group_col]

    if groups.isna().any():
        raise ValueError(
            f"Column '{group_col}' contains missing group IDs."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=random_state,
    )

    train_idx, val_idx = next(
        splitter.split(df, groups=groups)
    )

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    return train_df, val_df


def group_kfold_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "source1_entity_id",
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield leakage-safe GroupKFold train/validation splits.

    Every Source-1 entity appears in exactly one validation fold.
    """
    if group_col not in df.columns:
        raise ValueError(
            f"Missing required group column: {group_col}"
        )

    if df.empty:
        raise ValueError("Cannot split an empty dataframe.")

    if df[group_col].isna().any():
        raise ValueError(
            f"Column '{group_col}' contains missing group IDs."
        )

    unique_groups = df[group_col].nunique()

    if n_splits > unique_groups:
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of unique "
            f"groups={unique_groups}."
        )

    splitter = GroupKFold(n_splits=n_splits)

    for train_idx, val_idx in splitter.split(
        df,
        groups=df[group_col],
    ):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        yield train_df, val_df