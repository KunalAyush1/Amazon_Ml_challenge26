import pandas as pd
import pytest

from business_entity_resol.evaluation.cross_validation import (
    group_kfold_split,
    group_train_val_split,
)


@pytest.fixture
def pair_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source1_entity_id": [
                "S1-001",
                "S1-001",
                "S1-002",
                "S1-002",
                "S1-003",
                "S1-003",
                "S1-004",
                "S1-005",
            ],
            "candidate_entity_id": [
                "S2-001",
                "S2-002",
                "S2-003",
                "S2-004",
                "S3-001",
                "S3-002",
                "S2-005",
                "S3-003",
            ],
            "label": [1, 0, 1, 0, 0, 0, 1, 0],
        }
    )


def test_group_train_val_split_has_no_group_overlap(
    pair_dataframe: pd.DataFrame,
) -> None:
    train_df, val_df = group_train_val_split(
        pair_dataframe,
        val_size=0.4,
        random_state=42,
    )

    train_groups = set(train_df["source1_entity_id"])
    val_groups = set(val_df["source1_entity_id"])

    assert train_groups.isdisjoint(val_groups)


def test_group_train_val_split_preserves_all_rows(
    pair_dataframe: pd.DataFrame,
) -> None:
    train_df, val_df = group_train_val_split(
        pair_dataframe,
        val_size=0.4,
        random_state=42,
    )

    assert len(train_df) + len(val_df) == len(pair_dataframe)


def test_group_train_val_split_is_reproducible(
    pair_dataframe: pd.DataFrame,
) -> None:
    train_a, val_a = group_train_val_split(
        pair_dataframe,
        val_size=0.4,
        random_state=42,
    )

    train_b, val_b = group_train_val_split(
        pair_dataframe,
        val_size=0.4,
        random_state=42,
    )

    assert train_a.index.tolist() == train_b.index.tolist()
    assert val_a.index.tolist() == val_b.index.tolist()


def test_group_train_val_split_rejects_missing_group_column(
    pair_dataframe: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="Missing required group"):
        group_train_val_split(
            pair_dataframe,
            group_col="wrong_column",
        )


def test_group_train_val_split_rejects_missing_groups() -> None:
    df = pd.DataFrame(
        {
            "source1_entity_id": ["S1-001", None],
            "candidate_entity_id": ["S2-001", "S2-002"],
        }
    )

    with pytest.raises(ValueError, match="missing group"):
        group_train_val_split(df)


def test_group_kfold_has_no_group_overlap(
    pair_dataframe: pd.DataFrame,
) -> None:
    splits = list(
        group_kfold_split(
            pair_dataframe,
            n_splits=3,
        )
    )

    assert len(splits) == 3

    validation_groups = []

    for train_df, val_df in splits:
        train_groups = set(train_df["source1_entity_id"])
        val_groups = set(val_df["source1_entity_id"])

        assert train_groups.isdisjoint(val_groups)

        validation_groups.extend(
            val_df["source1_entity_id"].unique()
        )

    # Each Source-1 entity should appear in validation exactly once.
    assert len(validation_groups) == len(
        set(validation_groups)
    )


def test_group_kfold_rejects_too_many_splits(
    pair_dataframe: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        list(
            group_kfold_split(
                pair_dataframe,
                n_splits=10,
            )
        )