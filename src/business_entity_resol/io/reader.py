from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


class DataValidationError(ValueError):
    """Raised when an input dataset violates the expected schema."""


def read_source_file(path: str | Path) -> pd.DataFrame:
    """
    Read one challenge source TSV file.

    Parameters
    ----------
    path:
        Path to the TSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded and schema-validated source data.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype={"entity_id": "string"},
        keep_default_na=False,
    )

    validate_source_dataframe(df, path)

    return df


def validate_source_dataframe(
    df: pd.DataFrame,
    source: str | Path = "dataframe",
) -> None:
    """
    Validate that a source dataframe follows the challenge schema.
    """
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]

    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {missing_columns}"
        )

    if df["entity_id"].isna().any() or (df["entity_id"].str.strip() == "").any():
        raise DataValidationError(
            f"Empty entity_id found in {source}"
        )

    if df["entity_id"].duplicated().any():
        duplicates = (
            df.loc[df["entity_id"].duplicated(), "entity_id"]
            .astype(str)
            .tolist()
        )

        raise DataValidationError(
            f"Duplicate entity_id values found in {source}: {duplicates[:10]}"
        )


def read_train_sources(train_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load the three training source files.
    """
    train_dir = Path(train_dir)

    return {
        "source1": read_source_file(train_dir / "train_source1.tsv"),
        "source2": read_source_file(train_dir / "train_source2.tsv"),
        "source3": read_source_file(train_dir / "train_source3.tsv"),
    }


def read_test_sources(test_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load the three test source files.
    """
    test_dir = Path(test_dir)

    return {
        "source1": read_source_file(test_dir / "test_source1.tsv"),
        "source2": read_source_file(test_dir / "test_source2.tsv"),
        "source3": read_source_file(test_dir / "test_source3.tsv"),
    }