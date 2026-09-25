from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataWriteError(ValueError):
    """Raised when processed data cannot be written safely."""


def write_parquet(
    df: pd.DataFrame,
    path: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    """
    Write a DataFrame to a Parquet file.

    Parameters
    ----------
    df:
        DataFrame to write.
    path:
        Destination Parquet file.
    overwrite:
        Whether an existing file may be replaced.

    Returns
    -------
    Path
        The written file path.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataWriteError("Expected a pandas DataFrame.")

    path = Path(path)

    if path.suffix.lower() != ".parquet":
        raise DataWriteError(
            f"Expected a .parquet output path, got: {path}"
        )

    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(path, index=False)
    except Exception as exc:
        raise DataWriteError(
            f"Failed to write Parquet file: {path}"
        ) from exc

    return path


def write_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    """
    Write a DataFrame to a CSV file.

    This is provided for lightweight/interoperability outputs.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataWriteError("Expected a pandas DataFrame.")

    path = Path(path)

    if path.suffix.lower() != ".csv":
        raise DataWriteError(
            f"Expected a .csv output path, got: {path}"
        )

    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(path, index=False)
    except Exception as exc:
        raise DataWriteError(
            f"Failed to write CSV file: {path}"
        ) from exc

    return path