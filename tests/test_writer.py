from pathlib import Path

import pandas as pd
import pytest

from business_entity_resol.io.writer import DataWriteError, write_csv, write_parquet


def test_write_parquet(tmp_path: Path):
    df = pd.DataFrame(
        {
            "entity_id": ["S1-001", "S1-002"],
            "name_norm": ["abc store", "test cafe"],
        }
    )

    output_path = tmp_path / "output.parquet"

    result = write_parquet(df, output_path)

    assert result == output_path
    assert output_path.exists()

    loaded = pd.read_parquet(output_path)

    pd.testing.assert_frame_equal(loaded, df)


def test_write_parquet_rejects_wrong_extension(tmp_path: Path):
    df = pd.DataFrame({"entity_id": ["S1-001"]})

    with pytest.raises(DataWriteError):
        write_parquet(df, tmp_path / "output.csv")


def test_write_parquet_respects_overwrite_flag(tmp_path: Path):
    df = pd.DataFrame({"entity_id": ["S1-001"]})

    output_path = tmp_path / "output.parquet"

    write_parquet(df, output_path)

    with pytest.raises(FileExistsError):
        write_parquet(df, output_path, overwrite=False)


def test_write_csv(tmp_path: Path):
    df = pd.DataFrame(
        {
            "entity_id": ["S1-001"],
            "name_norm": ["abc store"],
        }
    )

    output_path = tmp_path / "output.csv"

    result = write_csv(df, output_path)

    assert result == output_path
    assert output_path.exists()

    loaded = pd.read_csv(output_path)

    pd.testing.assert_frame_equal(loaded, df)