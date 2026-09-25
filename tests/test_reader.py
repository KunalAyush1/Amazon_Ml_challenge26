import pandas as pd
import pytest

from business_entity_resol.io.ground_truth import (
    parse_matched_ids,
    read_ground_truth,
)
from business_entity_resol.io.reader import (
    DataValidationError,
    read_source_file,
)


def test_reader_parses_tsv(tmp_path):
    path = tmp_path / "source.tsv"

    path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00001\tABC Pvt. Ltd.\t21 MG Road\tIndia\n"
    )

    df = read_source_file(path)

    assert list(df.columns) == [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    assert len(df) == 1
    assert df.iloc[0]["entity_id"] == "S1-00001"


def test_entity_id_is_string(tmp_path):
    path = tmp_path / "source.tsv"

    path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00001\tABC\tDelhi\tIndia\n"
    )

    df = read_source_file(path)

    assert isinstance(df.iloc[0]["entity_id"], str)


def test_missing_required_column(tmp_path):
    path = tmp_path / "source.tsv"

    path.write_text(
        "entity_id\tbusiness_name\tcountry\n"
        "S1-00001\tABC\tIndia\n"
    )

    with pytest.raises(DataValidationError, match="business_address"):
        read_source_file(path)


def test_parse_empty_matches():
    assert parse_matched_ids("") == set()


def test_parse_multiple_matches():
    result = parse_matched_ids("S2-1,S2-2,S3-8")

    assert result == {"S2-1", "S2-2", "S3-8"}


def test_ground_truth_file(tmp_path):
    path = tmp_path / "ground_truth.tsv"

    path.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-00001\tS2-00047,S3-00812\n"
        "S1-00002\t\n"
    )

    result = read_ground_truth(path)

    assert result["S1-00001"] == {"S2-00047", "S3-00812"}
    assert result["S1-00002"] == set()