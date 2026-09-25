import pandas as pd

from business_entity_resol.datasets.candidate_features import (
    build_candidate_feature_dataframe,
    feature_columns,
)


def test_build_candidate_feature_dataframe():
    source1 = [
        {
            "entity_id": "S1-001",
            "business_name": "ABC Pvt Ltd",
            "business_address": "10 Main Road",
            "country": "India",
            "name_norm": "abc pvt ltd",
            "name_compact": "abcpvtltd",
            "address_norm": "10 main road",
            "address_compact": "10mainroad",
            "address_numbers": ["10"],
        }
    ]

    candidates = [
        {
            "source1_entity_id": "S1-001",
            "candidate_entity_id": "S2-001",
            "source": "S2",
            "blocker_provenance": ["exact", "numeric"],
            "ranks": {"numeric": 1},
            "num_blockers": 2,
            "best_rank": 1,
        }
    ]

    candidate_lookup = {
        "S2-001": {
            "entity_id": "S2-001",
            "business_name": "ABC Pvt Ltd",
            "business_address": "10 Main Road",
            "country": "India",
            "name_norm": "abc pvt ltd",
            "name_compact": "abcpvtltd",
            "address_norm": "10 main road",
            "address_compact": "10mainroad",
            "address_numbers": ["10"],
        }
    }

    df = build_candidate_feature_dataframe(
        source1,
        candidates,
        candidate_lookup,
    )

    assert len(df) == 1
    assert df.loc[0, "source1_entity_id"] == "S1-001"
    assert df.loc[0, "candidate_entity_id"] == "S2-001"

    assert df.loc[0, "candidate_is_s2"] == 1
    assert df.loc[0, "candidate_is_s3"] == 0

    assert df.loc[0, "name_norm_exact"] == 1
    assert df.loc[0, "address_norm_exact"] == 1

    assert df.loc[0, "blocker_num_blockers"] == 2
    assert df.loc[0, "blocker_best_rank"] == 1


def test_feature_columns_excludes_identifiers():
    df = pd.DataFrame(
        {
            "source1_entity_id": ["S1-001"],
            "candidate_entity_id": ["S2-001"],
            "candidate_source": ["S2"],
            "label": [1],
            "name_ratio": [0.9],
            "address_ratio": [0.8],
        }
    )

    columns = feature_columns(df)

    assert columns == [
        "name_ratio",
        "address_ratio",
    ]