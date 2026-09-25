from business_entity_resol.datasets.positive_pairs import generate_positive_pairs


def test_generate_positive_pairs(tmp_path):
    ground_truth = tmp_path / "ground_truth.tsv"

    ground_truth.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-001\tS2-001,S3-001\n"
        "S1-002\tS2-002\n"
        "S1-003\t\n"
        "S1-004\t-\n",
        encoding="utf-8",
    )

    pairs = list(generate_positive_pairs(ground_truth))

    assert pairs == [
        {
            "source1_entity_id": "S1-001",
            "candidate_entity_id": "S2-001",
            "label": 1,
        },
        {
            "source1_entity_id": "S1-001",
            "candidate_entity_id": "S3-001",
            "label": 1,
        },
        {
            "source1_entity_id": "S1-002",
            "candidate_entity_id": "S2-002",
            "label": 1,
        },
    ]


def test_generate_positive_pairs_requires_columns(tmp_path):
    ground_truth = tmp_path / "invalid.tsv"

    ground_truth.write_text(
        "wrong_column\tother_column\n"
        "S1-001\tS2-001\n",
        encoding="utf-8",
    )

    try:
        list(generate_positive_pairs(ground_truth))
    except ValueError as exc:
        assert "source1_entity_id" in str(exc)
        assert "matched_entity_ids" in str(exc)
    else:
        raise AssertionError("Expected ValueError")