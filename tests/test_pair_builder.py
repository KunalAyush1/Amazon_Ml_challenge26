from business_entity_resol.datasets.pair_builder import build_pair_dataset


def test_build_pair_dataset_creates_feature_rows():
    source1_records = [
        {
            "entity_id": "S1-001",
            "name_norm": "abc private limited",
            "name_compact": "abcprivatelimited",
            "address_norm": "12 main road",
            "address_compact": "12mainroad",
            "address_numbers": ["12"],
            "country": "India",
        }
    ]

    candidate_records = [
        {
            "source1_entity_id": "S1-001",
            "candidate_entity_id": "S2-001",
            "source": "S2",
            "blocker_provenance": {"exact"},
            "ranks": {},
        },
        {
            "source1_entity_id": "S1-001",
            "candidate_entity_id": "S2-002",
            "source": "S2",
            "blocker_provenance": {"tfidf"},
            "ranks": {"tfidf": 2},
        },
    ]

    candidate_lookup = {
        "S2-001": {
            "entity_id": "S2-001",
            "name_norm": "abc private limited",
            "name_compact": "abcprivatelimited",
            "address_norm": "12 main road",
            "address_compact": "12mainroad",
            "address_numbers": ["12"],
            "country": "India",
        },
        "S2-002": {
            "entity_id": "S2-002",
            "name_norm": "different company",
            "name_compact": "differentcompany",
            "address_norm": "99 other road",
            "address_compact": "99otherroad",
            "address_numbers": ["99"],
            "country": "India",
        },
    }

    ground_truth = {
        "S1-001": {"S2-001"},
    }

    rows = build_pair_dataset(
        source1_records,
        candidate_records,
        ground_truth,
        candidate_lookup,
        negative_ratio=1,
        random_seed=42,
    )

    assert len(rows) == 2

    labels = {
        row["candidate_entity_id"]: row["label"]
        for row in rows
    }

    assert labels["S2-001"] == 1
    assert labels["S2-002"] == 0

    positive_row = next(
        row for row in rows
        if row["candidate_entity_id"] == "S2-001"
    )

    assert positive_row["candidate_source"] == "S2"
    assert "name_ratio" in positive_row
    assert "address_ratio" in positive_row
    assert "numeric_jaccard" in positive_row
    assert "country_exact" in positive_row
    assert "blocker_num_blockers" in positive_row


def test_positive_pair_outside_candidate_pool_is_not_added():
    source1_records = [
        {"entity_id": "S1-001"},
    ]

    candidate_records = [
        {
            "source1_entity_id": "S1-001",
            "candidate_entity_id": "S2-002",
            "source": "S2",
        },
    ]

    candidate_lookup = {
        "S2-002": {
            "entity_id": "S2-002",
            "name_norm": "company",
            "address_norm": "road",
            "country": "India",
        },
    }

    ground_truth = {
        "S1-001": {"S2-001"},
    }

    rows = build_pair_dataset(
        source1_records,
        candidate_records,
        ground_truth,
        candidate_lookup,
    )

    assert all(
        row["candidate_entity_id"] != "S2-001"
        for row in rows
    )


def test_missing_candidate_record_is_skipped():
    source1_records = [
        {"entity_id": "S1-001"},
    ]

    candidate_records = [
        {
            "source1_entity_id": "S1-001",
            "candidate_entity_id": "S2-001",
            "source": "S2",
        },
    ]

    candidate_lookup = {}

    ground_truth = {
        "S1-001": {"S2-001"},
    }

    rows = build_pair_dataset(
        source1_records,
        candidate_records,
        ground_truth,
        candidate_lookup,
    )

    assert rows == []