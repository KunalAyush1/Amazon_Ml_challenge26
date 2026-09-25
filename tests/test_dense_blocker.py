from business_entity_resol.blocking.dense_blocker import DenseBlocker


def make_record(
    entity_id: str,
    name: str,
    address: str,
    country: str = "india",
) -> dict:
    return {
        "entity_id": entity_id,
        "name_norm": name,
        "address_norm": address,
        "country": country,
    }


def test_dense_blocker_fit_and_retrieve():
    source2 = [
        make_record(
            "S2_1",
            "amazon india",
            "bangalore india",
        ),
        make_record(
            "S2_2",
            "microsoft india",
            "hyderabad india",
        ),
    ]

    source3 = [
        make_record(
            "S3_1",
            "amazon",
            "bengaluru india",
        ),
        make_record(
            "S3_2",
            "apple india",
            "delhi india",
        ),
    ]

    source1 = make_record(
        "S1_1",
        "amazon india",
        "bangalore",
    )

    blocker = DenseBlocker(top_k=3)

    blocker.fit(source2, source3)

    assert len(blocker) == 4

    candidates = blocker.retrieve(source1)

    assert len(candidates) == 3
    assert "S2_1" in candidates


def test_dense_blocker_preserves_source():
    source2 = [
        make_record(
            "S2_1",
            "amazon india",
            "bangalore india",
        )
    ]

    source3 = [
        make_record(
            "S3_1",
            "amazon",
            "bengaluru india",
        )
    ]

    blocker = DenseBlocker(top_k=2)

    blocker.fit(source2, source3)

    s2_record = blocker.get_candidate_record("S2_1")
    s3_record = blocker.get_candidate_record("S3_1")

    assert s2_record["_source"] == "S2"
    assert s3_record["_source"] == "S3"


def test_dense_blocker_retrieve_many():
    source2 = [
        make_record(
            "S2_1",
            "amazon india",
            "bangalore india",
        ),
        make_record(
            "S2_2",
            "microsoft india",
            "hyderabad india",
        ),
    ]

    source3 = [
        make_record(
            "S3_1",
            "apple india",
            "delhi india",
        )
    ]

    source1 = [
        make_record(
            "S1_1",
            "amazon india",
            "bangalore",
        ),
        make_record(
            "S1_2",
            "microsoft india",
            "hyderabad",
        ),
    ]

    blocker = DenseBlocker(top_k=2)

    blocker.fit(source2, source3)

    results = blocker.retrieve_many(source1)

    assert set(results.keys()) == {
        "S1_1",
        "S1_2",
    }

    assert len(results["S1_1"]) == 2
    assert len(results["S1_2"]) == 2


def test_dense_blocker_never_returns_source1():
    source2 = [
        make_record(
            "S2_1",
            "amazon india",
            "bangalore india",
        )
    ]

    source3 = [
        make_record(
            "S3_1",
            "apple india",
            "delhi india",
        )
    ]

    source1 = make_record(
        "S1_1",
        "amazon india",
        "bangalore",
    )

    blocker = DenseBlocker(top_k=10)

    blocker.fit(source2, source3)

    candidates = blocker.retrieve(source1)

    assert "S1_1" not in candidates
    assert set(candidates).issubset({"S2_1", "S3_1"})


def test_dense_blocker_top_k_limit():
    source2 = [
        make_record(
            "S2_1",
            "amazon india",
            "bangalore",
        ),
        make_record(
            "S2_2",
            "amazon services",
            "mumbai",
        ),
        make_record(
            "S2_3",
            "amazon web services",
            "delhi",
        ),
    ]

    source3 = [
        make_record(
            "S3_1",
            "amazon",
            "hyderabad",
        ),
        make_record(
            "S3_2",
            "apple",
            "delhi",
        ),
    ]

    blocker = DenseBlocker(top_k=2)

    blocker.fit(source2, source3)

    source1 = make_record(
        "S1_1",
        "amazon india",
        "bangalore",
    )

    candidates = blocker.retrieve(source1)

    assert len(candidates) == 2