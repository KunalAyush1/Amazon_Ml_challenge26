from business_entity_resol.blocking.exact_blocker import ExactBlocker
from business_entity_resol.blocking.candidate_store import CandidateRecord
from business_entity_resol.blocking.candidate_union import CandidateUnion
from business_entity_resol.blocking.rare_token_blocker import (
    RareTokenBlocker,
)
from business_entity_resol.blocking.numeric_blocker import (
    NumericBlocker,
)

def test_exact_blocker_matches_normalized_name():
    source2 = [
        {
            "entity_id": "S2-001",
            "name_norm": "abc private limited",
            "name_compact": "abcprivatelimited",
            "address_norm": "21 mg road delhi",
            "address_compact": "21mgroaddelhi",
        }
    ]

    source3 = [
        {
            "entity_id": "S3-001",
            "name_norm": "xyz traders",
            "name_compact": "xyztraders",
            "address_norm": "10 main road",
            "address_compact": "10mainroad",
        }
    ]

    source1 = {
        "entity_id": "S1-001",
        "name_norm": "abc private limited",
        "name_compact": "abcprivatelimited",
        "address_norm": "different address",
        "address_compact": "differentaddress",
    }

    blocker = ExactBlocker()
    blocker.fit(source2, source3)

    candidates = blocker.retrieve(source1)

    candidate_ids = {candidate.candidate_id for candidate in candidates}

    assert candidate_ids == {"S2-001"}


def test_exact_blocker_matches_normalized_address():
    source2 = [
        {
            "entity_id": "S2-001",
            "name_norm": "different name",
            "name_compact": "differentname",
            "address_norm": "21 mg road delhi",
            "address_compact": "21mgroaddelhi",
        }
    ]

    source1 = {
        "entity_id": "S1-001",
        "name_norm": "abc business",
        "name_compact": "abcbusiness",
        "address_norm": "21 mg road delhi",
        "address_compact": "21mgroaddelhi",
    }

    blocker = ExactBlocker()
    blocker.fit(source2, [])

    candidates = blocker.retrieve(source1)

    assert len(candidates) == 1
    assert candidates[0].candidate_id == "S2-001"
    assert "address_norm" in candidates[0].matched_keys


def test_exact_blocker_unions_matches_from_multiple_fields():
    source2 = [
        {
            "entity_id": "S2-001",
            "name_norm": "abc business",
            "name_compact": "abcbusiness",
            "address_norm": "address one",
            "address_compact": "addressone",
        },
        {
            "entity_id": "S2-002",
            "name_norm": "different business",
            "name_compact": "differentbusiness",
            "address_norm": "address one",
            "address_compact": "addressone",
        },
    ]

    source1 = {
        "entity_id": "S1-001",
        "name_norm": "abc business",
        "name_compact": "abcbusiness",
        "address_norm": "address one",
        "address_compact": "addressone",
    }

    blocker = ExactBlocker()
    blocker.fit(source2, [])

    candidates = blocker.retrieve(source1)

    candidate_ids = {candidate.candidate_id for candidate in candidates}

    assert candidate_ids == {"S2-001", "S2-002"}

    first = next(c for c in candidates if c.candidate_id == "S2-001")

    assert "name_norm" in first.matched_keys
    assert "address_norm" in first.matched_keys


def test_exact_blocker_includes_both_s2_and_s3():
    source2 = [
        {
            "entity_id": "S2-001",
            "name_norm": "abc business",
        }
    ]

    source3 = [
        {
            "entity_id": "S3-001",
            "name_norm": "abc business",
        }
    ]

    source1 = {
        "entity_id": "S1-001",
        "name_norm": "abc business",
    }

    blocker = ExactBlocker()
    blocker.fit(source2, source3)

    candidates = blocker.retrieve(source1)

    assert {(c.candidate_id, c.source) for c in candidates} == {
        ("S2-001", "S2"),
        ("S3-001", "S3"),
    }


def test_exact_blocker_never_returns_source1():
    source2 = [
        {
            "entity_id": "S2-001",
            "name_norm": "abc business",
        }
    ]

    source3 = [
        {
            "entity_id": "S3-001",
            "name_norm": "abc business",
        }
    ]

    source1 = {
        "entity_id": "S1-001",
        "name_norm": "abc business",
    }

    blocker = ExactBlocker()
    blocker.fit(source2, source3)

    candidates = blocker.retrieve(source1)

    candidate_ids = {c.candidate_id for c in candidates}

    assert "S1-001" not in candidate_ids


def test_exact_blocker_skips_high_frequency_keys():
    source2 = [
        {
            "entity_id": f"S2-{i:03d}",
            "name_norm": "restaurant",
        }
        for i in range(5)
    ]

    source1 = {
        "entity_id": "S1-001",
        "name_norm": "restaurant",
    }

    blocker = ExactBlocker(max_frequency=3)
    blocker.fit(source2, [])

    candidates = blocker.retrieve(source1)

    assert candidates == []


def test_exact_blocker_handles_missing_values():
    source2 = [
        {
            "entity_id": "S2-001",
            "name_norm": None,
            "address_norm": "some address",
        }
    ]

    source1 = {
        "entity_id": "S1-001",
        "name_norm": None,
        "address_norm": None,
    }

    blocker = ExactBlocker()
    blocker.fit(source2, [])

    candidates = blocker.retrieve(source1)

    assert candidates == []


def test_exact_blocker_deduplicates_candidate():
    source2 = [
        {
            "entity_id": "S2-001",
            "name_norm": "abc business",
            "name_compact": "abcbusiness",
        }
    ]

    source1 = {
        "entity_id": "S1-001",
        "name_norm": "abc business",
        "name_compact": "abcbusiness",
    }

    blocker = ExactBlocker()
    blocker.fit(source2, [])

    candidates = blocker.retrieve(source1)

    assert len(candidates) == 1
    assert candidates[0].candidate_id == "S2-001"
    assert set(candidates[0].matched_keys) == {
        "name_norm",
        "name_compact",
    }
    
    from business_entity_resol.blocking.candidate_store import CandidateRecord
from business_entity_resol.blocking.candidate_union import CandidateUnion


def test_candidate_record_preserves_provenance():
    candidate = CandidateRecord(
        source1_entity_id="S1-001",
        candidate_entity_id="S2-001",
        source="S2",
    )

    candidate.add_provenance("exact")
    candidate.add_provenance("tfidf", rank=3)

    assert candidate.blocker_provenance == {"exact", "tfidf"}
    assert candidate.num_blockers == 2
    assert candidate.ranks["tfidf"] == 3
    assert candidate.best_rank == 3


def test_candidate_union_deduplicates_candidates():
    union = CandidateUnion()

    union.add(
        source1_entity_id="S1-001",
        candidate_entity_id="S2-001",
        source="S2",
        blocker="exact",
    )

    union.add(
        source1_entity_id="S1-001",
        candidate_entity_id="S2-001",
        source="S2",
        blocker="tfidf",
        rank=5,
    )

    candidates = union.get("S1-001")

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.candidate_entity_id == "S2-001"
    assert candidate.blocker_provenance == {"exact", "tfidf"}
    assert candidate.num_blockers == 2
    assert candidate.best_rank == 5


def test_candidate_union_keeps_different_candidates():
    union = CandidateUnion()

    union.add(
        "S1-001",
        "S2-001",
        "S2",
        "exact",
    )

    union.add(
        "S1-001",
        "S2-002",
        "S2",
        "tfidf",
        rank=2,
    )

    candidates = union.get("S1-001")

    assert len(candidates) == 2

    assert {
        candidate.candidate_entity_id
        for candidate in candidates
    } == {
        "S2-001",
        "S2-002",
    }


def test_candidate_union_keeps_s2_and_s3_separate():
    union = CandidateUnion()

    union.add(
        "S1-001",
        "S2-001",
        "S2",
        "exact",
    )

    union.add(
        "S1-001",
        "S3-001",
        "S3",
        "exact",
    )

    candidates = union.get("S1-001")

    assert len(candidates) == 2

    assert {
        (candidate.candidate_entity_id, candidate.source)
        for candidate in candidates
    } == {
        ("S2-001", "S2"),
        ("S3-001", "S3"),
    }


def test_candidate_union_rejects_source_change():
    union = CandidateUnion()

    union.add(
        "S1-001",
        "S2-001",
        "S2",
        "exact",
    )

    try:
        union.add(
            "S1-001",
            "S2-001",
            "S3",
            "tfidf",
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError when candidate source changes"
    )


def test_candidate_union_filters_by_source1_entity():
    union = CandidateUnion()

    union.add(
        "S1-001",
        "S2-001",
        "S2",
        "exact",
    )

    union.add(
        "S1-002",
        "S2-002",
        "S2",
        "exact",
    )

    candidates = union.get_for_entity("S1-001")

    assert len(candidates) == 1
    assert candidates[0].candidate_entity_id == "S2-001"


def test_candidate_union_add_many():
    union = CandidateUnion()

    candidates = [
        {
            "source1_entity_id": "S1-001",
            "candidate_entity_id": "S2-001",
            "source": "S2",
            "rank": 1,
        },
        {
            "source1_entity_id": "S1-001",
            "candidate_entity_id": "S2-002",
            "source": "S2",
            "rank": 2,
        },
    ]

    union.add_many(candidates, blocker="tfidf")

    result = union.get("S1-001")

    assert len(result) == 2
    assert all(
        candidate.blocker_provenance == {"tfidf"}
        for candidate in result
    )


def test_candidate_union_to_dicts():
    union = CandidateUnion()

    union.add(
        "S1-001",
        "S2-001",
        "S2",
        "exact",
    )

    result = union.to_dicts("S1-001")

    assert result == [
        {
            "source1_entity_id": "S1-001",
            "candidate_entity_id": "S2-001",
            "source": "S2",
            "blocker_provenance": ["exact"],
            "ranks": {},
            "num_blockers": 1,
            "best_rank": None,
        }
    ]
    
def test_rare_token_blocker_matches_rare_name_token():
    source2 = [
        {
            "entity_id": "S2-001",
            "name_tokens": ["shree", "krishna", "dental", "hospital"],
            "address_tokens": [],
        },
        {
            "entity_id": "S2-002",
            "name_tokens": ["city", "general", "hospital"],
            "address_tokens": [],
        },
    ]

    source1 = {
        "entity_id": "S1-001",
        "name_tokens": ["krishna", "dental", "clinic"],
        "address_tokens": [],
    }

    blocker = RareTokenBlocker(max_token_frequency=1)
    blocker.fit(source2, [])

    candidates = blocker.retrieve(source1)

    candidate_ids = {
        candidate.candidate_id
        for candidate in candidates
    }

    assert candidate_ids == {"S2-001"}


def test_rare_token_blocker_ignores_common_token():
    source2 = [
        {
            "entity_id": f"S2-{i:03d}",
            "name_tokens": ["hospital"],
            "address_tokens": [],
        }
        for i in range(5)
    ]

    source1 = {
        "entity_id": "S1-001",
        "name_tokens": ["hospital"],
        "address_tokens": [],
    }

    blocker = RareTokenBlocker(max_token_frequency=2)
    blocker.fit(source2, [])

    candidates = blocker.retrieve(source1)

    assert candidates == []


def test_rare_token_blocker_preserves_multiple_tokens():
    source2 = [
        {
            "entity_id": "S2-001",
            "name_tokens": ["krishna", "dental"],
            "address_tokens": ["delhi"],
        }
    ]

    source1 = {
        "entity_id": "S1-001",
        "name_tokens": ["krishna", "dental"],
        "address_tokens": ["delhi"],
    }

    blocker = RareTokenBlocker(max_token_frequency=10)
    blocker.fit(source2, [])

    candidates = blocker.retrieve(source1)

    assert len(candidates) == 1

    assert set(candidates[0].matched_tokens) == {
        "krishna",
        "dental",
        "delhi",
    }


def test_rare_token_blocker_supports_s2_and_s3():
    source2 = [
        {
            "entity_id": "S2-001",
            "name_tokens": ["krishna"],
            "address_tokens": [],
        }
    ]

    source3 = [
        {
            "entity_id": "S3-001",
            "name_tokens": ["krishna"],
            "address_tokens": [],
        }
    ]

    source1 = {
        "entity_id": "S1-001",
        "name_tokens": ["krishna"],
        "address_tokens": [],
    }

    blocker = RareTokenBlocker(max_token_frequency=10)
    blocker.fit(source2, source3)

    candidates = blocker.retrieve(source1)

    assert {
        (candidate.candidate_id, candidate.source)
        for candidate in candidates
    } == {
        ("S2-001", "S2"),
        ("S3-001", "S3"),
    }


def test_rare_token_blocker_deduplicates_tokens():
    source2 = [
        {
            "entity_id": "S2-001",
            "name_tokens": ["krishna"],
            "address_tokens": [],
        }
    ]

    source1 = {
        "entity_id": "S1-001",
        "name_tokens": ["krishna", "krishna"],
        "address_tokens": [],
    }

    blocker = RareTokenBlocker(max_token_frequency=10)
    blocker.fit(source2, [])

    candidates = blocker.retrieve(source1)

    assert len(candidates) == 1
    assert candidates[0].matched_tokens == ("krishna",)
def test_numeric_blocker_matches_common_number():
    source2 = [
        {
            "entity_id": "S2-001",
            "address_numbers": ["21", "110001"],
        }
    ]

    source1 = {
        "entity_id": "S1-001",
        "address_numbers": ["21"],
    }

    blocker = NumericBlocker()
    blocker.fit(source2, [])

    candidates = blocker.retrieve(source1)

    assert len(candidates) == 1
    assert candidates[0].candidate_id == "S2-001"
    assert candidates[0].source == "S2"
    assert candidates[0].matched_numbers == ("21",)


def test_numeric_blocker_matches_multiple_candidates():
    source2 = [
        {
            "entity_id": "S2-001",
            "address_numbers": ["21"],
        },
        {
            "entity_id": "S2-002",
            "address_numbers": ["21"],
        },
    ]

    source1 = {
        "entity_id": "S1-001",
        "address_numbers": ["21"],
    }

    blocker = NumericBlocker()
    blocker.fit(source2, [])

    candidates = blocker.retrieve(source1)

    assert {
        candidate.candidate_id
        for candidate in candidates
    } == {
        "S2-001",
        "S2-002",
    }


def test_numeric_blocker_supports_s2_and_s3():
    source2 = [
        {
            "entity_id": "S2-001",
            "address_numbers": ["21"],
        }
    ]

    source3 = [
        {
            "entity_id": "S3-001",
            "address_numbers": ["21"],
        }
    ]

    source1 = {
        "entity_id": "S1-001",
        "address_numbers": ["21"],
    }

    blocker = NumericBlocker()
    blocker.fit(source2, source3)

    candidates = blocker.retrieve(source1)

    assert {
        (candidate.candidate_id, candidate.source)
        for candidate in candidates
    } == {
        ("S2-001", "S2"),
        ("S3-001", "S3"),
    }


def test_numeric_blocker_ignores_non_overlapping_numbers():
    source2 = [
        {
            "entity_id": "S2-001",
            "address_numbers": ["100"],
        }
    ]

    source1 = {
        "entity_id": "S1-001",
        "address_numbers": ["21"],
    }

    blocker = NumericBlocker()
    blocker.fit(source2, [])

    candidates = blocker.retrieve(source1)

    assert candidates == []


def test_numeric_blocker_skips_high_frequency_numbers():
    source2 = [
        {
            "entity_id": f"S2-{i:03d}",
            "address_numbers": ["21"],
        }
        for i in range(5)
    ]

    source1 = {
        "entity_id": "S1-001",
        "address_numbers": ["21"],
    }

    blocker = NumericBlocker(max_frequency=3)
    blocker.fit(source2, [])

    candidates = blocker.retrieve(source1)

    assert candidates == []


def test_numeric_blocker_handles_missing_numbers():
    source2 = [
        {
            "entity_id": "S2-001",
            "address_numbers": None,
        }
    ]

    source1 = {
        "entity_id": "S1-001",
        "address_numbers": None,
    }

    blocker = NumericBlocker()
    blocker.fit(source2, [])

    candidates = blocker.retrieve(source1)

    assert candidates == []