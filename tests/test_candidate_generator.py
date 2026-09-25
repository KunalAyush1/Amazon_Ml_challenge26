from business_entity_resol.blocking.candidate_generator import (
    CandidateGenerator,
)


def test_candidate_generator_merges_blocker_results():
    source2 = [
        {
            "entity_id": "S2-1",
            "name_norm": "alpha cafe",
            "name_compact": "alphacafe",
            "name_tokens": ["alpha", "cafe"],
            "address_norm": "12 main road",
            "address_compact": "12mainroad",
            "address_tokens": ["12", "main", "road"],
            "address_numbers": ["12"],
        }
    ]

    source3 = [
        {
            "entity_id": "S3-1",
            "name_norm": "alpha cafe",
            "name_compact": "alphacafe",
            "name_tokens": ["alpha", "cafe"],
            "address_norm": "12 main road",
            "address_compact": "12mainroad",
            "address_tokens": ["12", "main", "road"],
            "address_numbers": ["12"],
        }
    ]

    source1 = [
        {
            "entity_id": "S1-1",
            "name_norm": "alpha cafe",
            "name_compact": "alphacafe",
            "name_tokens": ["alpha", "cafe"],
            "address_norm": "12 main road",
            "address_compact": "12mainroad",
            "address_tokens": ["12", "main", "road"],
            "address_numbers": ["12"],
        }
    ]

    generator = CandidateGenerator()
    generator.fit(source2, source3)

    candidates = generator.generate(source1)

    pairs = {
        (candidate.source1_entity_id, candidate.candidate_entity_id)
        for candidate in candidates.get()
    }

    assert ("S1-1", "S2-1") in pairs
    assert ("S1-1", "S3-1") in pairs
    assert ("S1-1", "S1-1") not in pairs


def test_candidate_generator_preserves_provenance():
    source2 = [
        {
            "entity_id": "S2-1",
            "name_norm": "alpha cafe",
            "name_compact": "alphacafe",
            "name_tokens": ["alpha", "cafe"],
            "address_norm": "12 main road",
            "address_compact": "12mainroad",
            "address_tokens": ["12", "main", "road"],
            "address_numbers": ["12"],
        }
    ]

    source1 = [
        {
            "entity_id": "S1-1",
            "name_norm": "alpha cafe",
            "name_compact": "alphacafe",
            "name_tokens": ["alpha", "cafe"],
            "address_norm": "12 main road",
            "address_compact": "12mainroad",
            "address_tokens": ["12", "main", "road"],
            "address_numbers": ["12"],
        }
    ]

    generator = CandidateGenerator()
    generator.fit(source2, [])

    candidates = generator.generate(source1)
    candidate = candidates.get_for_entity("S1-1")[0]

    assert candidate.candidate_entity_id == "S2-1"
    assert "exact" in candidate.blocker_provenance
    assert "rare_token" in candidate.blocker_provenance
    assert "numeric" in candidate.blocker_provenance
    assert "tfidf" in candidate.blocker_provenance