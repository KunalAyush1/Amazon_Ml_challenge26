from business_entity_resol.features.pair_features import pair_features


def test_pair_features_combines_feature_groups():
    source1 = {
        "name_norm": "abc private limited",
        "name_compact": "abcprivatelimited",
        "address_norm": "12 main road",
        "address_compact": "12mainroad",
        "address_numbers": ["12"],
        "country": "India",
    }

    candidate = {
        "name_norm": "abc pvt limited",
        "name_compact": "abcpvtlimited",
        "address_norm": "12 main road",
        "address_compact": "12mainroad",
        "address_numbers": ["12"],
        "country": "India",
        "blocker_provenance": ["exact", "tfidf"],
        "ranks": {"tfidf": 1},
    }

    features = pair_features(source1, candidate)

    assert features["name_norm_exact"] == 0
    assert features["address_norm_exact"] == 1
    assert features["numeric_exact_match"] == 1
    assert features["country_exact"] == 1

    assert features["blocker_num_blockers"] == 2
    assert features["blocker_best_rank"] == 1

    assert len(features) == 70


def test_pair_features_handles_missing_values():
    source1 = {
        "name_norm": "",
        "address_norm": "",
        "country": "",
    }

    candidate = {
        "name_norm": "",
        "address_norm": "",
        "country": "",
    }

    features = pair_features(source1, candidate)

    assert isinstance(features, dict)
    assert "name_norm_exact" in features
    assert "address_norm_exact" in features
    assert "numeric_jaccard" in features
    assert "country_exact" in features