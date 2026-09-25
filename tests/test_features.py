from business_entity_resol.features.name_features import name_features
from business_entity_resol.features.address_features import address_features


# =========================
# Name Features
# =========================

def test_name_features_exact_match():
    source1 = {
        "name": "Amazon India",
        "name_norm": "amazon india",
        "name_compact": "amazonindia",
    }

    candidate = {
        "name": "Amazon India",
        "name_norm": "amazon india",
        "name_compact": "amazonindia",
    }

    features = name_features(source1, candidate)

    assert features["name_exact"] == 1
    assert features["name_norm_exact"] == 1
    assert features["name_compact_exact"] == 1
    assert features["name_ratio"] == 1.0
    assert features["name_jaro_winkler"] == 1.0
    assert features["name_token_jaccard"] == 1.0


def test_name_features_similar_names():
    source1 = {
        "name": "Amazon India Private Limited",
        "name_norm": "amazon india private limited",
        "name_compact": "amazonindiaprivatelimited",
    }

    candidate = {
        "name": "Amazon India Pvt Ltd",
        "name_norm": "amazon india pvt ltd",
        "name_compact": "amazonindiapvtltd",
    }

    features = name_features(source1, candidate)

    assert features["name_exact"] == 0
    assert features["name_norm_exact"] == 0
    assert features["name_ratio"] > 0.5
    assert features["name_jaro_winkler"] > 0.5
    assert features["name_token_jaccard"] >= 0.0


def test_name_features_token_order():
    source1 = {
        "name": "India Amazon",
        "name_norm": "india amazon",
        "name_compact": "indiaamazon",
    }

    candidate = {
        "name": "Amazon India",
        "name_norm": "amazon india",
        "name_compact": "amazonindia",
    }

    features = name_features(source1, candidate)

    assert features["name_norm_exact"] == 0
    assert features["name_token_sort_ratio"] == 1.0
    assert features["name_token_set_ratio"] == 1.0
    assert features["name_token_jaccard"] == 1.0


def test_name_features_missing_values():
    source1 = {
        "name": None,
        "name_norm": None,
        "name_compact": None,
    }

    candidate = {
        "name": "Amazon India",
        "name_norm": "amazon india",
        "name_compact": "amazonindia",
    }

    features = name_features(source1, candidate)

    assert features["name_exact"] == 0
    assert features["name_norm_exact"] == 0
    assert features["name_compact_exact"] == 0
    assert features["name_ratio"] == 0.0
    assert features["name_token_jaccard"] == 0.0


# =========================
# Address Features
# =========================

def test_address_features_exact_match():
    source1 = {
        "address": "12 MG Road Delhi",
        "address_norm": "12 mg road delhi",
        "address_compact": "12mgroaddelhi",
    }

    candidate = {
        "address": "12 MG Road Delhi",
        "address_norm": "12 mg road delhi",
        "address_compact": "12mgroaddelhi",
    }

    features = address_features(source1, candidate)

    assert features["address_exact"] == 1
    assert features["address_norm_exact"] == 1
    assert features["address_compact_exact"] == 1
    assert features["address_ratio"] == 1.0
    assert features["address_token_jaccard"] == 1.0


def test_address_features_similar_addresses():
    source1 = {
        "address": "12 MG Road Delhi",
        "address_norm": "12 mg road delhi",
        "address_compact": "12mgroaddelhi",
    }

    candidate = {
        "address": "12 MG Rd Delhi",
        "address_norm": "12 mg rd delhi",
        "address_compact": "12mgrddelhi",
    }

    features = address_features(source1, candidate)

    assert features["address_exact"] == 0
    assert features["address_norm_exact"] == 0
    assert features["address_ratio"] > 0.5
    assert features["address_token_set_ratio"] > 0.5
    assert features["address_token_jaccard"] >= 0.0


def test_address_features_token_order():
    source1 = {
        "address": "Delhi MG Road 12",
        "address_norm": "delhi mg road 12",
        "address_compact": "delhimgroadd12",
    }

    candidate = {
        "address": "12 MG Road Delhi",
        "address_norm": "12 mg road delhi",
        "address_compact": "12mgroaddelhi",
    }

    features = address_features(source1, candidate)

    assert features["address_norm_exact"] == 0
    assert features["address_token_sort_ratio"] == 1.0
    assert features["address_token_set_ratio"] == 1.0
    assert features["address_token_jaccard"] == 1.0


def test_address_features_missing_values():
    source1 = {
        "address": None,
        "address_norm": None,
        "address_compact": None,
    }

    candidate = {
        "address": "12 MG Road Delhi",
        "address_norm": "12 mg road delhi",
        "address_compact": "12mgroaddelhi",
    }

    features = address_features(source1, candidate)

    assert features["address_exact"] == 0
    assert features["address_norm_exact"] == 0
    assert features["address_compact_exact"] == 0
    assert features["address_ratio"] == 0.0
    assert features["address_token_jaccard"] == 0.0