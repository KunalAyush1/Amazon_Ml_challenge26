from business_entity_resol.features.name_features import name_features
from business_entity_resol.features.address_features import address_features
from business_entity_resol.features.numeric_features import numeric_features
from business_entity_resol.features.country_features import country_features
from business_entity_resol.features.lexical_features import lexical_features
from business_entity_resol.features.blocker_features import blocker_features

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
    
# =========================
# Numeric Features
# =========================

def test_numeric_features_exact_match():
    source1 = {
        "address": "12 MG Road Delhi 110001",
    }

    candidate = {
        "address": "12 MG Road Delhi 110001",
    }

    features = numeric_features(source1, candidate)

    assert features["numeric_count_1"] == 2
    assert features["numeric_count_2"] == 2
    assert features["numeric_overlap_count"] == 2
    assert features["numeric_jaccard"] == 1.0
    assert features["numeric_exact_match"] == 1
    assert features["numeric_any_overlap"] == 1


def test_numeric_features_partial_overlap():
    source1 = {
        "address": "12 MG Road Delhi 110001",
    }

    candidate = {
        "address": "12 MG Road Delhi 110002",
    }

    features = numeric_features(source1, candidate)

    assert features["numeric_count_1"] == 2
    assert features["numeric_count_2"] == 2
    assert features["numeric_overlap_count"] == 1
    assert features["numeric_jaccard"] == 1 / 3
    assert features["numeric_exact_match"] == 0
    assert features["numeric_any_overlap"] == 1


def test_numeric_features_no_overlap():
    source1 = {
        "address": "12 MG Road Delhi",
    }

    candidate = {
        "address": "45 Park Road Mumbai",
    }

    features = numeric_features(source1, candidate)

    assert features["numeric_count_1"] == 1
    assert features["numeric_count_2"] == 1
    assert features["numeric_overlap_count"] == 0
    assert features["numeric_jaccard"] == 0.0
    assert features["numeric_exact_match"] == 0
    assert features["numeric_any_overlap"] == 0


def test_numeric_features_missing_values():
    source1 = {
        "address": None,
    }

    candidate = {
        "address": "12 MG Road Delhi",
    }

    features = numeric_features(source1, candidate)

    assert features["numeric_count_1"] == 0
    assert features["numeric_count_2"] == 1
    assert features["numeric_overlap_count"] == 0
    assert features["numeric_jaccard"] == 0.0
    assert features["numeric_exact_match"] == 0
    assert features["numeric_any_overlap"] == 0
    
# =========================
# Country Features
# =========================

def test_country_features_exact_match():
    source1 = {
        "country": "India",
        "country_norm": "india",
        "country_code": "IN",
    }

    candidate = {
        "country": "India",
        "country_norm": "india",
        "country_code": "IN",
    }

    features = country_features(source1, candidate)

    assert features["country_exact"] == 1
    assert features["country_norm_exact"] == 1
    assert features["country_code_exact"] == 1
    assert features["country_both_present"] == 1
    assert features["country_missing_1"] == 0
    assert features["country_missing_2"] == 0


def test_country_features_case_insensitive():
    source1 = {
        "country": "INDIA",
    }

    candidate = {
        "country": "india",
    }

    features = country_features(source1, candidate)

    assert features["country_exact"] == 1
    assert features["country_norm_exact"] == 1
    assert features["country_code_exact"] == 1


def test_country_features_different_countries():
    source1 = {
        "country": "India",
        "country_norm": "india",
        "country_code": "IN",
    }

    candidate = {
        "country": "United States",
        "country_norm": "united states",
        "country_code": "US",
    }

    features = country_features(source1, candidate)

    assert features["country_exact"] == 0
    assert features["country_norm_exact"] == 0
    assert features["country_code_exact"] == 0
    assert features["country_both_present"] == 1


def test_country_features_missing_values():
    source1 = {
        "country": None,
        "country_norm": None,
        "country_code": None,
    }

    candidate = {
        "country": "India",
        "country_norm": "india",
        "country_code": "IN",
    }

    features = country_features(source1, candidate)

    assert features["country_exact"] == 0
    assert features["country_norm_exact"] == 0
    assert features["country_code_exact"] == 0
    assert features["country_both_present"] == 0
    assert features["country_missing_1"] == 1
    assert features["country_missing_2"] == 0
    


# =========================
# Lexical Features
# =========================

def test_lexical_features_exact_match():
    source1 = {
        "name_norm": "amazon india",
    }

    candidate = {
        "name_norm": "amazon india",
    }

    features = lexical_features(
        source1,
        candidate,
        field="name_norm",
    )

    assert features["lexical_exact"] == 1
    assert features["lexical_ratio"] == 1.0
    assert features["lexical_partial_ratio"] == 1.0
    assert features["lexical_token_sort_ratio"] == 1.0
    assert features["lexical_token_set_ratio"] == 1.0
    assert features["lexical_token_jaccard"] == 1.0
    assert features["lexical_token_overlap_count"] == 2


def test_lexical_features_similar_text():
    source1 = {
        "name_norm": "amazon india private limited",
    }

    candidate = {
        "name_norm": "amazon india pvt ltd",
    }

    features = lexical_features(
        source1,
        candidate,
        field="name_norm",
    )

    assert features["lexical_exact"] == 0
    assert features["lexical_ratio"] > 0.5
    assert features["lexical_partial_ratio"] > 0.5
    assert features["lexical_token_set_ratio"] > 0.5
    assert features["lexical_token_jaccard"] >= 0.0


def test_lexical_features_token_order():
    source1 = {
        "name_norm": "india amazon",
    }

    candidate = {
        "name_norm": "amazon india",
    }

    features = lexical_features(
        source1,
        candidate,
        field="name_norm",
    )

    assert features["lexical_exact"] == 0
    assert features["lexical_token_sort_ratio"] == 1.0
    assert features["lexical_token_set_ratio"] == 1.0
    assert features["lexical_token_jaccard"] == 1.0


def test_lexical_features_missing_values():
    source1 = {
        "name_norm": None,
    }

    candidate = {
        "name_norm": "amazon india",
    }

    features = lexical_features(
        source1,
        candidate,
        field="name_norm",
    )

    assert features["lexical_exact"] == 0
    assert features["lexical_ratio"] == 0.0
    assert features["lexical_partial_ratio"] == 0.0
    assert features["lexical_token_jaccard"] == 0.0
    assert features["lexical_token_count_1"] == 0
    assert features["lexical_token_count_2"] == 2
# =========================
# Blocker Features
# =========================

def test_blocker_features_multiple_blockers():
    candidate = {
        "blocker_provenance": ["exact_name", "tfidf"],
        "ranks": {
            "tfidf": 3,
        },
    }

    features = blocker_features(candidate)

    assert features["blocker_num_blockers"] == 2
    assert features["blocker_best_rank"] == 3
    assert features["blocker_has_rank"] == 1
    assert features["blocker_rank_count"] == 1
    assert features["blocker_provenance_count"] == 2


def test_blocker_features_multiple_ranks():
    candidate = {
        "blocker_provenance": [
            "exact_name",
            "rare_token",
            "tfidf",
        ],
        "ranks": {
            "rare_token": 5,
            "tfidf": 2,
        },
    }

    features = blocker_features(candidate)

    assert features["blocker_num_blockers"] == 3
    assert features["blocker_best_rank"] == 2
    assert features["blocker_has_rank"] == 1
    assert features["blocker_rank_count"] == 2
    assert features["blocker_provenance_count"] == 3


def test_blocker_features_without_ranks():
    candidate = {
        "blocker_provenance": [
            "exact_name",
            "numeric",
        ],
        "ranks": {},
    }

    features = blocker_features(candidate)

    assert features["blocker_num_blockers"] == 2
    assert features["blocker_best_rank"] == 0
    assert features["blocker_has_rank"] == 0
    assert features["blocker_rank_count"] == 0
    assert features["blocker_provenance_count"] == 2


def test_blocker_features_empty_values():
    candidate = {
        "blocker_provenance": [],
        "ranks": {},
    }

    features = blocker_features(candidate)

    assert features["blocker_num_blockers"] == 0
    assert features["blocker_best_rank"] == 0
    assert features["blocker_has_rank"] == 0
    assert features["blocker_rank_count"] == 0
    assert features["blocker_provenance_count"] == 0