from business_entity_resol.preprocessing.address_normalizer import (
    normalize_address,
)
from business_entity_resol.preprocessing.name_normalizer import (
    compact_name,
    normalize_name,
    normalize_name_record,
)
from business_entity_resol.preprocessing.numeric_extractor import (
    extract_numbers,
)
from business_entity_resol.preprocessing.normalization import (
    normalize_record,
)
from business_entity_resol.preprocessing.tokenizer import tokenize


def test_name_case_normalization():
    assert normalize_name("ABC PVT LTD") == "abc pvt ltd"


def test_name_whitespace_normalization():
    assert normalize_name("ABC    PVT   LTD") == "abc pvt ltd"


def test_name_punctuation_normalization():
    assert normalize_name("ABC, Pvt. Ltd.") == "abc pvt ltd"


def test_unicode_normalization():
    assert normalize_name("Ｃａｆé") == "café"


def test_compact_name():
    assert compact_name("ABC Pvt Ltd") == "abcpvtltd"


def test_name_tokenization():
    result = normalize_name_record("ABC Motors Pvt Ltd")

    assert result["name_tokens"] == [
        "abc",
        "motors",
        "pvt",
        "ltd",
    ]


def test_address_normalization():
    result = normalize_address("21, M.G. Road, Delhi")

    assert "21" in result
    assert "road" in result
    assert "delhi" in result


def test_address_abbreviation():
    assert "road" in normalize_address("21 MG Rd Delhi")


def test_number_extraction():
    result = extract_numbers("21 MG Road Delhi 110001")

    assert result == ["21", "110001"]


def test_empty_values():
    assert normalize_name("") == ""
    assert normalize_name(None) == ""
    assert tokenize("") == []
    assert extract_numbers("") == []


def test_normalize_record():
    record = {
        "entity_id": "S1-00001",
        "business_name": "ABC Pvt. Ltd.",
        "business_address": "21 MG Rd, Delhi 110001",
        "country": "India",
    }

    result = normalize_record(record)

    assert result["entity_id"] == "S1-00001"
    assert result["business_name"] == "ABC Pvt. Ltd."
    assert result["name_norm"] == "abc pvt ltd"
    assert result["name_compact"] == "abcpvtltd"
    assert "110001" in result["address_numbers"]