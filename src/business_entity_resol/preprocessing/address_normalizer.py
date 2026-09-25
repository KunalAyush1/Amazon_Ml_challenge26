import re
import unicodedata

from business_entity_resol.preprocessing.numeric_extractor import (
    extract_numbers,
)
from business_entity_resol.preprocessing.tokenizer import tokenize


WHITESPACE_PATTERN = re.compile(r"\s+")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s'-]", re.UNICODE)


ABBREVIATIONS = {
    "rd": "road",
    "rd.": "road",
    "st": "street",
    "st.": "street",
    "ave": "avenue",
    "ave.": "avenue",
    "blvd": "boulevard",
    "blvd.": "boulevard",
    "ln": "lane",
    "ln.": "lane",
}


def normalize_address(value: object) -> str:
    """
    Apply safe, country-agnostic address normalization.
    """
    if value is None:
        return ""

    text = str(value)

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()

    text = PUNCTUATION_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    tokens = text.split()

    normalized_tokens = [
        ABBREVIATIONS.get(token, token)
        for token in tokens
    ]

    return " ".join(normalized_tokens)


def compact_address(value: object) -> str:
    """
    Create a compact address representation.
    """
    normalized = normalize_address(value)
    return normalized.replace(" ", "")


def normalize_address_record(value: object) -> dict[str, object]:
    """
    Return all useful address representations.
    """
    address_raw = "" if value is None else str(value)
    address_norm = normalize_address(value)

    return {
        "address_raw": address_raw,
        "address_norm": address_norm,
        "address_compact": address_norm.replace(" ", ""),
        "address_tokens": tokenize(address_norm),
        "address_numbers": extract_numbers(address_norm),
    }