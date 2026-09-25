import re
import unicodedata

from business_entity_resol.preprocessing.tokenizer import tokenize


WHITESPACE_PATTERN = re.compile(r"\s+")

# Only punctuation that is generally safe to normalize.
PUNCTUATION_PATTERN = re.compile(r"[^\w\s'-]", re.UNICODE)


def normalize_name(value: object) -> str:
    """
    Create a safe normalized representation of a business name.
    """
    if value is None:
        return ""

    text = str(value)

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()

    text = PUNCTUATION_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()


def compact_name(value: object) -> str:
    """
    Remove whitespace from the normalized name.
    """
    normalized = normalize_name(value)
    return normalized.replace(" ", "")


def normalize_name_record(value: object) -> dict[str, object]:
    """
    Return all useful name representations.
    """
    name_raw = "" if value is None else str(value)
    name_norm = normalize_name(value)

    return {
        "name_raw": name_raw,
        "name_norm": name_norm,
        "name_compact": name_norm.replace(" ", ""),
        "name_tokens": tokenize(name_norm),
    }