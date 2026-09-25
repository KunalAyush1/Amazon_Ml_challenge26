import re


NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")


def extract_numbers(text: str | None) -> list[str]:
    """
    Extract numeric tokens from text.

    The function does not attempt to determine whether a number
    represents a postal code, building number, phone number, etc.
    """
    if text is None:
        return []

    text = str(text)

    return NUMBER_PATTERN.findall(text)