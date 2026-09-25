import re


TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['-][^\W_]+)*", re.UNICODE)


def tokenize(text: str | None) -> list[str]:
    """
    Tokenize already-normalized text.

    Empty or missing values return an empty list.
    """
    if text is None:
        return []

    text = str(text).strip()

    if not text:
        return []

    return TOKEN_PATTERN.findall(text)