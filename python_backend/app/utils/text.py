import re


WHITESPACE_RE = re.compile(r"\s+")


def compact_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", (text or "").strip())


def truncate_text(text: str, max_len: int = 500) -> str:
    text = compact_text(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."

