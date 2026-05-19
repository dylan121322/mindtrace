import re


URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"[\w.\-+]+@[\w.\-]+\.\w+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s]{6,}\d)(?!\d)")


def mask_sensitive_text(text: str) -> str:
    text = URL_RE.sub("[链接]", text or "")
    text = EMAIL_RE.sub("[邮箱]", text)
    text = PHONE_RE.sub("[号码]", text)
    return text


def sanitize_snippet(text: str, max_len: int = 120) -> str:
    clean = " ".join(mask_sensitive_text(text).split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1] + "..."

