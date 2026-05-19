from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from app.config import get_settings


def _local_tz():
    try:
        return ZoneInfo(get_settings().timezone)
    except Exception:
        return timezone(timedelta(hours=8), name="Asia/Shanghai")


def parse_time_to_unix(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            return int(dt.replace(tzinfo=_local_tz()).timestamp())
        except ValueError:
            pass
    try:
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def unix_to_local_str(ts: int) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(int(ts), _local_tz()).strftime("%Y-%m-%d %H:%M:%S")


def date_part(dt_text: str) -> str:
    if not dt_text:
        return ""
    return dt_text[:10]
