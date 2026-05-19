from typing import Any, Dict, List, Optional, Tuple

from app.models import ChatMessage
from app.services.wechat_db import WeChatDBReader
from app.utils.time_utils import parse_time_to_unix


def _looks_like_group_key(target_key: str) -> bool:
    key = (target_key or "").lower()
    return "@chatroom" in key or "chatroom" in key


def list_messages(
    target_key: str,
    target_type: str = "contact",
    time_from=None,
    time_to=None,
    limit: int = 500,
) -> List[ChatMessage]:
    if target_type == "group" or _looks_like_group_key(target_key):
        return []
    reader = WeChatDBReader()
    return reader.read_messages(
        target_key=target_key,
        target_type=target_type,
        time_from=parse_time_to_unix(time_from),
        time_to=parse_time_to_unix(time_to),
        limit=limit,
    )


def list_messages_with_diagnostics(
    target_key: str,
    target_type: str = "contact",
    time_from=None,
    time_to=None,
    limit: int = 500,
) -> Tuple[List[ChatMessage], Dict[str, Any]]:
    if target_type == "group" or _looks_like_group_key(target_key):
        return [], {"source": "wechat_db", "target_type": target_type, "reason": "group_analysis_disabled"}
    reader = WeChatDBReader()
    return reader.read_messages_with_diagnostics(
        target_key=target_key,
        target_type=target_type,
        time_from=parse_time_to_unix(time_from),
        time_to=parse_time_to_unix(time_to),
        limit=limit,
    )


def list_message_targets(target_type: str = "all") -> List[Dict[str, Any]]:
    reader = WeChatDBReader()
    if target_type == "self":
        return [
            {
                "username": "__self__",
                "nickname": "本人",
                "remark": "本人",
                "avatar": "",
                "is_group": False,
                "is_self": True,
                "target_table": "",
                "message_count": 0,
                "first_message_time": "",
                "last_message_time": "",
            }
        ]
    targets = [item for item in reader.read_message_targets() if not item.get("is_group")]
    self_usernames = set(reader.read_self_usernames())
    if self_usernames:
        targets = [item for item in targets if item.get("username") not in self_usernames]
    if target_type == "contact":
        return targets
    if target_type == "group":
        return []
    return targets
