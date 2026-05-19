import re
from typing import Iterable, List

from app.models import ChatMessage
from app.utils.text import compact_text


MEDIA_MESSAGES = {"[图片]", "[语音]", "[视频]", "[红包]", "[转账]", "[动画表情]", "[红包/转账]", "[链接/文件]", "[小程序]"}
SHORT_CHITCHAT = {
    "哈哈",
    "哈哈哈",
    "hhh",
    "hh",
    "嗯",
    "嗯嗯",
    "好的",
    "好",
    "收到",
    "ok",
    "OK",
    "行",
    "可以",
}
URL_ONLY_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
SHARE_HINTS = ("<msg>", "<appmsg", "gh_", "mp.weixin.qq.com", "weappinfo", "miniprogram")


def normalize_text(text: str) -> str:
    return compact_text(text)


def is_meaningless_message(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return True
    if text in MEDIA_MESSAGES:
        return True
    if URL_ONLY_RE.match(text):
        return True
    lowered = text.lower()
    if any(hint in lowered for hint in SHARE_HINTS):
        return True
    if text in SHORT_CHITCHAT:
        return True
    if len(text) <= 2 and text in {"嗯", "哦", "啊", "哈"}:
        return True
    return False


def filter_messages(messages: Iterable[ChatMessage], only_mine: bool = True) -> List[ChatMessage]:
    out: List[ChatMessage] = []
    for msg in messages:
        if only_mine and not msg.is_mine:
            continue
        text = normalize_text(msg.content)
        if is_meaningless_message(text):
            continue
        out.append(
            ChatMessage(
                seq=msg.seq,
                datetime=msg.datetime,
                sender=msg.sender,
                content=text,
                is_mine=msg.is_mine,
                contact_key=msg.contact_key,
            )
        )
    return out

