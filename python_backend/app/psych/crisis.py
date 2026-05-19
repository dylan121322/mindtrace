from typing import Dict, List, Tuple

from app.models import ChatMessage, PsychEvidence
from app.psych.features import SELF_HARM_WORDS
from app.utils.privacy import sanitize_snippet


EXCLUSION_PHRASES = ("笑死了", "累死了", "社死了", "尴尬死了", "吓死了")
IMMINENT_WORDS = ("今晚", "明天", "已经准备", "准备好了", "买了药", "遗书", "告别", "最后一次", "撑不到")
QUOTE_CONTEXT_WORDS = ("新闻", "电影", "电视剧", "游戏", "剧情", "角色", "台词", "报道", "看到一个", "听说")

LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2, "crisis": 3}


def _max_level(a: str, b: str) -> str:
    return a if LEVEL_ORDER[a] >= LEVEL_ORDER[b] else b


def detect_crisis(messages: List[ChatMessage]) -> Tuple[Dict, List[PsychEvidence]]:
    level = "low"
    evidences: List[PsychEvidence] = []
    for msg in messages:
        text = msg.content or ""
        if any(phrase in text for phrase in EXCLUSION_PHRASES):
            continue
        hits = [word for word in SELF_HARM_WORDS if word in text]
        if not hits:
            continue
        quoted = any(word in text for word in QUOTE_CONTEXT_WORDS)
        imminent = any(word in text for word in IMMINENT_WORDS)
        msg_level = "crisis" if imminent else "high"
        if quoted and msg_level == "high":
            msg_level = "medium"
        level = _max_level(level, msg_level)
        evidences.append(
            PsychEvidence(
                seq=msg.seq,
                datetime=msg.datetime,
                sender=msg.sender,
                content=sanitize_snippet(text),
                evidence_type="crisis",
                severity=msg_level,
                reason="检测到自伤或轻生相关表达" + ("，且出现近期/准备线索" if imminent else ""),
            )
        )
    return {"level": level, "evidence_count": len(evidences)}, evidences

