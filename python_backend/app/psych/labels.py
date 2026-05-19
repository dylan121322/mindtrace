from __future__ import annotations

from typing import Any, Dict, Iterable, List

from app.models import ChatMessage
from app.utils.privacy import sanitize_snippet
from app.utils.time_utils import date_part


def _contains_any(text: str, words: Iterable[str]) -> List[str]:
    return [str(word) for word in words if str(word or "").strip() and str(word) in text]


def label_rules(config: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    section = (config or {}).get("symptom_labels", {})
    rules = section.get("labels", []) if isinstance(section, dict) else []
    return [item for item in rules if isinstance(item, dict) and item.get("enabled", True)]


def classify_message(text: str, config: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rule in label_rules(config):
        exclude_hits = _contains_any(text, rule.get("exclude_keywords", []))
        if exclude_hits:
            continue
        hits = _contains_any(text, rule.get("keywords", []))
        if not hits:
            continue
        out.append(
            {
                "key": str(rule.get("key") or ""),
                "label": str(rule.get("label") or rule.get("key") or ""),
                "category": str(rule.get("category") or ""),
                "weight": str(rule.get("weight") or ""),
                "weight_label": str(rule.get("weight_label") or ""),
                "risk_level": int(rule.get("risk_level") or 0),
                "protective": bool(rule.get("protective", False)),
                "modifier": bool(rule.get("modifier", False)),
                "dimension_keys": [str(item) for item in rule.get("dimension_keys", []) if str(item or "").strip()],
                "hit_words": hits[:8],
            }
        )
    return out


def classify_debug_message(msg: ChatMessage, config: Dict[str, Any] | None, max_len: int = 180) -> Dict[str, Any]:
    labels = classify_message(msg.content or "", config)
    return {
        "seq": msg.seq,
        "datetime": msg.datetime,
        "sender": msg.sender,
        "is_mine": bool(msg.is_mine),
        "contact_key": msg.contact_key,
        "content": sanitize_snippet(msg.content, max_len=max_len),
        "labels": [item["key"] for item in labels],
        "label_names": [item["label"] for item in labels],
        "label_hits": labels,
        "risk_level": max([int(item.get("risk_level") or 0) for item in labels] or [0]),
    }


def summarize_messages(messages: List[ChatMessage], config: Dict[str, Any] | None) -> Dict[str, Any]:
    label_map: Dict[str, Dict[str, Any]] = {}
    label_message_ids: Dict[str, set[str]] = {}
    label_days: Dict[str, set[str]] = {}
    for msg in messages:
        identity = f"{msg.seq}\x1f{msg.datetime}\x1f{msg.sender}\x1f{msg.content}"
        day = date_part(msg.datetime)
        for item in classify_message(msg.content or "", config):
            key = item["key"]
            if not key:
                continue
            current = label_map.setdefault(
                key,
                {
                    "key": key,
                    "label": item.get("label") or key,
                    "category": item.get("category") or "",
                    "weight": item.get("weight") or "",
                    "weight_label": item.get("weight_label") or "",
                    "risk_level": int(item.get("risk_level") or 0),
                    "protective": bool(item.get("protective", False)),
                    "modifier": bool(item.get("modifier", False)),
                    "count": 0,
                    "message_count": 0,
                    "active_days": 0,
                    "hit_words": [],
                },
            )
            current["count"] = int(current.get("count") or 0) + len(item.get("hit_words", []))
            words = list(current.get("hit_words", []))
            for word in item.get("hit_words", []):
                if word not in words:
                    words.append(word)
            current["hit_words"] = words[:12]
            label_message_ids.setdefault(key, set()).add(identity)
            if day:
                label_days.setdefault(key, set()).add(day)

    for key, item in label_map.items():
        item["message_count"] = len(label_message_ids.get(key, set()))
        item["active_days"] = len(label_days.get(key, set()))

    labels = sorted(
        label_map.values(),
        key=lambda item: (
            bool(item.get("protective") or item.get("modifier")),
            -int(item.get("risk_level") or 0),
            -int(item.get("message_count") or 0),
            str(item.get("label") or ""),
        ),
    )
    return {
        "labels": labels,
        "label_count": len(labels),
        "protective_count": len([item for item in labels if item.get("protective")]),
        "modifier_count": len([item for item in labels if item.get("modifier")]),
        "max_label_risk_level": max([int(item.get("risk_level") or 0) for item in labels] or [0]),
    }
