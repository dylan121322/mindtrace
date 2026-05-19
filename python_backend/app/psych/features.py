from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

from app.models import ChatMessage, PsychEvidence, PsychFeature
from app.psych import labels as label_classifier
from app.psych.scoring_config import DEFAULT_SCORING_CONFIG, enabled_dimensions, normalize_scoring_config
from app.utils.privacy import sanitize_snippet
from app.utils.time_utils import date_part


def _default_word_groups() -> Dict[str, Tuple[str, List[str], str, str]]:
    groups: Dict[str, Tuple[str, List[str], str, str]] = {}
    for dimension in DEFAULT_SCORING_CONFIG["dimensions"]:
        severity = "high" if dimension.get("redline") or int(dimension.get("max_points", 0)) >= 10 else "medium"
        groups[dimension["key"]] = (
            dimension["key"],
            list(dimension.get("keywords", [])),
            severity,
            str(dimension.get("label") or dimension["key"]),
        )
    return groups


WORD_GROUPS = _default_word_groups()
SELF_HARM_WORDS = WORD_GROUPS.get("self_harm_suicide_risk", ("", [], "high", ""))[1]


def _parse_hour(dt_text: str) -> int:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(dt_text[: len(fmt)], fmt).hour
        except ValueError:
            continue
    return -1


def _contains_any(text: str, words: Iterable[str]) -> List[str]:
    return [word for word in words if word and word in text]


def _subgroup_hits(text: str, subgroups: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name, words in subgroups.items():
        if not isinstance(words, list):
            continue
        hits = _contains_any(text, [str(word) for word in words])
        counts[name] = len(hits)
    return counts


def compute_features(
    messages: List[ChatMessage],
    config: Dict[str, Any] | None = None,
) -> Tuple[List[PsychFeature], List[PsychEvidence], Dict[str, float]]:
    scoring_config = normalize_scoring_config(config) if config else None
    dimensions = enabled_dimensions(scoring_config)
    values: Dict[str, float] = {
        "message_count": float(len(messages)),
        "avg_message_length": 0.0,
        "late_night_message_ratio": 0.0,
        "evidence_count": 0.0,
        "unique_active_days": 0.0,
        "relief_keyword_count": 0.0,
    }
    evidences: List[PsychEvidence] = []
    total_len = 0
    late_night_count = 0
    active_days = set()
    dimension_days: Dict[str, set[str]] = {item["key"]: set() for item in dimensions}
    dimension_messages: Dict[str, set[str]] = {item["key"]: set() for item in dimensions}
    dimension_day_hits: Dict[str, Dict[str, int]] = {item["key"]: {} for item in dimensions}
    label_rules = label_classifier.label_rules(scoring_config)
    label_days: Dict[str, set[str]] = {item["key"]: set() for item in label_rules}
    label_messages: Dict[str, set[str]] = {item["key"]: set() for item in label_rules}
    window_start = messages[0].datetime if messages else None
    window_end = messages[-1].datetime if messages else None
    relief_words = []
    if scoring_config:
        time_adjustment = scoring_config.get("time_adjustment", {})
        relief = time_adjustment.get("relief_reduction", {}) if isinstance(time_adjustment, dict) else {}
        relief_words = [str(word) for word in relief.get("keywords", []) if str(word or "").strip()]

    for dimension in dimensions:
        key = dimension["key"]
        values[f"{key}_keyword_count"] = 0.0
        values[f"{key}_message_count"] = 0.0
        values[f"{key}_active_days"] = 0.0
        values[f"{key}_strong_count"] = 0.0
        values[f"{key}_semantic_hit_count"] = 0.0
        values[f"{key}_semantic_active_days"] = 0.0
        values[f"{key}_evidence_day_span"] = 0.0
        values[f"{key}_worsening_trend"] = 0.0
        for subgroup in dimension.get("subgroups", {}):
            values[f"{key}_{subgroup}_count"] = 0.0
    for rule in label_rules:
        key = rule["key"]
        values[f"label_{key}_count"] = 0.0
        values[f"label_{key}_message_count"] = 0.0
        values[f"label_{key}_active_days"] = 0.0

    for msg in messages:
        text = msg.content or ""
        total_len += len(text)
        day = date_part(msg.datetime)
        if day:
            active_days.add(day)
        hour = _parse_hour(msg.datetime)
        if hour >= 23 or (0 <= hour < 5):
            late_night_count += 1
        if relief_words and _contains_any(text, relief_words):
            values["relief_keyword_count"] += 1.0

        identity = f"{msg.seq}\x1f{msg.datetime}\x1f{msg.sender}"
        message_label_hits = label_classifier.classify_message(text, scoring_config)
        message_label_keys = [item["key"] for item in message_label_hits if item.get("key")]
        message_risk_level = max([int(item.get("risk_level") or 0) for item in message_label_hits] or [0])
        for item in message_label_hits:
            key = item["key"]
            values[f"label_{key}_count"] = values.get(f"label_{key}_count", 0.0) + float(len(item.get("hit_words", [])))
            label_messages.setdefault(key, set()).add(identity)
            if day:
                label_days.setdefault(key, set()).add(day)
        for dimension in dimensions:
            key = dimension["key"]
            exclude_hits = _contains_any(text, dimension.get("exclude_keywords", []))
            if exclude_hits:
                continue
            keyword_hits = _contains_any(text, dimension.get("keywords", []))
            strong_hits = _contains_any(text, dimension.get("strong_keywords", []))
            subgroup_counts = _subgroup_hits(text, dimension.get("subgroups", {}))
            total_hits = len(keyword_hits) + sum(subgroup_counts.values())
            if total_hits <= 0:
                continue

            values[f"{key}_keyword_count"] += float(len(keyword_hits))
            values[f"{key}_strong_count"] += float(len(strong_hits))
            for subgroup, count in subgroup_counts.items():
                values[f"{key}_{subgroup}_count"] = values.get(f"{key}_{subgroup}_count", 0.0) + float(count)
            dimension_messages[key].add(identity)
            if day:
                dimension_days[key].add(day)
                dimension_day_hits[key][day] = dimension_day_hits[key].get(day, 0) + total_hits

            severity = "high" if dimension.get("redline") or int(dimension.get("max_points", 0)) >= 10 else "medium"
            evidences.append(
                PsychEvidence(
                    seq=msg.seq,
                    datetime=msg.datetime,
                    sender=msg.sender,
                    content=sanitize_snippet(text),
                    evidence_type=key,
                    severity=severity,
                    reason=f"{dimension.get('label', key)}: {', '.join((keyword_hits + strong_hits)[:5])}",
                    labels=message_label_keys,
                    risk_level=message_risk_level,
                )
            )

    message_count = len(messages)
    values.update(
        {
            "message_count": float(message_count),
            "avg_message_length": float(total_len / message_count) if message_count else 0.0,
            "late_night_message_ratio": float(late_night_count / message_count) if message_count else 0.0,
            "evidence_count": float(len(evidences)),
            "unique_active_days": float(len(active_days)),
        }
    )
    for dimension in dimensions:
        key = dimension["key"]
        values[f"{key}_message_count"] = float(len(dimension_messages[key]))
        values[f"{key}_active_days"] = float(len(dimension_days[key]))
        sorted_days = sorted(dimension_days[key])
        if sorted_days:
            try:
                first_day = datetime.strptime(sorted_days[0], "%Y-%m-%d")
                last_day = datetime.strptime(sorted_days[-1], "%Y-%m-%d")
                values[f"{key}_evidence_day_span"] = float((last_day - first_day).days + 1)
            except ValueError:
                values[f"{key}_evidence_day_span"] = float(len(sorted_days))
        if len(sorted_days) >= 3:
            midpoint = len(sorted_days) // 2
            early_days = sorted_days[:midpoint]
            late_days = sorted_days[midpoint:]
            early_hits = sum(dimension_day_hits[key].get(item, 0) for item in early_days)
            late_hits = sum(dimension_day_hits[key].get(item, 0) for item in late_days)
            early_avg = early_hits / max(1, len(early_days))
            late_avg = late_hits / max(1, len(late_days))
            if late_avg >= max(early_avg * 1.5, early_avg + 1):
                values[f"{key}_worsening_trend"] = 1.0
    for rule in label_rules:
        key = rule["key"]
        values[f"label_{key}_message_count"] = float(len(label_messages.get(key, set())))
        values[f"label_{key}_active_days"] = float(len(label_days.get(key, set())))

    features = [
        PsychFeature(
            group="psych_text",
            name=name,
            value=value,
            window_start=window_start,
            window_end=window_end,
        )
        for name, value in values.items()
    ]
    return features, evidences, values
