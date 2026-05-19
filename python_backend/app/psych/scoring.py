from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.models import PsychEvidence, PsychScore
from app.psych.scoring_config import load_scoring_config, normalize_scoring_config


LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2, "crisis": 3}


def _risk_from_score(score: int, thresholds: Dict[str, int]) -> str:
    if score >= int(thresholds.get("crisis", 85)):
        return "crisis"
    if score >= int(thresholds.get("high", 65)):
        return "high"
    if score >= int(thresholds.get("medium", 35)):
        return "medium"
    return "low"


def _max_risk(*levels: str) -> str:
    return max(levels, key=lambda item: LEVEL_ORDER.get(item, 0))


def _overall_from_numeric_level(level: int) -> str:
    if level >= 5:
        return "crisis"
    if level >= 4:
        return "high"
    if level >= 2:
        return "medium"
    return "low"


def _value(values: Dict[str, float], key: str) -> float:
    return float(values.get(key, 0) or 0)


def _risk_level_label(config: Dict[str, Any], level: int) -> str:
    section = config.get("symptom_labels", {})
    levels = section.get("risk_levels", []) if isinstance(section, dict) else []
    for item in levels:
        if int(item.get("level") or -1) == level:
            return str(item.get("label") or f"等级 {level}")
    return f"等级 {level}"


def _label_summary(values: Dict[str, float], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    section = config.get("symptom_labels", {})
    if not section.get("enabled", True):
        return []
    out: List[Dict[str, Any]] = []
    for rule in section.get("labels", []):
        if not rule.get("enabled", True):
            continue
        key = str(rule.get("key") or "")
        if not key:
            continue
        count = _value(values, f"label_{key}_count")
        messages = _value(values, f"label_{key}_message_count")
        days = _value(values, f"label_{key}_active_days")
        if count <= 0 and messages <= 0:
            continue
        out.append(
            {
                "key": key,
                "label": rule.get("label") or key,
                "category": rule.get("category") or "",
                "weight": rule.get("weight") or "",
                "weight_label": rule.get("weight_label") or "",
                "risk_level": int(rule.get("risk_level") or 0),
                "protective": bool(rule.get("protective", False)),
                "modifier": bool(rule.get("modifier", False)),
                "count": count,
                "message_count": messages,
                "active_days": days,
                "description": rule.get("description") or "",
            }
        )
    out.sort(
        key=lambda item: (
            bool(item.get("protective") or item.get("modifier")),
            -int(item.get("risk_level") or 0),
            -float(item.get("message_count") or 0),
            str(item.get("label") or ""),
        )
    )
    return out


def _dimension_counts(values: Dict[str, float], dimension: Dict[str, Any]) -> Dict[str, float]:
    key = dimension["key"]
    keyword_messages = _value(values, f"{key}_message_count")
    semantic_messages = _value(values, f"{key}_semantic_hit_count")
    keyword_days = _value(values, f"{key}_active_days")
    semantic_days = _value(values, f"{key}_semantic_active_days")
    return {
        "keyword_hits": _value(values, f"{key}_keyword_count"),
        "keyword_messages": keyword_messages,
        "keyword_days": keyword_days,
        "semantic_messages": semantic_messages,
        "semantic_days": semantic_days,
        "message_count": max(keyword_messages, semantic_messages),
        "active_days": max(keyword_days, semantic_days),
        "strong_hits": _value(values, f"{key}_strong_count"),
        "day_span": max(_value(values, f"{key}_evidence_day_span"), keyword_days, semantic_days),
        "worsening_trend": _value(values, f"{key}_worsening_trend"),
        "plan_hits": _value(values, f"{key}_plan_count"),
        "explicit_hits": _value(values, f"{key}_explicit_count"),
        "method_hits": _value(values, f"{key}_method_count"),
    }


def _subgroup_rule_matches(values: Dict[str, float], dimension_key: str, rule: Dict[str, Any]) -> bool:
    required = rule.get("min_subgroup_hits")
    if not isinstance(required, dict):
        return True
    for subgroup, minimum in required.items():
        if _value(values, f"{dimension_key}_{subgroup}_count") < float(minimum or 0):
            return False
    return True


def _rule_matches(values: Dict[str, float], dimension: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    key = dimension["key"]
    counts = _dimension_counts(values, dimension)
    checks = [
        ("min_hits", counts["keyword_hits"] + counts["semantic_messages"]),
        ("min_keyword_hits", counts["keyword_hits"]),
        ("min_semantic_hits", counts["semantic_messages"]),
        ("min_messages", counts["message_count"]),
        ("min_active_days", counts["active_days"]),
        ("min_strong_hits", counts["strong_hits"]),
    ]
    for name, actual in checks:
        if name in rule and actual < float(rule.get(name) or 0):
            return False
    return _subgroup_rule_matches(values, key, rule)


def _select_base_level(values: Dict[str, float], dimension: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    levels = sorted(dimension.get("levels", []), key=lambda item: int(item.get("score") or 0))
    if not levels:
        return 0, {"score": 0, "description": ""}
    selected_index = 0
    for index, level in enumerate(levels):
        rule = level.get("rule") if isinstance(level.get("rule"), dict) else {}
        if _rule_matches(values, dimension, rule):
            selected_index = index
    return selected_index, levels[selected_index]


def _strength_rule_matches(
    rule: Dict[str, Any],
    dimension: Dict[str, Any],
    counts: Dict[str, float],
) -> bool:
    if rule.get("redline") is True and not dimension.get("redline"):
        return False
    numeric_checks = [
        ("min_messages", counts["message_count"], True),
        ("max_messages", counts["message_count"], False),
        ("min_active_days", counts["active_days"], True),
        ("max_active_days", counts["active_days"], False),
        ("min_strong_hits", counts["strong_hits"], True),
        ("max_strong_hits", counts["strong_hits"], False),
        ("min_plan_hits", counts["plan_hits"], True),
    ]
    for name, actual, is_min in numeric_checks:
        if name not in rule:
            continue
        target = float(rule.get(name) or 0)
        if is_min and actual < target:
            return False
        if not is_min and actual > target:
            return False
    return True


def _evidence_strength(
    config: Dict[str, Any],
    dimension: Dict[str, Any],
    counts: Dict[str, float],
) -> Dict[str, Any]:
    section = config.get("evidence_strength", {})
    levels = section.get("levels", []) if isinstance(section, dict) else []
    if not section.get("enabled", True) or not levels:
        return {"key": "medium", "label": "中证据", "coefficient": 1.0, "description": "未启用证据强度修正"}

    selected = None
    for level in levels:
        rule = level.get("rule") if isinstance(level.get("rule"), dict) else {}
        if _strength_rule_matches(rule, dimension, counts):
            if selected is None or float(level.get("coefficient") or 0) >= float(selected.get("coefficient") or 0):
                selected = level
    if selected is None:
        selected = next((item for item in levels if item.get("key") == "medium"), levels[0])
    return {
        "key": selected.get("key") or "medium",
        "label": selected.get("label") or selected.get("key") or "中证据",
        "coefficient": float(selected.get("coefficient") or 1.0),
        "description": selected.get("description") or "",
    }


def _time_level(config: Dict[str, Any], counts: Dict[str, float], redline_acute: bool) -> Dict[str, Any]:
    section = config.get("time_adjustment", {})
    levels = section.get("levels", []) if isinstance(section, dict) else []
    if not section.get("enabled", True) or not levels:
        return {"key": "disabled", "label": "未启用", "coefficient": 1.0, "level_shift": 0, "score_delta": 0, "description": ""}

    days = max(1, int(counts["day_span"] or counts["active_days"] or 1))
    selected = None
    for level in levels:
        min_days = int(level.get("min_days") or 0)
        max_days = int(level.get("max_days") or 9999)
        if min_days <= days <= max_days:
            selected = level
            break
    if selected is None:
        selected = levels[-1]
    level_shift = int(selected.get("level_shift") or 0)
    coefficient = float(selected.get("coefficient") or 1.0)
    if redline_acute:
        level_shift = max(0, level_shift)
        coefficient = max(1.0, coefficient)
    return {
        "key": selected.get("key") or "",
        "label": selected.get("label") or "",
        "coefficient": coefficient,
        "level_shift": level_shift,
        "score_delta": int(selected.get("score_delta") or 0),
        "description": selected.get("description") or "",
        "duration_days": days,
    }


def _score_dimension(values: Dict[str, float], dimension: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    levels = sorted(dimension.get("levels", []), key=lambda item: int(item.get("score") or 0))
    selected_index, selected = _select_base_level(values, dimension)
    counts = _dimension_counts(values, dimension)
    redline_acute = bool(dimension.get("redline")) and (counts["plan_hits"] > 0 or counts["explicit_hits"] > 0 or counts["method_hits"] > 0)

    strength = _evidence_strength(config, dimension, counts)
    time_info = _time_level(config, counts, redline_acute)
    level_shift = int(time_info.get("level_shift") or 0)
    if level_shift < 0 and selected_index <= 1:
        level_shift = 0
    shifted_index = max(0, min(len(levels) - 1, selected_index + level_shift)) if levels else 0
    shifted_level = levels[shifted_index] if levels else selected

    base_score = max(0, min(int(dimension.get("max_points") or 0), int(selected.get("score") or 0)))
    level_adjusted_score = max(0, min(int(dimension.get("max_points") or 0), int(shifted_level.get("score") or 0)))
    adjusted = level_adjusted_score * float(strength.get("coefficient") or 1.0) * float(time_info.get("coefficient") or 1.0)
    adjusted += int(time_info.get("score_delta") or 0)
    final_score = int(round(max(0, min(int(dimension.get("max_points") or 0), adjusted))))

    key = dimension["key"]
    return {
        "key": key,
        "label": dimension.get("label") or key,
        "score": final_score,
        "base_score": base_score,
        "level_adjusted_score": level_adjusted_score,
        "max_points": int(dimension.get("max_points") or 0),
        "description": selected.get("description") or "",
        "adjusted_description": shifted_level.get("description") or selected.get("description") or "",
        "redline": bool(dimension.get("redline", False)),
        "evidence_strength": {
            "key": strength.get("key"),
            "label": strength.get("label"),
            "coefficient": strength.get("coefficient"),
            "description": strength.get("description"),
        },
        "time_adjustment": time_info,
        "applied_level_shift": level_shift,
        "keyword_count": counts["keyword_hits"],
        "keyword_message_count": counts["keyword_messages"],
        "keyword_active_days": counts["keyword_days"],
        "semantic_hit_count": counts["semantic_messages"],
        "semantic_active_days": counts["semantic_days"],
        "strong_count": counts["strong_hits"],
        "duration_days": time_info.get("duration_days", counts["day_span"]),
        "worsening_trend": bool(counts["worsening_trend"]),
    }


def _self_harm_risk_from_dimension(dimension_scores: List[Dict[str, Any]]) -> str:
    item = next((score for score in dimension_scores if score.get("key") == "self_harm_suicide_risk"), None)
    if not item:
        return "low"
    value = int(item.get("score") or 0)
    if value >= 10:
        return "crisis"
    if value >= 6:
        return "high"
    if value >= 3:
        return "medium"
    return "low"


def _overall_time_adjustments(values: Dict[str, float], dimension_scores: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    section = config.get("time_adjustment", {})
    bonus_config = section.get("worsening_bonus", {}) if isinstance(section, dict) else {}
    relief_config = section.get("relief_reduction", {}) if isinstance(section, dict) else {}

    worsening_dimensions = [item for item in dimension_scores if item.get("worsening_trend") and int(item.get("score") or 0) > 0]
    worsening_bonus = 0
    if section.get("enabled", True) and bonus_config.get("enabled", True) and worsening_dimensions:
        min_bonus = int(bonus_config.get("min_bonus") or 5)
        max_bonus = int(bonus_config.get("max_bonus") or 10)
        worsening_bonus = min(max_bonus, min_bonus + max(0, len(worsening_dimensions) - 1) * 2)

    relief_delta = 0
    if section.get("enabled", True) and relief_config.get("enabled", True) and _value(values, "relief_keyword_count") > 0:
        relief_delta = int(relief_config.get("score_delta") or -5)

    return {
        "worsening_bonus": worsening_bonus,
        "worsening_dimensions": [item.get("key") for item in worsening_dimensions],
        "relief_delta": relief_delta,
        "relief_keyword_count": _value(values, "relief_keyword_count"),
    }


def _rule_matches_message_days(rule: Dict[str, Any], message_count: float, active_days: float, hit_count: float = 0) -> bool:
    if message_count < float(rule.get("min_messages", 0) or 0):
        return False
    if active_days < float(rule.get("min_active_days", 0) or 0):
        return False
    if hit_count < float(rule.get("min_hits", 0) or 0):
        return False
    if "max_messages" in rule and message_count > float(rule.get("max_messages") or 0):
        return False
    if "max_active_days" in rule and active_days > float(rule.get("max_active_days") or 0):
        return False
    if "max_hits" in rule and hit_count > float(rule.get("max_hits") or 0):
        return False
    return True


def _protective_adjustment(
    symptom_labels: List[Dict[str, Any]],
    self_harm_risk: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    section = config.get("protective_adjustment", {})
    if not isinstance(section, dict) or not section.get("enabled", True):
        return {
            "protective_delta": 0,
            "protective_factor_delta": 0,
            "redline_safety_delta": 0,
            "redline_blocks_protective_reduction": False,
            "protective_factors": [],
        }

    labels_by_key = {str(item.get("key") or ""): item for item in symptom_labels}
    redline_present = self_harm_risk in {"medium", "high", "crisis"} or any(
        key in labels_by_key for key in {"self_harm_suicide", "existential_death_expression"}
    )
    factors: List[Dict[str, Any]] = []
    factor_delta = 0
    for factor in section.get("factors", []):
        if not isinstance(factor, dict):
            continue
        factor_label_keys = [str(item) for item in factor.get("label_keys", []) if str(item or "").strip()]
        matched = [labels_by_key[key] for key in factor_label_keys if key in labels_by_key]
        message_count = sum(float(item.get("message_count") or 0) for item in matched)
        active_days = max([float(item.get("active_days") or 0) for item in matched] or [0.0])
        hit_count = sum(float(item.get("count") or 0) for item in matched)
        selected = {"delta": 0, "description": "未命中保护性修正"}
        for level in factor.get("levels", []):
            if not isinstance(level, dict):
                continue
            rule = level.get("rule") if isinstance(level.get("rule"), dict) else {}
            if _rule_matches_message_days(rule, message_count, active_days, hit_count):
                if int(level.get("delta") or 0) <= int(selected.get("delta") or 0):
                    selected = level
        delta = max(int(factor.get("max_delta") or 0), int(selected.get("delta") or 0))
        factor_delta += delta
        if matched or delta:
            factors.append(
                {
                    "key": factor.get("key"),
                    "label": factor.get("label"),
                    "delta": delta,
                    "description": selected.get("description") or factor.get("description") or "",
                    "message_count": message_count,
                    "active_days": active_days,
                    "hit_count": hit_count,
                    "matched_labels": [
                        {
                            "key": item.get("key"),
                            "label": item.get("label"),
                            "message_count": item.get("message_count"),
                            "active_days": item.get("active_days"),
                        }
                        for item in matched
                    ],
                }
            )

    min_delta = int(section.get("min_delta") or -20)
    max_delta = int(section.get("max_delta") or 10)
    factor_delta = max(min_delta, min(0, factor_delta))
    blocked = bool(redline_present and section.get("redline_blocks_reduction", True) and factor_delta < 0)
    applied_factor_delta = 0 if blocked else factor_delta

    redline_bonus_config = section.get("redline_bonus", {}) if isinstance(section.get("redline_bonus"), dict) else {}
    redline_delta = 0
    if redline_present and redline_bonus_config.get("enabled", True):
        redline_delta = max(0, int(redline_bonus_config.get("delta") or 0))

    total_delta = max(min_delta, min(max_delta, applied_factor_delta + redline_delta))
    return {
        "protective_delta": total_delta,
        "protective_factor_delta": factor_delta,
        "applied_protective_factor_delta": applied_factor_delta,
        "positive_emotion_delta": next((item.get("delta", 0) for item in factors if item.get("key") == "positive_emotion"), 0),
        "positive_emotion_message_count": next((item.get("message_count", 0) for item in factors if item.get("key") == "positive_emotion"), 0),
        "positive_emotion_active_days": next((item.get("active_days", 0) for item in factors if item.get("key") == "positive_emotion"), 0),
        "positive_emotion_hit_count": next((item.get("hit_count", 0) for item in factors if item.get("key") == "positive_emotion"), 0),
        "redline_safety_delta": redline_delta,
        "redline_blocks_protective_reduction": blocked,
        "redline_present": redline_present,
        "protective_factors": factors,
        "protective_bounds": {"min_delta": min_delta, "max_delta": max_delta},
    }


def compute_score(
    values: Dict[str, float],
    evidences: List[PsychEvidence],
    crisis: Optional[Dict] = None,
    config: Optional[Dict[str, Any]] = None,
) -> PsychScore:
    scoring_config = normalize_scoring_config(config) if config else load_scoring_config()
    dimension_scores = [
        _score_dimension(values, dimension, scoring_config)
        for dimension in scoring_config.get("dimensions", [])
        if dimension.get("enabled", True)
    ]
    adjustment = _overall_time_adjustments(values, dimension_scores, scoring_config)
    raw_dimension_score = sum(int(item.get("score") or 0) for item in dimension_scores)

    self_harm_risk = _self_harm_risk_from_dimension(dimension_scores)
    symptom_labels = _label_summary(values, scoring_config)
    protective_adjustment = _protective_adjustment(symptom_labels, self_harm_risk, scoring_config)
    pre_protective_score = raw_dimension_score + int(adjustment.get("worsening_bonus") or 0) + int(adjustment.get("relief_delta") or 0)
    adjusted_raw_score = pre_protective_score + int(protective_adjustment.get("protective_delta") or 0)
    max_score = int(scoring_config.get("max_score") or 100)
    score = int(max(0, min(max_score, adjusted_raw_score)))

    overall = _risk_from_score(score, scoring_config.get("thresholds", {}))
    if self_harm_risk in {"high", "crisis"}:
        overall = _max_risk(overall, self_harm_risk)

    label_keys = {item.get("key") for item in symptom_labels}
    protective_count = len([item for item in symptom_labels if item.get("protective")])
    modifier_count = len([item for item in symptom_labels if item.get("modifier")])
    max_label_level = max([int(item.get("risk_level") or 0) for item in symptom_labels if not item.get("protective") and not item.get("modifier")] or [0])
    core_label_count = len(
        [
            item
            for item in symptom_labels
            if item.get("key") in {
                "low_mood",
                "interest_loss",
                "self_negation",
                "sleep_abnormal",
                "social_withdrawal",
                "function_impairment",
                "helpless_hopeless",
                "pressure_expression",
                "emotional_expression",
                "cognitive_expression",
                "existential_death_expression",
            }
            and float(item.get("message_count") or 0) > 0
        ]
    )
    function_item = next((item for item in dimension_scores if item.get("key") == "social_withdrawal_function_impairment"), {})
    worth_item = next((item for item in dimension_scores if item.get("key") == "self_blame_worthlessness"), {})
    max_duration = max([int(item.get("duration_days") or 0) for item in dimension_scores if int(item.get("score") or 0) > 0] or [0])
    numeric_risk_level = 0
    if self_harm_risk in {"high", "crisis"} or "self_harm_suicide" in label_keys or "existential_death_expression" in label_keys:
        numeric_risk_level = 5
    elif (
        max_duration >= 14
        and (int(function_item.get("score") or 0) >= 8 or int(worth_item.get("score") or 0) >= 10)
    ) or max_label_level >= 4:
        numeric_risk_level = 4
    elif core_label_count >= 2 or len([item for item in dimension_scores if int(item.get("score") or 0) > 0]) >= 3:
        numeric_risk_level = 3
    elif max_label_level >= 2 or score >= int(scoring_config.get("thresholds", {}).get("medium", 35)):
        numeric_risk_level = 2
    elif symptom_labels:
        numeric_risk_level = 1
    if numeric_risk_level in {1, 2} and protective_count > 0:
        numeric_risk_level = max(1, numeric_risk_level - 1)
    if numeric_risk_level in {1, 2} and modifier_count > 0 and score < int(scoring_config.get("thresholds", {}).get("medium", 35)):
        numeric_risk_level = max(0, numeric_risk_level - 1)

    overall = _max_risk(overall, _overall_from_numeric_level(numeric_risk_level))
    risk_level_label = _risk_level_label(scoring_config, numeric_risk_level)

    message_count = values.get("message_count", 0)
    active_days = values.get("unique_active_days", 0)
    confidence_config = scoring_config.get("confidence", {})
    confidence = min(
        1.0,
        float(confidence_config.get("base", 0.25))
        + min(message_count, float(confidence_config.get("message_cap", 80))) / float(confidence_config.get("message_divisor", 120))
        + min(active_days, float(confidence_config.get("active_days_cap", 14))) / float(confidence_config.get("active_days_divisor", 35)),
    )
    for cap in confidence_config.get("small_sample_caps", []):
        if message_count < float(cap.get("max_messages", 0)):
            confidence = min(confidence, float(cap.get("confidence_cap", confidence)))
            break
    confidence = round(float(confidence), 2)

    dimension_scores.sort(key=lambda item: int(item.get("score") or 0), reverse=True)
    main_signals = [
        f"{item['label']} {int(item['score'])}/{int(item['max_points'])}"
        for item in dimension_scores
        if int(item.get("score") or 0) > 0
    ][:6]
    if not main_signals:
        main_signals = ["所选文本中暂未观察到密集的抑郁相关信号"]

    if self_harm_risk in {"high", "crisis"}:
        summary = "所选文本中出现自伤或轻生相关红线信号；这不是诊断结论，建议立即联系可信任的人陪伴，并尽快寻求当地紧急救助或专业评估。"
    elif overall == "high":
        summary = "所选文本中存在较多抑郁相关信号，且证据强度或持续时间提示需要认真关注；建议尽快寻求专业人员进一步评估。"
    elif overall == "medium":
        summary = "所选文本中观察到一些抑郁相关信号，建议持续关注其持续时间、功能影响和现实诱因，并考虑专业评估。"
    else:
        summary = "基于当前所选文本，暂未观察到密集的抑郁相关信号；样本有限时仍需结合现实状态判断。"

    adjustment.update(
        {
            "raw_dimension_score": raw_dimension_score,
            "pre_protective_score": pre_protective_score,
            "adjusted_raw_score": adjusted_raw_score,
            "final_score": score,
            **protective_adjustment,
            "risk_level": numeric_risk_level,
            "risk_level_label": risk_level_label,
            "protective_label_count": protective_count,
            "context_modifier_count": modifier_count,
            "core_label_count": core_label_count,
            "max_label_risk_level": max_label_level,
        }
    )

    return PsychScore(
        depression_signal_score=score,
        self_harm_risk=self_harm_risk,
        overall_risk=overall,
        risk_level=numeric_risk_level,
        risk_level_label=risk_level_label,
        confidence=confidence,
        summary=summary,
        main_signals=main_signals,
        symptom_labels=symptom_labels,
        dimension_scores=dimension_scores,
        scoring_adjustments=adjustment,
    )
