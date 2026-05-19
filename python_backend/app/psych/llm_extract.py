import json
import re
from typing import Dict, List, Optional, Tuple

from app.models import ChatMessage, PsychEvidence, PsychFact
from app.psych.features import WORD_GROUPS
from app.services.llm_service import complete_chat
from app.utils.privacy import sanitize_snippet


PSYCH_FACT_PROMPT = """你是心理风险辅助筛查信息抽取器。
只基于本人明确发出的消息提取事实，不做医学诊断，不输出“确诊”“重度抑郁症”等结论，不提供药物建议。
输出 JSON 数组。每个对象包含：
- fact_type: signal/protective_factor/stressor/context/self_harm_risk/sleep/social/interest/fatigue/self_negation/hopelessness
- fact: 简短事实陈述，必须是辅助筛查表述
- severity: low/medium/high/crisis
- confidence: 0 到 1
- source_from: 起始消息 seq
- source_to: 结束消息 seq
可关注：持续痛苦、睡眠困扰、兴趣下降、自我否定、无望感、自伤或轻生风险线索、保护因素。
只返回 JSON 数组，不要额外解释。
"""

DIAGNOSIS_WORDS = ["确诊", "诊断为", "重度抑郁症", "中度抑郁症", "轻度抑郁症", "建议服药", "用药"]


def _extract_json_array(text: str) -> List[Dict]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", stripped, flags=re.S)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _severity_rank(severity: str) -> int:
    return {"low": 0, "medium": 1, "high": 2, "crisis": 3}.get(severity, 0)


def _normalize_severity(value: str) -> str:
    value = (value or "low").strip().lower()
    return value if value in {"low", "medium", "high", "crisis"} else "low"


def _clean_fact(text: str) -> str:
    fact = sanitize_snippet(text, max_len=180)
    for word in DIAGNOSIS_WORDS:
        fact = fact.replace(word, "相关信号")
    return fact


def _message_evidence(msg: ChatMessage, reason: str, severity: str = "medium") -> PsychEvidence:
    return PsychEvidence(
        seq=msg.seq,
        datetime=msg.datetime,
        sender=msg.sender,
        content=sanitize_snippet(msg.content),
        evidence_type="psych_fact_source",
        severity=severity,
        reason=reason,
    )


def _fallback_facts(messages: List[ChatMessage], evidences: Optional[List[PsychEvidence]] = None) -> List[PsychFact]:
    facts: List[PsychFact] = []
    grouped: Dict[str, List[PsychEvidence]] = {}
    for evidence in evidences or []:
        grouped.setdefault(evidence.evidence_type, []).append(evidence)

    for fact_type, items in grouped.items():
        if not items:
            continue
        highest = max((item.severity for item in items), key=_severity_rank, default="low")
        label = fact_type
        for _, (etype, _, _, reason) in WORD_GROUPS.items():
            if etype == fact_type:
                label = reason
                break
        facts.append(
            PsychFact(
                fact_type=fact_type,
                fact=f"检测到{label}，共 {len(items)} 条相关证据，建议结合上下文进一步专业评估。",
                severity=highest,
                confidence=min(0.85, 0.45 + len(items) * 0.08),
                evidence=items[:3],
                source_from=min(item.seq for item in items),
                source_to=max(item.seq for item in items),
            )
        )

    if facts:
        return facts[:12]

    for msg in messages[:8]:
        text = msg.content or ""
        if len(text) < 8:
            continue
        facts.append(
            PsychFact(
                fact_type="context",
                fact=f"候选消息提示需要结合上下文评估：{sanitize_snippet(text, max_len=80)}",
                severity="low",
                confidence=0.3,
                evidence=[_message_evidence(msg, "规则兜底事实来源", "low")],
                source_from=msg.seq,
                source_to=msg.seq,
            )
        )
    return facts[:5]


def _facts_from_llm_items(items: List[Dict], messages: List[ChatMessage]) -> List[PsychFact]:
    by_seq = {msg.seq: msg for msg in messages}
    out: List[PsychFact] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_fact = str(item.get("fact") or "").strip()
        if not raw_fact:
            continue
        source_from = int(item.get("source_from") or item.get("seq") or 0)
        source_to = int(item.get("source_to") or source_from or 0)
        evidence = []
        for msg in messages:
            if source_from and source_to and source_from <= msg.seq <= source_to:
                evidence.append(_message_evidence(msg, "大模型事实抽取来源", _normalize_severity(str(item.get("severity") or "medium"))))
            if len(evidence) >= 3:
                break
        if not evidence and source_from in by_seq:
            evidence = [_message_evidence(by_seq[source_from], "大模型事实抽取来源")]
        confidence = item.get("confidence", 0.5)
        try:
            confidence_value = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence_value = 0.5
        out.append(
            PsychFact(
                fact_type=str(item.get("fact_type") or "signal")[:40],
                fact=_clean_fact(raw_fact),
                severity=_normalize_severity(str(item.get("severity") or "medium")),
                confidence=confidence_value,
                evidence=evidence,
                source_from=source_from or None,
                source_to=source_to or None,
            )
        )
    return out[:20]


def extract_facts_with_metrics(
    messages: List[ChatMessage],
    evidences: Optional[List[PsychEvidence]] = None,
    options: Optional[Dict] = None,
) -> Tuple[List[PsychFact], Dict]:
    options = options or {}
    if not messages:
        return [], {"fact_extractor": "none", "fact_extract_reason": "no_messages"}

    limit = int(options.get("fact_extract_limit") or 50)
    candidates = messages[: max(1, limit)]
    llm_enabled = bool(options.get("llm_fact_extraction", True))
    if llm_enabled:
        payload = [
            {
                "seq": msg.seq,
                "datetime": msg.datetime,
                "sender": msg.sender,
                "text": sanitize_snippet(msg.content, max_len=180),
            }
            for msg in candidates
        ]
        try:
            reply = complete_chat(
                [
                    {"role": "system", "content": PSYCH_FACT_PROMPT},
                    {
                        "role": "user",
                        "content": "请抽取心理风险辅助筛查事实，只返回 JSON 数组：\n"
                        + json.dumps(payload, ensure_ascii=False),
                    },
                ],
                config=options.get("llm_config") or None,
            )
            facts = _facts_from_llm_items(_extract_json_array(reply), candidates)
            if facts:
                return facts, {
                    "fact_extractor": "llm",
                    "fact_candidate_count": len(candidates),
                    "fact_count": len(facts),
                }
        except Exception as error:
            fallback = _fallback_facts(candidates, evidences)
            return fallback, {
                "fact_extractor": "rule_fallback",
                "fact_candidate_count": len(candidates),
                "fact_count": len(fallback),
                "fact_llm_error": sanitize_snippet(str(error) or error.__class__.__name__, max_len=160),
            }

    fallback = _fallback_facts(candidates, evidences)
    return fallback, {
        "fact_extractor": "rule_fallback",
        "fact_candidate_count": len(candidates),
        "fact_count": len(fallback),
        "fact_extract_reason": "llm_disabled_or_empty",
    }


def extract_facts(messages: List[ChatMessage]) -> List[PsychFact]:
    facts, _ = extract_facts_with_metrics(messages)
    return facts
