import json
import re
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from app.config import get_settings
from app.models import ChatMessage, PsychEvidence, PsychFact
from app.psych import preprocess
from app.services.embedding_service import get_embedding, get_embeddings
from app.services.llm_service import complete_chat
from app.stores import memory_store
from app.utils.privacy import sanitize_snippet


FACT_KEYWORDS = {
    "preference": ("喜欢", "爱吃", "爱看", "讨厌", "不喜欢"),
    "experience": ("去过", "经历", "参加", "毕业", "入职", "离职"),
    "place": ("住在", "在北京", "在上海", "在深圳", "在广州", "老家"),
    "relationship": ("朋友", "同事", "家人", "对象", "男朋友", "女朋友"),
    "opinion": ("觉得", "认为", "看法", "我感觉"),
    "habit": ("每天", "经常", "总是", "习惯", "周末"),
}

PSYCH_FACT_SYSTEM_PROMPT = """你是心理风险辅助筛查的信息抽取器。
只基于“本人明确发出的消息”提取事实，不做医学诊断，不输出“确诊”“重度抑郁症”等疾病结论，不提供药物建议。
重点提取：持续痛苦、睡眠困扰、兴趣下降、自我否定、无望感、社交退缩、疲惫、保护因素、求助线索、压力源。
输出必须是 JSON 数组。每个对象包含：
- fact_type: signal/protective_factor/stressor/context/sleep/social/interest/fatigue/self_negation/hopelessness
- fact: 简短事实陈述，必须是“辅助筛查线索”表述
- severity: low/medium/high/crisis
- confidence: 0 到 1
- source_from: 起始消息 seq
- source_to: 结束消息 seq
如果没有可用事实，返回 []。"""

DIAGNOSIS_WORDS = ("确诊", "诊断为", "重度抑郁症", "中度抑郁症", "轻度抑郁症", "建议服药", "用药")


class MemoryBuildCancelled(RuntimeError):
    pass


ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


def _clean_fact(text: str) -> str:
    fact = sanitize_snippet(text, max_len=180)
    for word in DIAGNOSIS_WORDS:
        fact = fact.replace(word, "相关信号")
    return fact


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


def _normalize_severity(value: str) -> str:
    value = (value or "low").strip().lower()
    return value if value in {"low", "medium", "high", "crisis"} else "low"


def _chunks(items: List[ChatMessage], size: int) -> Iterable[List[ChatMessage]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _psych_fact_llm_config(options: Optional[Dict] = None) -> Dict:
    settings = get_settings()
    override = (options or {}).get("psych_fact_llm_config") or {}
    return {
        "provider": override.get("provider") or settings.psych_fact_llm_provider,
        "base_url": override.get("base_url") or settings.psych_fact_llm_base_url,
        "model": override.get("model") or settings.psych_fact_llm_model,
        "api_key": override.get("api_key") or settings.psych_fact_llm_api_key,
    }


def _message_evidence(msg: ChatMessage, reason: str, severity: str) -> Dict:
    return {
        "seq": msg.seq,
        "datetime": msg.datetime,
        "sender": msg.sender,
        "content": sanitize_snippet(msg.content, max_len=160),
        "evidence_type": "psych_memory_fact_source",
        "severity": severity,
        "reason": reason,
    }


def _facts_from_items(items: List[Dict], chunk: List[ChatMessage]) -> List[Dict]:
    out: List[Dict] = []
    by_seq = {msg.seq: msg for msg in chunk}
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_fact = str(item.get("fact") or "").strip()
        if not raw_fact:
            continue
        source_from = int(item.get("source_from") or item.get("seq") or 0)
        source_to = int(item.get("source_to") or source_from or 0)
        severity = _normalize_severity(str(item.get("severity") or "medium"))
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        evidence = []
        for msg in chunk:
            if source_from and source_to and source_from <= msg.seq <= source_to:
                evidence.append(_message_evidence(msg, "心理事实抽取来源", severity))
            if len(evidence) >= 3:
                break
        if not evidence and source_from in by_seq:
            evidence = [_message_evidence(by_seq[source_from], "心理事实抽取来源", severity)]
        out.append(
            {
                "fact_type": str(item.get("fact_type") or "signal")[:40],
                "fact": _clean_fact(raw_fact),
                "severity": severity,
                "confidence": confidence,
                "source_from": source_from,
                "source_to": source_to,
                "evidence": evidence,
            }
        )
    return out


def extract_facts_from_chunk(chunk: List[ChatMessage], llm_config: Dict) -> List[Dict]:
    payload = [
        {
            "seq": msg.seq,
            "datetime": msg.datetime,
            "sender": msg.sender,
            "text": sanitize_snippet(msg.content, max_len=180),
        }
        for msg in chunk
    ]
    reply = complete_chat(
        [
            {"role": "system", "content": PSYCH_FACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "请从以下聊天消息中抽取心理风险辅助筛查事实，只返回 JSON 数组：\n"
                + json.dumps(payload, ensure_ascii=False),
            },
        ],
        config=llm_config,
    )
    return _facts_from_items(_extract_json_array(reply), chunk)


def _fact_embedding_text(item: Dict) -> str:
    return (
        f"心理风险辅助筛查事实。类型：{item.get('fact_type') or 'signal'}；"
        f"严重度：{item.get('severity') or 'low'}；事实：{item.get('fact') or ''}；"
        f"置信度：{float(item.get('confidence') or 0):.2f}"
    )


def _dedupe_facts(facts: List[Dict]) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for fact in facts:
        key = (
            str(fact.get("fact_type") or ""),
            str(fact.get("fact") or ""),
            int(fact.get("source_from") or 0),
            int(fact.get("source_to") or 0),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(fact)
    return out


def build_psych_memory_facts(
    target_key: str,
    messages: List[ChatMessage],
    options: Optional[Dict] = None,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_callback: Optional[CancelCallback] = None,
) -> Dict:
    options = options or {}
    settings = get_settings()
    only_mine = bool(options.get("only_mine", True))
    message_scope = str(options.get("message_scope") or ("mine" if only_mine else "all")).lower()
    chunk_size = max(1, int(options.get("chunk_size") or settings.psych_fact_chunk_size or 80))
    filtered = preprocess.filter_messages(messages, only_mine=False)
    if message_scope in {"mine", "self"}:
        filtered = [msg for msg in filtered if msg.is_mine]
    elif message_scope == "other":
        filtered = [msg for msg in filtered if not msg.is_mine]
    llm_config = _psych_fact_llm_config(options)
    model_label = str(llm_config.get("model") or "")

    if cancel_callback and cancel_callback():
        raise MemoryBuildCancelled("psych_fact_build_cancelled")
    if progress_callback:
        progress_callback(0, len(filtered), f"已过滤出 {len(filtered)} 条可抽取消息，每 {chunk_size} 条一批")

    facts: List[Dict] = []
    chunks = list(_chunks(filtered, chunk_size))
    processed_messages = 0
    for index, chunk in enumerate(chunks, start=1):
        if cancel_callback and cancel_callback():
            raise MemoryBuildCancelled("psych_fact_build_cancelled")
        if progress_callback:
            progress_callback(processed_messages, len(filtered), f"正在抽取第 {index}/{len(chunks)} 批心理事实")
        chunk_facts = extract_facts_from_chunk(chunk, llm_config)
        facts.extend(chunk_facts)
        processed_messages += len(chunk)
        if progress_callback:
            progress_callback(processed_messages, len(filtered), f"第 {index}/{len(chunks)} 批完成，累计事实 {len(facts)} 条")

    facts = _dedupe_facts(facts)
    if facts:
        if progress_callback:
            progress_callback(len(filtered), len(filtered), f"正在为 {len(facts)} 条心理事实计算 embedding")
        embeddings = get_embeddings(
            [_fact_embedding_text(fact) for fact in facts],
            config=options.get("embedding_config") or None,
            input_type="document",
        )
        for fact, embedding in zip(facts, embeddings):
            fact["embedding"] = embedding
            fact["source_model"] = model_label

    memory_store.replace_facts(target_key, facts, source_kind="psych", source_model=model_label)
    return {
        "built": True,
        "contact_key": target_key,
        "source_message_count": len(messages),
        "filtered_message_count": len(filtered),
        "chunk_size": chunk_size,
        "chunk_count": len(chunks),
        "fact_count": len(facts),
        "model": model_label,
        "only_mine": only_mine,
        "message_scope": message_scope,
    }


def extract_memory_facts(target_key: str, messages: List[ChatMessage]) -> List[Dict]:
    facts = []
    seen = set()
    for msg in messages:
        text = (msg.content or "").strip()
        if len(text) < 4:
            continue
        for fact_type, keywords in FACT_KEYWORDS.items():
            if any(word in text for word in keywords):
                fact = f"{fact_type}: {sanitize_snippet(text, 80)}"
                if fact in seen:
                    continue
                seen.add(fact)
                facts.append({"fact": fact, "source_from": msg.seq, "source_to": msg.seq, "fact_type": fact_type})
                break
    if facts:
        try:
            embeddings = get_embeddings([f["fact"] for f in facts], input_type="document")
        except Exception:
            embeddings = [[] for _ in facts]
        for fact, embedding in zip(facts, embeddings):
            fact["embedding"] = embedding
    memory_store.replace_facts(target_key, facts, source_kind="memory")
    return [{"fact": f["fact"], "source_from": f["source_from"], "source_to": f["source_to"]} for f in facts]


def _search_facts(
    target_key: str,
    query: str,
    top_k: int = 5,
    source_kind: str = "memory",
    embedding_config: Optional[Dict] = None,
) -> List[Dict]:
    facts = memory_store.list_facts(target_key, source_kind=source_kind)
    if not facts or not query.strip():
        return []
    try:
        qvec = np.asarray(get_embedding(query, config=embedding_config, input_type="query"), dtype=np.float32)
    except Exception:
        q = query.lower()
        return [f for f in facts if q in f.get("fact", "").lower()][:top_k]
    qnorm = float(np.linalg.norm(qvec))
    scored = []
    for fact in facts:
        emb = fact.get("embedding_array")
        if emb is None or len(emb) == 0:
            continue
        denom = float(np.linalg.norm(emb)) * qnorm
        if denom:
            scored.append((float(np.dot(emb, qvec) / denom), fact))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [{"score": score, **{k: v for k, v in fact.items() if k != "embedding_array"}} for score, fact in scored[:top_k]]


def search_memory_facts(target_key: str, query: str, top_k: int = 5) -> List[Dict]:
    return _search_facts(target_key, query, top_k=top_k, source_kind="memory")


def search_psych_memory_facts(
    target_key: str,
    query: str,
    top_k: int = 8,
    embedding_config: Optional[Dict] = None,
) -> List[Dict]:
    return _search_facts(
        target_key,
        query,
        top_k=top_k,
        source_kind="psych",
        embedding_config=embedding_config,
    )


def psych_hits_to_facts(hits: List[Dict]) -> List[PsychFact]:
    facts: List[PsychFact] = []
    for hit in hits:
        evidence = []
        for item in hit.get("evidence") or []:
            try:
                evidence.append(PsychEvidence(**item))
            except Exception:
                continue
        facts.append(
            PsychFact(
                fact_type=str(hit.get("fact_type") or "signal"),
                fact=str(hit.get("fact") or ""),
                severity=str(hit.get("severity") or "low"),
                confidence=float(hit.get("confidence") or 0),
                evidence=evidence,
                source_from=int(hit.get("source_from") or 0) or None,
                source_to=int(hit.get("source_to") or 0) or None,
            )
        )
    return facts


def get_psych_fact_status(target_key: str) -> Dict:
    return memory_store.count_facts(target_key, source_kind="psych")
