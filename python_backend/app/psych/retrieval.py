import json
import re
from typing import Any, Callable, Dict, List, Tuple

from app.models import ChatMessage
from app.psych import labels as label_classifier
from app.psych.scoring_config import enabled_dimensions, normalize_scoring_config
from app.services.llm_service import complete_chat
from app.utils.privacy import sanitize_snippet


RISK_CONTEXT_RULES = """风险语境判断硬规则：
1. 单次出现“烦、累、崩溃、想死了”等网络化表达，若上下文明显是玩笑、吐槽、游戏、考试、堵车等具体事件，不直接判为高风险。
2. 但只要出现明确自杀意图、具体方法、时间、地点、工具、告别、遗书、财物安排或近期自伤行为，不允许被“哈哈”“开玩笑”“我没事”完全抵消，只能标记为需要人工复核。
3. 负面词只有在“持续出现 + 无明显缓解 + 功能受损 + 缺乏保护性信号”时，才提高风险等级。
4. 有积极计划、求助行为、社会支持、正常工作生活，可以作为保护性修正，但不能抵消明确自伤风险。
5. 风险等级 3 以上建议人工复核；风险等级 5 必须立即触发安全处理流程。"""


EXPRESSION_TYPE_RULES = """表达类型分层：
1. 事件型抱怨：如“今天堵车烦死了”，围绕具体事件吐槽，低权重。
2. 压力型表达：如“项目太赶，我快累瘫了”，看持续时间、功能影响和恢复线索。
3. 情绪型表达：如“最近一直很低落”，中权重，关注是否持续出现。
4. 认知型表达：如“我很失败，我没用”，高权重，关注自责、无价值感、拖累感。
5. 存在/死亡型表达：如“活着没意义，不想活了”，红线项；只做风险提示和求助建议，不做诊断。"""


SCREENING_SYSTEM_PROMPT = """你是心理风险辅助筛查的文本筛选器。
任务是对候选聊天消息做多标签分类，并判断是否有助于识别抑郁相关信号、自伤/轻生红线信号或保护因素。
你只做筛选，不做医学诊断，不输出疾病结论，不提供药物建议。
如果提供 context_before/context_after，请只把它们用于理解 candidate 的语义、玩笑、引用、新闻或剧情语境；不要把对方说的话直接当作本人心理状态。
一句 candidate 可以同时拥有多个标签。标签建议使用：
普通抱怨、压力疲劳、焦虑担忧、情绪低落、兴趣下降、无助绝望、自我否定、睡眠异常、食欲变化、注意力下降、社交退缩、功能受损、自伤轻生、正面情绪、希望感/未来感、积极计划、社会连接、求助意愿、功能保持、幽默调侃。
表达类型标签建议使用：事件型抱怨、压力型表达、情绪型表达、认知型表达、存在/死亡型表达。
最终为每条 candidate 给一个 0-5 风险等级：0 无明显信号，1 普通抱怨，2 明显压力或低落但有恢复线索，3 多项抑郁相关信号，4 持续且功能受损或强烈无价值感，5 自伤/轻生红线信号。
表达类型分层：
1. 事件型抱怨：如“今天堵车烦死了”，围绕具体事件吐槽，低权重。
2. 压力型表达：如“项目太赶，我快累瘫了”，看持续时间、功能影响和恢复线索。
3. 情绪型表达：如“最近一直很低落”，中权重，关注是否持续出现。
4. 认知型表达：如“我很失败，我没用”，高权重，关注自责、无价值感、拖累感。
5. 存在/死亡型表达：如“活着没意义，不想活了”，红线项；只做风险提示和求助建议，不做诊断。
风险语境判断硬规则：
1. 单次出现“烦、累、崩溃、想死了”等网络化表达，若上下文明显是玩笑、吐槽、游戏、考试、堵车等具体事件，不直接判为高风险。
2. 但只要出现明确自杀意图、具体方法、时间、地点、工具、告别、遗书、财物安排或近期自伤行为，不允许被“哈哈”“开玩笑”“我没事”完全抵消，只能标记为需要人工复核。
3. 负面词只有在“持续出现 + 无明显缓解 + 功能受损 + 缺乏保护性信号”时，才提高风险等级。
4. 有积极计划、求助行为、社会支持、正常工作生活，可以作为保护性修正，但不能抵消明确自伤风险。
5. 风险等级 3 以上建议人工复核；风险等级 5 必须立即触发安全处理流程。
只返回 JSON：{"items":[{"id":数字,"useful":true/false,"labels":["标签"...],"risk_level":0-5,"reason":"简短原因"}],"reason":"简短原因"}。
兼容旧格式时也可返回 {"useful_ids":[数字ID...],"reason":"简短原因"}。"""


CONTEXT_SCREENING_NOTE = (
    "请结合 candidate 与 context_before/context_after 做语义分析。"
    "candidate 是候选消息；context 只用于消歧，不作为独立心理证据。"
    "不要输出诊断或药物建议。"
)


def _message_identity(msg: ChatMessage) -> str:
    return f"{msg.seq}\x1f{msg.datetime}\x1f{msg.sender}\x1f{msg.content}"


def _context_item(msg: ChatMessage) -> Dict[str, Any]:
    return {
        "time": msg.datetime,
        "sender": msg.sender,
        "is_mine": bool(msg.is_mine),
        "text": sanitize_snippet(msg.content, max_len=120),
    }


def _context_positions(context_messages: List[ChatMessage]) -> Dict[str, int]:
    return {_message_identity(msg): index for index, msg in enumerate(context_messages)}


def _context_for_message(
    msg: ChatMessage,
    context_messages: List[ChatMessage],
    positions: Dict[str, int],
    window: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    index = positions.get(_message_identity(msg))
    if index is None:
        return [], []
    start = max(0, index - window)
    end = min(len(context_messages), index + window + 1)
    before = [_context_item(item) for item in context_messages[start:index]]
    after = [_context_item(item) for item in context_messages[index + 1 : end]]
    return before, after


def merge_messages(*groups: List[ChatMessage]) -> List[ChatMessage]:
    seen = set()
    out: List[ChatMessage] = []
    for group in groups:
        for msg in group:
            key = _message_identity(msg)
            if key in seen:
                continue
            seen.add(key)
            out.append(msg)
    return sorted(out, key=lambda item: (item.datetime or "", item.seq))


def keyword_retrieve(
    messages: List[ChatMessage],
    limit: int = 80,
    config: Dict[str, Any] | None = None,
) -> Tuple[List[ChatMessage], Dict[str, Any]]:
    scoring_config = normalize_scoring_config(config) if config else None
    dimensions = enabled_dimensions(scoring_config)
    scored: List[Tuple[int, ChatMessage, List[str]]] = []
    group_hits: Dict[str, int] = {}
    group_messages: Dict[str, int] = {}
    label_hits: Dict[str, int] = {}
    label_messages: Dict[str, int] = {}
    for msg in messages:
        text = msg.content or ""
        score = 0
        hit_groups = set()
        hit_words: List[str] = []
        hit_labels = set()
        for dimension in dimensions:
            key = str(dimension.get("key") or "")
            if any(word and word in text for word in dimension.get("exclude_keywords", [])):
                continue
            hits = [word for word in dimension.get("keywords", []) if word and word in text]
            if not hits:
                continue
            hit_words.extend(hits)
            hit_groups.add(key)
            group_hits[key] = group_hits.get(key, 0) + len(hits)
            boost = 4 if dimension.get("redline") else max(1, int(dimension.get("max_points", 0)) // 6)
            score += len(hits) * boost
        for item in label_classifier.classify_message(text, scoring_config):
            label_key = str(item.get("key") or "")
            if not label_key:
                continue
            hit_labels.add(label_key)
            hit_count = max(1, len(item.get("hit_words", [])))
            label_hits[label_key] = label_hits.get(label_key, 0) + hit_count
            if item.get("protective") or item.get("modifier"):
                score += 1
            else:
                score += hit_count * max(1, int(item.get("risk_level") or 1))
        for key in hit_groups:
            group_messages[key] = group_messages.get(key, 0) + 1
        for key in hit_labels:
            label_messages[key] = label_messages.get(key, 0) + 1
        if score > 0:
            scored.append((score, msg, hit_words))

    scored.sort(key=lambda item: (-item[0], item[1].datetime or "", item[1].seq))
    candidates = [item[1] for item in scored[: max(1, limit)]]
    return candidates, {
        "keyword_candidate_count": len(candidates),
        "keyword_total_hits": sum(group_hits.values()),
        "keyword_group_count": len(group_hits),
        "keyword_groups": group_hits,
        "keyword_group_messages": group_messages,
        "keyword_label_hits": label_hits,
        "keyword_label_messages": label_messages,
        "keyword_limit": limit,
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def screen_candidates_with_llm(
    messages: List[ChatMessage],
    stage: str,
    enabled: bool = True,
    limit: int = 40,
    config: Dict[str, Any] | None = None,
    context_messages: List[ChatMessage] | None = None,
    include_context: bool = False,
    context_window: int = 2,
    debug_callback: Callable[[str, Dict[str, Any]], None] | None = None,
    artifact_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> Tuple[List[ChatMessage], Dict[str, Any]]:
    if not messages:
        return [], {
            f"{stage}_screened_count": 0,
            f"{stage}_llm_screened": False,
            f"{stage}_screen_reason": "no_candidates",
        }

    candidates = messages[: max(1, limit)]
    context_source = context_messages or []
    context_window = max(0, min(int(context_window or 0), 5))
    use_context = bool(include_context and context_source and context_window > 0)
    positions = _context_positions(context_source) if use_context else {}
    context_items = 0
    if debug_callback:
        debug_callback(
            "select_llm_candidates",
            {
                "debug_phase": "select_llm_candidates",
                f"{stage}_candidate_input_count": len(messages),
                f"{stage}_candidate_payload_count": len(candidates),
                f"{stage}_screen_limit": limit,
            },
        )

    if not enabled:
        return candidates, {
            f"{stage}_screened_count": len(candidates),
            f"{stage}_llm_screened": False,
            f"{stage}_screen_reason": "disabled",
            f"{stage}_context_enabled": use_context,
            f"{stage}_context_window": context_window if use_context else 0,
            f"{stage}_context_items": context_items,
        }

    payload = []
    for index, msg in enumerate(candidates):
        item: Dict[str, Any] = {
            "id": index,
            "time": msg.datetime,
            "sender": msg.sender,
            "is_mine": bool(msg.is_mine),
            "candidate": sanitize_snippet(msg.content, max_len=160),
            "rule_labels": [
                {
                    "key": label.get("key"),
                    "label": label.get("label"),
                    "category": label.get("category"),
                    "weight": label.get("weight_label") or label.get("weight"),
                    "risk_level": label.get("risk_level"),
                    "protective": label.get("protective", False),
                    "modifier": label.get("modifier", False),
                    "hit_words": label.get("hit_words", []),
                }
                for label in label_classifier.classify_message(msg.content or "", config)
            ],
        }
        if use_context:
            before, after = _context_for_message(msg, context_source, positions, context_window)
            item["context_before"] = before
            item["context_after"] = after
            context_items += len(before) + len(after)
        payload.append(item)
    if artifact_callback:
        artifact_callback(
            {
                "payload": payload,
                "context_enabled": use_context,
                "context_window": context_window if use_context else 0,
                "context_items": context_items,
            }
        )
    if debug_callback:
        debug_callback(
            "build_llm_payload",
            {
                "debug_phase": "build_context_payload",
                f"{stage}_payload_count": len(payload),
                f"{stage}_context_enabled": use_context,
                f"{stage}_context_window": context_window if use_context else 0,
                f"{stage}_context_items": context_items,
                f"{stage}_payload_chars": len(json.dumps(payload, ensure_ascii=False)),
            },
        )

    user_prompt = (
        "请从下面候选消息中筛选对心理风险辅助筛查有用的项，并为每条候选做多标签分类。"
        "如果有上下文，请结合上下文做语义判断，但证据仍以 candidate 为主。"
        "可以给同一句话多个标签；同时给一个 0-5 单一风险等级。"
        "只返回 JSON，不要解释，不要诊断。\n"
        f"{EXPRESSION_TYPE_RULES}\n"
        f"{RISK_CONTEXT_RULES}\n"
        f"{CONTEXT_SCREENING_NOTE}\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    if artifact_callback:
        artifact_callback(
            {
                "system_prompt": SCREENING_SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "prompt_chars": len(user_prompt),
            }
        )
    try:
        if debug_callback:
            debug_callback(
                "call_llm",
                {
                    "debug_phase": "call_llm_screening",
                    f"{stage}_payload_count": len(payload),
                    f"{stage}_prompt_chars": len(user_prompt),
                },
            )
        reply = complete_chat(
            [
                {"role": "system", "content": SCREENING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            config=config,
        )
        parsed = _extract_json_object(reply)
        items = parsed.get("items", [])
        selected_ids = set()
        label_annotations: Dict[int, Dict[str, Any]] = {}
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                raw_id = item.get("id")
                if not (isinstance(raw_id, int) or str(raw_id).isdigit()):
                    continue
                item_id = int(raw_id)
                labels = item.get("labels", [])
                if not isinstance(labels, list):
                    labels = []
                risk_level = max(0, min(5, int(item.get("risk_level") or 0)))
                normalized_labels = [str(label) for label in labels if str(label or "").strip()]
                has_redline_label = any(label in {"自伤轻生", "self_harm_suicide", "自伤/轻生", "轻生风险"} for label in normalized_labels)
                if has_redline_label:
                    risk_level = max(risk_level, 5)
                useful = bool(item.get("useful", risk_level > 0 or normalized_labels))
                if risk_level >= 3:
                    useful = True
                label_annotations[item_id] = {
                    "labels": normalized_labels,
                    "risk_level": risk_level,
                    "reason": sanitize_snippet(str(item.get("reason") or ""), max_len=120),
                    "useful": useful,
                    "manual_review_required": risk_level >= 3,
                    "safety_action_required": risk_level >= 5,
                }
                if useful:
                    selected_ids.add(item_id)
        if not selected_ids:
            ids = parsed.get("useful_ids", [])
            if not isinstance(ids, list):
                ids = []
            selected_ids = {int(item) for item in ids if isinstance(item, int) or str(item).isdigit()}
        selected = [msg for index, msg in enumerate(candidates) if index in selected_ids]
        if artifact_callback:
            artifact_callback(
                {
                    "reply": sanitize_snippet(reply or "", max_len=1200),
                    "selected_ids": sorted(selected_ids),
                    "label_annotations": label_annotations,
                    "selected": [
                        {
                            "id": index,
                            "seq": msg.seq,
                            "datetime": msg.datetime,
                            "sender": msg.sender,
                            "is_mine": bool(msg.is_mine),
                            "content": sanitize_snippet(msg.content, max_len=180),
                            "labels": label_annotations.get(index, {}).get("labels", []),
                            "risk_level": label_annotations.get(index, {}).get("risk_level", 0),
                            "ai_reason": label_annotations.get(index, {}).get("reason", ""),
                            "manual_review_required": label_annotations.get(index, {}).get("manual_review_required", False),
                            "safety_action_required": label_annotations.get(index, {}).get("safety_action_required", False),
                        }
                        for index, msg in enumerate(candidates)
                        if index in selected_ids
                    ],
                }
            )
        if debug_callback:
            debug_callback(
                "parse_llm_result",
                {
                    "debug_phase": "parse_llm_result",
                    f"{stage}_reply_chars": len(reply or ""),
                    f"{stage}_selected_id_count": len(selected_ids),
                    f"{stage}_screened_count": len(selected),
                    f"{stage}_labeled_item_count": len(label_annotations),
                    f"{stage}_manual_review_count": len([item for item in label_annotations.values() if item.get("manual_review_required")]),
                    f"{stage}_safety_action_count": len([item for item in label_annotations.values() if item.get("safety_action_required")]),
                },
            )
        return selected, {
            f"{stage}_screened_count": len(selected),
            f"{stage}_llm_screened": True,
            f"{stage}_screen_reason": sanitize_snippet(str(parsed.get("reason") or ""), max_len=80),
            f"{stage}_labeled_item_count": len(label_annotations),
            f"{stage}_max_ai_risk_level": max([int(item.get("risk_level") or 0) for item in label_annotations.values()] or [0]),
            f"{stage}_manual_review_count": len([item for item in label_annotations.values() if item.get("manual_review_required")]),
            f"{stage}_safety_action_count": len([item for item in label_annotations.values() if item.get("safety_action_required")]),
            f"{stage}_context_enabled": use_context,
            f"{stage}_context_window": context_window if use_context else 0,
            f"{stage}_context_items": context_items,
        }
    except Exception as error:
        error_text = sanitize_snippet(str(error) or error.__class__.__name__, max_len=300)
        if artifact_callback:
            artifact_callback({"error": error_text, "fallback_selected_all": True})
        if debug_callback:
            debug_callback(
                "llm_screen_fallback",
                {
                    "debug_phase": "llm_screen_fallback",
                    f"{stage}_screened_count": len(candidates),
                    f"{stage}_llm_error": error_text,
                },
            )
        return candidates, {
            f"{stage}_screened_count": len(candidates),
            f"{stage}_llm_screened": False,
            f"{stage}_screen_reason": "llm_unavailable_fallback",
            f"{stage}_llm_error": error_text,
            f"{stage}_context_enabled": use_context,
            f"{stage}_context_window": context_window if use_context else 0,
            f"{stage}_context_items": context_items,
        }
