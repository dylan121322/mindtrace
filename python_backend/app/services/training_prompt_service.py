import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from app.config import BASE_DIR


TRAINING_PROMPT_CONFIG_PATH = BASE_DIR / "training_prompt_config.json"
TRAINING_PROMPT_CONFIG_VERSION = 1


DEFAULT_TRAINING_SYSTEM_PROMPT = """你是一个心理风险辅助筛查规则优化助手。
你只能提出评分规则、关键词、向量查询、阈值和训练 Prompt 的优化建议，不能给出医学诊断，不能建议药物。
所有输出必须保持“辅助筛查、建议专业评估、人工复核”的边界。
请根据人工审核样本找出模型误判原因，输出 JSON，不要输出 Markdown。
JSON 格式：
{
  "summary": "一句话概括",
  "suggestions": [
    {
      "type": "add_keywords|remove_keywords|add_vector_query|adjust_threshold|set_max_points|update_description|append_reason|append_training_prompt_instruction|update_training_prompt|append_auto_review_prompt_instruction|update_auto_review_prompt",
      "dimension_key": "可选，评分维度 key",
      "label_key": "可选，多标签 key",
      "keywords": ["可选"],
      "query": "可选",
      "threshold": "medium|high|crisis，可选",
      "value": 35,
      "prompt_key": "training_system_prompt|auto_review_system_prompt，可选",
      "prompt_text": "可选，替换后的完整 Prompt",
      "instruction": "可选，要追加到 Prompt 的指令",
      "description": "可选，新的规则说明或判断标准",
      "reason": "为什么这么改；若没有其他可应用字段，也会作为优化理由写入对应规则说明"
    }
  ],
  "cautions": ["需要人工复核的注意点"]
}
Prompt 修改建议只能作为待审核草案输出，不能假设会自动应用。"""


DEFAULT_AUTO_REVIEW_SYSTEM_PROMPT = """你是一个心理风险辅助筛查结果复盘助手。
你的任务是审核一次聊天文本辅助筛查结果是否可能漏判或误判，并指出评分规则可优化点。
边界要求：
1. 不输出医学诊断，不使用“确诊”“重度抑郁症”等结论。
2. 不提供药物建议。
3. 自伤/轻生红线信号只能提示风险和人工复核，不能被正面词或玩笑完全抵消。
4. 你只能基于提供的脱敏摘要、证据片段、标签和评分明细判断，不要臆造聊天内容。
5. 输出必须是 JSON，不要输出 Markdown。

JSON 格式：
{
  "accurate": true,
  "suggested_risk_level": 0,
  "suggested_score": 0,
  "missed_labels": ["可选"],
  "false_positive_labels": ["可选"],
  "suggested_keywords": ["可选"],
  "suggested_negative_keywords": ["可选"],
  "summary": "一句话审核结论",
  "optimization_notes": "可优化点，需人工确认"
}
"""


DEFAULT_TRAINING_PROMPT_CONFIG: Dict[str, Any] = {
    "version": TRAINING_PROMPT_CONFIG_VERSION,
    "allow_ai_prompt_suggestions": True,
    "training_system_prompt": DEFAULT_TRAINING_SYSTEM_PROMPT,
    "auto_review_system_prompt": DEFAULT_AUTO_REVIEW_SYSTEM_PROMPT,
    "manual_notes": "AI 可提出 Prompt 修改建议，但必须人工审核后才会应用。",
}


PROMPT_FIELDS = {"training_system_prompt", "auto_review_system_prompt"}


def load_training_prompt_config() -> Dict[str, Any]:
    if TRAINING_PROMPT_CONFIG_PATH.exists():
        try:
            raw = json.loads(TRAINING_PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
    else:
        raw = {}
    config = normalize_training_prompt_config(raw)
    if config != raw:
        save_training_prompt_config(config)
    return config


def normalize_training_prompt_config(value: Dict[str, Any] | None) -> Dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    config = deepcopy(DEFAULT_TRAINING_PROMPT_CONFIG)
    config["allow_ai_prompt_suggestions"] = bool(raw.get("allow_ai_prompt_suggestions", config["allow_ai_prompt_suggestions"]))
    for key in PROMPT_FIELDS:
        text = str(raw.get(key) or "").strip()
        if text:
            config[key] = text[:12000]
    notes = str(raw.get("manual_notes") or "").strip()
    if notes:
        config["manual_notes"] = notes[:2000]
    config["version"] = TRAINING_PROMPT_CONFIG_VERSION
    return config


def save_training_prompt_config(value: Dict[str, Any]) -> Dict[str, Any]:
    config = normalize_training_prompt_config(value)
    TRAINING_PROMPT_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return config


def reset_training_prompt_config() -> Dict[str, Any]:
    return save_training_prompt_config(DEFAULT_TRAINING_PROMPT_CONFIG)


def compact_training_prompt_config(config: Dict[str, Any]) -> Dict[str, Any]:
    config = normalize_training_prompt_config(config)
    return {
        "allow_ai_prompt_suggestions": config.get("allow_ai_prompt_suggestions", True),
        "training_system_prompt_preview": str(config.get("training_system_prompt", ""))[:1200],
        "auto_review_system_prompt_preview": str(config.get("auto_review_system_prompt", ""))[:800],
        "manual_notes": config.get("manual_notes", ""),
    }


def apply_prompt_suggestions(config: Dict[str, Any], suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
    next_config = normalize_training_prompt_config(deepcopy(config))
    if not next_config.get("allow_ai_prompt_suggestions", True):
        return next_config
    for suggestion in suggestions:
        stype = str(suggestion.get("type") or "").strip()
        prompt_key = _prompt_key_for_suggestion(suggestion, stype)
        if not prompt_key:
            continue
        if stype in {"update_training_prompt", "set_training_prompt", "update_auto_review_prompt", "set_auto_review_prompt"}:
            prompt_text = str(suggestion.get("prompt_text") or suggestion.get("description") or "").strip()
            if prompt_text:
                next_config[prompt_key] = _ensure_prompt_guardrails(prompt_text[:12000])
        elif stype in {"append_training_prompt_instruction", "append_auto_review_prompt_instruction", "append_prompt_instruction"}:
            instruction = str(suggestion.get("instruction") or suggestion.get("prompt_text") or suggestion.get("description") or suggestion.get("reason") or "").strip()
            if instruction:
                next_config[prompt_key] = _append_prompt_instruction(next_config[prompt_key], instruction)
    return normalize_training_prompt_config(next_config)


def diff_training_prompt_config(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    before = normalize_training_prompt_config(before)
    after = normalize_training_prompt_config(after)
    changes: Dict[str, Any] = {}
    for key in ("allow_ai_prompt_suggestions", "training_system_prompt", "auto_review_system_prompt", "manual_notes"):
        if before.get(key) != after.get(key):
            changes[key] = {"before": before.get(key), "after": after.get(key)}
    return {"prompts": {"prompt_config": changes} if changes else {}, "changed": bool(changes)}


def _prompt_key_for_suggestion(suggestion: Dict[str, Any], stype: str) -> str:
    raw = str(suggestion.get("prompt_key") or suggestion.get("prompt") or "").strip()
    if raw in PROMPT_FIELDS:
        return raw
    if "auto_review" in stype:
        return "auto_review_system_prompt"
    if "training" in stype or "prompt" in stype:
        return "training_system_prompt"
    return ""


def _append_prompt_instruction(current: str, instruction: str) -> str:
    instruction = " ".join(str(instruction or "").split()).strip()
    if not instruction:
        return current
    line = f"补充规则：{instruction}"
    if line in current or instruction in current:
        return current
    return _ensure_prompt_guardrails(f"{current.rstrip()}\n{line}"[:12000])


def _ensure_prompt_guardrails(prompt: str) -> str:
    guardrails = [
        "不得输出医学诊断结论。",
        "不得提供药物建议。",
        "所有规则修改必须作为待人工审核草案输出，不能自动应用。",
    ]
    text = str(prompt or "").strip()
    for item in guardrails:
        if item not in text:
            text = f"{text}\n{item}" if text else item
    return text[:12000]
