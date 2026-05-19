from typing import Any, Dict, List, Optional

from app.models import PsychEvidence, PsychFact, PsychFeature, PsychScore


DISCLAIMER = "本报告仅基于聊天文本进行心理风险辅助筛查，不构成医学诊断。若存在持续痛苦、自伤或轻生想法，请尽快联系专业人员或当地紧急救助服务。"


def _feature_value(features: List[PsychFeature], name: str) -> float:
    for feature in features:
        if feature.name == name:
            return feature.value
    return 0.0


def _fmt_float(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def generate_report(
    features: List[PsychFeature],
    evidences: List[PsychEvidence],
    facts: List[PsychFact],
    score: PsychScore,
    process_steps: Optional[List[Any]] = None,
    fact_vector_hits: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    adjustments = score.scoring_adjustments or {}
    lines = [
        "# 心理风险辅助筛查报告",
        "",
        DISCLAIMER,
        "",
        "## 风险概览",
        f"- 抑郁相关信号分数：{score.depression_signal_score}",
        f"- 综合风险提示：{score.overall_risk}",
        f"- 多标签风险等级：{score.risk_level}（{score.risk_level_label or '未标注'}）",
        f"- 自伤/轻生红线提示：{score.self_harm_risk}",
        f"- 置信度：{score.confidence}",
        f"- 摘要：{score.summary}",
        "",
        "## 评分修正",
        f"- 维度原始合计：{adjustments.get('raw_dimension_score', score.depression_signal_score)}",
        f"- 趋势加分：{adjustments.get('worsening_bonus', 0)}",
        f"- 缓解修正：{adjustments.get('relief_delta', 0)}",
        f"- 保护性修正：{adjustments.get('protective_delta', 0)}",
        f"- 正面情绪修正量：{adjustments.get('positive_emotion_delta', 0)} "
        f"（消息数 {int(float(adjustments.get('positive_emotion_message_count', 0) or 0))}，"
        f"命中词 {int(float(adjustments.get('positive_emotion_hit_count', 0) or 0))}）",
        f"- 红线保护规则：{'已阻止正面词自动降分' if adjustments.get('redline_blocks_protective_reduction') else '未触发'}",
        "",
        "## 评分维度",
    ]
    if score.dimension_scores:
        for item in score.dimension_scores:
            label = item.get("label") or item.get("key") or ""
            strength = item.get("evidence_strength") if isinstance(item.get("evidence_strength"), dict) else {}
            time_info = item.get("time_adjustment") if isinstance(item.get("time_adjustment"), dict) else {}
            lines.append(
                "- "
                f"{label}: {int(item.get('score') or 0)}/{int(item.get('max_points') or 0)} "
                f"(基础 {int(item.get('base_score') or 0)}，证据 {strength.get('label', '-')}"
                f"×{_fmt_float(strength.get('coefficient'), 1)}，时间 {time_info.get('label', '-')}"
                f"×{_fmt_float(time_info.get('coefficient'), 1)}) - {item.get('adjusted_description') or item.get('description') or ''}"
            )
    else:
        lines.append("- 暂无维度评分。")

    lines.extend(["", "## 主要信号"])
    for signal in score.main_signals:
        lines.append(f"- {signal}")

    lines.extend(["", "## 多标签分类"])
    if score.symptom_labels:
        for item in score.symptom_labels[:16]:
            suffix = []
            if item.get("protective"):
                suffix.append("保护性")
            if item.get("modifier"):
                suffix.append("语境修正")
            suffix_text = f" | {'/'.join(suffix)}" if suffix else ""
            lines.append(
                f"- {item.get('label') or item.get('key')} | 权重：{item.get('weight_label') or item.get('weight')} "
                f"| 风险等级：{item.get('risk_level')} | 消息数：{int(float(item.get('message_count') or 0))}{suffix_text}"
            )
    else:
        lines.append("- 暂未命中可配置症状标签。")

    lines.extend(["", "## 保护性修正因子"])
    protective_factors = adjustments.get("protective_factors", [])
    if protective_factors:
        for item in protective_factors:
            lines.append(
                f"- {item.get('label') or item.get('key')}：{item.get('delta', 0)} 分"
                f" | 消息数：{int(float(item.get('message_count') or 0))}"
                f" | {item.get('description') or ''}"
            )
    else:
        lines.append("- 暂未命中明显保护性修正因子。")

    lines.extend(["", "## 心理事实摘要"])
    if facts:
        for fact in facts[:10]:
            lines.append(f"- {fact.fact_type} | {fact.severity} | 置信度 {fact.confidence:.2f} | {fact.fact}")
    else:
        lines.append("- 暂未从事实库召回明确心理事实。")

    if fact_vector_hits:
        lines.extend(["", "## 事实向量检索重点"])
        for hit in fact_vector_hits[:8]:
            content = str(hit.get("content") or "")
            score_value = hit.get("score", "")
            lines.append(f"- 相似度 {score_value} | {content}")

    lines.extend(
        [
            "",
            "## 统计特征",
            f"- 有效消息数：{int(_feature_value(features, 'message_count'))}",
            f"- 活跃天数：{int(_feature_value(features, 'unique_active_days'))}",
            f"- 夜间消息比例：{_feature_value(features, 'late_night_message_ratio'):.2f}",
            f"- 证据条数：{int(_feature_value(features, 'evidence_count'))}",
            "",
            "## 文本证据摘录",
        ]
    )
    for evidence in evidences[:12]:
        lines.append(f"- {evidence.datetime} | {evidence.evidence_type} | {evidence.severity} | {evidence.content}")
    if not evidences:
        lines.append("- 暂无明显文本证据。")

    lines.extend(
        [
            "",
            "## 建议",
            "- 将结果视为心理风险辅助筛查线索，而不是医学结论；请结合近期睡眠、饮食、工作学习功能、人际支持和现实压力综合判断。",
            "- 若痛苦持续、功能明显受影响，建议联系心理咨询师、精神心理科或当地心理援助热线进行进一步专业评估。",
            "- 若出现自伤或轻生想法、计划或准备行为，请立即联系可信任的人陪伴，并寻求当地紧急救助服务。",
        ]
    )
    report_md = "\n".join(lines)
    report_json = {
        "disclaimer": DISCLAIMER,
        "score": score.dict(),
        "feature_summary": {feature.name: feature.value for feature in features},
        "evidence_count": len(evidences),
        "fact_count": len(facts),
        "facts": [fact.dict() for fact in facts],
        "fact_vector_hits": fact_vector_hits or [],
        "pipeline": [
            {
                "key": getattr(step, "key", ""),
                "name": getattr(step, "name", ""),
                "duration_ms": getattr(step, "duration_ms", 0),
                "metrics": getattr(step, "metrics", {}),
            }
            for step in (process_steps or [])
        ],
    }
    return {"report_md": report_md, "report_json": report_json}
