import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.models import PsychAnalyzeRequest, TrainingSampleCreateRequest, TrainingSampleUpdateRequest
from app.psych.scoring_config import load_scoring_config, normalize_scoring_config, save_scoring_config
from app.services.llm_service import LLMServiceError, complete_chat
from app.services.training_prompt_service import (
    apply_prompt_suggestions,
    compact_training_prompt_config,
    diff_training_prompt_config,
    load_training_prompt_config,
    save_training_prompt_config,
)
from app.stores import training_store
from app.utils.privacy import sanitize_snippet


TRAINING_SYSTEM_PROMPT = """你是一个心理风险辅助筛查规则优化助手。
你只能提出评分规则、关键词、向量查询和阈值的优化建议，不能给出医学诊断，不能建议药物。
所有输出必须保持“辅助筛查、建议专业评估、人工复核”的边界。
请根据人工审核样本找出模型误判原因，输出 JSON，不要输出 Markdown。
JSON 格式：
{
  "summary": "一句话概括",
  "suggestions": [
    {
      "type": "add_keywords|remove_keywords|add_vector_query|adjust_threshold|set_max_points|update_description|append_reason",
      "dimension_key": "可选，评分维度 key",
      "label_key": "可选，多标签 key",
      "keywords": ["可选"],
      "query": "可选",
      "threshold": "medium|high|crisis，可选",
      "value": 35,
      "description": "可选，新的规则说明或判断标准",
      "reason": "为什么这么改；若没有其他可应用字段，也会作为优化理由写入对应规则说明"
    }
  ],
  "cautions": ["需要人工复核的注意点"]
}
"""

AUTO_REVIEW_SYSTEM_PROMPT = """你是一个心理风险辅助筛查结果复盘助手。
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


def create_sample(request: TrainingSampleCreateRequest) -> Dict[str, Any]:
    analysis = _compact_analysis(request.analysis_result)
    review = _review_dict(request.human_review)
    return training_store.create_sample(
        name=request.name.strip() or _default_sample_name(analysis),
        target_key=request.target_key.strip(),
        target_type=request.target_type.strip(),
        analysis_task_id=request.analysis_task_id.strip() or str(analysis.get("task_id") or ""),
        analysis_json=analysis,
        human_review=review,
        notes=request.notes.strip(),
    )


def update_sample(sample_id: str, request: TrainingSampleUpdateRequest) -> Optional[Dict[str, Any]]:
    review = _review_dict(request.human_review) if request.human_review is not None else None
    return training_store.update_sample(
        sample_id,
        name=request.name,
        human_review=review,
        notes=request.notes,
    )


def list_samples(
    limit: int = 100,
    reviewed_only: bool = False,
    target_type: str = "",
    target_key: str = "",
) -> List[Dict[str, Any]]:
    return training_store.list_samples(
        limit=limit,
        reviewed_only=reviewed_only,
        target_type=target_type,
        target_key=target_key,
    )


def delete_sample(sample_id: str) -> bool:
    return training_store.delete_sample(sample_id)


def generate_proposal(
    *,
    sample_ids: Optional[List[str]] = None,
    max_samples: Any = 30,
    use_llm: bool = True,
    target_type: str = "all",
    target_key: str = "",
) -> Dict[str, Any]:
    scope = _training_scope(target_type, target_key)
    samples = _select_samples(
        sample_ids,
        max_samples=max_samples,
        target_type=scope["target_type"],
        target_key=scope["target_key"],
    )
    if not samples:
        return training_store.create_proposal(
            sample_count=0,
            model_provider="",
            model_name="",
            summary=f"{scope['label']}没有可用的人审样本，无法生成优化草案。",
            suggestions=[],
            proposed_config=load_scoring_config(),
            diagnostics={
                "reason": "no_reviewed_samples",
                "training_scope": scope,
                "requires_human_review": True,
                "repeat_training_guidance": _repeat_training_guidance(),
            },
        )

    current_config = load_scoring_config()
    llm_payload: Dict[str, Any] = {}
    llm_error = ""
    if use_llm:
        try:
            llm_payload = _ask_llm_for_suggestions(samples, current_config)
        except Exception as error:
            llm_error = _safe_error_text(error)

    suggestions = _normalize_suggestions(llm_payload.get("suggestions", []))
    heuristic_suggestions = _heuristic_suggestions(samples, current_config)
    suggestions = _merge_suggestions([*suggestions, *heuristic_suggestions])
    proposed_config = _apply_suggestions(current_config, suggestions)
    settings = get_settings()
    summary = str(llm_payload.get("summary") or "").strip()
    if not summary:
        summary = _fallback_summary(samples, suggestions, llm_error)
    diagnostics = {
        "llm_used": bool(use_llm and not llm_error),
        "llm_error": llm_error,
        "sample_ids": [sample["sample_id"] for sample in samples],
        "sample_count": len(samples),
        "suggestion_count": len(suggestions),
        "cautions": llm_payload.get("cautions", []) if isinstance(llm_payload.get("cautions"), list) else [],
        "human_review_only": True,
        "requires_human_review": True,
        "training_scope": scope,
        "repeat_training_guidance": _repeat_training_guidance(),
    }
    return training_store.create_proposal(
        sample_count=len(samples),
        model_provider=settings.llm_provider,
        model_name=settings.llm_model,
        summary=summary,
        suggestions=suggestions,
        proposed_config=proposed_config,
        diagnostics=diagnostics,
    )


def list_proposals(
    limit: int = 50,
    target_type: str = "",
    target_key: str = "",
) -> List[Dict[str, Any]]:
    proposals = training_store.list_proposals(limit=limit)
    scope = _training_scope(target_type, target_key)
    if scope["target_type"] == "all":
        return proposals
    out: List[Dict[str, Any]] = []
    for proposal in proposals:
        diagnostics = proposal.get("diagnostics") if isinstance(proposal.get("diagnostics"), dict) else {}
        proposal_scope = diagnostics.get("training_scope") if isinstance(diagnostics.get("training_scope"), dict) else {}
        if proposal_scope.get("target_type") != scope["target_type"]:
            continue
        if scope["target_type"] != "self" and scope["target_key"] and proposal_scope.get("target_key") != scope["target_key"]:
            continue
        out.append(proposal)
    return out


def get_proposal(proposal_id: str) -> Optional[Dict[str, Any]]:
    return training_store.get_proposal(proposal_id)


def preview_proposal(
    proposal_id: str,
    selected_suggestion_indexes: Optional[List[int]] = None,
) -> Optional[Dict[str, Any]]:
    proposal = training_store.get_proposal(proposal_id)
    if not proposal:
        return None
    before = load_scoring_config()
    prompt_before = load_training_prompt_config()
    suggestions = proposal.get("suggestions") if isinstance(proposal.get("suggestions"), list) else []
    selected_suggestions = _select_suggestions(suggestions, selected_suggestion_indexes)
    proposed_config = proposal.get("proposed_config")
    if selected_suggestion_indexes is not None or selected_suggestions:
        proposed_config = _apply_suggestions(before, selected_suggestions)
    if not isinstance(proposed_config, dict):
        return None
    after = normalize_scoring_config(proposed_config)
    prompt_after = apply_prompt_suggestions(prompt_before, selected_suggestions)
    diff = _merge_preview_diffs(
        _diff_scoring_config(before, after),
        diff_training_prompt_config(prompt_before, prompt_after),
    )
    return {
        "proposal_id": proposal_id,
        "selected_suggestion_indexes": _normalized_indexes(suggestions, selected_suggestion_indexes),
        "selected_suggestions": selected_suggestions,
        "preview_config": after,
        "prompt_preview_config": prompt_after,
        "preview_diff": diff,
        "changed": bool(diff.get("changed")),
    }


def apply_proposal(
    proposal_id: str,
    selected_suggestion_indexes: Optional[List[int]] = None,
    human_review_confirmed: bool = False,
) -> Optional[Dict[str, Any]]:
    if not human_review_confirmed:
        raise ValueError("training proposal must be manually reviewed before applying")
    preview = preview_proposal(proposal_id, selected_suggestion_indexes)
    if not preview:
        return None
    saved = save_scoring_config(preview["preview_config"])
    saved_prompts = save_training_prompt_config(preview["prompt_preview_config"])
    applied = training_store.mark_proposal_applied(proposal_id)
    if applied:
        applied["applied_config"] = saved
        applied["applied_prompt_config"] = saved_prompts
        applied["applied_diff"] = preview["preview_diff"]
        applied["selected_suggestion_indexes"] = preview["selected_suggestion_indexes"]
        applied["selected_suggestions"] = preview["selected_suggestions"]
    return applied


def auto_review_analysis(response: Any, request: Any) -> Dict[str, Any]:
    return _auto_review_analysis(response, request)


def _auto_review_analysis(
    response: Any,
    request: Any,
    *,
    force_enabled: bool = False,
    create_proposal_after: Optional[bool] = None,
) -> Dict[str, Any]:
    """Create an AI-reviewed training sample and optional proposal.

    The function is deliberately non-authoritative: it stores a review draft and
    an optimization proposal, but never applies scoring changes automatically.
    """
    settings = get_settings()
    if not force_enabled and not getattr(settings, "training_auto_review_enabled", False):
        return {"enabled": False, "reason": "disabled"}

    try:
        response_dict = _model_to_dict(response)
        compact = _compact_analysis(response_dict)
        current_config = load_scoring_config()
        llm_payload: Dict[str, Any] = {}
        llm_error = ""
        use_llm = bool(getattr(settings, "training_auto_review_use_llm", True))
        if use_llm:
            try:
                llm_payload = _ask_llm_for_auto_review(compact, current_config)
            except Exception as error:
                llm_error = _safe_error_text(error)

        review = _auto_review_dict(compact, llm_payload, llm_error)
        task_id = str(compact.get("task_id") or response_dict.get("task_id") or "")
        sample = training_store.create_sample(
            name=_auto_sample_name(compact),
            target_key=str(getattr(request, "target_key", "") or compact.get("target_key") or ""),
            target_type=str(getattr(request, "target_type", "") or compact.get("target_type") or ""),
            analysis_task_id=task_id,
            analysis_json=compact,
            human_review=review,
            notes=(
                "AI 自动审核样本，仅作为评分规则优化线索；"
                "应用任何草案前仍需人工复核。"
            ),
        )

        proposal: Dict[str, Any] = {}
        should_create_proposal = (
            bool(getattr(settings, "training_auto_proposal_enabled", True))
            if create_proposal_after is None
            else bool(create_proposal_after)
        )
        if should_create_proposal:
            proposal = generate_proposal(
                sample_ids=[sample.get("sample_id", "")],
                max_samples=getattr(settings, "training_auto_max_samples", 1),
                use_llm=use_llm,
                target_type=str(sample.get("target_type") or "all"),
                target_key=str(sample.get("target_key") or ""),
            )

        return {
            "enabled": True,
            "status": "completed",
            "sample_id": sample.get("sample_id", ""),
            "proposal_id": proposal.get("proposal_id", ""),
            "proposal_status": proposal.get("status", ""),
            "suggestion_count": len(proposal.get("suggestions", []) or []),
            "llm_used": bool(use_llm and not llm_error),
            "llm_error": llm_error,
            "review_summary": review.get("notes", "")[:300],
            "requires_human_confirmation": True,
        }
    except Exception as error:
        return {
            "enabled": True,
            "status": "failed",
            "error": _safe_error_text(error),
            "requires_human_confirmation": True,
        }


def repeat_after_apply(
    proposal_id: str,
    *,
    max_samples: Any = "max",
    use_llm: bool = True,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Re-run reference samples after a reviewed proposal was applied.

    The loop is intentionally human-in-the-loop: it creates fresh auto-reviewed
    samples and one next-round proposal, but it never applies that proposal.
    """
    proposal = training_store.get_proposal(proposal_id)
    if not proposal:
        raise ValueError("training proposal not found")
    diagnostics = proposal.get("diagnostics") if isinstance(proposal.get("diagnostics"), dict) else {}
    sample_ids = diagnostics.get("sample_ids") if isinstance(diagnostics.get("sample_ids"), list) else []
    scope_raw = diagnostics.get("training_scope") if isinstance(diagnostics.get("training_scope"), dict) else {}
    scope = _training_scope(str(scope_raw.get("target_type") or "all"), str(scope_raw.get("target_key") or ""))
    samples = _select_samples(
        [str(item) for item in sample_ids],
        max_samples=max_samples,
        target_type=scope["target_type"],
        target_key=scope["target_key"],
    )
    if not samples:
        return {
            "source_proposal_id": proposal_id,
            "status": "completed",
            "reanalyzed": 0,
            "auto_reviewed": 0,
            "skipped": 0,
            "proposal": generate_proposal(
                sample_ids=[],
                max_samples=1,
                use_llm=False,
                target_type=scope["target_type"],
                target_key=scope["target_key"],
            ),
            "reason": "no_reference_samples",
        }

    def report(stage: str, message: str, progress: int, **extra: Any) -> None:
        if progress_callback:
            progress_callback(stage, message, progress, extra)

    from app.services import psych_service
    from app.stores import psych_task_store

    new_sample_ids: List[str] = []
    skipped: List[Dict[str, str]] = []
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        request = _request_from_training_sample(sample)
        if not request:
            skipped.append({"sample_id": str(sample.get("sample_id") or ""), "reason": "missing_replay_target"})
            continue
        psych_task = psych_task_store.create_task(
            {
                "source": "training_retest",
                "training_sample_id": str(sample.get("sample_id") or ""),
                "target_key": request.target_key or "",
                "target_type": request.target_type,
                "message": f"正在复测训练样本 {index}/{total}",
            }
        )
        psych_task_id = str(psych_task.get("task_id") or "")

        def on_psych_progress(steps: List[Any], current_key: str, state: str) -> None:
            completed = len([step for step in steps if getattr(step, "status", "") == "completed"])
            running = next((step for step in steps if getattr(step, "status", "") == "running"), None)
            psych_progress = min(99, int((completed / 9) * 100))
            psych_message = getattr(running, "name", "") or current_key or "复测分析中"
            psych_task_store.update_task(
                psych_task_id,
                status="running",
                stage=current_key,
                message=psych_message,
                progress=psych_progress,
                process_steps=[_model_to_dict(step) for step in steps],
            )
            mapped_progress = 10 + int(((index - 1 + psych_progress / 100) / max(1, total)) * 45)
            report(
                "reanalyze",
                f"正在复测样本 {index}/{total}：{psych_message}",
                min(55, mapped_progress),
                sample_id=sample.get("sample_id", ""),
                target_type=request.target_type,
                psych_task_id=psych_task_id,
                current_psych_task_id=psych_task_id,
                psych_progress=psych_progress,
                current_step=current_key,
            )

        report(
            "reanalyze",
            f"正在按新评分规则复测样本 {index}/{total}",
            10 + int((index - 1) / max(1, total) * 45),
            sample_id=sample.get("sample_id", ""),
            target_type=request.target_type,
            psych_task_id=psych_task_id,
            current_psych_task_id=psych_task_id,
            psych_progress=0,
        )
        try:
            response = psych_service.analyze_psych(
                request,
                task_id=psych_task_id,
                progress_callback=on_psych_progress,
            )
        except Exception as error:
            psych_task_store.update_task(
                psych_task_id,
                status="failed",
                stage="failed",
                message="training retest analysis failed",
                progress=100,
                error=_safe_error_text(error),
            )
            raise
        psych_task_store.update_task(
            psych_task_id,
            status="completed",
            stage="completed",
            message="training retest analysis completed",
            progress=100,
            process_steps=[_model_to_dict(step) for step in response.process_steps],
            result=_model_to_dict(response),
            auto_review={
                "enabled": True,
                "status": "queued",
                "stage": "training_auto_review",
                "message": "复测分析已完成，正在进入自动审核。",
                "requires_human_confirmation": True,
            },
        )
        report(
            "auto_review",
            f"正在自动审核复测结果 {index}/{total}",
            55 + int((index - 1) / max(1, total) * 25),
            task_id=response.task_id,
            psych_task_id=psych_task_id,
            current_psych_task_id=psych_task_id,
        )
        psych_task_store.update_task(
            psych_task_id,
            auto_review={
                "enabled": True,
                "status": "running",
                "stage": "training_auto_review",
                "message": "正在自动审核复测结果，并生成下一轮规则优化线索。",
                "requires_human_confirmation": True,
            },
        )
        review_info = _auto_review_analysis(
            response,
            request,
            force_enabled=True,
            create_proposal_after=False,
        )
        psych_task_store.update_task(
            psych_task_id,
            auto_review={
                "enabled": True,
                "status": str(review_info.get("status") or "completed"),
                "stage": "training_auto_review",
                "message": "复测自动审核完成",
                "sample_id": review_info.get("sample_id", ""),
                "proposal_id": review_info.get("proposal_id", ""),
                "suggestion_count": int(review_info.get("suggestion_count") or 0),
                "llm_used": bool(review_info.get("llm_used")),
                "llm_error": review_info.get("llm_error", ""),
                "error": review_info.get("error", ""),
                "review_summary": review_info.get("review_summary", ""),
                "requires_human_confirmation": True,
            },
        )
        sample_id = str(review_info.get("sample_id") or "")
        if sample_id:
            new_sample_ids.append(sample_id)
        else:
            skipped.append({
                "sample_id": str(sample.get("sample_id") or ""),
                "reason": str(review_info.get("error") or review_info.get("reason") or "auto_review_failed"),
            })

    report("generate_next_proposal", "正在根据复测审核结果生成下一轮待审核草案", 88)
    next_proposal = generate_proposal(
        sample_ids=new_sample_ids,
        max_samples="max",
        use_llm=use_llm,
        target_type=scope["target_type"],
        target_key=scope["target_key"],
    )
    return {
        "source_proposal_id": proposal_id,
        "status": "completed",
        "training_scope": scope,
        "reanalyzed": total - len(skipped),
        "auto_reviewed": len(new_sample_ids),
        "skipped": len(skipped),
        "skipped_items": skipped[:20],
        "new_sample_ids": new_sample_ids,
        "proposal": next_proposal,
        "requires_human_review": True,
    }


def _model_to_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        data = value.model_dump()
    elif hasattr(value, "dict"):
        data = value.dict()
    elif isinstance(value, dict):
        data = value
    else:
        data = {}
    return data if isinstance(data, dict) else {}


def _ask_llm_for_auto_review(analysis: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    prompt_config = load_training_prompt_config()
    user_prompt = json.dumps(
        {
            "analysis_result": analysis,
            "current_scoring_config": _compact_config(config),
            "review_focus": [
                "判断模型风险等级和分数是否可能偏高或偏低。",
                "列出可能漏判的标签、误判的标签、应补充的关键词或应降低触发的词。",
                "注意事件型抱怨、幽默调侃、保护性信号对风险等级的修正。",
                "明确自伤/轻生意图、方法、时间、工具、告别等红线信号必须建议人工复核。",
                "只输出规则优化线索，不输出医学诊断结论。",
            ],
        },
        ensure_ascii=False,
    )
    text = complete_chat(
        [
            {"role": "system", "content": str(prompt_config.get("auto_review_system_prompt") or AUTO_REVIEW_SYSTEM_PROMPT)},
            {"role": "user", "content": user_prompt},
        ]
    )
    return _parse_json_object(text)


def _auto_review_dict(analysis: Dict[str, Any], payload: Dict[str, Any], llm_error: str) -> Dict[str, Any]:
    score = analysis.get("score", {}) if isinstance(analysis.get("score"), dict) else {}
    model_level = max(0, min(5, int(score.get("risk_level") or 0)))
    model_score = _optional_int(score.get("depression_signal_score")) or 0
    suggested_level = _optional_int(payload.get("suggested_risk_level"))
    suggested_score = _optional_int(payload.get("suggested_score"))
    missed = _string_list(payload.get("missed_labels"))
    false_positive = _string_list(payload.get("false_positive_labels"))
    accurate_value = payload.get("accurate")
    if isinstance(accurate_value, bool):
        accurate = accurate_value
    else:
        accurate = (
            (suggested_level is None or suggested_level == model_level)
            and not missed
            and not false_positive
            and not llm_error
        )
    summary = str(payload.get("summary") or "").strip()
    notes = str(payload.get("optimization_notes") or "").strip()
    if not notes:
        if llm_error:
            notes = f"AI 自动审核调用失败，暂以模型原始结果作为训练样本；错误：{llm_error}"
        else:
            notes = summary or "AI 自动审核未发现明确可优化点，仍需人工确认。"
    else:
        notes = f"{summary} {notes}".strip()
    return {
        "accurate": bool(accurate),
        "human_risk_level": max(0, min(5, int(suggested_level if suggested_level is not None else model_level))),
        "human_score": max(0, min(100, int(suggested_score if suggested_score is not None else model_score))),
        "missed_labels": missed,
        "false_positive_labels": false_positive,
        "suggested_keywords": _string_list(payload.get("suggested_keywords")),
        "suggested_negative_keywords": _string_list(payload.get("suggested_negative_keywords")),
        "notes": ("AI 自动审核（需人工确认）： " + notes)[:1200],
        "review_source": "ai_auto_review",
        "llm_error": llm_error,
    }


def _auto_sample_name(analysis: Dict[str, Any]) -> str:
    score = analysis.get("score", {}) if isinstance(analysis.get("score"), dict) else {}
    task_id = str(analysis.get("task_id") or "")[:8]
    return f"AI 自动审核 L{score.get('risk_level', 0)} {task_id}".strip()


def _compact_analysis(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        result = {}
    score = result.get("score") if isinstance(result.get("score"), dict) else {}
    evidences = result.get("evidences") if isinstance(result.get("evidences"), list) else []
    facts = result.get("facts") if isinstance(result.get("facts"), list) else []
    report_json = result.get("report_json") if isinstance(result.get("report_json"), dict) else {}
    return {
        "task_id": result.get("task_id") or "",
        "status": result.get("status") or "",
        "score": {
            "depression_signal_score": score.get("depression_signal_score", 0),
            "overall_risk": score.get("overall_risk", ""),
            "risk_level": score.get("risk_level", 0),
            "risk_level_label": score.get("risk_level_label", ""),
            "confidence": score.get("confidence", 0),
            "main_signals": score.get("main_signals", []),
            "symptom_labels": score.get("symptom_labels", [])[:20],
            "dimension_scores": score.get("dimension_scores", [])[:12],
            "scoring_adjustments": score.get("scoring_adjustments", {}),
        },
        "evidences": [
            {
                "seq": item.get("seq"),
                "datetime": item.get("datetime"),
                "evidence_type": item.get("evidence_type"),
                "severity": item.get("severity"),
                "reason": item.get("reason"),
                "labels": item.get("labels", []),
                "risk_level": item.get("risk_level", 0),
                "content": sanitize_snippet(str(item.get("content") or ""), max_len=160),
            }
            for item in evidences[:80]
            if isinstance(item, dict)
        ],
        "facts": [
            {
                "fact_type": item.get("fact_type"),
                "fact": sanitize_snippet(str(item.get("fact") or ""), max_len=180),
                "severity": item.get("severity"),
                "confidence": item.get("confidence"),
            }
            for item in facts[:30]
            if isinstance(item, dict)
        ],
        "pipeline_metrics": report_json.get("process_steps", [])[:12],
        "analysis_request": report_json.get("analysis_request", {}),
        "target_key": report_json.get("target_key", ""),
        "target_type": report_json.get("target_type", ""),
        "time_from": report_json.get("time_from"),
        "time_to": report_json.get("time_to"),
    }


def _review_dict(review) -> Dict[str, Any]:
    if hasattr(review, "model_dump"):
        data = review.model_dump()
    elif hasattr(review, "dict"):
        data = review.dict()
    elif isinstance(review, dict):
        data = review
    else:
        data = {}
    return {
        "accurate": bool(data.get("accurate", False)),
        "human_risk_level": max(0, min(5, int(data.get("human_risk_level") or 0))),
        "human_score": _optional_int(data.get("human_score")),
        "missed_labels": _string_list(data.get("missed_labels")),
        "false_positive_labels": _string_list(data.get("false_positive_labels")),
        "suggested_keywords": _string_list(data.get("suggested_keywords")),
        "suggested_negative_keywords": _string_list(data.get("suggested_negative_keywords")),
        "notes": str(data.get("notes") or "").strip()[:1200],
    }


def _resolve_max_samples(max_samples: Any) -> int:
    if isinstance(max_samples, str) and max_samples.strip().lower() == "max":
        return 5000
    try:
        return max(1, min(int(max_samples or 30), 5000))
    except (TypeError, ValueError):
        return 30


def _select_samples(
    sample_ids: Optional[List[str]],
    max_samples: Any,
    target_type: str = "all",
    target_key: str = "",
) -> List[Dict[str, Any]]:
    max_samples = _resolve_max_samples(max_samples)
    if sample_ids:
        out = []
        for sample_id in sample_ids[:max_samples]:
            sample = training_store.get_sample(sample_id)
            if sample and _sample_in_scope(sample, target_type, target_key):
                out.append(sample)
        return out
    return training_store.list_samples(
        limit=max_samples,
        reviewed_only=True,
        target_type=target_type,
        target_key=target_key,
    )


def _training_scope(target_type: str = "all", target_key: str = "") -> Dict[str, str]:
    target_type = (target_type or "all").strip() or "all"
    target_key = (target_key or "").strip()
    if target_type not in {"all", "self", "contact"}:
        target_type = "all"
        target_key = ""
    if target_type in {"all", "self"}:
        target_key = ""
    labels = {
        "all": "全部人审样本",
        "self": "本人样本人审集",
        "contact": f"特定联系人样本人审集 {target_key[:12]}" if target_key else "特定联系人样本人审集",
    }
    return {
        "target_type": target_type,
        "target_key": target_key,
        "label": labels.get(target_type, "全部人审样本"),
    }


def _sample_in_scope(sample: Dict[str, Any], target_type: str, target_key: str) -> bool:
    if target_type in ("", "all"):
        return True
    if sample.get("target_type") != target_type:
        return False
    if target_type == "self":
        return True
    return not target_key or sample.get("target_key") == target_key


def _request_from_training_sample(sample: Dict[str, Any]) -> Optional[PsychAnalyzeRequest]:
    analysis = sample.get("analysis_json") if isinstance(sample.get("analysis_json"), dict) else {}
    saved_request = analysis.get("analysis_request") if isinstance(analysis.get("analysis_request"), dict) else {}
    target_type = str(saved_request.get("target_type") or sample.get("target_type") or analysis.get("target_type") or "").strip()
    target_key = str(saved_request.get("target_key") or sample.get("target_key") or analysis.get("target_key") or "").strip()
    if target_type not in {"self", "contact"}:
        return None
    if target_type == "contact" and not target_key:
        return None
    options = saved_request.get("options") if isinstance(saved_request.get("options"), dict) else {}
    replay_options = dict(options)
    replay_options["training_replay"] = True
    replay_options["llm_screening"] = options.get("llm_screening", True)
    replay_options["use_vector"] = options.get("use_vector", True)
    return PsychAnalyzeRequest(
        target_key=target_key or None,
        target_type=target_type,  # type: ignore[arg-type]
        time_from=saved_request.get("time_from", analysis.get("time_from")),
        time_to=saved_request.get("time_to", analysis.get("time_to")),
        only_mine=bool(saved_request.get("only_mine", True)),
        include_context=bool(saved_request.get("include_context", True)),
        options=replay_options,
    )


def _repeat_training_guidance() -> List[str]:
    return [
        "先按目标范围保存足够的人审样本，尤其补齐人工风险等级、人工分数和审核说明。",
        "重复训练时优先编辑旧样本的评语和评分，再重新生成草案，不要让草案直接覆盖规则。",
        "每次草案都需要人工审核差异和参考样本，勾选确认后才能应用。",
        "应用后重新跑同一联系人或本人样本，若仍不符合预期，继续补充样本并再次生成草案。",
    ]


def _ask_llm_for_suggestions(samples: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    prompt_config = load_training_prompt_config()
    compact_config = _compact_config(config)
    compact_samples = [_compact_sample_for_llm(sample) for sample in samples]
    user_prompt = json.dumps(
        {
            "current_scoring_config": compact_config,
            "current_prompt_config": compact_training_prompt_config(prompt_config),
            "reviewed_samples": compact_samples,
            "rules": [
                "优先优化关键词、排除词、向量查询和阈值，不要输出诊断结论。",
                "若人审样本说明当前规则优化 Prompt 有遗漏，可以提出 append_training_prompt_instruction 或 update_training_prompt，但仍必须作为待人工审核草案。",
                "风险等级 3 以上建议人工复核，风险等级 5 必须触发安全处理流程。",
                "若只是事件型抱怨或幽默调侃，不应直接上调为高风险。",
                "明确自伤/轻生意图、方法、时间、工具、告别行为不能被正面词抵消。",
                "本次输出只是待人工审核草案，不能假设会自动应用。",
            ],
        },
        ensure_ascii=False,
    )
    text = complete_chat(
        [
            {"role": "system", "content": str(prompt_config.get("training_system_prompt") or TRAINING_SYSTEM_PROMPT)},
            {"role": "user", "content": user_prompt},
        ]
    )
    return _parse_json_object(text)


def _compact_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "thresholds": config.get("thresholds", {}),
        "dimensions": [
            {
                "key": item.get("key"),
                "label": item.get("label"),
                "max_points": item.get("max_points"),
                "keywords": item.get("keywords", [])[:30],
                "exclude_keywords": item.get("exclude_keywords", [])[:15],
                "vector_queries": item.get("vector_queries", [])[:5],
            }
            for item in config.get("dimensions", [])
            if isinstance(item, dict)
        ],
        "labels": [
            {
                "key": item.get("key"),
                "label": item.get("label"),
                "risk_level": item.get("risk_level"),
                "keywords": item.get("keywords", [])[:20],
                "exclude_keywords": item.get("exclude_keywords", [])[:10],
            }
            for item in config.get("symptom_labels", {}).get("labels", [])
            if isinstance(item, dict)
        ],
    }


def _compact_sample_for_llm(sample: Dict[str, Any]) -> Dict[str, Any]:
    analysis = sample.get("analysis_json", {})
    review = sample.get("human_review", {})
    score = analysis.get("score", {}) if isinstance(analysis, dict) else {}
    return {
        "sample_id": sample.get("sample_id"),
        "name": sample.get("name"),
        "target_type": sample.get("target_type"),
        "target_key": sample.get("target_key"),
        "model_risk_level": score.get("risk_level", 0),
        "model_score": score.get("depression_signal_score", 0),
        "human_review": review,
        "model_labels": [
            {
                "key": item.get("key"),
                "label": item.get("label"),
                "message_count": item.get("message_count"),
                "risk_level": item.get("risk_level"),
            }
            for item in score.get("symptom_labels", [])[:12]
            if isinstance(item, dict)
        ],
        "top_dimensions": [
            {
                "key": item.get("key"),
                "label": item.get("label"),
                "score": item.get("score"),
                "max_points": item.get("max_points"),
            }
            for item in score.get("dimension_scores", [])[:8]
            if isinstance(item, dict)
        ],
        "evidence_snippets": [
            {
                "type": item.get("evidence_type"),
                "severity": item.get("severity"),
                "content": item.get("content"),
                "reason": item.get("reason"),
            }
            for item in analysis.get("evidences", [])[:20]
            if isinstance(item, dict)
        ],
    }


def _heuristic_suggestions(samples: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    for sample in samples:
        review = sample.get("human_review", {})
        analysis = sample.get("analysis_json", {})
        score = analysis.get("score", {}) if isinstance(analysis, dict) else {}
        model_level = int(score.get("risk_level") or 0)
        human_level = int(review.get("human_risk_level") or 0)
        suggested_keywords = _string_list(review.get("suggested_keywords"))
        negative_keywords = _string_list(review.get("suggested_negative_keywords"))
        missed_labels = _string_list(review.get("missed_labels"))
        false_positive_labels = _string_list(review.get("false_positive_labels"))

        if suggested_keywords and missed_labels:
            target_key = _find_dimension_key(config, missed_labels[0])
            if target_key:
                suggestions.append(
                    {
                        "type": "add_keywords",
                        "dimension_key": target_key,
                        "keywords": suggested_keywords[:20],
                        "reason": f"样本 {sample.get('name') or sample.get('sample_id')} 人审标记漏判，补充关键词。",
                    }
                )
        if negative_keywords and false_positive_labels:
            target_key = _find_dimension_key(config, false_positive_labels[0])
            if target_key:
                suggestions.append(
                    {
                        "type": "remove_keywords",
                        "dimension_key": target_key,
                        "keywords": negative_keywords[:20],
                        "reason": f"样本 {sample.get('name') or sample.get('sample_id')} 人审标记误判，降低这些词的触发。",
                    }
                )
        if human_level - model_level >= 2 and missed_labels:
            target_key = _find_dimension_key(config, missed_labels[0])
            if target_key:
                suggestions.append(
                    {
                        "type": "add_vector_query",
                        "dimension_key": target_key,
                        "query": f"{missed_labels[0]}：用户真实表达可能比较隐性，请召回同义、行为变化和持续性语境。",
                        "reason": "人审风险等级显著高于模型，增强该维度向量召回。",
                    }
                )
    return suggestions


def _normalized_indexes(suggestions: List[Dict[str, Any]], indexes: Optional[List[int]]) -> List[int]:
    if indexes is None:
        return list(range(len(suggestions)))
    out: List[int] = []
    for value in indexes:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(suggestions) and index not in out:
            out.append(index)
    return out


def _select_suggestions(
    suggestions: List[Dict[str, Any]],
    indexes: Optional[List[int]],
) -> List[Dict[str, Any]]:
    selected_indexes = _normalized_indexes(suggestions, indexes)
    return [suggestions[index] for index in selected_indexes]


def _apply_suggestions(config: Dict[str, Any], suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
    next_config = normalize_scoring_config(deepcopy(config))
    dimensions = {str(item.get("key") or ""): item for item in next_config.get("dimensions", []) if isinstance(item, dict)}
    labels = {
        str(item.get("key") or ""): item
        for item in next_config.get("symptom_labels", {}).get("labels", [])
        if isinstance(item, dict)
    }
    for suggestion in suggestions:
        stype = str(suggestion.get("type") or "").strip()
        dimension_key = str(suggestion.get("dimension_key") or "").strip()
        label_key = str(suggestion.get("label_key") or "").strip()
        target = dimensions.get(dimension_key) or labels.get(label_key)
        keywords = _string_list(suggestion.get("keywords"))
        if stype in {"add_keywords", "add_keyword"} and target and _suggestion_prefers_excludes(suggestion):
            _move_to_excludes(target, keywords)
        elif stype in {"add_keywords", "add_keyword"} and target:
            _append_unique(target, "keywords", _string_list(suggestion.get("keywords")))
        elif stype in {"remove_keywords", "remove_keyword", "exclude_keywords", "add_exclude_keywords", "lower_trigger"} and target:
            _move_to_excludes(target, keywords)
        elif stype in {"add_vector_query", "add_vector_queries"} and dimension_key in dimensions:
            query = str(suggestion.get("query") or "").strip()
            if query:
                _append_unique(dimensions[dimension_key], "vector_queries", [query])
        elif stype in {"adjust_threshold", "set_threshold"}:
            threshold = str(suggestion.get("threshold") or "").strip()
            if threshold in {"medium", "high", "crisis"}:
                value = _optional_int(suggestion.get("value"))
                if value is not None:
                    next_config.setdefault("thresholds", {})[threshold] = value
        elif stype in {"adjust_dimension_points", "adjust_dimension_weight", "set_max_points"} and dimension_key in dimensions:
            value = _optional_int(suggestion.get("value"))
            if value is not None:
                dimensions[dimension_key]["max_points"] = max(0, min(100, value))
        if target:
            _apply_rule_text_update(target, suggestion)
    return normalize_scoring_config(next_config)


def _suggestion_prefers_excludes(suggestion: Dict[str, Any]) -> bool:
    text = " ".join(
        str(suggestion.get(key) or "")
        for key in ("reason", "description", "note", "summary")
    )
    stype = str(suggestion.get("type") or "")
    if stype in {"exclude_keywords", "add_exclude_keywords", "lower_trigger"}:
        return True
    return any(
        token in text
        for token in (
            "排除",
            "误判",
            "误触发",
            "假阳性",
            "降低触发",
            "降低这些词",
            "不应",
            "不代表",
            "非自杀",
            "非轻生",
            "减少误报",
            "减少虚警",
            "降低虚警",
        )
    )


def _apply_rule_text_update(target: Dict[str, Any], suggestion: Dict[str, Any]) -> None:
    stype = str(suggestion.get("type") or "").strip()
    description = _first_text(
        suggestion,
        "description",
        "rule_description",
        "judgement_rule",
        "judgment_rule",
        "criteria",
        "standard",
    )
    reason = _first_text(
        suggestion,
        "reason",
        "rationale",
        "why",
        "note",
        "notes",
        "analysis",
    )
    if description and stype in {
        "update_description",
        "set_description",
        "replace_description",
        "update_rule_description",
        "set_rule_description",
        "adjust_rule_description",
    }:
        target["description"] = description[:1200]
    elif description:
        _append_description_line(target, f"规则补充：{description}")
    if reason:
        _append_description_line(target, f"优化理由：{reason}")


def _append_description_line(target: Dict[str, Any], line: str) -> None:
    line = re.sub(r"\s+", " ", str(line or "")).strip()
    if not line:
        return
    current = str(target.get("description") or "").strip()
    existing_lines = [item.strip() for item in re.split(r"[\n；;]+", current) if item.strip()]
    if line in existing_lines or line in current:
        return
    next_text = f"{current}\n{line}" if current else line
    target["description"] = next_text[:1600]


def _first_text(item: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:800]
    return ""


def _diff_scoring_config(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    before = normalize_scoring_config(before)
    after = normalize_scoring_config(after)
    diff: Dict[str, Any] = {"thresholds": {}, "dimensions": {}, "labels": {}, "changed": False}
    for key in ("medium", "high", "crisis"):
        old = before.get("thresholds", {}).get(key)
        new = after.get("thresholds", {}).get(key)
        if old != new:
            diff["thresholds"][key] = {"before": old, "after": new}
            diff["changed"] = True

    before_dims = {item.get("key"): item for item in before.get("dimensions", []) if isinstance(item, dict)}
    after_dims = {item.get("key"): item for item in after.get("dimensions", []) if isinstance(item, dict)}
    for key, new_item in after_dims.items():
        old_item = before_dims.get(key, {})
        changes = _diff_rule_item(old_item, new_item)
        if changes:
            diff["dimensions"][key] = changes
            diff["changed"] = True

    before_labels = {
        item.get("key"): item
        for item in before.get("symptom_labels", {}).get("labels", [])
        if isinstance(item, dict)
    }
    after_labels = {
        item.get("key"): item
        for item in after.get("symptom_labels", {}).get("labels", [])
        if isinstance(item, dict)
    }
    for key, new_item in after_labels.items():
        old_item = before_labels.get(key, {})
        changes = _diff_rule_item(old_item, new_item)
        if changes:
            diff["labels"][key] = changes
            diff["changed"] = True
    return diff


def _merge_preview_diffs(scoring_diff: Dict[str, Any], prompt_diff: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(scoring_diff)
    prompts = prompt_diff.get("prompts") if isinstance(prompt_diff.get("prompts"), dict) else {}
    merged["prompts"] = prompts
    merged["changed"] = bool(scoring_diff.get("changed") or prompt_diff.get("changed"))
    return merged


def _diff_rule_item(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    changes: Dict[str, Any] = {}
    for field in ("max_points", "enabled", "redline", "risk_level", "description"):
        if field in after and before.get(field) != after.get(field):
            changes[field] = {"before": before.get(field), "after": after.get(field)}
    for field in ("keywords", "exclude_keywords", "strong_keywords", "vector_queries"):
        before_values = [str(item) for item in before.get(field, []) if str(item or "").strip()]
        after_values = [str(item) for item in after.get(field, []) if str(item or "").strip()]
        added = [item for item in after_values if item not in before_values]
        removed = [item for item in before_values if item not in after_values]
        if added or removed:
            changes[field] = {
                "added": added[:50],
                "removed": removed[:50],
                "before_count": len(before_values),
                "after_count": len(after_values),
            }
    return changes


def _merge_suggestions(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_suggestion(item)
        key = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out[:80]


def _normalize_suggestions(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_normalize_suggestion(item) for item in value if isinstance(item, dict)]


def _normalize_suggestion(item: Dict[str, Any]) -> Dict[str, Any]:
    stype = str(item.get("type") or item.get("action") or item.get("operation") or "").strip()
    reason = _first_text(item, "reason", "rationale", "why", "note", "notes", "analysis")
    description = _first_text(
        item,
        "description",
        "rule_description",
        "judgement_rule",
        "judgment_rule",
        "criteria",
        "standard",
    )
    if not stype and (reason or description):
        stype = "append_reason"
    out = {
        "type": stype,
        "dimension_key": str(item.get("dimension_key") or item.get("dimension") or item.get("dimensionKey") or "").strip(),
        "label_key": str(item.get("label_key") or item.get("label") or item.get("labelKey") or "").strip(),
        "prompt_key": str(item.get("prompt_key") or item.get("prompt") or item.get("promptKey") or "").strip(),
        "keywords": _string_list(item.get("keywords") if "keywords" in item else item.get("keyword")),
        "query": str(item.get("query") or item.get("vector_query") or item.get("vectorQuery") or "").strip(),
        "threshold": str(item.get("threshold") or "").strip(),
        "value": _optional_int(item.get("value") if "value" in item else item.get("new_value") or item.get("score") or item.get("max_points")),
        "prompt_text": str(item.get("prompt_text") or item.get("promptText") or item.get("new_prompt") or "").strip()[:12000],
        "instruction": str(item.get("instruction") or item.get("prompt_instruction") or item.get("promptInstruction") or "").strip()[:2000],
        "description": description,
        "reason": reason,
    }
    return {key: value for key, value in out.items() if value not in ("", [], None)}


def _find_dimension_key(config: Dict[str, Any], label_or_key: str) -> str:
    token = str(label_or_key or "").strip().lower()
    if not token:
        return ""
    for item in config.get("dimensions", []):
        key = str(item.get("key") or "")
        label = str(item.get("label") or "")
        if token in {key.lower(), label.lower()} or token in label.lower():
            return key
    for item in config.get("symptom_labels", {}).get("labels", []):
        key = str(item.get("key") or "")
        label = str(item.get("label") or "")
        if token in {key.lower(), label.lower()} or token in label.lower():
            dims = item.get("dimension_keys") or []
            if dims:
                return str(dims[0])
    return ""


def _append_unique(target: Dict[str, Any], field: str, values: List[str]) -> None:
    current = [str(item) for item in target.get(field, []) if str(item or "").strip()]
    for value in values:
        if value and value not in current:
            current.append(value)
    target[field] = current


def _move_to_excludes(target: Dict[str, Any], keywords: List[str]) -> None:
    current = [str(item) for item in target.get("keywords", []) if str(item or "").strip()]
    excludes = [str(item) for item in target.get("exclude_keywords", []) if str(item or "").strip()]
    for keyword in keywords:
        if keyword in current:
            current = [item for item in current if item != keyword]
        if keyword and keyword not in excludes:
            excludes.append(keyword)
    target["keywords"] = current
    target["exclude_keywords"] = excludes


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    elif "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _default_sample_name(analysis: Dict[str, Any]) -> str:
    score = analysis.get("score", {}) if isinstance(analysis, dict) else {}
    task_id = str(analysis.get("task_id") or "")[:8]
    return f"训练样本 L{score.get('risk_level', 0)} {task_id}".strip()


def _fallback_summary(samples: List[Dict[str, Any]], suggestions: List[Dict[str, Any]], error: str) -> str:
    if error:
        return f"大模型优化失败，已根据 {len(samples)} 条人审样本生成本地规则草案。"
    return f"已根据 {len(samples)} 条人审样本生成 {len(suggestions)} 条评分标准优化建议。"


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        raw = re.split(r"[\n,，;；]+", value)
    elif isinstance(value, list):
        raw = value
    else:
        raw = []
    out: List[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _safe_error_text(error: Exception) -> str:
    text = str(error).replace("\r", " ").replace("\n", " ").strip()
    if not text:
        text = error.__class__.__name__
    return text[:300]
