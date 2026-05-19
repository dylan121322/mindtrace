import time
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException

from app.config import get_settings
from app.models import PsychAnalyzeRequest, PsychAnalyzeResponse, PsychEvidence, PsychProcessStep, PsychScore
from app.psych.scoring_config import load_scoring_config, reset_scoring_config, save_scoring_config
from app.services.psych_service import analyze_psych
from app.stores import psych_store, psych_task_store, vector_task_store


router = APIRouter(tags=["psych"])


def _model_dump(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _safe_error_text(error: Exception) -> str:
    text = str(error).replace("\r", " ").replace("\n", " ").strip()
    return text[:300] if text else error.__class__.__name__


def _auto_review_disabled_status() -> Dict[str, Any]:
    return {
        "enabled": False,
        "status": "disabled",
        "message": "自动审核未启用",
    }


def _auto_review_queued_status() -> Dict[str, Any]:
    return {
        "enabled": True,
        "status": "queued",
        "stage": "queued",
        "message": "分析已完成，自动审核已进入独立队列",
        "started_at": 0,
        "updated_at": int(time.time()),
        "requires_human_confirmation": True,
    }


def _run_auto_review_task(task_id: str, request: PsychAnalyzeRequest, response: PsychAnalyzeResponse) -> None:
    if not get_settings().training_auto_review_enabled:
        psych_task_store.update_task(task_id, auto_review=_auto_review_disabled_status())
        return

    started_at = int(time.time())
    psych_task_store.update_task(
        task_id,
        auto_review={
            "enabled": True,
            "status": "running",
            "stage": "auto_review",
            "message": "自动审核与规则优化草案生成中；分析结果已完成可查看",
            "started_at": started_at,
            "updated_at": started_at,
            "requires_human_confirmation": True,
        },
    )
    try:
        from app.services import training_service

        info = training_service.auto_review_analysis(response, request)
        now = int(time.time())
        status = str(info.get("status") or "completed")
        psych_task_store.update_task(
            task_id,
            auto_review={
                "enabled": True,
                "status": status,
                "stage": "auto_review",
                "message": "自动审核完成" if status == "completed" else "自动审核未完成",
                "sample_id": info.get("sample_id", ""),
                "proposal_id": info.get("proposal_id", ""),
                "proposal_status": info.get("proposal_status", ""),
                "suggestion_count": int(info.get("suggestion_count") or 0),
                "llm_used": bool(info.get("llm_used")),
                "llm_error": info.get("llm_error", ""),
                "error": info.get("error", ""),
                "review_summary": info.get("review_summary", ""),
                "requires_human_confirmation": bool(info.get("requires_human_confirmation", True)),
                "started_at": started_at,
                "updated_at": now,
                "duration_seconds": max(0, now - started_at),
            },
        )
    except Exception as exc:
        now = int(time.time())
        psych_task_store.update_task(
            task_id,
            auto_review={
                "enabled": True,
                "status": "failed",
                "stage": "auto_review",
                "message": "自动审核失败，分析结果不受影响",
                "error": f"{exc.__class__.__name__}: {_safe_error_text(exc)}",
                "requires_human_confirmation": True,
                "started_at": started_at,
                "updated_at": now,
                "duration_seconds": max(0, now - started_at),
            },
        )


@router.post("/psych/analyze", response_model=PsychAnalyzeResponse)
def post_analyze(request: PsychAnalyzeRequest) -> PsychAnalyzeResponse:
    wait_result = vector_task_store.wait_for_active_tasks(timeout_seconds=3600, poll_seconds=1.0)
    if wait_result.get("timed_out"):
        raise HTTPException(
            status_code=409,
            detail="向量数据库仍在建立中，心理分析已等待但尚未完成。请稍后重试，或在数据库页面查看向量任务进度。",
        )
    response = analyze_psych(request)
    if wait_result.get("waited"):
        response.report_json["waited_for_vector_index"] = wait_result
        response.process_steps.insert(
            0,
            PsychProcessStep(
                key="wait_vector_index",
                name="等待向量数据库",
                status="completed",
                duration_ms=int(wait_result.get("wait_seconds", 0)) * 1000,
                detail="检测到向量数据库正在建立，已等待任务完成后再开始心理分析。",
                metrics=wait_result,
            ),
        )
    return response


@router.post("/psych/analyze/start")
def start_analyze(request: PsychAnalyzeRequest, background_tasks: BackgroundTasks) -> dict:
    task = psych_task_store.create_task(
        {
            "target_key": request.target_key or "",
            "target_type": request.target_type,
            "message": "analysis task queued",
        }
    )
    background_tasks.add_task(_run_psych_task, task["task_id"], request)
    return task


def _run_psych_task(task_id: str, request: PsychAnalyzeRequest) -> None:
    wait_prefix: List[PsychProcessStep] = []
    try:
        psych_task_store.update_task(
            task_id,
            status="running",
            stage="wait_vector_index",
            message="checking vector index tasks",
            progress=1,
        )
        wait_result = vector_task_store.wait_for_active_tasks(timeout_seconds=3600, poll_seconds=1.0)
        if wait_result.get("timed_out"):
            psych_task_store.update_task(
                task_id,
                status="failed",
                stage="wait_vector_index",
                message="vector index task timed out",
                error="向量数据库仍在建立中，心理分析等待超时。请在数据库页面查看向量任务进度后重试。",
            )
            return
        if wait_result.get("waited"):
            wait_prefix = [
                PsychProcessStep(
                    key="wait_vector_index",
                    name="等待向量数据库",
                    status="completed",
                    duration_ms=int(wait_result.get("wait_seconds", 0)) * 1000,
                    detail="检测到向量数据库正在建立，已等待任务完成后再开始心理分析。",
                    metrics=wait_result,
                )
            ]

        def on_progress(steps: List[PsychProcessStep], current_key: str, state: str) -> None:
            merged = [*wait_prefix, *steps]
            total = max(1, len(wait_prefix) + 9)
            completed = len([step for step in merged if step.status == "completed"])
            running = next((step for step in merged if step.status == "running"), None)
            progress = min(99, int((completed / total) * 100))
            psych_task_store.update_task(
                task_id,
                status="running",
                stage=current_key,
                message=running.name if running else current_key,
                progress=progress,
                process_steps=[_model_dump(step) for step in merged],
            )

        response = analyze_psych(request, task_id=task_id, progress_callback=on_progress)
        if wait_prefix:
            response.process_steps = [*wait_prefix, *response.process_steps]
            response.report_json["waited_for_vector_index"] = wait_prefix[0].metrics
            response.report_json["process_steps"] = [_model_dump(step) for step in response.process_steps]
        psych_task_store.update_task(
            task_id,
            status="completed",
            stage="completed",
            message="analysis completed",
            progress=100,
            process_steps=[_model_dump(step) for step in response.process_steps],
            result=_model_dump(response),
            auto_review=_auto_review_queued_status()
            if get_settings().training_auto_review_enabled
            else _auto_review_disabled_status(),
        )
        _run_auto_review_task(task_id, request, response)
    except Exception as exc:
        psych_task_store.update_task(
            task_id,
            status="failed",
            stage="failed",
            message="analysis failed",
            error=f"{exc.__class__.__name__}: {_safe_error_text(exc)}",
        )


@router.get("/psych/tasks/{task_id}/progress")
def get_progress(task_id: str) -> dict:
    live_task = psych_task_store.get_task(task_id)
    if live_task:
        return live_task
    task = psych_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return {"task_id": task_id, "status": task["status"], "updated_at": task["updated_at"]}


@router.get("/psych/tasks/{task_id}/score", response_model=PsychScore)
def get_score(task_id: str) -> PsychScore:
    score = psych_store.get_score(task_id)
    if not score:
        raise HTTPException(status_code=404, detail="score not found")
    return score


@router.get("/psych/tasks/{task_id}/report")
def get_report(task_id: str) -> dict:
    report = psych_store.get_report(task_id)
    if not report:
        raise HTTPException(status_code=404, detail="report not found")
    return report


@router.get("/psych/tasks/{task_id}/evidence", response_model=List[PsychEvidence])
def get_evidence(task_id: str) -> List[PsychEvidence]:
    task = psych_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return psych_store.get_evidence(task_id)


@router.get("/psych/scoring-config")
def get_scoring_config() -> dict:
    return load_scoring_config()


@router.put("/psych/scoring-config")
def put_scoring_config(config: dict = Body(...)) -> dict:
    return save_scoring_config(config)


@router.post("/psych/scoring-config/reset")
def post_reset_scoring_config() -> dict:
    return reset_scoring_config()
