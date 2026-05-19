from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.models import (
    TrainingProposalApplyRequest,
    TrainingProposalRequest,
    TrainingSampleCreateRequest,
    TrainingSampleUpdateRequest,
)
from app.services import training_service
from app.services.training_prompt_service import (
    load_training_prompt_config,
    reset_training_prompt_config,
    save_training_prompt_config,
)
from app.stores import training_proposal_task_store


router = APIRouter(tags=["training"])


@router.get("/psych/training/samples")
def get_training_samples(
    limit: int = Query(100, ge=1, le=500),
    reviewed_only: bool = False,
    target_type: str = "all",
    target_key: str = "",
) -> list[dict]:
    return training_service.list_samples(
        limit=limit,
        reviewed_only=reviewed_only,
        target_type=target_type,
        target_key=target_key,
    )


@router.post("/psych/training/samples")
def post_training_sample(request: TrainingSampleCreateRequest) -> dict:
    if not request.analysis_result:
        raise HTTPException(status_code=400, detail="analysis_result is required")
    return training_service.create_sample(request)


@router.put("/psych/training/samples/{sample_id}")
def put_training_sample(sample_id: str, request: TrainingSampleUpdateRequest) -> dict:
    sample = training_service.update_sample(sample_id, request)
    if not sample:
        raise HTTPException(status_code=404, detail="training sample not found")
    return sample


@router.delete("/psych/training/samples/{sample_id}")
def delete_training_sample(sample_id: str) -> dict:
    deleted = training_service.delete_sample(sample_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="training sample not found")
    return {"deleted": True, "sample_id": sample_id}


@router.get("/psych/training/prompts")
def get_training_prompts() -> dict:
    return load_training_prompt_config()


@router.put("/psych/training/prompts")
def put_training_prompts(config: dict) -> dict:
    return save_training_prompt_config(config)


@router.post("/psych/training/prompts/reset")
def post_reset_training_prompts() -> dict:
    return reset_training_prompt_config()


@router.get("/psych/training/proposals")
def get_training_proposals(
    limit: int = Query(50, ge=1, le=200),
    target_type: str = "all",
    target_key: str = "",
) -> list[dict]:
    return training_service.list_proposals(limit=limit, target_type=target_type, target_key=target_key)


@router.get("/psych/training/proposals/{proposal_id}")
def get_training_proposal(proposal_id: str) -> dict:
    proposal = training_service.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="training proposal not found")
    return proposal


@router.post("/psych/training/proposals/generate")
def post_generate_training_proposal(request: TrainingProposalRequest) -> dict:
    return training_service.generate_proposal(
        sample_ids=request.sample_ids,
        max_samples=request.max_samples,
        use_llm=request.use_llm,
        target_type=request.target_type,
        target_key=request.target_key,
    )


@router.post("/psych/training/proposals/generate/start")
def post_start_training_proposal(
    request: TrainingProposalRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    task = training_proposal_task_store.create_task(
        {
            "use_llm": request.use_llm,
            "max_samples": request.max_samples,
            "sample_ids": request.sample_ids or [],
            "target_type": request.target_type,
            "target_key": request.target_key,
            "requires_human_review": True,
        }
    )
    background_tasks.add_task(_run_proposal_task, task["task_id"], request)
    return task


@router.get("/psych/training/proposals/generate/tasks/{task_id}/progress")
def get_training_proposal_task(task_id: str) -> dict:
    task = training_proposal_task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="training proposal task not found")
    return task


@router.post("/psych/training/proposals/apply")
def post_apply_training_proposal(request: TrainingProposalApplyRequest, background_tasks: BackgroundTasks) -> dict:
    try:
        proposal = training_service.apply_proposal(
            request.proposal_id,
            selected_suggestion_indexes=request.selected_suggestion_indexes,
            human_review_confirmed=request.human_review_confirmed,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not proposal:
        raise HTTPException(status_code=404, detail="training proposal not found")
    if request.auto_repeat_after_apply:
        task = training_proposal_task_store.create_task(
            {
                "kind": "repeat_after_apply",
                "source_proposal_id": request.proposal_id,
                "use_llm": request.repeat_use_llm,
                "max_samples": request.repeat_max_samples,
                "requires_human_review": True,
            }
        )
        background_tasks.add_task(_run_repeat_after_apply_task, task["task_id"], request)
        proposal["repeat_task"] = task
    return proposal


@router.post("/psych/training/proposals/preview")
def post_preview_training_proposal(request: TrainingProposalApplyRequest) -> dict:
    preview = training_service.preview_proposal(
        request.proposal_id,
        selected_suggestion_indexes=request.selected_suggestion_indexes,
    )
    if not preview:
        raise HTTPException(status_code=404, detail="training proposal not found")
    return preview


def _run_proposal_task(task_id: str, request: TrainingProposalRequest) -> None:
    try:
        training_proposal_task_store.update_task(
            task_id,
            status="running",
            stage="select_samples",
            message="正在读取人审样本",
            progress=10,
        )
        training_proposal_task_store.update_task(
            task_id,
            stage="generate_with_llm" if request.use_llm else "generate_with_rules",
            message="正在调用大模型生成草案" if request.use_llm else "正在使用本地规则生成草案",
            progress=35,
        )
        proposal = training_service.generate_proposal(
            sample_ids=request.sample_ids,
            max_samples=request.max_samples,
            use_llm=request.use_llm,
            target_type=request.target_type,
            target_key=request.target_key,
        )
        training_proposal_task_store.update_task(
            task_id,
            status="completed",
            stage="completed",
            message="优化草案已生成",
            progress=100,
            result=proposal,
        )
    except Exception as error:
        training_proposal_task_store.update_task(
            task_id,
            status="failed",
            stage="failed",
            message="优化草案生成失败",
            progress=100,
            error=str(error)[:500] or error.__class__.__name__,
        )


def _run_repeat_after_apply_task(task_id: str, request: TrainingProposalApplyRequest) -> None:
    def on_progress(stage: str, message: str, progress: int, extra: dict) -> None:
        training_proposal_task_store.update_task(
            task_id,
            status="running",
            stage=stage,
            message=message,
            progress=progress,
            **extra,
        )

    try:
        training_proposal_task_store.update_task(
            task_id,
            status="running",
            stage="collect_reference_samples",
            message="正在读取已审核草案的参考样本",
            progress=5,
        )
        result = training_service.repeat_after_apply(
            request.proposal_id,
            max_samples=request.repeat_max_samples,
            use_llm=request.repeat_use_llm,
            progress_callback=on_progress,
        )
        training_proposal_task_store.update_task(
            task_id,
            status="completed",
            stage="completed",
            message="复测、自动审核和下一轮草案生成完成；下一轮草案仍需人工审核",
            progress=100,
            result=result,
        )
    except Exception as error:
        training_proposal_task_store.update_task(
            task_id,
            status="failed",
            stage="failed",
            message="自动复测闭环失败",
            progress=100,
            error=str(error)[:500] or error.__class__.__name__,
        )
