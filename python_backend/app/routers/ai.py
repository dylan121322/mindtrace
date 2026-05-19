from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field
from requests import exceptions as request_exceptions

from app.config import get_settings
from app.models import ChatMessage
from app.services import memory_service, message_service, vector_service
from app.services.embedding_service import get_embedding, get_embedding_profile
from app.services.llm_service import LLMServiceError, complete_chat
from app.stores import fact_task_store, vector_task_store


router = APIRouter(tags=["ai"])


class BuildIndexRequest(BaseModel):
    target_key: str = ""
    target_type: str = "contact"
    time_from: Optional[str] = None
    time_to: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    options: dict = Field(default_factory=dict)


class LLMTestRequest(BaseModel):
    prompt: str = "你好，请用一句话回复。"
    profile: str = "main"


def _messages_from_request(body: BuildIndexRequest) -> List[ChatMessage]:
    if body.target_type == "group":
        raise HTTPException(status_code=400, detail="群聊分析已关闭，请选择本人或联系人。")
    if body.messages is not None:
        return body.messages
    return message_service.list_messages(
        body.target_key,
        body.target_type,
        body.time_from,
        body.time_to,
        0,
    )


def _vector_key(body: BuildIndexRequest) -> str:
    key = (body.target_key or "").strip()
    if key:
        return key
    return body.target_type or "all"


@router.post("/ai/vec/build-index")
def build_vec_index(body: BuildIndexRequest, background_tasks: BackgroundTasks) -> dict:
    if body.target_type == "group":
        raise HTTPException(status_code=400, detail="群聊向量索引已关闭，请选择本人、联系人或全部联系人消息。")
    key = _vector_key(body)
    task = vector_task_store.create_task(
        {
            "contact_key": key,
            "target_type": body.target_type,
            "target_key": body.target_key,
            "time_from": body.time_from,
            "time_to": body.time_to,
        }
    )
    background_tasks.add_task(_run_vector_index_task, task["task_id"], body)
    return task


def _run_vector_index_task(task_id: str, body: BuildIndexRequest) -> None:
    key = _vector_key(body)
    try:
        if vector_task_store.is_cancel_requested(task_id):
            raise vector_service.VectorBuildCancelled()
        vector_task_store.update_task(
            task_id,
            status="running",
            stage="reading_messages",
            message="正在读取所选时间范围内的聊天消息",
            progress=3,
        )
        messages = _messages_from_request(body)
        if vector_task_store.is_cancel_requested(task_id):
            raise vector_service.VectorBuildCancelled()
        total_messages = len(messages)
        vector_task_store.update_task(
            task_id,
            stage="embedding",
            message=f"已读取 {total_messages} 条消息，开始调用向量模型",
            progress=8,
            processed=0,
            total=total_messages,
            source_message_count=total_messages,
        )

        def on_progress(done: int, total: int, message: str) -> None:
            if total <= 0:
                progress = 10
            else:
                progress = 10 + int((done / total) * 80)
            vector_task_store.update_task(
                task_id,
                status="running",
                stage="embedding",
                message=message,
                progress=min(90, progress),
                processed=done,
                total=total,
            )

        status = vector_service.build_vector_index(
            key,
            messages,
            body.options.get("embedding_config"),
            progress_callback=on_progress,
            cancel_callback=lambda: vector_task_store.is_cancel_requested(task_id),
        )
    except vector_service.VectorBuildCancelled:
        vector_task_store.update_task(
            task_id,
            status="canceled",
            stage="canceled",
            message="向量任务已停止，已完成写入的索引会保留",
            error="",
        )
        return
    except vector_service.VectorEmbeddingError as exc:
        vector_task_store.update_task(
            task_id,
            status="failed",
            stage="failed",
            message="鍚戦噺妯″瀷杩炵画璇锋眰澶辫触",
            error=f"向量模型连续请求失败：{str(exc) or exc.__class__.__name__}。已自动拆分批次重试，但仍超过失败阈值。请检查模型是否稳定、调小最大 batch 或改用本地 hash 兜底。",
        )
        return
    except request_exceptions.Timeout as exc:
        vector_task_store.update_task(
            task_id,
            status="failed",
            stage="failed",
            message="鍚戦噺妯″瀷璇锋眰瓒呮椂",
            error="向量模型请求超时：Ollama 或兼容 Embedding 接口在限定时间内没有返回。请确认模型已启动、缩短时间范围，或在 .env 中增大 EMBEDDING_TIMEOUT。",
        )
        return
    except request_exceptions.RequestException as exc:
        vector_task_store.update_task(
            task_id,
            status="failed",
            stage="failed",
            message="向量模型请求失败",
            error=f"向量模型请求失败：{exc.__class__.__name__}",
        )
        return
    except vector_service.VectorEmbeddingError as exc:
        vector_task_store.update_task(
            task_id,
            status="failed",
            stage="failed",
            message="向量模型请求失败",
            error=(
                "向量模型请求失败：已自动重试、拆分批次并跳过少量失败消息，"
                f"但失败数量超过容错阈值。详情：{exc}"
            ),
        )
        return
    except Exception as exc:
        vector_task_store.update_task(
            task_id,
            status="failed",
            stage="failed",
            message="向量数据库建立失败",
            error=f"鍚戦噺鏁版嵁搴撳缓绔嬪け璐ワ細{exc.__class__.__name__}",
        )
        return
    status.update(
        {
            "contact_key": key,
            "source_message_count": len(messages),
            "target_type": body.target_type,
            "target_key": body.target_key,
            "time_from": body.time_from,
            "time_to": body.time_to,
            "message_limit_applied": False,
        }
    )
    vector_task_store.update_task(
        task_id,
        status="completed",
        stage="completed",
        message="向量数据库建立完成",
        progress=100,
        processed=status.get("msg_count", len(messages)),
        total=len(messages),
        result=status,
    )


@router.get("/ai/vec/build-index/{task_id}/progress")
def get_vec_build_progress(task_id: str) -> dict:
    task = vector_task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="向量任务不存在或已过期")
    return task


@router.post("/ai/vec/build-index/{task_id}/cancel")
def cancel_vec_build(task_id: str) -> dict:
    task = vector_task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="向量任务不存在或已过期")
    if task.get("status") not in ("queued", "running"):
        return task
    updated = vector_task_store.request_cancel(task_id)
    return updated or task


@router.get("/ai/vec/status")
def get_vec_status(key: str = Query(...)) -> dict:
    return vector_service.get_vector_index_status(key)


@router.get("/ai/vec/search")
def search_vec(
    key: str = Query(...),
    q: str = Query(...),
    top_k: int = Query(5, ge=1, le=50),
) -> List[dict]:
    try:
        return vector_service.search_vector(key, q, top_k=top_k)
    except vector_service.VectorEmbeddingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/ai/psych-facts/build")
def build_psych_facts(body: BuildIndexRequest, background_tasks: BackgroundTasks) -> dict:
    if body.target_type == "group":
        raise HTTPException(status_code=400, detail="群聊心理事实库已关闭，请选择本人、联系人或全部联系人消息。")
    key = _vector_key(body)
    task = fact_task_store.create_task(
        {
            "contact_key": key,
            "target_type": body.target_type,
            "target_key": body.target_key,
            "time_from": body.time_from,
            "time_to": body.time_to,
        }
    )
    background_tasks.add_task(_run_psych_fact_task, task["task_id"], body)
    return task


def _run_psych_fact_task(task_id: str, body: BuildIndexRequest) -> None:
    key = _vector_key(body)
    try:
        fact_task_store.update_task(
            task_id,
            status="running",
            stage="reading_messages",
            message="正在读取所选时间范围内的聊天消息",
            progress=3,
        )
        messages = _messages_from_request(body)
        total_messages = len(messages)
        fact_task_store.update_task(
            task_id,
            status="running",
            stage="extracting",
            message=f"已读取 {total_messages} 条消息，开始按批抽取心理事实",
            progress=8,
            processed=0,
            total=total_messages,
            source_message_count=total_messages,
        )

        def on_progress(done: int, total: int, message: str) -> None:
            progress = 10 if total <= 0 else 10 + int((done / total) * 75)
            fact_task_store.update_task(
                task_id,
                status="running",
                stage="extracting",
                message=message,
                progress=min(90, progress),
                processed=done,
                total=total,
            )

        fact_options = dict(body.options or {})
        if "message_scope" not in fact_options:
            if body.target_type == "self":
                fact_options["message_scope"] = "mine"
            elif body.target_type == "contact":
                fact_options["message_scope"] = "other"
            elif "only_mine" not in fact_options:
                fact_options["message_scope"] = "mine"
        status = memory_service.build_psych_memory_facts(
            key,
            messages,
            fact_options,
            progress_callback=on_progress,
            cancel_callback=lambda: fact_task_store.is_cancel_requested(task_id),
        )
    except memory_service.MemoryBuildCancelled:
        fact_task_store.update_task(
            task_id,
            status="canceled",
            stage="canceled",
            message="心理事实库构建已停止，已完成写入的数据会保留",
            error="",
        )
        return
    except Exception as exc:
        fact_task_store.update_task(
            task_id,
            status="failed",
            stage="failed",
            message="心理事实库构建失败",
            error=f"{exc.__class__.__name__}: {str(exc)[:200]}",
        )
        return
    fact_task_store.update_task(
        task_id,
        status="completed",
        stage="completed",
        message="心理事实库构建完成",
        progress=100,
        processed=status.get("filtered_message_count", len(messages)),
        total=status.get("filtered_message_count", len(messages)),
        result=status,
    )


@router.get("/ai/psych-facts/build/{task_id}/progress")
def get_psych_fact_progress(task_id: str) -> dict:
    task = fact_task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="心理事实库任务不存在或已过期")
    return task


@router.post("/ai/psych-facts/build/{task_id}/cancel")
def cancel_psych_fact_build(task_id: str) -> dict:
    task = fact_task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="心理事实库任务不存在或已过期")
    if task.get("status") not in ("queued", "running"):
        return task
    updated = fact_task_store.request_cancel(task_id)
    return updated or task


@router.get("/ai/psych-facts/status")
def get_psych_fact_status(key: str = Query(...)) -> dict:
    return memory_service.get_psych_fact_status(key)


@router.get("/ai/psych-facts/search")
def search_psych_facts(
    key: str = Query(...),
    q: str = Query(...),
    top_k: int = Query(8, ge=1, le=50),
) -> List[dict]:
    return memory_service.search_psych_memory_facts(key, q, top_k=top_k)


@router.post("/ai/mem/build")
def build_memory(body: BuildIndexRequest) -> dict:
    messages = _messages_from_request(body)
    facts = memory_service.extract_memory_facts(body.target_key, messages)
    return {"fact_count": len(facts), "facts": facts}


@router.get("/ai/mem/search")
def search_memory(
    key: str = Query(...),
    q: str = Query(...),
    top_k: int = Query(5, ge=1, le=50),
) -> List[dict]:
    return memory_service.search_memory_facts(key, q, top_k=top_k)


@router.post("/ai/embedding/test")
def test_embedding(input_type: str = Query("query", pattern="^(query|document)$")) -> dict:
    profile = get_embedding_profile()
    text = "心理风险辅助筛查向量连接测试"
    vec = get_embedding(text, input_type=input_type)
    return {
        "ok": True,
        "dims": len(vec),
        "provider": profile["provider"],
        "model": profile["model"],
        "model_identity": profile["model_identity"],
        "input_type": input_type,
        "uses_search_prefix": bool(profile.get("uses_search_prefix")),
        "search_prefix_mode": profile.get("search_prefix_mode") or "auto",
    }


def _llm_test_config(profile: str) -> tuple[str, dict, str]:
    settings = get_settings()
    name = (profile or "main").strip().lower()
    if name in {"psych_fact", "fact", "facts"}:
        return (
            "psych_fact",
            {
                "provider": settings.psych_fact_llm_provider,
                "base_url": settings.psych_fact_llm_base_url,
                "model": settings.psych_fact_llm_model,
                "api_key": settings.psych_fact_llm_api_key,
            },
            "请回复：心理事实抽取模型连接正常",
        )
    if name in {"training", "auto_review", "proposal"}:
        return (
            "training",
            {
                "provider": settings.llm_provider,
                "base_url": settings.llm_base_url,
                "model": settings.llm_model,
                "api_key": settings.llm_api_key,
            },
            "请回复：训练优化复盘模型连接正常",
        )
    return (
        "main",
        {
            "provider": settings.llm_provider,
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
            "api_key": settings.llm_api_key,
        },
        "请回复：心理分析模型连接正常",
    )


@router.post("/ai/llm/test")
def test_llm(body: LLMTestRequest) -> dict:
    profile_name, config, default_prompt = _llm_test_config(body.profile)
    try:
        reply = complete_chat([{"role": "user", "content": body.prompt or default_prompt}], config=config)
    except LLMServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc.__class__.__name__}") from exc
    return {
        "ok": True,
        "profile": profile_name,
        "provider": config.get("provider", ""),
        "model": config.get("model", ""),
        "reply": reply,
    }
