import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import get_settings
from app.stores.ai_store import ping_ai_store
from app.utils.hardware import get_hardware_info
from app.utils.logging import current_log_time


router = APIRouter(tags=["app"])
frontend_logger = logging.getLogger("frontend")


class FrontendLogPayload(BaseModel):
    logs: list[dict] = Field(default_factory=list)


@router.get("/app/info")
def app_info() -> dict:
    settings = get_settings()
    data = settings.safe_public_dict()
    data.update(
        {
            "service": "python_backend",
            "status": "ok",
            "data_dir_exists": settings.data_dir.exists(),
            "ai_store_ready": ping_ai_store(),
            "hardware": get_hardware_info(),
        }
    )
    return data


@router.get("/status")
def status() -> dict:
    settings = get_settings()
    return {
        "status": "ready",
        "python_backend": True,
        "data_dir_exists": settings.data_dir.exists(),
        "is_initialized": True,
        "is_indexing": False,
        "total_cached": 0,
        "progress": None,
        "hardware": get_hardware_info(),
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_use_search_prefix": settings.embedding_use_search_prefix,
        "last_error": None if settings.data_dir.exists() else "DATA_DIR not found; contacts/messages may be empty",
    }


@router.post("/app/frontend-log")
def frontend_log(payload: FrontendLogPayload) -> dict:
    received_at = current_log_time()
    count = 0
    for item in payload.logs[:200]:
        if not isinstance(item, dict):
            continue
        level = str(item.get("level") or "info").lower()
        event_time = str(item.get("time") or item.get("local_time") or "")
        message = str(item.get("message") or "").replace("\r", " ").replace("\n", "\\n")[:2000]
        url = str(item.get("url") or "")[:300]
        log_level = logging.ERROR if level == "error" else logging.WARNING if level == "warn" else logging.INFO
        frontend_logger.log(
            log_level,
            "frontend event_time=%s received_at=%s url=%s message=%s",
            event_time or "-",
            received_at,
            url or "-",
            message,
        )
        count += 1
    return {"status": "ok", "received_at": received_at, "count": count}
