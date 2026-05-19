from typing import List, Optional

from fastapi import APIRouter, Query

from app.models import ChatMessage
from app.services.message_service import list_message_targets, list_messages, list_messages_with_diagnostics


router = APIRouter(tags=["messages"])


@router.get("/messages", response_model=List[ChatMessage])
def get_messages(
    target_key: str = Query("", description="Contact username or contact:username key"),
    target_type: str = Query("contact", pattern="^(self|contact|all)$"),
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    limit: int = Query(500, ge=1, le=10000),
) -> List[ChatMessage]:
    return list_messages(
        target_key=target_key,
        target_type=target_type,
        time_from=time_from,
        time_to=time_to,
        limit=limit,
    )


@router.get("/messages/targets")
def get_message_targets(
    target_type: str = Query("all", pattern="^(self|contact|all)$"),
    limit: int = Query(1000, ge=1, le=10000),
) -> List[dict]:
    return list_message_targets(target_type)[:limit]


@router.get("/messages/diagnostics")
def get_message_diagnostics(
    target_key: str = Query("", description="Contact username or contact:username key"),
    target_type: str = Query("contact", pattern="^(self|contact|all)$"),
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
) -> dict:
    messages, diagnostics = list_messages_with_diagnostics(
        target_key=target_key,
        target_type=target_type,
        time_from=time_from,
        time_to=time_to,
        limit=1,
    )
    diagnostics["returned_sample_count"] = len(messages)
    return diagnostics


@router.get("/contacts/messages", response_model=List[ChatMessage])
def get_contact_messages(
    username: str = Query(""),
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    limit: int = Query(500, ge=1, le=10000),
) -> List[ChatMessage]:
    return list_messages(username, "contact", time_from, time_to, limit)
