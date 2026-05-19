import time
from threading import Lock
from typing import Any, Dict, Optional
from uuid import uuid4


_LOCK = Lock()
_TASKS: Dict[str, Dict[str, Any]] = {}


def create_task(meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    task_id = str(uuid4())
    now = int(time.time())
    task = {
        "task_id": task_id,
        "status": "queued",
        "stage": "queued",
        "message": "草案生成任务已创建",
        "progress": 0,
        "created_at": now,
        "updated_at": now,
        "error": "",
        "result": None,
        **(meta or {}),
    }
    with _LOCK:
        _TASKS[task_id] = task
    return task.copy()


def update_task(task_id: str, **updates: Any) -> Dict[str, Any]:
    with _LOCK:
        task = _TASKS.get(task_id)
        if not task:
            return {}
        task.update(updates)
        task["updated_at"] = int(time.time())
        return task.copy()


def get_task(task_id: str) -> Dict[str, Any]:
    with _LOCK:
        task = _TASKS.get(task_id)
        return task.copy() if task else {}
