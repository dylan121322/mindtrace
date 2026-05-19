import time
from threading import Lock
from typing import Any, Dict
from uuid import uuid4


_LOCK = Lock()
_TASKS: Dict[str, Dict[str, Any]] = {}


def create_task(meta: Dict[str, Any]) -> Dict[str, Any]:
    task_id = str(uuid4())
    task = {
        "task_id": task_id,
        "status": "queued",
        "stage": "queued",
        "message": "心理事实库构建任务已创建",
        "progress": 0,
        "processed": 0,
        "total": 0,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "error": "",
        "result": None,
        "cancel_requested": False,
        **meta,
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


def request_cancel(task_id: str) -> Dict[str, Any]:
    return update_task(
        task_id,
        cancel_requested=True,
        message="已请求停止心理事实库构建，等待当前批次结束",
    )


def is_cancel_requested(task_id: str) -> bool:
    with _LOCK:
        task = _TASKS.get(task_id)
        return bool(task and task.get("cancel_requested"))
