import time
from time import sleep
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
        "message": "向量数据库任务已创建",
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
        message="已请求停止向量任务，等待当前批次结束",
    )


def is_cancel_requested(task_id: str) -> bool:
    with _LOCK:
        task = _TASKS.get(task_id)
        return bool(task and task.get("cancel_requested"))


def get_active_tasks() -> list[Dict[str, Any]]:
    with _LOCK:
        return [
            task.copy()
            for task in _TASKS.values()
            if task.get("status") in ("queued", "running")
        ]


def wait_for_active_tasks(timeout_seconds: int = 3600, poll_seconds: float = 1.0) -> Dict[str, Any]:
    started = time.time()
    waited_for: list[str] = []
    while True:
        active = get_active_tasks()
        if not active:
            return {
                "waited": bool(waited_for),
                "wait_seconds": int(time.time() - started),
                "task_ids": waited_for,
                "timed_out": False,
            }
        waited_for = sorted({*waited_for, *[str(task.get("task_id", "")) for task in active if task.get("task_id")]})
        if time.time() - started >= timeout_seconds:
            return {
                "waited": True,
                "wait_seconds": int(time.time() - started),
                "task_ids": waited_for,
                "timed_out": True,
                "active_count": len(active),
            }
        sleep(poll_seconds)
