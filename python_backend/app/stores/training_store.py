import json
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.database import get_connection


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def _json_loads(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value or "")
    except (TypeError, json.JSONDecodeError):
        return fallback


def create_sample(
    *,
    name: str,
    target_key: str,
    target_type: str,
    analysis_task_id: str,
    analysis_json: Dict[str, Any],
    human_review: Dict[str, Any],
    notes: str,
) -> Dict[str, Any]:
    now = int(time.time())
    sample_id = str(uuid4())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO psych_training_samples(
                sample_id, name, target_key, target_type, analysis_task_id,
                status, analysis_json, human_review_json, notes, created_at, updated_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                sample_id,
                name,
                target_key,
                target_type,
                analysis_task_id,
                "reviewed",
                _json_dumps(analysis_json),
                _json_dumps(human_review),
                notes,
                now,
                now,
            ),
        )
    sample = get_sample(sample_id)
    return sample or {"sample_id": sample_id}


def update_sample(
    sample_id: str,
    *,
    name: Optional[str] = None,
    human_review: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    current = get_sample(sample_id)
    if not current:
        return None
    next_name = current.get("name", "") if name is None else name
    next_review = current.get("human_review", {}) if human_review is None else human_review
    next_notes = current.get("notes", "") if notes is None else notes
    now = int(time.time())
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE psych_training_samples
            SET name = ?, human_review_json = ?, notes = ?, updated_at = ?
            WHERE sample_id = ?
            """,
            (next_name, _json_dumps(next_review), next_notes, now, sample_id),
        )
    return get_sample(sample_id)


def get_sample(sample_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM psych_training_samples WHERE sample_id = ?",
            (sample_id,),
        ).fetchone()
    if not row:
        return None
    return _row_to_sample(dict(row))


def list_samples(
    limit: int = 100,
    reviewed_only: bool = False,
    target_type: str = "",
    target_key: str = "",
) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit or 100), 5000))
    clauses: List[str] = []
    params: List[Any] = []
    if reviewed_only:
        clauses.append("status = 'reviewed'")
    target_type = (target_type or "").strip()
    target_key = (target_key or "").strip()
    if target_type and target_type != "all":
        clauses.append("target_type = ?")
        params.append(target_type)
    if target_key and target_type not in ("", "all", "self"):
        clauses.append("target_key = ?")
        params.append(target_key)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM psych_training_samples
            {where}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
    return [_row_to_sample(dict(row)) for row in rows]


def delete_sample(sample_id: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM psych_training_samples WHERE sample_id = ?", (sample_id,))
    return cur.rowcount > 0


def create_proposal(
    *,
    sample_count: int,
    model_provider: str,
    model_name: str,
    summary: str,
    suggestions: List[Dict[str, Any]],
    proposed_config: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    now = int(time.time())
    proposal_id = str(uuid4())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO psych_training_proposals(
                proposal_id, status, sample_count, model_provider, model_name,
                summary, suggestions_json, proposed_config_json, diagnostics_json,
                created_at, applied_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,0)
            """,
            (
                proposal_id,
                "proposed",
                sample_count,
                model_provider,
                model_name,
                summary,
                _json_dumps(suggestions),
                _json_dumps(proposed_config),
                _json_dumps(diagnostics),
                now,
            ),
        )
    proposal = get_proposal(proposal_id)
    return proposal or {"proposal_id": proposal_id}


def get_proposal(proposal_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM psych_training_proposals WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
    if not row:
        return None
    return _row_to_proposal(dict(row))


def list_proposals(limit: int = 50) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit or 50), 200))
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM psych_training_proposals
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_row_to_proposal(dict(row)) for row in rows]


def mark_proposal_applied(proposal_id: str) -> Optional[Dict[str, Any]]:
    now = int(time.time())
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE psych_training_proposals
            SET status = 'applied', applied_at = ?
            WHERE proposal_id = ?
            """,
            (now, proposal_id),
        )
    return get_proposal(proposal_id)


def _row_to_sample(row: Dict[str, Any]) -> Dict[str, Any]:
    row["analysis_json"] = _json_loads(row.get("analysis_json", "{}"), {})
    row["human_review"] = _json_loads(row.get("human_review_json", "{}"), {})
    row.pop("human_review_json", None)
    return row


def _row_to_proposal(row: Dict[str, Any]) -> Dict[str, Any]:
    row["suggestions"] = _json_loads(row.get("suggestions_json", "[]"), [])
    row["proposed_config"] = _json_loads(row.get("proposed_config_json", "{}"), {})
    row["diagnostics"] = _json_loads(row.get("diagnostics_json", "{}"), {})
    row.pop("suggestions_json", None)
    row.pop("proposed_config_json", None)
    row.pop("diagnostics_json", None)
    return row
