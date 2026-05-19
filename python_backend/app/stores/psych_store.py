import json
import time
from typing import Dict, List, Optional

from app.database import get_connection
from app.models import PsychAnalyzeResponse, PsychEvidence, PsychScore


def save_response(response: PsychAnalyzeResponse, target_key: str, target_type: str, options: Dict) -> None:
    now = int(time.time())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO psych_tasks(task_id, target_key, target_type, status, created_at, updated_at, options_json)
            VALUES(?,?,?,?,?,?,?)
            """,
            (response.task_id, target_key or "", target_type, response.status, now, now, json.dumps(options, ensure_ascii=False)),
        )
        for table in ("psych_features", "psych_facts", "psych_evidence"):
            conn.execute(f"DELETE FROM {table} WHERE task_id = ?", (response.task_id,))
        conn.execute("DELETE FROM psych_scores WHERE task_id = ?", (response.task_id,))
        conn.execute("DELETE FROM psych_reports WHERE task_id = ?", (response.task_id,))
        conn.executemany(
            """
            INSERT INTO psych_features(task_id, group_name, name, value, window_start, window_end)
            VALUES(?,?,?,?,?,?)
            """,
            [
                (response.task_id, f.group, f.name, f.value, f.window_start, f.window_end)
                for f in response.features
            ],
        )
        conn.executemany(
            """
            INSERT INTO psych_evidence(task_id, seq, datetime, sender, content, evidence_type, severity, reason)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            [
                (response.task_id, e.seq, e.datetime, e.sender, e.content, e.evidence_type, e.severity, e.reason)
                for e in response.evidences
            ],
        )
        conn.executemany(
            """
            INSERT INTO psych_facts(task_id, fact_type, fact, severity, confidence, evidence_json, source_from, source_to)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            [
                (
                    response.task_id,
                    f.fact_type,
                    f.fact,
                    f.severity,
                    f.confidence,
                    json.dumps([e.dict() for e in f.evidence], ensure_ascii=False),
                    f.source_from,
                    f.source_to,
                )
                for f in response.facts
            ],
        )
        conn.execute(
            """
            INSERT INTO psych_scores(task_id, depression_signal_score, self_harm_risk, overall_risk, confidence, summary, main_signals_json)
            VALUES(?,?,?,?,?,?,?)
            """,
            (
                response.task_id,
                response.score.depression_signal_score,
                response.score.self_harm_risk,
                response.score.overall_risk,
                response.score.confidence,
                response.score.summary,
                json.dumps(response.score.main_signals, ensure_ascii=False),
            ),
        )
        conn.execute(
            """
            INSERT INTO psych_reports(task_id, report_md, report_json, created_at)
            VALUES(?,?,?,?)
            """,
            (response.task_id, response.report_md, json.dumps(response.report_json, ensure_ascii=False), now),
        )


def get_task(task_id: str) -> Optional[Dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM psych_tasks WHERE task_id = ?", (task_id,)).fetchone()
    return dict(row) if row else None


def get_score(task_id: str) -> Optional[PsychScore]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM psych_scores WHERE task_id = ?", (task_id,)).fetchone()
    if not row:
        return None
    data = dict(row)
    return PsychScore(
        depression_signal_score=data["depression_signal_score"],
        self_harm_risk=data["self_harm_risk"],
        overall_risk=data["overall_risk"],
        confidence=data["confidence"],
        summary=data["summary"],
        main_signals=json.loads(data["main_signals_json"] or "[]"),
    )


def get_report(task_id: str) -> Optional[Dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT report_md, report_json FROM psych_reports WHERE task_id = ?", (task_id,)).fetchone()
    if not row:
        return None
    return {"report_md": row["report_md"], "report_json": json.loads(row["report_json"] or "{}")}


def get_evidence(task_id: str) -> List[PsychEvidence]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT seq, datetime, sender, content, evidence_type, severity, reason
            FROM psych_evidence WHERE task_id = ?
            ORDER BY id ASC
            """,
            (task_id,),
        ).fetchall()
    return [PsychEvidence(**dict(row)) for row in rows]

