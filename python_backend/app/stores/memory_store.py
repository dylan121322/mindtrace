import json
import time
from typing import Dict, List

from app.database import get_connection, init_ai_db
from app.stores.vector_store import decode_embedding, encode_embedding


def replace_facts(
    contact_key: str,
    facts: List[Dict],
    source_kind: str = "memory",
    source_model: str = "",
) -> None:
    init_ai_db()
    now = int(time.time())
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM mem_facts WHERE contact_key = ? AND source_kind = ?",
            (contact_key, source_kind),
        )
        conn.executemany(
            """
            INSERT INTO mem_facts(
                contact_key, fact, source_from, source_to, embedding, pinned,
                created_at, updated_at, fact_type, severity, confidence,
                evidence_json, source_kind, source_model
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    contact_key,
                    item["fact"],
                    int(item.get("source_from") or 0),
                    int(item.get("source_to") or 0),
                    encode_embedding(item.get("embedding") or []),
                    int(item.get("pinned") or 0),
                    now,
                    now,
                    str(item.get("fact_type") or source_kind),
                    str(item.get("severity") or "low"),
                    float(item.get("confidence") or 0),
                    json.dumps(item.get("evidence") or [], ensure_ascii=False),
                    source_kind,
                    str(item.get("source_model") or source_model or ""),
                )
                for item in facts
            ],
        )


def list_facts(contact_key: str, source_kind: str | None = "memory") -> List[Dict]:
    init_ai_db()
    where = "contact_key = ?"
    params: list = [contact_key]
    if source_kind:
        where += " AND source_kind = ?"
        params.append(source_kind)
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT id, contact_key, fact, source_from, source_to, embedding, pinned,
                   created_at, updated_at, fact_type, severity, confidence,
                   evidence_json, source_kind, source_model
            FROM mem_facts WHERE {where}
            ORDER BY pinned DESC, id ASC
            """,
            params,
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["embedding_array"] = decode_embedding(row["embedding"]) if row["embedding"] else None
        try:
            item["evidence"] = json.loads(row["evidence_json"] or "[]")
        except json.JSONDecodeError:
            item["evidence"] = []
        item.pop("embedding", None)
        item.pop("evidence_json", None)
        out.append(item)
    return out


def count_facts(contact_key: str, source_kind: str = "psych") -> Dict:
    init_ai_db()
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS fact_count, MAX(updated_at) AS updated_at
            FROM mem_facts
            WHERE contact_key = ? AND source_kind = ?
            """,
            (contact_key, source_kind),
        ).fetchone()
    return {
        "contact_key": contact_key,
        "source_kind": source_kind,
        "fact_count": int(row["fact_count"] or 0) if row else 0,
        "updated_at": int(row["updated_at"] or 0) if row and row["updated_at"] else 0,
    }
