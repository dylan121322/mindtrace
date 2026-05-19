import hashlib
import time
from typing import Dict, List

import numpy as np

from app.database import get_vector_connection, init_vector_db
from app.models import ChatMessage


def encode_embedding(values: List[float]) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


def decode_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def content_hash(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def message_identity(source_key: str, seq: int, datetime: str, sender: str) -> str:
    stable = f"{source_key or ''}\x1f{seq}\x1f{datetime or ''}\x1f{sender or ''}"
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _source_key(msg: ChatMessage, index_key: str) -> str:
    return msg.contact_key or index_key


def _source_id(msg: ChatMessage, index_key: str) -> str:
    return message_identity(_source_key(msg, index_key), msg.seq, msg.datetime, msg.sender)


def load_existing_vector_signatures(index_key: str) -> Dict[str, str]:
    init_vector_db()
    with get_vector_connection() as conn:
        rows = conn.execute(
            """
            SELECT source_id, content_hash
            FROM vec_messages
            WHERE index_key = ?
            """,
            (index_key,),
        ).fetchall()
    return {str(row["source_id"]): str(row["content_hash"] or "") for row in rows}


def replace_vectors(index_key: str, messages: List[ChatMessage], embeddings: List[List[float]], model: str) -> None:
    init_vector_db()
    dims = len(embeddings[0]) if embeddings else 0
    now = int(time.time())
    with get_vector_connection() as conn:
        conn.execute("DELETE FROM vec_messages WHERE index_key = ?", (index_key,))
        conn.execute("DELETE FROM vec_index_status WHERE index_key = ?", (index_key,))
        conn.executemany(
            """
            INSERT INTO vec_messages(
                index_key, source_key, source_id, seq, datetime, sender, content,
                content_hash, embedding, created_at, updated_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    index_key,
                    _source_key(msg, index_key),
                    _source_id(msg, index_key),
                    msg.seq,
                    msg.datetime,
                    msg.sender,
                    msg.content,
                    content_hash(msg.content),
                    encode_embedding(embedding),
                    now,
                    now,
                )
                for msg, embedding in zip(messages, embeddings)
            ],
        )
        conn.execute(
            """
            INSERT INTO vec_index_status(index_key, msg_count, built_at, updated_at, model, dims)
            VALUES(?,?,?,?,?,?)
            """,
            (index_key, len(messages), now, now, model, dims),
        )


def upsert_vectors(index_key: str, messages: List[ChatMessage], embeddings: List[List[float]], model: str) -> Dict:
    init_vector_db()
    dims = len(embeddings[0]) if embeddings else 0
    now = int(time.time())
    with get_vector_connection() as conn:
        if messages:
            conn.executemany(
                """
                INSERT INTO vec_messages(
                    index_key, source_key, source_id, seq, datetime, sender, content,
                    content_hash, embedding, created_at, updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(index_key, source_id) DO UPDATE SET
                    source_key = excluded.source_key,
                    seq = excluded.seq,
                    datetime = excluded.datetime,
                    sender = excluded.sender,
                    content = excluded.content,
                    content_hash = excluded.content_hash,
                    embedding = excluded.embedding,
                    updated_at = excluded.updated_at
                """,
                [
                    (
                        index_key,
                        _source_key(msg, index_key),
                        _source_id(msg, index_key),
                        msg.seq,
                        msg.datetime,
                        msg.sender,
                        msg.content,
                        content_hash(msg.content),
                        encode_embedding(embedding),
                        now,
                        now,
                    )
                    for msg, embedding in zip(messages, embeddings)
                ],
            )
        row = conn.execute(
            "SELECT COUNT(*) AS msg_count FROM vec_messages WHERE index_key = ?",
            (index_key,),
        ).fetchone()
        msg_count = int(row["msg_count"] if row else 0)
        old = conn.execute(
            "SELECT dims FROM vec_index_status WHERE index_key = ?",
            (index_key,),
        ).fetchone()
        if dims <= 0 and old:
            dims = int(old["dims"] or 0)
        conn.execute("DELETE FROM vec_index_status WHERE index_key = ?", (index_key,))
        conn.execute(
            """
            INSERT INTO vec_index_status(index_key, msg_count, built_at, updated_at, model, dims)
            VALUES(?,?,?,?,?,?)
            """,
            (index_key, msg_count, now, now, model, dims),
        )
    return get_index_status(index_key)


def load_vectors(index_key: str) -> List[Dict]:
    init_vector_db()
    with get_vector_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, index_key, source_key, source_id, seq, datetime, sender, content, content_hash, embedding
            FROM vec_messages WHERE index_key = ?
            ORDER BY seq ASC, id ASC
            """,
            (index_key,),
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["contact_key"] = item.pop("index_key")
        item["embedding"] = decode_embedding(row["embedding"])
        out.append(item)
    return out


def get_context(index_key: str, seq: int, window: int = 2, source_key: str = "") -> List[Dict]:
    init_vector_db()
    with get_vector_connection() as conn:
        if source_key:
            rows = conn.execute(
                """
                SELECT seq, datetime, sender, content, source_key
                FROM vec_messages
                WHERE index_key = ? AND source_key = ? AND seq BETWEEN ? AND ?
                ORDER BY seq ASC, id ASC
                """,
                (index_key, source_key, seq - window, seq + window),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT seq, datetime, sender, content, source_key
                FROM vec_messages
                WHERE index_key = ? AND seq BETWEEN ? AND ?
                ORDER BY seq ASC, id ASC
                """,
                (index_key, seq - window, seq + window),
            ).fetchall()
    return [dict(row) for row in rows]


def get_index_status(index_key: str) -> Dict:
    init_vector_db()
    with get_vector_connection() as conn:
        row = conn.execute(
            "SELECT index_key, msg_count, built_at, updated_at, model, dims FROM vec_index_status WHERE index_key = ?",
            (index_key,),
        ).fetchone()
        count_row = conn.execute(
            "SELECT COUNT(*) AS actual_vector_count FROM vec_messages WHERE index_key = ?",
            (index_key,),
        ).fetchone()
    actual_vector_count = int(count_row["actual_vector_count"] if count_row else 0)
    if not row:
        return {
            "built": False,
            "contact_key": index_key,
            "index_key": index_key,
            "msg_count": 0,
            "actual_vector_count": actual_vector_count,
            "built_at": 0,
            "updated_at": 0,
            "model": "",
            "dims": 0,
        }
    data = dict(row)
    data["contact_key"] = data.get("index_key")
    data["actual_vector_count"] = actual_vector_count
    data["built"] = (
        bool(data.get("built_at"))
        and int(data.get("msg_count") or 0) > 0
        and int(data.get("dims") or 0) > 0
        and actual_vector_count > 0
    )
    return data
