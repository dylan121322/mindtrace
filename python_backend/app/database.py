import sqlite3
from pathlib import Path

from app.config import get_settings


def get_connection() -> sqlite3.Connection:
    settings = get_settings()
    settings.ai_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.ai_db_path))
    conn.row_factory = sqlite3.Row
    return conn


def get_vector_connection() -> sqlite3.Connection:
    settings = get_settings()
    settings.vector_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.vector_db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_ai_db() -> None:
    settings = get_settings()
    settings.ai_db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS mem_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contact_key TEXT NOT NULL,
                fact TEXT NOT NULL,
                source_from INTEGER NOT NULL DEFAULT 0,
                source_to INTEGER NOT NULL DEFAULT 0,
                embedding BLOB NOT NULL,
                pinned INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_mem_facts_contact
                ON mem_facts(contact_key);

            CREATE TABLE IF NOT EXISTS psych_tasks (
                task_id TEXT PRIMARY KEY,
                target_key TEXT NOT NULL DEFAULT '',
                target_type TEXT NOT NULL DEFAULT 'contact',
                status TEXT NOT NULL DEFAULT 'completed',
                created_at INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL DEFAULT 0,
                options_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS psych_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                group_name TEXT NOT NULL,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                window_start TEXT,
                window_end TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_psych_features_task
                ON psych_features(task_id);

            CREATE TABLE IF NOT EXISTS psych_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                fact TEXT NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_json TEXT NOT NULL DEFAULT '[]',
                source_from INTEGER,
                source_to INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_psych_facts_task
                ON psych_facts(task_id);

            CREATE TABLE IF NOT EXISTS psych_scores (
                task_id TEXT PRIMARY KEY,
                depression_signal_score INTEGER NOT NULL,
                self_harm_risk TEXT NOT NULL,
                overall_risk TEXT NOT NULL,
                confidence REAL NOT NULL,
                summary TEXT NOT NULL,
                main_signals_json TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS psych_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                datetime TEXT NOT NULL,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                reason TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_psych_evidence_task
                ON psych_evidence(task_id);

            CREATE TABLE IF NOT EXISTS psych_reports (
                task_id TEXT PRIMARY KEY,
                report_md TEXT NOT NULL,
                report_json TEXT NOT NULL DEFAULT '{}',
                created_at INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS psych_training_samples (
                sample_id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                target_key TEXT NOT NULL DEFAULT '',
                target_type TEXT NOT NULL DEFAULT '',
                analysis_task_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'reviewed',
                analysis_json TEXT NOT NULL DEFAULT '{}',
                human_review_json TEXT NOT NULL DEFAULT '{}',
                notes TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_psych_training_samples_updated
                ON psych_training_samples(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_psych_training_samples_target
                ON psych_training_samples(target_type, target_key);

            CREATE TABLE IF NOT EXISTS psych_training_proposals (
                proposal_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'proposed',
                sample_count INTEGER NOT NULL DEFAULT 0,
                model_provider TEXT NOT NULL DEFAULT '',
                model_name TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                suggestions_json TEXT NOT NULL DEFAULT '[]',
                proposed_config_json TEXT NOT NULL DEFAULT '{}',
                diagnostics_json TEXT NOT NULL DEFAULT '{}',
                created_at INTEGER NOT NULL DEFAULT 0,
                applied_at INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_psych_training_proposals_created
                ON psych_training_proposals(created_at DESC);
            """
        )
        _ensure_columns(
            conn,
            "mem_facts",
            {
                "fact_type": "TEXT NOT NULL DEFAULT 'memory'",
                "severity": "TEXT NOT NULL DEFAULT 'low'",
                "confidence": "REAL NOT NULL DEFAULT 0",
                "evidence_json": "TEXT NOT NULL DEFAULT '[]'",
                "source_kind": "TEXT NOT NULL DEFAULT 'memory'",
                "source_model": "TEXT NOT NULL DEFAULT ''",
            },
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mem_facts_contact_kind
                ON mem_facts(contact_key, source_kind)
            """
        )


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def init_vector_db() -> None:
    settings = get_settings()
    settings.vector_db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_vector_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vec_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_key TEXT NOT NULL,
                source_key TEXT NOT NULL DEFAULT '',
                source_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                datetime TEXT NOT NULL,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL DEFAULT 0,
                UNIQUE(index_key, source_id)
            );
            CREATE INDEX IF NOT EXISTS idx_vec_messages_index
                ON vec_messages(index_key);
            CREATE INDEX IF NOT EXISTS idx_vec_messages_index_seq
                ON vec_messages(index_key, seq);
            CREATE INDEX IF NOT EXISTS idx_vec_messages_index_source
                ON vec_messages(index_key, source_id);

            CREATE TABLE IF NOT EXISTS vec_index_status (
                index_key TEXT PRIMARY KEY,
                msg_count INTEGER NOT NULL DEFAULT 0,
                built_at INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL DEFAULT 0,
                model TEXT NOT NULL DEFAULT '',
                dims INTEGER NOT NULL DEFAULT 0
            );
            """
        )
