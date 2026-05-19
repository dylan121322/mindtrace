from app.database import get_connection


def ping_ai_store() -> bool:
    with get_connection() as conn:
        conn.execute("SELECT 1").fetchone()
    return True

