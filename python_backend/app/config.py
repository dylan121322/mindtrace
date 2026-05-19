import os
from pathlib import Path
from functools import lru_cache

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency guard
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"


if load_dotenv:
    load_dotenv(ENV_PATH, override=True)


def _read_env_file_values() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    values: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1].replace('\\"', '"')
        values[key.strip()] = value
    return values


def _env(name: str, default: str = "") -> str:
    file_values = _read_env_file_values()
    if name in file_values:
        return file_values[name]
    value = os.getenv(name)
    if value is None:
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = _env(name, str(default)).strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = _env(name, str(default)).strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name, "true" if default else "false").strip().lower()
    if value in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if value in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    return default


def _env_int_or_max(name: str, default: int) -> int | str:
    raw = _env(name, str(default)).strip().lower()
    if raw == "max":
        return "max"
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _looks_like_data_dir(path: Path) -> bool:
    return (path / "contact" / "contact.db").exists() and (path / "message").exists()


def _resolve_data_dir(raw: str) -> Path:
    path = Path(raw).expanduser()
    if _looks_like_data_dir(path):
        return path
    candidates = [
        BASE_DIR.parent / "decrypted",
        BASE_DIR.parent.parent / "decrypted",
        BASE_DIR.parent.parent.parent / "decrypted",
    ]
    for candidate in candidates:
        if _looks_like_data_dir(candidate):
            return candidate
    return path


class Settings:
    def __init__(self) -> None:
        data_dir = _env("DATA_DIR") or _env("WELINK_DATA_DIR") or str(BASE_DIR.parent / "decrypted")
        self.data_dir = _resolve_data_dir(data_dir)
        ai_db = _env("AI_DB_PATH") or str(BASE_DIR / "ai_analysis.db")
        self.ai_db_path = Path(ai_db).expanduser()
        vector_db = _env("VECTOR_DB_PATH") or str(BASE_DIR / "vector_index.db")
        self.vector_db_path = Path(vector_db).expanduser()

        self.llm_provider = _env("LLM_PROVIDER", "ollama")
        self.llm_base_url = _env("LLM_BASE_URL", "http://localhost:11434/v1")
        self.llm_model = _env("LLM_MODEL", "qwen2.5:7b")
        self.llm_api_key = _env("LLM_API_KEY")
        self.psych_fact_llm_provider = _env("PSYCH_FACT_LLM_PROVIDER", self.llm_provider)
        self.psych_fact_llm_base_url = _env("PSYCH_FACT_LLM_BASE_URL", self.llm_base_url)
        self.psych_fact_llm_model = _env("PSYCH_FACT_LLM_MODEL", self.llm_model)
        self.psych_fact_llm_api_key = _env("PSYCH_FACT_LLM_API_KEY", self.llm_api_key)
        self.psych_fact_chunk_size = max(1, _env_int("PSYCH_FACT_CHUNK_SIZE", 80))
        self.training_auto_review_enabled = _env_bool("TRAINING_AUTO_REVIEW_ENABLED", False)
        self.training_auto_review_use_llm = _env_bool("TRAINING_AUTO_REVIEW_USE_LLM", True)
        self.training_auto_proposal_enabled = _env_bool("TRAINING_AUTO_PROPOSAL_ENABLED", True)
        self.training_auto_max_samples = _env_int_or_max("TRAINING_AUTO_MAX_SAMPLES", 1)

        self.embedding_provider = _env("EMBEDDING_PROVIDER", "ollama")
        self.embedding_base_url = _env("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
        self.embedding_model = _env("EMBEDDING_MODEL", "nomic-embed-text")
        self.embedding_api_key = _env("EMBEDDING_API_KEY")
        self.embedding_use_search_prefix = _env("EMBEDDING_USE_SEARCH_PREFIX", "auto")
        self.embedding_document_prefix = _env("EMBEDDING_DOCUMENT_PREFIX", "search_document: ")
        self.embedding_query_prefix = _env("EMBEDDING_QUERY_PREFIX", "search_query: ")
        self.embedding_batch_size = _env_int("EMBEDDING_BATCH_SIZE", 8)
        self.embedding_timeout = _env_int("EMBEDDING_TIMEOUT", 180)
        self.embedding_preprocess_workers = max(
            1,
            _env_int("EMBEDDING_PREPROCESS_WORKERS", min(8, os.cpu_count() or 1)),
        )
        self.embedding_max_batch_size = max(1, _env_int("EMBEDDING_MAX_BATCH_SIZE", 32))
        self.embedding_max_batch_tokens = max(1, _env_int("EMBEDDING_MAX_BATCH_TOKENS", 2048))
        self.embedding_http_retries = max(0, _env_int("EMBEDDING_HTTP_RETRIES", 2))
        self.embedding_retry_backoff = max(0.1, _env_float("EMBEDDING_RETRY_BACKOFF", 1.5))
        self.embedding_max_failed_items = max(0, _env_int("EMBEDDING_MAX_FAILED_ITEMS", 0))
        self.embedding_max_failed_ratio = max(0.0, _env_float("EMBEDDING_MAX_FAILED_RATIO", 0.02))

        self.host = _env("HOST", "127.0.0.1")
        self.port = _env_int("PORT", 8000)
        self.timezone = _env("TIMEZONE", "Asia/Shanghai")

    def safe_public_dict(self) -> dict:
        return {
            "data_dir_configured": bool(str(self.data_dir)),
            "ai_db_configured": bool(str(self.ai_db_path)),
            "vector_db_configured": bool(str(self.vector_db_path)),
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "psych_fact_llm_provider": self.psych_fact_llm_provider,
            "psych_fact_llm_model": self.psych_fact_llm_model,
            "psych_fact_chunk_size": self.psych_fact_chunk_size,
            "training_auto_review_enabled": self.training_auto_review_enabled,
            "training_auto_review_use_llm": self.training_auto_review_use_llm,
            "training_auto_proposal_enabled": self.training_auto_proposal_enabled,
            "training_auto_max_samples": self.training_auto_max_samples,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_use_search_prefix": self.embedding_use_search_prefix,
            "embedding_document_prefix": self.embedding_document_prefix,
            "embedding_query_prefix": self.embedding_query_prefix,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_timeout": self.embedding_timeout,
            "embedding_preprocess_workers": self.embedding_preprocess_workers,
            "embedding_max_batch_size": self.embedding_max_batch_size,
            "embedding_max_batch_tokens": self.embedding_max_batch_tokens,
            "embedding_http_retries": self.embedding_http_retries,
            "embedding_retry_backoff": self.embedding_retry_backoff,
            "embedding_max_failed_items": self.embedding_max_failed_items,
            "embedding_max_failed_ratio": self.embedding_max_failed_ratio,
            "timezone": self.timezone,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
