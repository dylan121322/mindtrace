import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

from app.config import BASE_DIR, reset_settings_cache


ENV_PATH = BASE_DIR / ".env"

ENV_KEYS = [
    "DATA_DIR",
    "AI_DB_PATH",
    "VECTOR_DB_PATH",
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_API_KEY",
    "PSYCH_FACT_LLM_PROVIDER",
    "PSYCH_FACT_LLM_BASE_URL",
    "PSYCH_FACT_LLM_MODEL",
    "PSYCH_FACT_LLM_API_KEY",
    "PSYCH_FACT_CHUNK_SIZE",
    "TRAINING_AUTO_REVIEW_ENABLED",
    "TRAINING_AUTO_REVIEW_USE_LLM",
    "TRAINING_AUTO_PROPOSAL_ENABLED",
    "TRAINING_AUTO_MAX_SAMPLES",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_MODEL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_USE_SEARCH_PREFIX",
    "EMBEDDING_DOCUMENT_PREFIX",
    "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_TIMEOUT",
    "EMBEDDING_PREPROCESS_WORKERS",
    "EMBEDDING_MAX_BATCH_SIZE",
    "EMBEDDING_MAX_BATCH_TOKENS",
    "EMBEDDING_HTTP_RETRIES",
    "EMBEDDING_RETRY_BACKOFF",
    "EMBEDDING_MAX_FAILED_ITEMS",
    "EMBEDDING_MAX_FAILED_RATIO",
    "HOST",
    "PORT",
    "TIMEZONE",
]

SECRET_KEYS = {"LLM_API_KEY", "PSYCH_FACT_LLM_API_KEY", "EMBEDDING_API_KEY"}
SECRET_PLACEHOLDER = "__HAS_KEY__"
DEFAULT_ENV_VALUES = {
    "VECTOR_DB_PATH": str(BASE_DIR / "vector_index.db"),
    "LLM_PROVIDER": "ollama",
    "LLM_BASE_URL": "http://localhost:11434/v1",
    "LLM_MODEL": "qwen2.5:7b",
    "PSYCH_FACT_LLM_PROVIDER": "ollama",
    "PSYCH_FACT_LLM_BASE_URL": "http://localhost:11434/v1",
    "PSYCH_FACT_LLM_MODEL": "qwen2.5:7b",
    "PSYCH_FACT_CHUNK_SIZE": "80",
    "TRAINING_AUTO_REVIEW_ENABLED": "false",
    "TRAINING_AUTO_REVIEW_USE_LLM": "true",
    "TRAINING_AUTO_PROPOSAL_ENABLED": "true",
    "TRAINING_AUTO_MAX_SAMPLES": "1",
    "EMBEDDING_PROVIDER": "ollama",
    "EMBEDDING_BASE_URL": "http://localhost:11434/v1",
    "EMBEDDING_MODEL": "nomic-embed-text",
    "EMBEDDING_USE_SEARCH_PREFIX": "auto",
    "EMBEDDING_DOCUMENT_PREFIX": "search_document: ",
    "EMBEDDING_QUERY_PREFIX": "search_query: ",
    "EMBEDDING_BATCH_SIZE": "8",
    "EMBEDDING_TIMEOUT": "180",
    "EMBEDDING_PREPROCESS_WORKERS": "8",
    "EMBEDDING_MAX_BATCH_SIZE": "32",
    "EMBEDDING_MAX_BATCH_TOKENS": "2048",
    "EMBEDDING_HTTP_RETRIES": "2",
    "EMBEDDING_RETRY_BACKOFF": "1.5",
    "EMBEDDING_MAX_FAILED_ITEMS": "0",
    "EMBEDDING_MAX_FAILED_RATIO": "0.02",
    "HOST": "127.0.0.1",
    "PORT": "8000",
    "TIMEZONE": "Asia/Shanghai",
}

MODEL_PRESETS = {
    "deepseek-v4-pro": {
        "label": "DeepSeek V4 Pro",
        "llm_provider": "deepseek",
        "llm_base_url": "https://api.deepseek.com/v1",
        "llm_model": "deepseek-v4-pro",
    },
    "deepseek-v4-flash": {
        "label": "DeepSeek V4 Flash",
        "llm_provider": "deepseek",
        "llm_base_url": "https://api.deepseek.com/v1",
        "llm_model": "deepseek-v4-flash",
    },
    "kimi-k2.5": {
        "label": "Kimi K2.5",
        "llm_provider": "kimi",
        "llm_base_url": "https://api.moonshot.cn/v1",
        "llm_model": "kimi-k2.5",
    },
    "kimi-k2-thinking": {
        "label": "Kimi K2 Thinking",
        "llm_provider": "kimi",
        "llm_base_url": "https://api.moonshot.cn/v1",
        "llm_model": "kimi-k2-thinking",
    },
}


def _parse_env_line(line: str) -> Tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    return key.strip(), value.strip()


def read_env_file() -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not ENV_PATH.exists():
        return data
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if not parsed:
            continue
        key, value = parsed
        if key in ENV_KEYS:
            data[key] = value
    return data


def public_env() -> Dict:
    values = read_env_file()
    for key in ENV_KEYS:
        values.setdefault(key, os.getenv(key, DEFAULT_ENV_VALUES.get(key, "")))
    for key, value in values.items():
        if key in ENV_KEYS:
            os.environ[key] = value
    reset_settings_cache()
    masked = dict(values)
    has_secret = {}
    for key in SECRET_KEYS:
        has_secret[key] = bool(values.get(key))
        if values.get(key):
            masked[key] = SECRET_PLACEHOLDER
    return {
        "env_path": str(ENV_PATH),
        "values": masked,
        "has_secret": has_secret,
        "presets": MODEL_PRESETS,
    }


def _env_quote(value: str) -> str:
    value = value.replace("\r", "").replace("\n", "")
    if value.startswith(" ") or value.endswith(" ") or "#" in value:
        return '"' + value.replace('"', '\\"') + '"'
    return value


def save_env_file(updates: Dict[str, str]) -> Dict:
    current = read_env_file()
    next_values = {key: current.get(key, os.getenv(key, DEFAULT_ENV_VALUES.get(key, ""))) for key in ENV_KEYS}
    for key, raw_value in updates.items():
        if key not in ENV_KEYS:
            continue
        value = "" if raw_value is None else str(raw_value)
        if key in SECRET_KEYS and value == SECRET_PLACEHOLDER:
            continue
        next_values[key] = value.replace("\r", "").replace("\n", "")

    lines = []
    for key in ENV_KEYS:
        if key in ("LLM_PROVIDER", "PSYCH_FACT_LLM_PROVIDER", "EMBEDDING_PROVIDER", "HOST"):
            lines.append("")
        lines.append(f"{key}={_env_quote(next_values.get(key, ''))}")
    ENV_PATH.write_text("\n".join(lines).lstrip() + "\n", encoding="utf-8")

    # Make the running process pick up the saved values immediately. This keeps
    # the settings window useful without forcing a backend restart.
    for key, value in next_values.items():
        os.environ[key] = value
    reset_settings_cache()
    return public_env()
