import hashlib
import re
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from app.config import get_settings


DEFAULT_DOCUMENT_PREFIX = "search_document: "
DEFAULT_QUERY_PREFIX = "search_query: "
SEARCH_PREFIX_MARKERS = ("search_document:", "search_query:")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _headers(api_key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def _ollama_native_base_url(base_url: str) -> str:
    value = (base_url or "").rstrip("/")
    lower = value.lower()
    if lower.endswith("/v1"):
        return value[:-3].rstrip("/")
    return value


def _cfg_value(cfg: dict, *keys: str) -> Any:
    for key in keys:
        if key in cfg and cfg[key] not in (None, ""):
            return cfg[key]
    return None


def _mode_enabled(mode: Any, model: str, provider: str) -> bool:
    text = str(mode if mode is not None else "auto").strip().lower()
    if text in {"0", "false", "no", "off", "none", "disabled", "disable"}:
        return False
    if text in {"1", "true", "yes", "on", "enabled", "enable", "always"}:
        return True
    if provider.lower() in {"local", "hash", "none"}:
        return False
    model_lower = (model or "").lower()
    return "nomic" in model_lower and "embed" in model_lower


def _normalize_prefix(value: Any, default: str) -> str:
    prefix = str(value if value not in (None, "") else default)
    if prefix and not prefix.endswith((" ", "\n", "\t")):
        prefix = prefix + " "
    return prefix


def _input_role(input_type: str) -> str:
    value = (input_type or "document").strip().lower()
    if value in {"query", "search_query", "q", "question"}:
        return "query"
    return "document"


def get_embedding_profile(config: Optional[dict] = None) -> Dict[str, Any]:
    settings = get_settings()
    cfg = config or {}
    provider = (cfg.get("provider") or settings.embedding_provider or "ollama").lower()
    base_url = cfg.get("base_url") or settings.embedding_base_url
    model = cfg.get("model") or settings.embedding_model
    api_key = cfg.get("api_key") or settings.embedding_api_key
    mode = _cfg_value(cfg, "use_search_prefix", "use_nomic_search_prefix", "search_prefix_mode")
    if mode is None:
        mode = settings.embedding_use_search_prefix
    uses_search_prefix = _mode_enabled(mode, model, provider)
    document_prefix = _normalize_prefix(
        _cfg_value(cfg, "document_prefix", "search_document_prefix", "embedding_document_prefix")
        or settings.embedding_document_prefix,
        DEFAULT_DOCUMENT_PREFIX,
    )
    query_prefix = _normalize_prefix(
        _cfg_value(cfg, "query_prefix", "search_query_prefix", "embedding_query_prefix")
        or settings.embedding_query_prefix,
        DEFAULT_QUERY_PREFIX,
    )
    model_identity = model
    if uses_search_prefix:
        prefix_hash = hashlib.sha1(f"{document_prefix}|{query_prefix}".encode("utf-8")).hexdigest()[:8]
        model_identity = f"{model}|search_prefix:{prefix_hash}"
    return {
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "model_identity": model_identity,
        "api_key": api_key,
        "uses_search_prefix": uses_search_prefix,
        "search_prefix_mode": str(mode or "auto"),
        "document_prefix": document_prefix,
        "query_prefix": query_prefix,
    }


def format_embedding_input(text: str, input_type: str = "document", config: Optional[dict] = None) -> str:
    profile = get_embedding_profile(config)
    return _apply_search_prefix(str(text), _input_role(input_type), profile)


def _apply_search_prefix(text: str, role: str, profile: Dict[str, Any]) -> str:
    text = CONTROL_RE.sub(" ", text).strip()
    if not profile.get("uses_search_prefix"):
        return text
    if text.lstrip().startswith(SEARCH_PREFIX_MARKERS):
        return text
    prefix = profile["query_prefix"] if role == "query" else profile["document_prefix"]
    return f"{prefix}{text}"


def local_hash_embedding(text: str, dims: int = 256) -> List[float]:
    vec = np.zeros(dims, dtype=np.float32)
    for token in text:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "little") % dims
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec.astype(np.float32).tolist()


def get_embeddings(
    texts: Iterable[str],
    config: Optional[dict] = None,
    input_type: str = "document",
) -> List[List[float]]:
    settings = get_settings()
    cfg = config or {}
    profile = get_embedding_profile(cfg)
    provider = profile["provider"]
    base_url = profile["base_url"]
    model = profile["model"]
    api_key = profile["api_key"]
    batch_size = max(1, int(cfg.get("batch_size") or settings.embedding_batch_size or 8))
    timeout = max(1, int(cfg.get("timeout") or settings.embedding_timeout or 180))
    retries = max(0, int(cfg.get("http_retries") or settings.embedding_http_retries or 0))
    retry_backoff = max(0.1, float(cfg.get("retry_backoff") or settings.embedding_retry_backoff or 1.5))
    role = _input_role(str(_cfg_value(cfg, "input_type", "input_role") or input_type))
    inputs = [_apply_search_prefix(str(t), role, profile) for t in texts]
    if not inputs:
        return []
    if provider in ("local", "hash", "none"):
        return [local_hash_embedding(t) for t in inputs]
    out: List[List[float]] = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size]
        if provider == "ollama":
            if "/v1" in (base_url or "").lower():
                try:
                    out.extend(
                        _openai_compatible_embeddings(
                            base_url,
                            model,
                            api_key,
                            batch,
                            timeout=timeout,
                            retries=retries,
                            retry_backoff=retry_backoff,
                        )
                    )
                    continue
                except Exception as exc:
                    status = int(getattr(getattr(exc, "response", None), "status_code", 0) or 0)
                    if status not in {400, 404, 422}:
                        raise
                    out.extend(
                        _ollama_embeddings(
                            _ollama_native_base_url(base_url),
                            model,
                            batch,
                            timeout=timeout,
                            retries=retries,
                            retry_backoff=retry_backoff,
                        )
                    )
                    continue
            out.extend(
                _ollama_embeddings(
                    _ollama_native_base_url(base_url),
                    model,
                    batch,
                    timeout=timeout,
                    retries=retries,
                    retry_backoff=retry_backoff,
                )
            )
        else:
            out.extend(_openai_compatible_embeddings(base_url, model, api_key, batch, timeout=timeout, retries=retries, retry_backoff=retry_backoff))
    return out


def get_embedding(text: str, config: Optional[dict] = None, input_type: str = "document") -> List[float]:
    return get_embeddings([text], config=config, input_type=input_type)[0]


def _openai_compatible_embeddings(
    base_url: str,
    model: str,
    api_key: str,
    inputs: List[str],
    timeout: int,
    retries: int,
    retry_backoff: float,
) -> List[List[float]]:
    url = _join_url(base_url, "/embeddings")
    resp = _post_json_with_retries(
        url,
        headers=_headers(api_key),
        json={"model": model, "input": inputs},
        timeout=timeout,
        retries=retries,
        retry_backoff=retry_backoff,
    )
    resp.raise_for_status()
    payload = resp.json()
    data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
    embeddings = [item["embedding"] for item in data]
    if len(embeddings) != len(inputs):
        raise RuntimeError(f"embedding_count_mismatch:{len(embeddings)}!={len(inputs)}")
    return embeddings


def _ollama_embeddings(
    base_url: str,
    model: str,
    inputs: List[str],
    timeout: int,
    retries: int,
    retry_backoff: float,
) -> List[List[float]]:
    url = _join_url(base_url, "/api/embed")
    resp = _post_json_with_retries(
        url,
        json={"model": model, "input": inputs},
        timeout=timeout,
        retries=retries,
        retry_backoff=retry_backoff,
    )
    if resp.status_code < 400:
        payload = resp.json()
        embeddings = payload.get("embeddings")
        if embeddings and len(embeddings) == len(inputs):
            return embeddings
    out = []
    legacy_url = _join_url(base_url, "/api/embeddings")
    for text in inputs:
        legacy = _post_json_with_retries(
            legacy_url,
            json={"model": model, "prompt": text},
            timeout=timeout,
            retries=retries,
            retry_backoff=retry_backoff,
        )
        legacy.raise_for_status()
        out.append(legacy.json()["embedding"])
    return out


def _post_json_with_retries(
    url: str,
    json: dict,
    timeout: int,
    retries: int,
    retry_backoff: float,
    headers: Optional[dict] = None,
):
    import requests

    retryable_status = {429, 500, 502, 503, 504}
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=json, timeout=timeout)
            if resp.status_code not in retryable_status:
                return resp
            if attempt >= retries:
                return resp
            time.sleep(retry_backoff * (2**attempt))
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            time.sleep(retry_backoff * (2**attempt))
    if last_exc:
        raise last_exc
    raise RuntimeError("embedding_request_failed")
