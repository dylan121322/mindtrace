from typing import Dict, List, Optional
from urllib.parse import urlparse

from app.config import get_settings


class LLMServiceError(RuntimeError):
    """Safe LLM error for API responses and process metrics."""


class LLMHTTPError(LLMServiceError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def _endpoint_url(base_url: str, endpoint: str) -> str:
    clean = (base_url or "").strip().rstrip("/")
    endpoint = "/" + endpoint.strip("/")
    if clean.endswith(endpoint):
        return clean
    return _join_url(clean, endpoint)


def _headers(api_key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _safe_host(base_url: str) -> str:
    parsed = urlparse(base_url or "")
    host = parsed.netloc or parsed.path.split("/")[0]
    return host[:120] if host else "unknown"


def _safe_response_text(resp) -> str:
    text = (resp.text or "").replace("\r", " ").replace("\n", " ").strip()
    if not text:
        return ""
    return text[:300]


def _raise_for_status(resp, provider: str, model: str, endpoint: str, base_url: str) -> None:
    if resp.status_code < 400:
        return
    detail = _safe_response_text(resp)
    message = (
        f"LLM HTTP {resp.status_code} "
        f"(provider={provider or 'unknown'}, model={model or 'unknown'}, "
        f"endpoint={endpoint}, host={_safe_host(base_url)})"
    )
    if detail:
        message += f": {detail}"
    raise LLMHTTPError(message, resp.status_code)


def _post_json(url: str, *, provider: str, model: str, endpoint: str, base_url: str, **kwargs):
    import requests

    try:
        resp = requests.post(url, **kwargs)
    except requests.Timeout as exc:
        raise LLMServiceError(
            f"LLM timeout (provider={provider}, model={model}, endpoint={endpoint}, host={_safe_host(base_url)})"
        ) from exc
    except requests.ConnectionError as exc:
        raise LLMServiceError(
            f"LLM connection failed (provider={provider}, model={model}, endpoint={endpoint}, host={_safe_host(base_url)})"
        ) from exc
    except requests.RequestException as exc:
        raise LLMServiceError(
            f"LLM request failed (provider={provider}, model={model}, endpoint={endpoint}, error={exc.__class__.__name__})"
        ) from exc
    _raise_for_status(resp, provider, model, endpoint, base_url)
    return resp


def _openai_compatible_chat(
    provider: str,
    base_url: str,
    model: str,
    api_key: str,
    messages: List[Dict[str, str]],
) -> str:
    endpoint = "/chat/completions"
    resp = _post_json(
        _endpoint_url(base_url, endpoint),
        provider=provider,
        model=model,
        endpoint=endpoint,
        base_url=base_url,
        headers=_headers(api_key),
        json={"model": model, "messages": messages, "stream": False},
        timeout=120,
    )
    payload = resp.json()
    choices = payload.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "")


def _ollama_native_chat(base_url: str, model: str, messages: List[Dict[str, str]]) -> str:
    endpoint = "/api/chat"
    resp = _post_json(
        _endpoint_url(base_url, endpoint),
        provider="ollama",
        model=model,
        endpoint=endpoint,
        base_url=base_url,
        json={"model": model, "messages": messages, "stream": False},
        timeout=120,
    )
    payload = resp.json()
    return payload.get("message", {}).get("content", "")


def _ollama_native_base(base_url: str) -> str:
    clean = (base_url or "").strip().rstrip("/")
    if clean.endswith("/v1"):
        return clean[:-3]
    return clean


def complete_chat(messages: List[Dict[str, str]], config: Optional[dict] = None) -> str:
    settings = get_settings()
    cfg = config or {}
    provider = (cfg.get("provider") or settings.llm_provider or "ollama").lower().strip()
    base_url = (cfg.get("base_url") or settings.llm_base_url or "").strip()
    model = (cfg.get("model") or settings.llm_model or "").strip()
    api_key = cfg.get("api_key") or settings.llm_api_key

    if not base_url:
        raise LLMServiceError("LLM base_url is empty")
    if not model:
        raise LLMServiceError("LLM model is empty")

    if provider == "ollama" and "/v1" not in base_url:
        return _ollama_native_chat(base_url, model, messages)

    try:
        return _openai_compatible_chat(provider, base_url, model, api_key, messages)
    except LLMHTTPError as error:
        if provider == "ollama" and error.status_code in {404, 405}:
            return _ollama_native_chat(_ollama_native_base(base_url), model, messages)
        raise
