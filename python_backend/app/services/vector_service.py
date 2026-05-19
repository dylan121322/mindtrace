import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from app.config import get_settings
from app.models import ChatMessage
from app.services.embedding_service import get_embedding_profile, get_embeddings
from app.stores import vector_store


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]")


class VectorPreprocessError(RuntimeError):
    pass


class VectorBuildCancelled(RuntimeError):
    pass


class VectorEmbeddingError(RuntimeError):
    pass


def _safe_exception_label(error: Exception) -> str:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    reason = str(getattr(response, "reason", "") or "").strip()
    label = error.__class__.__name__
    body = ""
    if response is not None:
        try:
            body = str(getattr(response, "text", "") or "").strip()
        except Exception:
            body = ""
    body = re.sub(r"\s+", " ", body)
    if len(body) > 160:
        body = body[:160] + "..."
    detail = f"{status} {reason}".strip() if status else ""
    if body:
        detail = f"{detail}; {body}".strip("; ")
    return f"{label}({detail})" if detail else label


def _exception_status(error: Exception) -> int:
    return int(getattr(getattr(error, "response", None), "status_code", 0) or 0)


def _message_text(msg: ChatMessage) -> str:
    return f"{msg.sender}: {msg.content}"


def _estimate_token_count(text: str) -> int:
    return max(1, len(TOKEN_RE.findall(text)))


def _preprocess_vector_chunk(rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for row in rows:
        content = str(row.get("content") or "").strip()
        if not content:
            continue
        sender = str(row.get("sender") or "")
        text = f"{sender}: {content}"[:800]
        out.append(
            {
                "seq": int(row.get("seq") or 0),
                "datetime": str(row.get("datetime") or ""),
                "sender": sender,
                "content": content,
                "is_mine": bool(row.get("is_mine")),
                "contact_key": str(row.get("contact_key") or ""),
                "text": text,
                "token_count": _estimate_token_count(text),
            }
        )
    return out


def _row_identity(row: Dict) -> str:
    return vector_store.message_identity(
        str(row.get("contact_key") or ""),
        int(row.get("seq") or 0),
        str(row.get("datetime") or ""),
        str(row.get("sender") or ""),
    )


def _chunked(items: List[Dict], chunk_size: int) -> Iterable[List[Dict]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _iter_process_pool_chunks(
    rows: List[Dict],
    workers: int,
    chunk_size: int,
) -> Iterable[Tuple[int, int, List[Dict]]]:
    chunks = list(_chunked(rows, chunk_size))
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_preprocess_vector_chunk, chunk) for chunk in chunks]
            completed = 0
            for future in as_completed(futures):
                try:
                    items = future.result()
                except Exception as exc:
                    raise VectorPreprocessError("process_pool_chunk_failed") from exc
                completed += 1
                yield completed, len(futures), items
    except VectorPreprocessError:
        raise
    except Exception as exc:
        raise VectorPreprocessError("process_pool_unavailable") from exc


def _item_to_message(item: Dict) -> ChatMessage:
    return ChatMessage(
        seq=int(item.get("seq") or 0),
        datetime=str(item.get("datetime") or ""),
        sender=str(item.get("sender") or ""),
        content=str(item.get("content") or ""),
        is_mine=bool(item.get("is_mine")),
        contact_key=str(item.get("contact_key") or ""),
    )


def _extract_dynamic_batches(
    pending: List[Dict],
    max_batch_size: int,
    max_batch_tokens: int,
    force: bool,
) -> Tuple[List[List[Dict]], List[Dict]]:
    if not pending:
        return [], []
    if not force and len(pending) < max_batch_size:
        return [], pending

    sorted_items = sorted(pending, key=lambda item: int(item.get("token_count") or 1))
    batches: List[List[Dict]] = []
    batch: List[Dict] = []
    batch_max_tokens = 0

    def projected_cost(item: Dict) -> int:
        next_max = max(batch_max_tokens, int(item.get("token_count") or 1))
        return next_max * (len(batch) + 1)

    for item in sorted_items:
        would_exceed_size = len(batch) >= max_batch_size
        would_exceed_tokens = bool(batch) and projected_cost(item) > max_batch_tokens
        if would_exceed_size or would_exceed_tokens:
            batches.append(batch)
            batch = []
            batch_max_tokens = 0
        batch.append(item)
        batch_max_tokens = max(batch_max_tokens, int(item.get("token_count") or 1))

    if batch:
        if force or len(batch) >= max_batch_size or batch_max_tokens * len(batch) >= max_batch_tokens:
            batches.append(batch)
            batch = []

    return batches, batch


def build_vector_index(
    target_key: str,
    messages: List[ChatMessage],
    embedding_config: Optional[dict] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> Dict:
    source_rows = [
        {
            "seq": msg.seq,
            "datetime": msg.datetime,
            "sender": msg.sender,
            "content": msg.content,
            "is_mine": msg.is_mine,
            "contact_key": msg.contact_key,
        }
        for msg in messages
        if (msg.content or "").strip()
    ]
    if not source_rows:
        return {"built": False, "msg_count": 0, "error": "no text messages to index"}

    cfg = embedding_config or {}
    settings = get_settings()
    max_batch_size = max(
        1,
        int(cfg.get("max_batch_size") or cfg.get("batch_size") or settings.embedding_max_batch_size),
    )
    max_batch_tokens = max(1, int(cfg.get("max_batch_tokens") or settings.embedding_max_batch_tokens))
    requested_workers = max(1, int(cfg.get("preprocess_workers") or settings.embedding_preprocess_workers))
    workers = min(requested_workers, os.cpu_count() or requested_workers)
    timeout = max(1, int(cfg.get("timeout") or settings.embedding_timeout))
    embedding_profile = get_embedding_profile(cfg)
    model = embedding_profile["model_identity"]
    force_rebuild = bool(cfg.get("force_rebuild") or cfg.get("rebuild"))
    existing_status = get_vector_index_status(target_key, cfg)
    rebuild_reason = ""
    invalid_reasons = existing_status.get("invalid_reasons") or []
    if invalid_reasons and int(existing_status.get("actual_vector_count") or 0) > 0:
        force_rebuild = True
        rebuild_reason = ",".join(str(item) for item in invalid_reasons)

    if force_rebuild:
        raw_rows = source_rows
        skipped_existing = 0
        incremental = False
    else:
        existing = vector_store.load_existing_vector_signatures(target_key)
        raw_rows = [
            row
            for row in source_rows
            if existing.get(_row_identity(row)) != vector_store.content_hash(str(row.get("content") or ""))
        ]
        skipped_existing = len(source_rows) - len(raw_rows)
        incremental = True

    if not raw_rows:
        status = vector_store.upsert_vectors(target_key, [], [], model)
        status.update(get_vector_index_status(target_key, cfg))
        status["incremental"] = incremental
        status["force_rebuild"] = force_rebuild
        status["indexed_count"] = 0
        status["skipped_existing"] = skipped_existing
        status["source_message_count"] = len(source_rows)
        status["rebuild_reason"] = rebuild_reason
        status["pipeline"] = "incremental_noop"
        status["embedding_failed_count"] = 0
        status["embedding_failure_limit"] = 0
        status["embedding_failed_ratio"] = 0
        status["last_embedding_error"] = ""
        status["embedding_model"] = embedding_profile["model"]
        status["embedding_model_identity"] = model
        status["embedding_uses_search_prefix"] = bool(embedding_profile.get("uses_search_prefix"))
        status["embedding_search_prefix_mode"] = embedding_profile.get("search_prefix_mode") or "auto"
        return status

    embeddings: List[List[float]] = []
    vector_messages: List[ChatMessage] = []
    indexed_count = 0
    total = len(raw_rows)
    batch_count = 0
    embedding_failed_count = 0
    last_embedding_error = ""
    configured_failed_items = int(cfg.get("max_failed_items") or settings.embedding_max_failed_items or 0)
    failed_ratio = float(cfg.get("max_failed_ratio") or settings.embedding_max_failed_ratio or 0.02)
    auto_failed_items = max(1000, int(math.ceil(total * failed_ratio)))
    max_failed_items = configured_failed_items if configured_failed_items > 0 else auto_failed_items
    chunk_size = max(128, max_batch_size * 8)
    preprocess_chunks = max(1, math.ceil(total / chunk_size))
    pending_items: List[Dict] = []
    pipeline = "process_pool_dynamic_batch"
    streaming_upsert = not force_rebuild

    if progress_callback:
        action = "全量重建" if force_rebuild else f"增量构建，跳过 {skipped_existing} 条已索引消息"
        progress_callback(0, total, f"{action}；准备 CPU 预处理，worker={workers}")

    def check_cancel() -> None:
        if cancel_callback and cancel_callback():
            raise VectorBuildCancelled("vector_build_cancelled")

    def consume_batches(force: bool = False) -> None:
        nonlocal pending_items, batch_count, embedding_failed_count, indexed_count, last_embedding_error
        check_cancel()
        batches, pending_items = _extract_dynamic_batches(
            pending_items,
            max_batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
            force=force,
        )

        def embed_batch_items(batch_items: List[Dict]) -> List[Tuple[Dict, List[float]]]:
            nonlocal embedding_failed_count, last_embedding_error
            check_cancel()
            batch_texts = [str(item["text"]) for item in batch_items]
            batch_config = {**cfg, "batch_size": len(batch_items), "timeout": timeout}
            try:
                batch_embeddings = get_embeddings(batch_texts, config=batch_config, input_type="document")
                check_cancel()
                if len(batch_embeddings) != len(batch_items):
                    raise VectorEmbeddingError("embedding_count_mismatch")
                return list(zip(batch_items, batch_embeddings))
            except Exception as exc:
                check_cancel()
                if len(batch_items) > 1:
                    mid = max(1, len(batch_items) // 2)
                    return embed_batch_items(batch_items[:mid]) + embed_batch_items(batch_items[mid:])
                if _exception_status(exc) in {401, 403, 404}:
                    raise VectorEmbeddingError(f"fatal_embedding_http_error:{_safe_exception_label(exc)}") from exc
                embedding_failed_count += 1
                last_embedding_error = _safe_exception_label(exc)
                if embedding_failed_count > max_failed_items:
                    raise VectorEmbeddingError(f"too_many_embedding_failures:{last_embedding_error}") from exc
                if progress_callback:
                    progress_callback(
                        indexed_count + len(vector_messages) + embedding_failed_count,
                        total,
                        f"跳过 1 条向量化失败消息，累计失败 {embedding_failed_count}",
                    )
                return []

        for batch_items in batches:
            check_cancel()
            embedded_pairs = embed_batch_items(batch_items)
            batch_embeddings = [embedding for _, embedding in embedded_pairs]
            batch_messages = [_item_to_message(item) for item, _ in embedded_pairs]
            if batch_embeddings:
                existing_dims = int(existing_status.get("dims") or 0)
                batch_dims = len(batch_embeddings[0])
                if streaming_upsert and incremental and existing_dims and batch_dims != existing_dims:
                    raise VectorEmbeddingError(
                        f"dims_changed:{existing_dims}->{batch_dims};please_force_rebuild"
                    )
                if streaming_upsert:
                    vector_store.upsert_vectors(target_key, batch_messages, batch_embeddings, model)
                    indexed_count += len(batch_messages)
                else:
                    embeddings.extend(batch_embeddings)
                    vector_messages.extend(batch_messages)
            batch_count += 1
            if progress_callback:
                progress_callback(
                    indexed_count + len(vector_messages) + embedding_failed_count,
                    total,
                    f"动态批处理 {indexed_count + len(vector_messages) + embedding_failed_count}/{total}，batch={len(batch_items)}",
                )

    if workers <= 1 or total < 256:
        pipeline = "single_process_dynamic_batch"
        check_cancel()
        pending_items.extend(_preprocess_vector_chunk(raw_rows))
        consume_batches(force=True)
    else:
        try:
            for completed_chunks, total_chunks, chunk_items in _iter_process_pool_chunks(raw_rows, workers, chunk_size):
                check_cancel()
                pending_items.extend(chunk_items)
                if progress_callback and completed_chunks < total_chunks:
                    progress_callback(
                        indexed_count + len(vector_messages),
                        total,
                        f"CPU 预处理 {completed_chunks}/{total_chunks}，模型服务并行消费中",
                    )
                consume_batches(force=False)
            consume_batches(force=True)
        except VectorPreprocessError as exc:
            pipeline = "single_process_dynamic_batch_fallback"
            embeddings.clear()
            vector_messages.clear()
            indexed_count = 0
            check_cancel()
            pending_items = _preprocess_vector_chunk(raw_rows)
            batch_count = 0
            if progress_callback:
                progress_callback(
                    0,
                    total,
                    f"多进程预处理不可用，已降级为单进程：{exc.__class__.__name__}",
                )
            consume_batches(force=True)

    check_cancel()
    successful_count = indexed_count + len(vector_messages)
    if successful_count <= 0:
        raise VectorEmbeddingError(f"all_embedding_requests_failed:{last_embedding_error or 'unknown'}")

    existing_dims = int(existing_status.get("dims") or 0)
    new_dims = len(embeddings[0]) if embeddings else 0
    if not streaming_upsert and incremental and existing_dims and new_dims and existing_dims != new_dims:
        if progress_callback:
            progress_callback(
                0,
                len(source_rows),
                f"向量维度从 {existing_dims} 变为 {new_dims}，转为全量重建以避免混合索引",
            )
        rebuilt = build_vector_index(
            target_key,
            messages,
            {**cfg, "force_rebuild": True},
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
        rebuilt["rebuild_reason"] = "dims_changed"
        return rebuilt

    if progress_callback:
        progress_callback(total, total, "正在写入向量数据库" if force_rebuild else "已分批写入向量数据库")
    if force_rebuild:
        vector_store.replace_vectors(target_key, vector_messages, embeddings, model)
        status = get_vector_index_status(target_key, cfg)
    elif streaming_upsert:
        status = get_vector_index_status(target_key, cfg)
    else:
        status = vector_store.upsert_vectors(target_key, vector_messages, embeddings, model)
        status.update(get_vector_index_status(target_key, cfg))
    status["embedding_batch_size"] = max_batch_size
    status["embedding_max_batch_size"] = max_batch_size
    status["embedding_max_batch_tokens"] = max_batch_tokens
    status["embedding_timeout"] = timeout
    status["embedding_model"] = embedding_profile["model"]
    status["embedding_model_identity"] = model
    status["embedding_uses_search_prefix"] = bool(embedding_profile.get("uses_search_prefix"))
    status["embedding_search_prefix_mode"] = embedding_profile.get("search_prefix_mode") or "auto"
    status["preprocess_workers"] = workers
    status["preprocess_requested_workers"] = requested_workers
    status["dynamic_batch_count"] = batch_count
    status["preprocess_chunks"] = preprocess_chunks
    status["pipeline"] = pipeline
    status["embedding_failed_count"] = embedding_failed_count
    status["embedding_failure_limit"] = max_failed_items
    status["embedding_failed_ratio"] = round(embedding_failed_count / max(1, total), 6)
    status["last_embedding_error"] = last_embedding_error
    status["incremental"] = incremental
    status["force_rebuild"] = force_rebuild
    status["indexed_count"] = successful_count
    status["skipped_existing"] = skipped_existing
    status["source_message_count"] = len(source_rows)
    status["rebuild_reason"] = rebuild_reason
    return status


def get_vector_index_status(target_key: str, embedding_config: Optional[dict] = None) -> Dict:
    status = vector_store.get_index_status(target_key)
    profile = get_embedding_profile(embedding_config or {})
    expected_model = profile["model_identity"]
    actual_count = int(status.get("actual_vector_count") or 0)
    msg_count = int(status.get("msg_count") or 0)
    dims = int(status.get("dims") or 0)
    stored_model = str(status.get("model") or "")
    invalid_reasons: List[str] = []
    warnings: List[str] = []

    if actual_count <= 0:
        invalid_reasons.append("empty_index")
    if msg_count <= 0:
        invalid_reasons.append("empty_status_count")
    if dims <= 0:
        invalid_reasons.append("missing_dims")
    if not stored_model:
        invalid_reasons.append("missing_model")
    elif stored_model != expected_model:
        invalid_reasons.append("model_mismatch")
    if msg_count != actual_count:
        invalid_reasons.append("count_mismatch")
    if actual_count > 0 and not status.get("built_at"):
        invalid_reasons.append("missing_status_row")
    if status.get("built_at") and actual_count == 0:
        warnings.append("status_exists_but_no_vectors")

    status["expected_model"] = expected_model
    status["embedding_model"] = profile["model"]
    status["embedding_provider"] = profile["provider"]
    status["embedding_uses_search_prefix"] = bool(profile.get("uses_search_prefix"))
    status["embedding_search_prefix_mode"] = profile.get("search_prefix_mode") or "auto"
    status["valid"] = bool(status.get("built")) and not invalid_reasons
    status["invalid_reasons"] = invalid_reasons
    status["warnings"] = warnings
    return status


def search_vector_multi(
    target_key: str,
    queries: List[str],
    top_k: int = 5,
    final_top_k: Optional[int] = None,
    embedding_config: Optional[dict] = None,
) -> List[Dict]:
    status = get_vector_index_status(target_key, embedding_config)
    if not status.get("valid"):
        reasons = ",".join(str(item) for item in status.get("invalid_reasons") or [])
        raise VectorEmbeddingError(f"vector_index_invalid:{reasons or 'unknown'}")
    rows = vector_store.load_vectors(target_key)
    clean_queries = [str(query).strip() for query in queries if str(query).strip()]
    if not rows or not clean_queries:
        return []
    query_vectors = get_embeddings(clean_queries, config=embedding_config, input_type="query")
    if not query_vectors:
        return []

    row_vectors = [(row, row["embedding"].astype(np.float32), float(np.linalg.norm(row["embedding"]))) for row in rows]
    merged: Dict[str, Dict] = {}
    per_query_top_k = max(1, top_k)
    for query_index, query_vec_values in enumerate(query_vectors):
        query_vec = np.asarray(query_vec_values, dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm == 0:
            continue
        scored = []
        for row, emb, emb_norm in row_vectors:
            denom = emb_norm * query_norm
            if denom == 0:
                continue
            score = float(np.dot(emb, query_vec) / denom)
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        for rank, (score, row) in enumerate(scored[:per_query_top_k], start=1):
            identity = str(row.get("source_id") or f"{row.get('source_key')}:{row.get('seq')}:{row.get('datetime')}:{row.get('sender')}")
            item = merged.get(identity)
            if not item:
                item = {
                    "row": row,
                    "max_score": score,
                    "total_score": 0.0,
                    "hit_count": 0,
                    "query_matches": [],
                }
                merged[identity] = item
            item["max_score"] = max(float(item["max_score"]), score)
            item["total_score"] = float(item["total_score"]) + score
            item["hit_count"] = int(item["hit_count"]) + 1
            item["query_matches"].append(
                {
                    "query_index": query_index,
                    "query": clean_queries[query_index],
                    "rank": rank,
                    "score": score,
                }
            )

    reranked = []
    for item in merged.values():
        hit_count = max(1, int(item["hit_count"]))
        avg_score = float(item["total_score"]) / hit_count
        rerank_score = float(item["max_score"]) + min(0.12, 0.025 * (hit_count - 1)) + 0.01 * avg_score
        item["avg_score"] = avg_score
        item["rerank_score"] = rerank_score
        reranked.append(item)
    reranked.sort(key=lambda item: (float(item["rerank_score"]), float(item["max_score"]), int(item["hit_count"])), reverse=True)

    results = []
    limit = max(1, final_top_k or top_k)
    for item in reranked[:limit]:
        row = item["row"]
        results.append(
            {
                "score": float(item["max_score"]),
                "rerank_score": float(item["rerank_score"]),
                "avg_score": float(item["avg_score"]),
                "hit_count": int(item["hit_count"]),
                "query_matches": item["query_matches"],
                "message": {
                    "seq": row["seq"],
                    "datetime": row["datetime"],
                    "sender": row["sender"],
                    "content": row["content"],
                    "contact_key": row.get("source_key") or row["contact_key"],
                    "index_key": row["contact_key"],
                },
                "context": vector_store.get_context(
                    target_key,
                    int(row["seq"]),
                    window=2,
                    source_key=str(row.get("source_key") or ""),
                ),
            }
        )
    return results


def search_vector(target_key: str, query: str, top_k: int = 5, embedding_config: Optional[dict] = None) -> List[Dict]:
    return search_vector_multi(
        target_key,
        [query],
        top_k=top_k,
        final_top_k=top_k,
        embedding_config=embedding_config,
    )
