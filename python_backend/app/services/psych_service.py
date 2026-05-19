from time import perf_counter
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from app.config import get_settings
from app.models import ChatMessage, PsychAnalyzeRequest, PsychAnalyzeResponse, PsychProcessStep
from app.psych import features, labels as label_classifier, llm_extract, preprocess, report, retrieval, scoring
from app.psych.scoring_config import load_scoring_config, vector_queries_from_config
from app.services import memory_service, message_service, vector_service
from app.stores import psych_store
from app.utils.privacy import sanitize_snippet
from app.utils.time_utils import date_part


DEFAULT_PSYCH_VECTOR_QUERY = (
    "心理风险辅助筛查语义检索：低落情绪、无望感、自我否定、社交退缩、"
    "睡眠困扰、兴趣下降、疲惫、持续痛苦和生活动力下降。"
)

DEFAULT_PSYCH_VECTOR_QUERIES = [
    "低落情绪，指心情持续低沉、悲伤、失落、空虚、没动力等负面情绪状态。",
    "心情不好，情绪低沉，悲伤，沮丧，经常觉得委屈或想哭。",
    "最近什么都不想做，没有动力，提不起劲，兴趣下降，对原本喜欢的事情也没感觉。",
    "感觉空虚、失落、委屈、想哭，觉得自己很累或很难撑下去。",
    "睡不好、疲惫、焦虑、压抑，夜里难受，白天没有精神。",
    "抑郁相关信号，持续情绪低落、消极想法、自我否定、无望感和生活动力下降。",
]

DEFAULT_FACT_VECTOR_QUERY = (
    "心理风险辅助筛查事实检索：情绪低落、绝望感、兴趣下降、睡眠异常、疲乏、"
    "食欲体重变化、自责无价值感、注意力下降、行为迟滞或激越、社交退缩、"
    "自伤或轻生红线信号、保护因素、求助线索和现实压力源。"
)


def _load_messages(request: PsychAnalyzeRequest) -> tuple[List[ChatMessage], Dict[str, Any]]:
    if request.messages is not None:
        return request.messages, {
            "source": "manual_messages",
            "raw_message_count": len(request.messages),
            "decoded_message_count": len(request.messages),
        }
    if request.target_type == "group" or "@chatroom" in str(request.target_key or "").lower():
        return [], {"source": "wechat_db", "reason": "group_analysis_disabled"}
    if not request.target_key and request.target_type != "self":
        return [], {"source": "wechat_db", "reason": "empty_target_key"}
    messages, diagnostics = message_service.list_messages_with_diagnostics(
        target_key=request.target_key,
        target_type=request.target_type,
        time_from=request.time_from,
        time_to=request.time_to,
        limit=0,
    )
    diagnostics["message_limit_applied"] = False
    diagnostics["source"] = "wechat_db"
    return messages, diagnostics


def _message_scope(request: PsychAnalyzeRequest) -> str:
    raw_scope = str((request.options or {}).get("message_scope") or "").strip().lower()
    if raw_scope in {"mine", "self", "other", "all"}:
        return "mine" if raw_scope == "self" else raw_scope
    if request.target_type == "self":
        return "mine"
    if request.target_type == "contact":
        return "other"
    return "mine"


def _filter_by_scope(messages: List[ChatMessage], scope: str) -> List[ChatMessage]:
    if scope == "mine":
        return [msg for msg in messages if msg.is_mine]
    if scope == "other":
        return [msg for msg in messages if not msg.is_mine]
    return messages


def _safe_error_text(error: Exception) -> str:
    text = str(error).replace("\n", " ").strip()
    return text[:180] if text else error.__class__.__name__


def _message_identity(msg: ChatMessage) -> str:
    return f"{msg.seq}\x1f{msg.datetime}\x1f{msg.sender}\x1f{msg.content}"


def _debug_message(msg: ChatMessage, max_len: int = 180) -> Dict[str, Any]:
    return {
        "seq": msg.seq,
        "datetime": msg.datetime,
        "sender": msg.sender,
        "is_mine": bool(msg.is_mine),
        "contact_key": msg.contact_key,
        "content": sanitize_snippet(msg.content, max_len=max_len),
    }


def _debug_message_list(messages: List[ChatMessage], max_items: int = 80) -> List[Dict[str, Any]]:
    return [_debug_message(msg) for msg in messages[:max_items]]


def _keyword_debug_candidates(messages: List[ChatMessage], scoring_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    dimensions = scoring_config.get("dimensions", [])
    for msg in messages:
        groups = []
        words = []
        text = msg.content or ""
        for dimension in dimensions:
            feature_name = str(dimension.get("key") or "")
            if any(word and word in text for word in dimension.get("exclude_keywords", [])):
                continue
            hits = [word for word in dimension.get("keywords", []) if word in text]
            if hits:
                groups.append(feature_name)
                words.extend(hits[:5])
        item = _debug_message(msg)
        label_item = label_classifier.classify_debug_message(msg, scoring_config)
        item.update(
            {
                "hit_groups": groups,
                "hit_words": words[:10],
                "labels": label_item.get("labels", []),
                "label_names": label_item.get("label_names", []),
                "label_hits": label_item.get("label_hits", []),
                "risk_level": label_item.get("risk_level", 0),
            }
        )
        out.append(item)
    return out


def _vector_key_for_request(request: PsychAnalyzeRequest) -> str:
    key = (request.target_key or "").strip()
    if key:
        return key
    return request.target_type or "all"


def _psych_vector_queries(options: Dict[str, Any], scoring_config: Dict[str, Any]) -> tuple[List[str], List[Dict[str, str]]]:
    raw_queries = options.get("psych_vector_queries")
    if isinstance(raw_queries, list):
        queries = [str(item).strip() for item in raw_queries if str(item).strip()]
        if queries:
            return queries, [{"dimension_key": "custom", "dimension_label": "自定义查询", "query": item} for item in queries]
    if isinstance(raw_queries, str) and raw_queries.strip():
        queries = [line.strip() for line in raw_queries.splitlines() if line.strip()]
        if queries:
            return queries, [{"dimension_key": "custom", "dimension_label": "自定义查询", "query": item} for item in queries]
    raw_query = str(options.get("psych_vector_query") or "").strip()
    if raw_query:
        return [raw_query], [{"dimension_key": "custom", "dimension_label": "自定义查询", "query": raw_query}]
    query_items = vector_queries_from_config(scoring_config)
    if query_items:
        return [item["query"] for item in query_items], query_items
    return DEFAULT_PSYCH_VECTOR_QUERIES, [
        {"dimension_key": "default", "dimension_label": "默认查询", "query": item}
        for item in DEFAULT_PSYCH_VECTOR_QUERIES
    ]


def _vector_semantic_retrieve(
    request: PsychAnalyzeRequest,
    messages: List[ChatMessage],
    options: Dict[str, Any],
    scoring_config: Dict[str, Any],
    debug_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    artifact_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> tuple[List[ChatMessage], Dict[str, Any]]:
    if not messages:
        return [], {"vector_skipped_reason": "no_filtered_messages", "semantic_hit_count": 0}
    if options.get("use_vector") is False:
        return [], {"vector_skipped_reason": "disabled_by_options", "semantic_hit_count": 0}

    vector_key = _vector_key_for_request(request)
    queries, query_items = _psych_vector_queries(options, scoring_config)
    query = "\n".join(queries)
    top_k = int(options.get("psych_vector_top_k") or 30)
    per_query_top_k = int(options.get("psych_vector_top_k_per_query") or top_k)
    final_top_k = int(options.get("psych_vector_final_top_k") or top_k)
    if artifact_callback:
        artifact_callback(
            {
                "query": sanitize_snippet(query, max_len=500),
                "queries": [sanitize_snippet(item, max_len=220) for item in queries],
                "query_items": [
                    {
                        "dimension_key": item.get("dimension_key", ""),
                        "dimension_label": item.get("dimension_label", ""),
                        "query": sanitize_snippet(item.get("query", ""), max_len=220),
                    }
                    for item in query_items
                ],
                "query_count": len(queries),
                "top_k": top_k,
                "top_k_per_query": per_query_top_k,
                "final_top_k": final_top_k,
                "input_message_count": len(messages),
                "vector_key": vector_key,
                "embedding_input_role": "search_query",
            }
        )
    try:
        if debug_callback:
            debug_callback(
                "read_index_status",
                {
                    "debug_phase": "read_vector_index_status",
                    "vector_key": vector_key,
                    "vector_input_messages": len(messages),
                    "vector_top_k": top_k,
                    "vector_top_k_per_query": per_query_top_k,
                    "vector_final_top_k": final_top_k,
                    "vector_query_count": len(queries),
                    "vector_query_chars": len(query),
                },
            )
        before_status = vector_service.get_vector_index_status(vector_key)
        if debug_callback:
            debug_callback(
                "ensure_vector_index",
                {
                    "debug_phase": "ensure_incremental_vector_index",
                    "vector_key": vector_key,
                    "vector_index_preexisting": bool(before_status.get("built")),
                    "indexed_message_count": int(before_status.get("msg_count") or 0),
                },
            )

        def on_vector_progress(done: int, total: int, message: str) -> None:
            if debug_callback:
                debug_callback(
                    "vector_index_progress",
                    {
                        "debug_phase": "vector_index_building",
                        "vector_progress_message": message,
                        "vector_processed": done,
                        "vector_total": total,
                    },
                )

        ensure_status = vector_service.build_vector_index(
            vector_key,
            messages,
            options.get("embedding_config") or None,
            progress_callback=on_vector_progress,
        )
        status = vector_service.get_vector_index_status(vector_key)
        if not status.get("valid"):
            return [], {
                "vector_index_built": False,
                "vector_skipped_reason": "ensure_failed",
                "semantic_hit_count": 0,
                "vector_key": vector_key,
                "vector_invalid_reasons": status.get("invalid_reasons") or [],
                "vector_expected_model": status.get("expected_model") or "",
                "vector_actual_model": status.get("model") or "",
            }
        allowed = {_message_identity(msg) for msg in messages}
        if debug_callback:
            debug_callback(
                "semantic_search",
                {
                    "debug_phase": "semantic_vector_search",
                    "vector_key": vector_key,
                    "indexed_message_count": int(status.get("msg_count") or 0),
                    "embedding_model": status.get("model") or "",
                    "embedding_dims": int(status.get("dims") or 0),
                    "vector_top_k": top_k,
                    "vector_top_k_per_query": per_query_top_k,
                    "vector_query_count": len(queries),
                },
            )
        hits = vector_service.search_vector_multi(
            vector_key,
            queries,
            top_k=per_query_top_k,
            final_top_k=final_top_k,
            embedding_config=options.get("embedding_config") or None,
        )
        if artifact_callback:
            artifact_callback(
                {
                    "raw_hits": [
                        {
                            "score": round(float(hit.get("score") or 0), 4),
                            "rerank_score": round(float(hit.get("rerank_score") or hit.get("score") or 0), 4),
                            "hit_count": int(hit.get("hit_count") or 1),
                            "query_matches": [
                                {
                                    "query_index": int(match.get("query_index") or 0),
                                    "rank": int(match.get("rank") or 0),
                                    "score": round(float(match.get("score") or 0), 4),
                                    "query": sanitize_snippet(str(match.get("query") or ""), max_len=160),
                                }
                                for match in hit.get("query_matches", [])[:8]
                            ],
                            **_debug_message(
                                ChatMessage(
                                    seq=int(hit.get("message", {}).get("seq") or 0),
                                    datetime=str(hit.get("message", {}).get("datetime") or ""),
                                    sender=str(hit.get("message", {}).get("sender") or ""),
                                    content=str(hit.get("message", {}).get("content") or ""),
                                    is_mine=True,
                                    contact_key=str(hit.get("message", {}).get("contact_key") or vector_key),
                                )
                            ),
                        }
                        for hit in hits
                    ],
                }
            )
        out: List[ChatMessage] = []
        semantic_message_ids: Dict[str, set[str]] = {}
        semantic_message_dimensions: Dict[str, set[str]] = {}
        semantic_days: Dict[str, set[str]] = {}
        for hit in hits:
            item = hit.get("message", {})
            msg = ChatMessage(
                seq=int(item.get("seq") or 0),
                datetime=str(item.get("datetime") or ""),
                sender=str(item.get("sender") or ""),
                content=str(item.get("content") or ""),
                is_mine=True,
                contact_key=str(item.get("contact_key") or vector_key),
            )
            if _message_identity(msg) in allowed:
                out.append(msg)
                day = date_part(msg.datetime)
                for match in hit.get("query_matches", []):
                    index = int(match.get("query_index") or 0)
                    if index < 0 or index >= len(query_items):
                        continue
                    dimension_key = query_items[index].get("dimension_key") or "unknown"
                    message_key = _message_identity(msg)
                    semantic_message_ids.setdefault(dimension_key, set()).add(message_key)
                    semantic_message_dimensions.setdefault(message_key, set()).add(dimension_key)
                    if day:
                        semantic_days.setdefault(dimension_key, set()).add(day)
        semantic_dimension_hits = {key: len(value) for key, value in semantic_message_ids.items()}
        semantic_dimension_days = {key: len(value) for key, value in semantic_days.items()}
        semantic_message_dimension_map = {
            key: sorted(value)
            for key, value in semantic_message_dimensions.items()
        }
        if debug_callback:
            debug_callback(
                "semantic_filter_scope",
                {
                    "debug_phase": "filter_to_selected_scope",
                    "semantic_raw_hit_count": len(hits),
                    "semantic_hit_count": len(out),
                    "allowed_message_count": len(allowed),
                    "semantic_dimension_hits": semantic_dimension_hits,
                },
            )
        if artifact_callback:
            artifact_callback({"scoped_hits": [_debug_message(msg) for msg in out]})
        return out, {
            "vector_index_built": True,
            "vector_index_valid": bool(status.get("valid")),
            "vector_invalid_reasons": status.get("invalid_reasons") or [],
            "vector_index_preexisting": bool(before_status.get("built")),
            "vector_ensure_incremental": bool(ensure_status.get("incremental", True)),
            "vector_ensure_indexed": int(ensure_status.get("indexed_count") or 0),
            "vector_ensure_skipped": int(ensure_status.get("skipped_existing") or 0),
            "indexed_message_count": int(status.get("msg_count") or 0),
            "actual_vector_count": int(status.get("actual_vector_count") or 0),
            "expected_embedding_model": status.get("expected_model") or "",
            "embedding_model": status.get("model") or "",
            "embedding_dims": int(status.get("dims") or 0),
            "semantic_query_count": len(queries),
            "semantic_top_k_per_query": per_query_top_k,
            "semantic_final_top_k": final_top_k,
            "semantic_hit_count": len(out),
            "semantic_raw_hit_count": len(hits),
            "semantic_dimension_hits": semantic_dimension_hits,
            "semantic_dimension_days": semantic_dimension_days,
            "semantic_message_dimensions": semantic_message_dimension_map,
            "index_scope": "existing_vector_index_filtered_to_selected_messages",
            "vector_key": vector_key,
        }
    except Exception as error:
        status = vector_service.get_vector_index_status(vector_key)
        return [], {
            "vector_index_built": False,
            "indexed_message_count": int(status.get("msg_count") or 0),
            "embedding_error": _safe_error_text(error),
            "semantic_hit_count": 0,
            "vector_key": vector_key,
        }


def _screen_step(
    messages: List[ChatMessage],
    stage: str,
    options: Dict[str, Any],
    context_messages: List[ChatMessage],
    include_context: bool,
    debug_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    artifact_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> tuple[List[ChatMessage], Dict[str, Any]]:
    settings = get_settings()
    if debug_callback:
        debug_callback(
            "prepare_llm_screen",
            {
                "debug_phase": "prepare_llm_screening",
                f"{stage}_input_candidates": len(messages),
                f"{stage}_llm_provider": settings.llm_provider,
                f"{stage}_llm_model": settings.llm_model,
                f"{stage}_context_enabled": include_context,
                f"{stage}_context_message_count": len(context_messages) if include_context else 0,
            },
        )
    selected, metrics = retrieval.screen_candidates_with_llm(
        messages,
        stage=stage,
        enabled=bool(options.get("llm_screening", True)),
        limit=int(options.get("llm_screen_limit") or 40),
        config=options.get("llm_config") or None,
        context_messages=context_messages,
        include_context=include_context,
        context_window=int(options.get("llm_screen_context_window") or 2),
        debug_callback=debug_callback,
        artifact_callback=artifact_callback,
    )
    metrics[f"{stage}_llm_provider"] = settings.llm_provider
    metrics[f"{stage}_llm_model"] = settings.llm_model
    return selected, metrics


def _fact_to_vector_messages(task_id: str, facts: List[Any]) -> List[ChatMessage]:
    messages: List[ChatMessage] = []
    for index, fact in enumerate(facts, start=1):
        messages.append(
            ChatMessage(
                seq=index,
                datetime="",
                sender="psych_fact",
                content=(
                    "心理风险辅助筛查事实。"
                    f"类型：{fact.fact_type}；严重度：{fact.severity}；"
                    f"事实：{fact.fact}；置信度：{fact.confidence:.2f}"
                ),
                is_mine=True,
                contact_key=f"psych_facts:{task_id}",
            )
        )
    return messages


def _fact_vector_index_and_search(
    task_id: str,
    facts: List[Any],
    options: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not facts:
        return [], {
            "fact_vector_index_built": False,
            "fact_vector_skipped_reason": "no_facts",
            "fact_vector_hit_count": 0,
        }
    fact_key = f"psych_facts:{task_id}"
    fact_messages = _fact_to_vector_messages(task_id, facts)
    query = str(
        options.get("psych_fact_vector_query")
        or DEFAULT_FACT_VECTOR_QUERY
    )
    if not options.get("psych_fact_vector_query"):
        query = DEFAULT_FACT_VECTOR_QUERY
    top_k = int(options.get("psych_fact_vector_top_k") or 8)
    try:
        status = vector_service.build_vector_index(
            fact_key,
            fact_messages,
            options.get("embedding_config") or None,
        )
        hits = vector_service.search_vector(
            fact_key,
            query,
            top_k=top_k,
            embedding_config=options.get("embedding_config") or None,
        )
        compact_hits = [
            {
                "score": round(float(hit.get("score") or 0), 4),
                "seq": hit.get("message", {}).get("seq"),
                "content": hit.get("message", {}).get("content"),
            }
            for hit in hits
        ]
        return compact_hits, {
            "fact_vector_index_built": bool(status.get("built")),
            "fact_vector_key": fact_key,
            "fact_vector_count": len(fact_messages),
            "fact_vector_indexed": int(status.get("indexed_count") or 0),
            "fact_vector_skipped": int(status.get("skipped_existing") or 0),
            "fact_vector_hit_count": len(compact_hits),
            "fact_vector_model": status.get("model") or "",
            "fact_vector_dims": int(status.get("dims") or 0),
        }
    except Exception as error:
        return [], {
            "fact_vector_index_built": False,
            "fact_vector_key": fact_key,
            "fact_vector_count": len(fact_messages),
            "fact_vector_hit_count": 0,
            "fact_vector_error": _safe_error_text(error),
        }


ProgressCallback = Callable[[List[PsychProcessStep], str, str], None]


def analyze_psych(
    request: PsychAnalyzeRequest,
    task_id: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> PsychAnalyzeResponse:
    process_steps: List[PsychProcessStep] = []
    task_id = task_id or str(uuid4())
    options = request.options or {}
    scoring_config = load_scoring_config()
    debug_steps: Dict[str, Dict[str, Any]] = {
        "load_messages": {},
        "preprocess": {},
        "keyword_search": {},
        "keyword_llm_screen": {},
        "vector_semantic_search": {},
        "vector_llm_screen": {},
        "fact_memory_search": {},
        "scoring": {},
        "report": {},
    }

    def emit_step_debug(key: str, name: str, detail: str, metrics: Dict[str, Any]) -> None:
        if not progress_callback:
            return
        progress_callback(
            [
                *process_steps,
                PsychProcessStep(
                    key=key,
                    name=name,
                    status="running",
                    duration_ms=0,
                    detail=detail,
                    metrics=metrics,
                ),
            ],
            key,
            "running",
        )

    def merge_debug_artifact(step_key: str, data: Dict[str, Any]) -> None:
        current = debug_steps.setdefault(step_key, {})
        for key, value in data.items():
            if key == "events":
                current.setdefault("events", []).extend(value if isinstance(value, list) else [value])
            elif key == "metrics" and isinstance(value, dict):
                current.setdefault("metrics", {}).update(value)
            elif key == "step" and isinstance(value, dict):
                current.setdefault("step", {}).update(value)
            else:
                current[key] = value

    def run_step(key: str, name: str, detail: str, fn):
        started = perf_counter()
        if progress_callback:
            progress_callback(
                [
                    *process_steps,
                    PsychProcessStep(
                        key=key,
                        name=name,
                        status="running",
                        duration_ms=0,
                        detail=detail,
                        metrics={},
                    ),
                ],
                key,
                "running",
            )
        result = fn()
        duration_ms = int((perf_counter() - started) * 1000)
        metrics: Dict[str, Any] = {}
        if isinstance(result, tuple) and result and isinstance(result[-1], dict) and result[-1].get("__metrics__"):
            *payload, metric_data = result
            result = tuple(payload) if len(payload) != 1 else payload[0]
            metrics = {k: v for k, v in metric_data.items() if k != "__metrics__"}
        process_steps.append(
            PsychProcessStep(
                key=key,
                name=name,
                status="completed",
                duration_ms=duration_ms,
                detail=detail,
                metrics=metrics,
            )
        )
        merge_debug_artifact(
            key,
            {
                "step": {
                    "key": key,
                    "name": name,
                    "status": "completed",
                    "duration_ms": duration_ms,
                    "detail": detail,
                },
                "metrics": metrics,
            },
        )
        if progress_callback:
            progress_callback(process_steps, key, "completed")
        return result

    raw_messages, load_diagnostics = run_step(
        "load_messages",
        "读取聊天消息",
        "从请求或本地微信数据库读取用户明确选择的聊天范围。",
        lambda: (*_load_messages(request), {"__metrics__": True}),
    )
    if process_steps:
        process_steps[-1].metrics.update(load_diagnostics)
        process_steps[-1].metrics["raw_message_count"] = len(raw_messages)
    merge_debug_artifact(
        "load_messages",
        {
            "metrics": {**load_diagnostics, "raw_message_count": len(raw_messages)},
            "diagnostics": load_diagnostics,
            "messages": _debug_message_list(raw_messages),
        },
    )

    speaker_scope = _message_scope(request)
    filtered = run_step(
        "preprocess",
        "预处理与隐私过滤",
        "过滤空消息、媒体占位、链接分享和短寒暄，并按分析对象保留本人或对方消息。",
        lambda: (
            _filter_by_scope(preprocess.filter_messages(raw_messages, only_mine=False), speaker_scope),
            {"__metrics__": True, "message_scope": speaker_scope, "only_mine": speaker_scope == "mine"},
        ),
    )
    if process_steps:
        process_steps[-1].metrics.update(
            {
                "filtered_message_count": len(filtered),
                "removed_message_count": max(0, len(raw_messages) - len(filtered)),
            }
        )
    merge_debug_artifact(
        "preprocess",
        {
            "metrics": {
                "message_scope": speaker_scope,
                "raw_message_count": len(raw_messages),
                "filtered_message_count": len(filtered),
                "removed_message_count": max(0, len(raw_messages) - len(filtered)),
            },
            "input_messages": _debug_message_list(raw_messages),
            "output_messages": _debug_message_list(filtered),
        },
    )
    screen_include_context = bool(options.get("llm_screen_include_context", request.include_context))
    screen_context_messages = (
        preprocess.filter_messages(raw_messages, only_mine=False)
        if screen_include_context
        else filtered
    )
    if process_steps:
        process_steps[-1].metrics.update(
            {
                "llm_context_enabled": screen_include_context,
                "llm_context_message_count": len(screen_context_messages) if screen_include_context else 0,
                "llm_context_window": int(options.get("llm_screen_context_window") or 2) if screen_include_context else 0,
            }
        )
    merge_debug_artifact(
        "preprocess",
        {
            "metrics": {
                "llm_context_enabled": screen_include_context,
                "llm_context_message_count": len(screen_context_messages) if screen_include_context else 0,
                "llm_context_window": int(options.get("llm_screen_context_window") or 2) if screen_include_context else 0,
            },
            "context_messages": _debug_message_list(screen_context_messages) if screen_include_context else [],
        },
    )

    def run_keyword_retrieval():
        emit_step_debug(
            "keyword_search",
            "关键词检索",
            "正在统计规则特征和关键词命中。",
            {
                "debug_phase": "compute_rule_features",
                "filtered_message_count": len(filtered),
                "keyword_dictionary_groups": len(scoring_config.get("dimensions", [])),
                "keyword_limit": int(options.get("keyword_limit") or 80),
            },
        )
        merge_debug_artifact(
            "keyword_search",
            {
                "input_message_count": len(filtered),
                "keyword_limit": int(options.get("keyword_limit") or 80),
                "dictionary_groups": [item.get("key") for item in scoring_config.get("dimensions", [])],
            },
        )
        psych_features, feature_evidences, values = features.compute_features(filtered, config=scoring_config)
        emit_step_debug(
            "keyword_search",
            "关键词检索",
            "规则特征统计完成，开始按词典召回候选消息。",
            {
                "debug_phase": "keyword_candidate_recall",
                "feature_count": len(psych_features),
                "feature_evidence_count": len(feature_evidences),
                "active_days": int(values.get("unique_active_days", 0)),
            },
        )
        candidates, keyword_metrics = retrieval.keyword_retrieve(
            filtered,
            limit=int(options.get("keyword_limit") or 80),
            config=scoring_config,
        )
        merge_debug_artifact(
            "keyword_search",
            {
                "metrics": {
                    **keyword_metrics,
                    "feature_count": len(psych_features),
                    "feature_evidence_count": len(feature_evidences),
                    "active_days": int(values.get("unique_active_days", 0)),
                },
                "candidates": _keyword_debug_candidates(candidates, scoring_config),
            },
        )
        emit_step_debug(
            "keyword_search",
            "关键词检索",
            "关键词候选召回完成。",
            {
                "debug_phase": "keyword_recall_completed",
                **keyword_metrics,
                "feature_count": len(psych_features),
                "feature_evidence_count": len(feature_evidences),
                "active_days": int(values.get("unique_active_days", 0)),
            },
        )
        return (
            candidates,
            psych_features,
            feature_evidences,
            values,
            {
                "__metrics__": True,
                **keyword_metrics,
                "feature_count": len(psych_features),
                "feature_evidence_count": len(feature_evidences),
                "active_days": int(values.get("unique_active_days", 0)),
            },
        )

    keyword_candidates, psych_features, feature_evidences, values = run_step(
        "keyword_search",
        "关键词检索",
        "用心理风险辅助筛查词典召回候选消息，同时统计基础规则特征。",
        run_keyword_retrieval,
    )

    def run_keyword_screen():
        settings = get_settings()
        def keyword_debug(phase: str, data: Dict[str, Any]) -> None:
            merge_debug_artifact("keyword_llm_screen", {"events": [{"phase": phase, "metrics": data}]})
            emit_step_debug(
                "keyword_llm_screen",
                "大模型筛选（辨别是否有用）",
                f"关键词候选上下文语义筛选：{phase}",
                {
                    **data,
                    "keyword_llm_provider": settings.llm_provider,
                    "keyword_llm_model": settings.llm_model,
                },
            )

        def keyword_artifact(data: Dict[str, Any]) -> None:
            merge_debug_artifact("keyword_llm_screen", data)

        selected, metrics = _screen_step(
            keyword_candidates,
            "keyword",
            options,
            screen_context_messages,
            screen_include_context,
            debug_callback=keyword_debug,
            artifact_callback=keyword_artifact,
        )
        merge_debug_artifact(
            "keyword_llm_screen",
            {
                "input_candidates": _debug_message_list(keyword_candidates),
                "output_candidates": _keyword_debug_candidates(selected, scoring_config),
            },
        )
        return selected, {"__metrics__": True, **metrics}

    keyword_screened = run_step(
        "keyword_llm_screen",
        "大模型筛选（辨别是否有用）",
        "让本地或兼容大模型结合前后上下文做语义判断，辨别关键词候选是否有用。",
        run_keyword_screen,
    )

    def run_vector_retrieval():
        def vector_debug(phase: str, data: Dict[str, Any]) -> None:
            merge_debug_artifact("vector_semantic_search", {"events": [{"phase": phase, "metrics": data}]})
            emit_step_debug(
                "vector_semantic_search",
                "向量语义检索",
                f"向量检索调试：{phase}",
                data,
            )

        vector_candidates, metrics = _vector_semantic_retrieve(
            request,
            filtered,
            options,
            scoring_config,
            debug_callback=vector_debug,
            artifact_callback=lambda data: merge_debug_artifact("vector_semantic_search", data),
        )
        merge_debug_artifact("vector_semantic_search", {"candidates": _debug_message_list(vector_candidates)})
        return vector_candidates, {"__metrics__": True, **metrics}

    vector_candidates = run_step(
        "vector_semantic_search",
        "向量语义检索",
        "读取或补建向量索引，按心理风险相关语义召回候选消息，并限制在本次选择范围内。",
        run_vector_retrieval,
    )
    vector_retrieval_metrics = process_steps[-1].metrics if process_steps else {}

    def run_vector_screen():
        settings = get_settings()
        def vector_screen_debug(phase: str, data: Dict[str, Any]) -> None:
            merge_debug_artifact("vector_llm_screen", {"events": [{"phase": phase, "metrics": data}]})
            emit_step_debug(
                "vector_llm_screen",
                "大模型筛选",
                f"向量召回上下文语义筛选：{phase}",
                {
                    **data,
                    "vector_llm_provider": settings.llm_provider,
                    "vector_llm_model": settings.llm_model,
                },
            )

        def vector_screen_artifact(data: Dict[str, Any]) -> None:
            merge_debug_artifact("vector_llm_screen", data)

        selected, metrics = _screen_step(
            vector_candidates,
            "vector",
            options,
            screen_context_messages,
            screen_include_context,
            debug_callback=vector_screen_debug,
            artifact_callback=vector_screen_artifact,
        )
        merge_debug_artifact(
            "vector_llm_screen",
            {
                "input_candidates": _debug_message_list(vector_candidates),
                "output_candidates": _keyword_debug_candidates(selected, scoring_config),
            },
        )
        return selected, {"__metrics__": True, **metrics}

    vector_screened = run_step(
        "vector_llm_screen",
        "大模型筛选",
        "对向量语义召回结果结合上下文做二次语义筛选，减少误召回。",
        run_vector_screen,
    )
    vector_screen_metrics = process_steps[-1].metrics if process_steps else {}
    semantic_message_dimensions = (
        vector_retrieval_metrics.get("semantic_message_dimensions")
        if isinstance(vector_retrieval_metrics.get("semantic_message_dimensions"), dict)
        else {}
    )
    if vector_screen_metrics.get("vector_llm_screened"):
        semantic_ids: Dict[str, set[str]] = {}
        semantic_days: Dict[str, set[str]] = {}
        semantic_label_ids: Dict[str, set[str]] = {}
        semantic_label_days: Dict[str, set[str]] = {}
        for msg in vector_screened:
            message_key = _message_identity(msg)
            for dimension_key in semantic_message_dimensions.get(message_key, []):
                day = date_part(msg.datetime)
                if dimension_key.startswith("label:"):
                    label_key = dimension_key.split(":", 1)[1]
                    semantic_label_ids.setdefault(label_key, set()).add(message_key)
                    if day:
                        semantic_label_days.setdefault(label_key, set()).add(day)
                else:
                    semantic_ids.setdefault(dimension_key, set()).add(message_key)
                    if day:
                        semantic_days.setdefault(dimension_key, set()).add(day)
        for key, items in semantic_ids.items():
            values[f"{key}_semantic_hit_count"] = float(len(items))
        for key, items in semantic_days.items():
            values[f"{key}_semantic_active_days"] = float(len(items))
        for key, items in semantic_label_ids.items():
            values[f"label_{key}_message_count"] = max(values.get(f"label_{key}_message_count", 0.0), float(len(items)))
            values[f"label_{key}_count"] = max(values.get(f"label_{key}_count", 0.0), float(len(items)))
        for key, items in semantic_label_days.items():
            values[f"label_{key}_active_days"] = max(values.get(f"label_{key}_active_days", 0.0), float(len(items)))
        vector_screen_metrics["semantic_scoring_applied"] = True
        vector_screen_metrics["semantic_scoring_dimension_hits"] = {key: len(value) for key, value in semantic_ids.items()}
        vector_screen_metrics["semantic_scoring_label_hits"] = {key: len(value) for key, value in semantic_label_ids.items()}
    else:
        vector_screen_metrics["semantic_scoring_applied"] = False

    analysis_messages = retrieval.merge_messages(keyword_screened, vector_screened)
    if not analysis_messages:
        analysis_messages = retrieval.merge_messages(keyword_candidates, vector_candidates)
    if not analysis_messages:
        analysis_messages = filtered

    all_evidences = feature_evidences

    def run_fact_memory_search():
        fact_key = _vector_key_for_request(request)
        query = str(options.get("psych_fact_query") or DEFAULT_FACT_VECTOR_QUERY)
        top_k = int(options.get("psych_fact_top_k") or options.get("psych_fact_vector_top_k") or 8)
        hits = memory_service.search_psych_memory_facts(
            fact_key,
            query,
            top_k=top_k,
            embedding_config=options.get("embedding_config") or None,
        )
        facts = memory_service.psych_hits_to_facts(hits)
        compact_hits = [
            {
                "score": round(float(hit.get("score") or 0), 4),
                "fact_type": hit.get("fact_type") or "",
                "severity": hit.get("severity") or "",
                "confidence": hit.get("confidence") or 0,
                "source_from": hit.get("source_from") or 0,
                "source_to": hit.get("source_to") or 0,
                "content": sanitize_snippet(str(hit.get("fact") or ""), max_len=180),
            }
            for hit in hits
        ]
        merge_debug_artifact(
            "fact_memory_search",
            {
                "query": sanitize_snippet(query, max_len=500),
                "fact_hits": compact_hits,
                "facts": [
                    {
                        "fact_type": fact.fact_type,
                        "severity": fact.severity,
                        "confidence": round(float(fact.confidence or 0), 4),
                        "fact": sanitize_snippet(fact.fact, max_len=220),
                    }
                    for fact in facts[:50]
                ],
            },
        )
        return (facts, compact_hits), {
            "__metrics__": True,
            "fact_store_key": fact_key,
            "fact_query_chars": len(query),
            "fact_top_k": top_k,
            "fact_count": len(facts),
            "fact_memory_hit_count": len(hits),
            "fact_source": "mem_facts",
        }

    fact_payload = run_step(
        "fact_memory_search",
        "心理事实库检索",
        "从数据库页面预先构建的 mem_facts 中按语义召回相关心理事实，分析阶段不再临时抽取事实。",
        run_fact_memory_search,
    )
    facts, fact_vector_hits = fact_payload

    score = run_step(
        "scoring",
        "综合评分",
        "按可配置规则权重计算抑郁相关信号分、综合风险和置信度。",
        lambda: (
            scoring.compute_score(values, all_evidences, config=scoring_config),
            {
                "__metrics__": True,
                "evidence_count": len(all_evidences),
                "scoring_config_dimensions": len(scoring_config.get("dimensions", [])),
                "evidence_strength_enabled": bool(scoring_config.get("evidence_strength", {}).get("enabled", True)),
                "time_adjustment_enabled": bool(scoring_config.get("time_adjustment", {}).get("enabled", True)),
                "multi_label_enabled": bool(scoring_config.get("symptom_labels", {}).get("enabled", True)),
            },
        ),
    )
    if process_steps:
        process_steps[-1].metrics.update(
            {
                "depression_signal_score": score.depression_signal_score,
                "overall_risk": score.overall_risk,
                "risk_level": score.risk_level,
                "risk_level_label": score.risk_level_label,
                "symptom_label_count": len(score.symptom_labels),
                "raw_dimension_score": score.scoring_adjustments.get("raw_dimension_score"),
                "worsening_bonus": score.scoring_adjustments.get("worsening_bonus"),
                "relief_delta": score.scoring_adjustments.get("relief_delta"),
                "confidence": score.confidence,
            }
        )
    merge_debug_artifact(
        "scoring",
        {
            "metrics": {
                "depression_signal_score": score.depression_signal_score,
                "overall_risk": score.overall_risk,
                "risk_level": score.risk_level,
                "risk_level_label": score.risk_level_label,
                "confidence": score.confidence,
                "symptom_label_count": len(score.symptom_labels),
                "dimension_score_count": len(score.dimension_scores),
            },
            "dimension_scores": score.dimension_scores,
            "symptom_labels": score.symptom_labels,
            "scoring_adjustments": score.scoring_adjustments,
            "main_signals": score.main_signals,
        },
    )

    report_result = run_step(
        "report",
        "生成报告",
        "生成带免责声明的 Markdown 和 JSON 报告。",
        lambda: (
            report.generate_report(
                psych_features,
                all_evidences,
                facts,
                score,
                process_steps=process_steps,
                fact_vector_hits=fact_vector_hits,
            ),
            {"__metrics__": True},
        ),
    )
    if process_steps:
        process_steps[-1].metrics["report_chars"] = len(report_result.get("report_md", ""))
    merge_debug_artifact(
        "report",
        {
            "metrics": {
                "report_chars": len(report_result.get("report_md", "")),
                "evidence_count": len(all_evidences),
                "fact_count": len(facts),
                "feature_count": len(psych_features),
            },
            "report_preview": sanitize_snippet(report_result.get("report_md", ""), max_len=1600),
            "report_sections": list(report_result.get("report_json", {}).keys()),
        },
    )
    report_result["report_json"]["process_steps"] = [step.dict() for step in process_steps]
    report_result["report_json"]["fact_vector_hits"] = fact_vector_hits
    report_result["report_json"]["debug_steps"] = debug_steps
    report_result["report_json"]["scoring_config"] = scoring_config
    report_result["report_json"]["analysis_request"] = {
        "target_key": request.target_key or "",
        "target_type": request.target_type,
        "time_from": request.time_from,
        "time_to": request.time_to,
        "only_mine": request.only_mine,
        "include_context": request.include_context,
        "options": request.options or {},
    }
    report_result["report_json"]["target_key"] = request.target_key or (filtered[0].contact_key if filtered else "")
    report_result["report_json"]["target_type"] = request.target_type
    report_result["report_json"]["time_from"] = request.time_from
    report_result["report_json"]["time_to"] = request.time_to
    response = PsychAnalyzeResponse(
        task_id=task_id,
        status="completed",
        process_steps=process_steps,
        features=psych_features,
        evidences=all_evidences,
        facts=facts,
        score=score,
        report_md=report_result["report_md"],
        report_json=report_result["report_json"],
    )
    psych_store.save_response(
        response,
        target_key=request.target_key or (filtered[0].contact_key if filtered else ""),
        target_type=request.target_type,
        options=request.options or {},
    )
    return response
