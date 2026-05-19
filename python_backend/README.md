# MindTrace Python Backend

License: AGPL-3.0-only. MindTrace is based on and modified from [runzhliu/welink](https://github.com/runzhliu/welink).

AI-assisted development notice: parts of this project were generated with AI coding assistance and then reviewed, modified, and integrated by the maintainer.

这是 MindTrace 的第一阶段 Python 后端，用 FastAPI + SQLite 实现微信聊天记录读取、向量索引、记忆事实占位提取，以及心理风险辅助筛查。

本后端是独立服务，目标是逐步替代 Go backend 中的新 API 逻辑。React 前端可继续保留，后续将 `API_BASE_URL` 指向 Python 服务即可。

## 安全边界

- 本系统只做心理风险辅助筛查、抑郁相关信号识别和进一步专业评估建议。
- 不输出医学诊断结论，不做疾病分级判断。
- 不提供用药建议。
- 自伤或轻生相关内容只做风险提示和求助建议。
- 默认本地处理聊天数据，不上传原始聊天记录。
- 只分析用户明确选择的联系人、群聊、时间段，或请求体中手工传入的消息。
- 心理分析默认优先使用 `is_mine=true` 的本人消息。
- 日志不打印完整聊天原文、API Key 或敏感路径。

## 快速启动

```bash
cd python_backend
python -m venv .venv
pip install -r requirements.txt
python run.py
```

默认监听：`http://127.0.0.1:8000`

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

## 配置

支持 `.env` 和环境变量：

```env
DATA_DIR=../decrypted
AI_DB_PATH=./ai_analysis.db
VECTOR_DB_PATH=./vector_index.db

LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:7b
LLM_API_KEY=

EMBEDDING_PROVIDER=ollama
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_API_KEY=
EMBEDDING_USE_SEARCH_PREFIX=auto
EMBEDDING_DOCUMENT_PREFIX=search_document:
EMBEDDING_QUERY_PREFIX=search_query:
EMBEDDING_BATCH_SIZE=8
EMBEDDING_TIMEOUT=180
EMBEDDING_PREPROCESS_WORKERS=8
EMBEDDING_MAX_BATCH_SIZE=32
EMBEDDING_MAX_BATCH_TOKENS=2048
EMBEDDING_HTTP_RETRIES=2
EMBEDDING_RETRY_BACKOFF=1.5
EMBEDDING_MAX_FAILED_ITEMS=0
EMBEDDING_MAX_FAILED_RATIO=0.02
```

向量索引构建会自动检测 CPU/GPU 型号，并采用多进程 CPU 预处理 + 动态 batch 的流水线方式调用 Embedding 服务。向量索引现在单独写入 `VECTOR_DB_PATH`，默认是 `python_backend/vector_index.db`；心理任务、报告和评分仍写入 `AI_DB_PATH`。`EMBEDDING_MAX_BATCH_SIZE` 控制单次最大消息数，`EMBEDDING_MAX_BATCH_TOKENS` 控制按估算 token 数组合 batch 的上限。

硬件检测采用 best-effort 策略，兼容 Windows、macOS、Linux 的常见命令；检测不到 GPU 时会显示 CPU 模式，不影响后端运行。多进程预处理不可用时会自动降级为单进程动态 batch。

Embedding 批次请求失败时会自动重试、拆分批次；拆到单条仍失败时会跳过该条并继续。默认容错上限为 `max(1000, 本次待索引消息数 * EMBEDDING_MAX_FAILED_RATIO)`，也可以用 `EMBEDDING_MAX_FAILED_ITEMS` 指定固定上限。增量构建会按批次即时合并写入 `VECTOR_DB_PATH`，即使后续任务失败，下次重试也能跳过已经成功写入的消息。任务结果会返回 `embedding_failed_count`、`embedding_failure_limit`、`embedding_failed_ratio` 和 `last_embedding_error`，不会记录完整聊天原文。

重复建立同一索引时默认走增量模式：每条消息使用 `source_key + seq + datetime + sender` 生成稳定源 ID，并保存 `content_hash`；源 ID 已存在且内容哈希未变化的消息会跳过，只对新增或变化的消息重新计算向量。当前模型变化时会自动全量重建，以避免同一索引混入不同维度或不同模型的 embedding。构建中的任务可通过 `POST /api/ai/vec/build-index/{task_id}/cancel` 请求停止，停止会在当前批次结束后生效。

心理分析流程会按“读取聊天消息 -> 预处理与隐私过滤 -> 关键词检索 -> 大模型筛选 -> 向量语义检索 -> 大模型筛选 -> 自伤/轻生风险识别 -> 心理事实抽取 -> 事实向量索引/检索 -> 综合评分 -> 生成报告”执行。向量语义检索会先确保本次选择范围已有索引：未建立或缺失的消息会走增量构建并合并进 SQLite，已存在且内容未变化的消息会跳过；补建失败时会记录原因并继续后续规则分析。

心理事实抽取会优先调用本地 Ollama 或 OpenAI-compatible LLM 抽取结构化事实；模型不可用时会基于规则证据生成兜底事实。抽取后的事实会使用独立索引键 `psych_facts:{task_id}` 再做一次向量索引与检索，检索命中会写入报告 JSON 的 `fact_vector_hits`，用于辅助定位重点事实。

`DATA_DIR` 期望兼容 MindTrace 解密后的目录结构：

```text
decrypted/
  contact/contact.db
  message/message_*.db
```

## 主要 API

- `GET /health`
- `GET /api/app/info`
- `GET /api/contacts`
- `GET /api/messages?target_key=wxid_xxx&target_type=contact&limit=500`
- `POST /api/psych/analyze`
- `GET /api/psych/tasks/{task_id}/progress`
- `GET /api/psych/tasks/{task_id}/score`
- `GET /api/psych/tasks/{task_id}/report`
- `GET /api/psych/tasks/{task_id}/evidence`
- `POST /api/ai/vec/build-index`
- `GET /api/ai/vec/search?key=xxx&q=关键词`
- `POST /api/ai/mem/build`
- `GET /api/ai/mem/search?key=xxx&q=关键词`

## 训练优化闭环

训练功能采用“分析 -> 人工审核 -> 生成草案 -> 人工应用”的闭环，不会让大模型自动悄悄覆盖评分标准。

- `GET /api/psych/training/samples`：查看人审样本。
- `POST /api/psych/training/samples`：保存一次分析结果和人工审核结论。
- `PUT /api/psych/training/samples/{sample_id}`：更新样本审核。
- `DELETE /api/psych/training/samples/{sample_id}`：删除样本。
- `POST /api/psych/training/proposals/generate`：基于人审样本和当前评分标准生成优化草案。
- `GET /api/psych/training/proposals`：查看历史草案。
- `POST /api/psych/training/proposals/apply`：人工确认后应用草案到 `psych_scoring_config.json`。

优化草案主要调整关键词、排除词、向量查询和风险阈值。大模型 Prompt 明确禁止输出医学诊断结论或药物建议；若大模型不可用，会退回本地启发式建议。

可在设置页开启“训练自动复盘”，或在 `.env` 中配置：

```env
TRAINING_AUTO_REVIEW_ENABLED=false
TRAINING_AUTO_REVIEW_USE_LLM=true
TRAINING_AUTO_PROPOSAL_ENABLED=true
TRAINING_AUTO_MAX_SAMPLES=1
```

`TRAINING_AUTO_MAX_SAMPLES` 可填数字，也可填 `max` 表示生成草案时参考全部已审核样本。开启后，心理分析完成会自动生成一条 AI 审核训练样本，并可自动生成评分标准优化草案。草案不会自动应用，仍需在“训练优化”页面人工确认。

## 手工消息心理分析示例

```bash
curl -X POST http://127.0.0.1:8000/api/psych/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "target_key": "manual:test",
    "target_type": "contact",
    "only_mine": true,
    "messages": [
      {
        "seq": 1,
        "datetime": "2026-05-06 01:30:00",
        "sender": "我",
        "content": "最近一直睡不着，感觉很累，什么都不想做",
        "is_mine": true,
        "contact_key": "manual:test"
      }
    ]
  }'
```

返回内容包含 `features`、`evidences`、`score`、`report_md`。报告中会包含免责声明。

## 微信数据库兼容说明

当前读取逻辑是适配式基础版本：

1. 先查询 `sqlite_master` 获取表名；
2. 再用 `PRAGMA table_info` 获取字段；
3. 根据常见字段名匹配联系人、消息内容、时间、发送者、是否本人发送等字段；
4. 对 MindTrace 常见 `Msg_<md5(username)>` 表和 `Name2Id` 映射做了基础兼容。

不同微信版本、解密工具或 WCDB 导出方式可能存在字段差异。如果 `/api/contacts` 或 `/api/messages` 返回空列表，需要根据实际 SQLite schema 调整 `app/services/wechat_db.py` 中的字段候选列表和解码逻辑。

部分微信消息内容会以 zstd 压缩 BLOB 存储，后端已接入 `zstandard` 做轻量解压。更新代码后请重新执行 `pip install -r requirements.txt`，否则压缩文本会退回 best-effort 解码，可能只能读到行数而读不到真实文本。
