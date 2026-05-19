# Python Backend

这是 MindTrace 的 Python 后端，使用 FastAPI + SQLite 实现微信聊天记录读取、向量索引、心理事实库、评分标准配置、训练样本审核和心理风险辅助筛查。

License: AGPL-3.0-only. MindTrace is based on and modified from [runzhliu/welink](https://github.com/runzhliu/welink).

AI-assisted development notice: parts of this project were generated with AI coding assistance and then reviewed, modified, and integrated by the maintainer.

## 安全边界

- 本系统只做心理风险辅助筛查和抑郁相关信号识别，不构成医学诊断。
- 不输出“确诊抑郁症”“重度抑郁症”等医学诊断结论。
- 不提供药物建议。
- 高风险自伤/轻生相关表达只做风险提示和求助建议。
- 默认本地处理聊天数据，不上传原始聊天记录。
- 日志不打印完整聊天原文、API Key 或敏感路径。

## 启动

```powershell
cd python_backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python run.py
```

默认监听：

```text
http://127.0.0.1:8000
```

健康检查：

```powershell
curl http://127.0.0.1:8000/health
```

## 配置

`.env.example` 提供了本地示例配置。常用字段：

- `DATA_DIR`：解密后的微信数据库目录。
- `AI_DB_PATH`：心理分析、训练样本和报告 SQLite 数据库。
- `VECTOR_DB_PATH`：独立向量索引 SQLite 数据库。
- `LLM_PROVIDER / LLM_BASE_URL / LLM_MODEL / LLM_API_KEY`：大模型服务配置。
- `EMBEDDING_PROVIDER / EMBEDDING_BASE_URL / EMBEDDING_MODEL`：向量模型配置。

Ollama 默认不需要 API Key。

## 数据目录

`DATA_DIR` 期望类似：

```text
decrypted/
  contact/contact.db
  message/message_*.db
```

这些数据属于本地敏感数据，不应提交到 GitHub。

## 主要 API

- `GET /health`
- `GET /api/status`
- `GET /api/contacts`
- `GET /api/messages/targets`
- `GET /api/messages/diagnostics`
- `POST /api/psych/analyze/start`
- `GET /api/psych/tasks/{task_id}/progress`
- `GET /api/psych/tasks/{task_id}/score`
- `GET /api/psych/tasks/{task_id}/report`
- `GET /api/psych/tasks/{task_id}/evidence`
- `GET /api/psych/scoring-config`
- `PUT /api/psych/scoring-config`
- `POST /api/ai/vec/build-index`
- `GET /api/ai/vec/build-index/{task_id}/progress`
- `POST /api/ai/vec/build-index/{task_id}/cancel`
- `POST /api/ai/mem/build`
- `GET /api/ai/mem/search`
- `GET /api/psych/training/samples`
- `POST /api/psych/training/samples`
- `POST /api/psych/training/proposals/generate`
- `POST /api/psych/training/proposals/apply`

## 分析流程

当前心理分析流程：

```text
读取聊天消息
预处理与隐私过滤
关键词检索
大模型筛选
向量语义检索
大模型筛选
心理事实库检索
综合评分
生成报告
```

评分体系包含多标签分类、100 分抑郁相关信号评分、证据强度系数、时间维度修正、保护性修正因子和单一风险等级。所有规则可在前端“评分标准”页面编辑。

## 训练优化闭环

训练功能采用：

```text
分析 -> 人工审核 -> 生成规则草案 -> 人工应用
```

大模型只能生成规则优化草案，不会自动覆盖评分标准。每次应用草案前都需要人工确认。
