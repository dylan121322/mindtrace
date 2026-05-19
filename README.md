# MindTrace

> License: AGPL-3.0-only  
> Based on the original project [runzhliu/welink](https://github.com/runzhliu/welink). This repository is a modified derivative focused on a Python FastAPI backend and psychological risk auxiliary screening workflow.
> AI-assisted development notice: parts of this project were generated with AI coding assistance and then reviewed, modified, and integrated by the maintainer.

本项目是一个本地化的微信聊天记录心理风险辅助筛查系统，后端使用 Python FastAPI，前端使用 React。系统用于识别聊天文本中的抑郁相关信号、保护性信号和风险线索，输出可解释评分、证据、可视化图表、规则调试和人工审核训练草案。

重要边界：

- 本项目不提供医学诊断，不输出“确诊抑郁症”“重度抑郁症”等结论。
- 报告仅定位为“心理风险辅助筛查”“抑郁相关信号识别”“建议进一步专业评估”。
- 不提供药物建议。
- 高风险自伤/轻生相关表达只做风险提示和求助建议，不作为诊断。
- 默认本地化处理，聊天数据、向量库、分析数据库不应上传到云端或 GitHub。

## 目录结构

```text
.
├── frontend/          # React 前端
├── python_backend/    # FastAPI 后端
├── logo.svg
├── LICENSE
└── README.md
```

## 后端启动

```powershell
cd python_backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python run.py
```

默认后端地址：

```text
http://127.0.0.1:8000
```

健康检查：

```powershell
curl http://127.0.0.1:8000/health
```

## 前端启动

```powershell
cd frontend
npm install
npm run dev
```

默认前端地址以 Vite 输出为准。

## 配置说明

请复制：

```text
python_backend/.env.example
```

为：

```text
python_backend/.env
```

然后按本机环境修改：

- `DATA_DIR`：解密后的微信数据库目录，上层目录应包含 `contact/contact.db` 和 `message/message_*.db`
- `AI_DB_PATH`：心理分析、训练样本、报告等本地 SQLite 数据库
- `VECTOR_DB_PATH`：独立向量索引 SQLite 数据库
- `LLM_*`：大模型配置，可使用 Ollama、DeepSeek、Kimi 或 OpenAI-compatible 服务
- `EMBEDDING_*`：向量模型配置，Ollama 使用 `nomic-embed-text` 时不需要 API Key

不要提交 `.env`、微信数据库、分析数据库、向量数据库、日志或训练产生的本地数据。

## 功能概览

- 联系人数据库读取与诊断
- 心理分析异步任务与进度可视化
- 关键词检索、大模型筛选、向量语义检索
- 多标签分类与单一风险等级
- 100 分抑郁相关信号评分表
- 证据强度系数、时间维度修正、保护性修正因子
- 评分标准可视化编辑
- 细节调试页面，展示第 3-6 步候选、Prompt 和筛选结果
- 向量数据库独立构建、增量索引、停止任务
- 心理事实库构建与检索
- 人工审核训练样本与 AI 规则优化草案
- 可视化分析页面

## 隐私与安全

本仓库只应包含源码和示例配置。以下内容属于敏感本地数据，不应上传：

- `python_backend/.env`
- `decrypted/`
- `ai_analysis.db`
- `vector_index.db`
- `*.sqlite`
- `*.db`
- `logs/`
- 微信聊天原文导出文件

## 免责声明

本系统仅基于聊天文本进行心理风险辅助筛查，不构成医学诊断。若存在持续痛苦、自伤或轻生想法，请尽快联系专业人员或当地紧急救助服务。

## License And Attribution

MindTrace is released under the GNU Affero General Public License v3.0 only (`AGPL-3.0-only`). See [LICENSE](./LICENSE).

This project is based on and modified from [runzhliu/welink](https://github.com/runzhliu/welink). The original project is also distributed under the GNU Affero General Public License. MindTrace keeps the same copyleft license and preserves attribution to the upstream project.
