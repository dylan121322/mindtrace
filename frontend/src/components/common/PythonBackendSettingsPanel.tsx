import React, { useCallback, useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import {
  AlertCircle,
  Bot,
  CheckCircle2,
  Database,
  FolderOpen,
  KeyRound,
  Loader2,
  RefreshCw,
  Save,
  Server,
  Sparkles,
} from 'lucide-react';

const LS_PYTHON_BACKEND_URL = 'mindtrace:python-backend-url';
const SECRET_PLACEHOLDER = '__HAS_KEY__';

const ENV_KEYS = [
  'DATA_DIR',
  'AI_DB_PATH',
  'VECTOR_DB_PATH',
  'LLM_PROVIDER',
  'LLM_BASE_URL',
  'LLM_MODEL',
  'LLM_API_KEY',
  'PSYCH_FACT_LLM_PROVIDER',
  'PSYCH_FACT_LLM_BASE_URL',
  'PSYCH_FACT_LLM_MODEL',
  'PSYCH_FACT_LLM_API_KEY',
  'PSYCH_FACT_CHUNK_SIZE',
  'TRAINING_AUTO_REVIEW_ENABLED',
  'TRAINING_AUTO_REVIEW_USE_LLM',
  'TRAINING_AUTO_PROPOSAL_ENABLED',
  'TRAINING_AUTO_MAX_SAMPLES',
  'EMBEDDING_PROVIDER',
  'EMBEDDING_BASE_URL',
  'EMBEDDING_MODEL',
  'EMBEDDING_API_KEY',
  'EMBEDDING_USE_SEARCH_PREFIX',
  'EMBEDDING_DOCUMENT_PREFIX',
  'EMBEDDING_QUERY_PREFIX',
  'EMBEDDING_BATCH_SIZE',
  'EMBEDDING_TIMEOUT',
  'EMBEDDING_PREPROCESS_WORKERS',
  'EMBEDDING_MAX_BATCH_SIZE',
  'EMBEDDING_MAX_BATCH_TOKENS',
  'EMBEDDING_HTTP_RETRIES',
  'EMBEDDING_RETRY_BACKOFF',
  'EMBEDDING_MAX_FAILED_ITEMS',
  'EMBEDDING_MAX_FAILED_RATIO',
  'HOST',
  'PORT',
  'TIMEZONE',
] as const;

type EnvKey = typeof ENV_KEYS[number];
type EnvValues = Record<EnvKey, string>;

interface PythonConfigResponse {
  env_path: string;
  values: Partial<Record<EnvKey, string>>;
  has_secret?: Partial<Record<EnvKey, boolean>>;
}

const emptyValues = (): EnvValues => ({
  DATA_DIR: '',
  AI_DB_PATH: '',
  VECTOR_DB_PATH: '',
  LLM_PROVIDER: 'ollama',
  LLM_BASE_URL: 'http://localhost:11434/v1',
  LLM_MODEL: 'qwen2.5:7b',
  LLM_API_KEY: '',
  PSYCH_FACT_LLM_PROVIDER: 'ollama',
  PSYCH_FACT_LLM_BASE_URL: 'http://localhost:11434/v1',
  PSYCH_FACT_LLM_MODEL: 'qwen2.5:7b',
  PSYCH_FACT_LLM_API_KEY: '',
  PSYCH_FACT_CHUNK_SIZE: '80',
  TRAINING_AUTO_REVIEW_ENABLED: 'false',
  TRAINING_AUTO_REVIEW_USE_LLM: 'true',
  TRAINING_AUTO_PROPOSAL_ENABLED: 'true',
  TRAINING_AUTO_MAX_SAMPLES: '1',
  EMBEDDING_PROVIDER: 'ollama',
  EMBEDDING_BASE_URL: 'http://localhost:11434/v1',
  EMBEDDING_MODEL: 'nomic-embed-text',
  EMBEDDING_API_KEY: '',
  EMBEDDING_USE_SEARCH_PREFIX: 'auto',
  EMBEDDING_DOCUMENT_PREFIX: 'search_document: ',
  EMBEDDING_QUERY_PREFIX: 'search_query: ',
  EMBEDDING_BATCH_SIZE: '8',
  EMBEDDING_TIMEOUT: '180',
  EMBEDDING_PREPROCESS_WORKERS: '8',
  EMBEDDING_MAX_BATCH_SIZE: '32',
  EMBEDDING_MAX_BATCH_TOKENS: '2048',
  EMBEDDING_HTTP_RETRIES: '2',
  EMBEDDING_RETRY_BACKOFF: '1.5',
  EMBEDDING_MAX_FAILED_ITEMS: '0',
  EMBEDDING_MAX_FAILED_RATIO: '0.02',
  HOST: '127.0.0.1',
  PORT: '8000',
  TIMEZONE: 'Asia/Shanghai',
});

const MODEL_PRESETS = [
  {
    key: 'deepseek-v4-pro',
    label: 'DeepSeek V4 Pro',
    provider: 'deepseek',
    baseURL: 'https://api.deepseek.com/v1',
    model: 'deepseek-v4-pro',
  },
  {
    key: 'kimi',
    label: 'Kimi',
    provider: 'kimi',
    baseURL: 'https://api.moonshot.cn/v1',
    model: 'kimi-k2.5',
  },
  {
    key: 'kimi-thinking',
    label: 'Kimi Thinking',
    provider: 'kimi',
    baseURL: 'https://api.moonshot.cn/v1',
    model: 'kimi-k2-thinking',
  },
  {
    key: 'ollama',
    label: 'Ollama 本地',
    provider: 'ollama',
    baseURL: 'http://localhost:11434/v1',
    model: 'qwen2.5:7b',
  },
];

function joinApi(baseURL: string, path: string): string {
  return baseURL.replace(/\/+$/, '') + path;
}

function normalizeBaseURL(raw: string): string {
  const value = raw.trim();
  if (!value) return window.location.origin;
  return value.replace(/\/+$/, '');
}

function defaultBackendURL(): string {
  const saved = localStorage.getItem(LS_PYTHON_BACKEND_URL);
  const publicPage = !['localhost', '127.0.0.1', '::1'].includes(window.location.hostname);
  if (publicPage && saved && /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?/i.test(saved)) {
    return window.location.origin;
  }
  return saved || window.location.origin;
}

function isOllamaProvider(provider: string): boolean {
  return provider.trim().toLowerCase() === 'ollama';
}

function valuesForSave(values: EnvValues): EnvValues {
  return {
    ...values,
    LLM_API_KEY: isOllamaProvider(values.LLM_PROVIDER) ? '' : values.LLM_API_KEY,
    PSYCH_FACT_LLM_API_KEY: isOllamaProvider(values.PSYCH_FACT_LLM_PROVIDER) ? '' : values.PSYCH_FACT_LLM_API_KEY,
    EMBEDDING_API_KEY: isOllamaProvider(values.EMBEDDING_PROVIDER) ? '' : values.EMBEDDING_API_KEY,
  };
}

function modelProfileLabel(profile: 'main' | 'psych_fact' | 'training'): string {
  if (profile === 'psych_fact') return '心理事实抽取模型';
  if (profile === 'training') return '训练优化模型';
  return '心理分析模型';
}

export const PythonBackendSettingsPanel: React.FC = () => {
  const [backendURL, setBackendURL] = useState(defaultBackendURL);
  const [values, setValues] = useState<EnvValues>(emptyValues);
  const [envPath, setEnvPath] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState<
    'health' | 'main-llm' | 'fact-llm' | 'training-llm' | 'embedding-query' | 'embedding-document' | null
  >(null);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);

  const apiBase = useMemo(() => normalizeBaseURL(backendURL), [backendURL]);

  const setField = (key: EnvKey, value: string) => {
    setValues(prev => ({ ...prev, [key]: value }));
  };

  const loadConfig = useCallback(async () => {
    setLoading(true);
    setMessage(null);
    try {
      const resp = await axios.get<PythonConfigResponse>(joinApi(apiBase, '/api/python-config'), { timeout: 10000 });
      const next = emptyValues();
      for (const key of ENV_KEYS) {
        const value = resp.data.values?.[key];
        if (typeof value === 'string') next[key] = value;
      }
      setValues(next);
      setEnvPath(resp.data.env_path || '');
      localStorage.setItem(LS_PYTHON_BACKEND_URL, apiBase);
      setMessage({ ok: true, text: '已读取 Python 后端配置' });
    } catch (error) {
      const text = axios.isAxiosError(error)
        ? (error.response?.data?.detail || error.message)
        : '读取失败';
      setMessage({ ok: false, text: `无法连接 Python 后端：${text}` });
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  const applyPreset = (preset: typeof MODEL_PRESETS[number]) => {
    setValues(prev => ({
      ...prev,
      LLM_PROVIDER: preset.provider,
      LLM_BASE_URL: preset.baseURL,
      LLM_MODEL: preset.model,
      LLM_API_KEY: isOllamaProvider(preset.provider) ? '' : prev.LLM_API_KEY,
    }));
  };

  const saveConfig = async (): Promise<boolean> => {
    setSaving(true);
    setMessage(null);
    try {
      const resp = await axios.put<PythonConfigResponse>(
        joinApi(apiBase, '/api/python-config'),
        { values: valuesForSave(values) },
        { timeout: 10000 },
      );
      const next = emptyValues();
      for (const key of ENV_KEYS) {
        const value = resp.data.values?.[key];
        if (typeof value === 'string') next[key] = value;
      }
      setValues(next);
      setEnvPath(resp.data.env_path || envPath);
      localStorage.setItem(LS_PYTHON_BACKEND_URL, apiBase);
      setMessage({ ok: true, text: '已保存到 Python 后端 .env。HOST/PORT 变更需要重启 Python 后端进程。' });
      return true;
    } catch (error) {
      const text = axios.isAxiosError(error)
        ? (error.response?.data?.detail || error.message)
        : '保存失败';
      setMessage({ ok: false, text: String(text) });
      return false;
    } finally {
      setSaving(false);
    }
  };

  const testHealth = async () => {
    setTesting('health');
    setMessage(null);
    try {
      const resp = await axios.get<{ status: string }>(joinApi(apiBase, '/health'), { timeout: 8000 });
      setMessage({ ok: resp.data.status === 'ok', text: `Python 后端状态：${resp.data.status}` });
    } catch (error) {
      const text = axios.isAxiosError(error) ? error.message : '连接失败';
      setMessage({ ok: false, text });
    } finally {
      setTesting(null);
    }
  };

  const testLLM = async (profile: 'main' | 'psych_fact' | 'training') => {
    const testingKey = profile === 'main' ? 'main-llm' : profile === 'psych_fact' ? 'fact-llm' : 'training-llm';
    setTesting(testingKey);
    setMessage(null);
    try {
      const saved = await saveConfig();
      if (!saved) return;
      const prompt =
        profile === 'psych_fact'
          ? '请回复：心理事实抽取模型连接正常'
          : profile === 'training'
            ? '请回复：训练优化复盘模型连接正常'
            : '请回复：心理分析模型连接正常';
      const resp = await axios.post<{ ok: boolean; reply: string; provider?: string; model?: string; profile?: string }>(
        joinApi(apiBase, '/api/ai/llm/test'),
        { prompt, profile },
        { timeout: 120000 },
      );
      setMessage({
        ok: resp.data.ok,
        text: `${modelProfileLabel(profile)}连接正常：${resp.data.provider || ''}/${resp.data.model || ''} · ${resp.data.reply || 'ok'}`,
      });
    } catch (error) {
      const text = axios.isAxiosError(error)
        ? (error.response?.data?.detail || error.response?.data?.error || error.message)
        : `${modelProfileLabel(profile)}测试失败`;
      setMessage({ ok: false, text: String(text) });
    } finally {
      setTesting(null);
    }
  };

  const testEmbedding = async (inputType: 'query' | 'document') => {
    const testingKey = inputType === 'query' ? 'embedding-query' : 'embedding-document';
    setTesting(testingKey);
    setMessage(null);
    try {
      const saved = await saveConfig();
      if (!saved) return;
      const resp = await axios.post<{ ok: boolean; dims: number; provider?: string; model?: string; input_type?: string; uses_search_prefix?: boolean }>(
        joinApi(apiBase, `/api/ai/embedding/test?input_type=${inputType}`),
        {},
        { timeout: 120000 },
      );
      setMessage({
        ok: resp.data.ok,
        text: `Embedding ${inputType === 'query' ? '查询' : '文档'}连接正常：${resp.data.provider || ''}/${resp.data.model || ''}，维度 ${resp.data.dims}，RAG 前缀 ${resp.data.uses_search_prefix ? '启用' : '未启用'}`,
      });
    } catch (error) {
      const text = axios.isAxiosError(error)
        ? (error.response?.data?.detail || error.response?.data?.error || error.message)
        : 'Embedding 测试失败';
      setMessage({ ok: false, text: String(text) });
    } finally {
      setTesting(null);
    }
  };

  const clearSecret = (key: 'LLM_API_KEY' | 'PSYCH_FACT_LLM_API_KEY' | 'EMBEDDING_API_KEY') => {
    setValues(prev => ({ ...prev, [key]: '' }));
  };

  const llmUsesOllama = isOllamaProvider(values.LLM_PROVIDER);
  const psychFactLlmUsesOllama = isOllamaProvider(values.PSYCH_FACT_LLM_PROVIDER);
  const embeddingUsesOllama = isOllamaProvider(values.EMBEDDING_PROVIDER);

  return (
    <section className="mb-8" data-settings-tags="python backend fastapi env deepseek kimi moonshot 心理分析">
      <div className="flex items-center gap-2 mb-3">
        <Server size={18} className="text-[#07c160]" />
        <h3 className="text-base font-bold text-[#1d1d1f] dk-text">Python 后端配置</h3>
      </div>
      <p className="text-sm text-gray-400 mb-4">
        修改 FastAPI 后端的 <code className="font-mono">.env</code>，用于心理风险辅助筛查、向量检索和 Python 侧 LLM 调用。
      </p>

      <div className="bg-white rounded-3xl border border-gray-100 shadow-sm p-6 dk-card dk-border space-y-5">
        <div className="grid grid-cols-1 lg:grid-cols-[1.2fr_0.8fr] gap-4">
          <div>
            <label className="flex items-center gap-1.5 text-xs font-bold text-gray-500 mb-1 uppercase tracking-wide">
              <Server size={13} />
              Python API 地址
            </label>
            <div className="flex gap-2">
              <input
                value={backendURL}
                onChange={e => setBackendURL(e.target.value)}
                placeholder={window.location.origin}
                className="flex-1 px-3 py-2 rounded-xl bg-gray-50 dark:bg-white/5 border border-gray-200 dark:border-white/10 text-sm font-mono dk-text outline-none focus:border-[#07c160]"
              />
              <button
                type="button"
                onClick={loadConfig}
                disabled={loading}
                className="px-3 py-2 rounded-xl bg-gray-100 dark:bg-white/10 text-gray-600 dark:text-gray-300 text-sm font-semibold hover:bg-gray-200 dark:hover:bg-white/15 disabled:opacity-50 flex items-center gap-1.5"
              >
                {loading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                读取
              </button>
            </div>
            {envPath && <p className="mt-1 text-[11px] text-gray-400 font-mono break-all">.env：{envPath}</p>}
          </div>

          <div className="flex items-end gap-2 flex-wrap">
            <button
              type="button"
              onClick={testHealth}
              disabled={testing !== null || saving || loading}
              className="px-3 py-2 rounded-xl border border-gray-200 dark:border-white/10 text-gray-600 dark:text-gray-300 text-sm font-semibold hover:border-[#07c160] hover:text-[#07c160] disabled:opacity-50 flex items-center gap-1.5"
            >
              {testing === 'health' ? <Loader2 size={14} className="animate-spin" /> : <CheckCircle2 size={14} />}
              测试后端
            </button>
            <button
              type="button"
              onClick={saveConfig}
              disabled={saving || loading}
              className="px-4 py-2 rounded-xl bg-[#07c160] text-white text-sm font-bold hover:bg-[#06ad56] disabled:opacity-50 flex items-center gap-1.5"
            >
              {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
              保存 .env
            </button>
          </div>
        </div>

        {message && (
          <div className={`flex items-start gap-2 rounded-2xl px-4 py-3 border ${
            message.ok
              ? 'bg-green-50 dark:bg-green-500/10 border-green-200 dark:border-green-500/30'
              : 'bg-red-50 dark:bg-red-500/10 border-red-200 dark:border-red-500/30'
          }`}>
            {message.ok ? <CheckCircle2 size={16} className="text-green-500 flex-shrink-0 mt-0.5" /> : <AlertCircle size={16} className="text-red-500 flex-shrink-0 mt-0.5" />}
            <p className={`text-sm break-all ${message.ok ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'}`}>{message.text}</p>
          </div>
        )}

        <div>
          <div className="flex items-center gap-1.5 mb-2">
            <Sparkles size={14} className="text-[#07c160]" />
            <span className="text-xs font-bold text-gray-500 uppercase tracking-wide">模型预设</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {MODEL_PRESETS.map(preset => (
              <button
                key={preset.key}
                type="button"
                onClick={() => applyPreset(preset)}
                className="px-3 py-1.5 rounded-full border border-gray-200 dark:border-white/10 bg-gray-50 dark:bg-white/5 text-xs font-bold text-gray-600 dark:text-gray-300 hover:border-[#07c160] hover:text-[#07c160] hover:bg-[#07c160]/5 transition-colors"
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="space-y-3">
            <div className="flex items-center gap-1.5">
              <Database size={14} className="text-[#07c160]" />
              <h4 className="text-sm font-bold text-[#1d1d1f] dark:text-white">数据与服务</h4>
            </div>
            <TextField label="微信解密数据目录" icon={<FolderOpen size={13} />} value={values.DATA_DIR} onChange={v => setField('DATA_DIR', v)} placeholder="E:/.../decrypted" />
            <TextField label="AI 数据库路径" icon={<Database size={13} />} value={values.AI_DB_PATH} onChange={v => setField('AI_DB_PATH', v)} placeholder="E:/.../python_backend/ai_analysis.db" />
            <TextField label="向量索引库路径" icon={<Database size={13} />} value={values.VECTOR_DB_PATH} onChange={v => setField('VECTOR_DB_PATH', v)} placeholder="E:/.../python_backend/vector_index.db" />
            <div className="grid grid-cols-2 gap-2">
              <TextField label="Host" value={values.HOST} onChange={v => setField('HOST', v)} placeholder="127.0.0.1" />
              <TextField label="Port" value={values.PORT} onChange={v => setField('PORT', v)} placeholder="8000" />
            </div>
            <TextField label="时区" value={values.TIMEZONE} onChange={v => setField('TIMEZONE', v)} placeholder="Asia/Shanghai" />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-1.5">
                <Bot size={14} className="text-[#07c160]" />
                <h4 className="text-sm font-bold text-[#1d1d1f] dark:text-white">LLM</h4>
              </div>
              <button
                type="button"
                onClick={() => testLLM('main')}
                disabled={testing !== null || saving || loading}
                className="px-3 py-1.5 rounded-lg border border-gray-200 dark:border-white/10 text-xs font-bold text-gray-500 hover:border-[#07c160] hover:text-[#07c160] disabled:opacity-50 flex items-center gap-1"
              >
                {testing === 'main-llm' ? <Loader2 size={12} className="animate-spin" /> : <AlertCircle size={12} />}
                测试心理分析模型
              </button>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <TextField label="Provider" value={values.LLM_PROVIDER} onChange={v => setField('LLM_PROVIDER', v)} placeholder="deepseek / kimi / ollama" />
              <TextField label="Model" value={values.LLM_MODEL} onChange={v => setField('LLM_MODEL', v)} placeholder="deepseek-v4-pro" />
            </div>
            <TextField label="Base URL" value={values.LLM_BASE_URL} onChange={v => setField('LLM_BASE_URL', v)} placeholder="https://api.deepseek.com/v1" />
            <SecretField
              label="API Key"
              value={values.LLM_API_KEY}
              hasKey={values.LLM_API_KEY === SECRET_PLACEHOLDER}
              onChange={v => setField('LLM_API_KEY', v)}
              onClear={() => clearSecret('LLM_API_KEY')}
              disabled={llmUsesOllama}
              helpText={llmUsesOllama ? 'Ollama 本地服务不需要 API Key，保存时会自动留空。' : undefined}
            />
          </div>
        </div>

        <div className="rounded-2xl bg-gray-50 dark:bg-white/5 border border-gray-100 dark:border-white/10 p-4 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5">
              <Bot size={14} className="text-[#07c160]" />
              <h4 className="text-sm font-bold text-[#1d1d1f] dark:text-white">心理事实抽取 LLM</h4>
            </div>
            <button
              type="button"
              onClick={() => testLLM('psych_fact')}
              disabled={testing !== null || saving || loading}
              className="px-3 py-1.5 rounded-lg border border-gray-200 dark:border-white/10 text-xs font-bold text-gray-500 hover:border-[#07c160] hover:text-[#07c160] disabled:opacity-50 flex items-center gap-1"
            >
              {testing === 'fact-llm' ? <Loader2 size={12} className="animate-spin" /> : <AlertCircle size={12} />}
              测试事实模型
            </button>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <TextField label="Provider" value={values.PSYCH_FACT_LLM_PROVIDER} onChange={v => setField('PSYCH_FACT_LLM_PROVIDER', v)} placeholder="deepseek / kimi / ollama" />
            <TextField label="Model" value={values.PSYCH_FACT_LLM_MODEL} onChange={v => setField('PSYCH_FACT_LLM_MODEL', v)} placeholder="deepseek-v4-pro" />
            <TextField label="Base URL" value={values.PSYCH_FACT_LLM_BASE_URL} onChange={v => setField('PSYCH_FACT_LLM_BASE_URL', v)} placeholder="https://api.deepseek.com/v1" />
            <SecretField
              label="API Key"
              value={values.PSYCH_FACT_LLM_API_KEY}
              hasKey={values.PSYCH_FACT_LLM_API_KEY === SECRET_PLACEHOLDER}
              onChange={v => setField('PSYCH_FACT_LLM_API_KEY', v)}
              onClear={() => clearSecret('PSYCH_FACT_LLM_API_KEY')}
              disabled={psychFactLlmUsesOllama}
              helpText={psychFactLlmUsesOllama ? 'Ollama 本地服务不需要 API Key，保存时会自动留空。' : undefined}
            />
            <TextField label="每批消息数" value={values.PSYCH_FACT_CHUNK_SIZE} onChange={v => setField('PSYCH_FACT_CHUNK_SIZE', v)} placeholder="80" />
          </div>
          <p className="text-xs text-gray-500">
            数据库页会按这个模型每批抽取心理事实，写入 <code className="font-mono">mem_facts</code>；心理分析只检索已入库事实。
          </p>
        </div>

        <div className="rounded-2xl bg-gray-50 dark:bg-white/5 border border-gray-100 dark:border-white/10 p-4 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5">
              <Sparkles size={14} className="text-[#07c160]" />
              <h4 className="text-sm font-bold text-[#1d1d1f] dark:text-white">训练自动复盘</h4>
            </div>
            <button
              type="button"
              onClick={() => testLLM('training')}
              disabled={testing !== null || saving || loading}
              className="px-3 py-1.5 rounded-lg border border-gray-200 dark:border-white/10 text-xs font-bold text-gray-500 hover:border-[#07c160] hover:text-[#07c160] disabled:opacity-50 flex items-center gap-1"
            >
              {testing === 'training-llm' ? <Loader2 size={12} className="animate-spin" /> : <AlertCircle size={12} />}
              测试训练模型
            </button>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
            <ToggleField
              label="分析后自动审核"
              description="心理分析完成后自动生成一条 AI 审核训练样本，仍需人工确认。"
              checked={values.TRAINING_AUTO_REVIEW_ENABLED === 'true'}
              onChange={checked => setField('TRAINING_AUTO_REVIEW_ENABLED', checked ? 'true' : 'false')}
            />
            <ToggleField
              label="自动审核使用大模型"
              description="关闭后仅按当前分析结果生成样本，不调用 LLM 复盘。"
              checked={values.TRAINING_AUTO_REVIEW_USE_LLM === 'true'}
              onChange={checked => setField('TRAINING_AUTO_REVIEW_USE_LLM', checked ? 'true' : 'false')}
            />
            <ToggleField
              label="自动生成优化草案"
              description="根据自动审核样本生成评分标准优化草案，但不会自动应用。"
              checked={values.TRAINING_AUTO_PROPOSAL_ENABLED === 'true'}
              onChange={checked => setField('TRAINING_AUTO_PROPOSAL_ENABLED', checked ? 'true' : 'false')}
            />
          </div>
          <TextField
            label="草案参考样本数"
            value={values.TRAINING_AUTO_MAX_SAMPLES}
            onChange={v => setField('TRAINING_AUTO_MAX_SAMPLES', v)}
            placeholder="1 或 max"
          />
          <p className="text-xs text-gray-500">
            自动复盘只会写入训练样本和优化草案。草案参考样本数可填数字，也可填 max 使用全部已审核样本；评分标准必须在“训练优化”页面人工确认后才会应用。
          </p>
        </div>

        <div className="rounded-2xl bg-gray-50 dark:bg-white/5 border border-gray-100 dark:border-white/10 p-4 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5">
              <KeyRound size={14} className="text-[#07c160]" />
              <h4 className="text-sm font-bold text-[#1d1d1f] dark:text-white">Embedding</h4>
            </div>
            <button
              type="button"
              onClick={() => testEmbedding('query')}
              disabled={testing !== null || saving || loading}
              className="px-3 py-1.5 rounded-lg border border-gray-200 dark:border-white/10 text-xs font-bold text-gray-500 hover:border-[#07c160] hover:text-[#07c160] disabled:opacity-50 flex items-center gap-1"
            >
              {testing === 'embedding-query' ? <Loader2 size={12} className="animate-spin" /> : <AlertCircle size={12} />}
              测试查询向量
            </button>
            <button
              type="button"
              onClick={() => testEmbedding('document')}
              disabled={testing !== null || saving || loading}
              className="px-3 py-1.5 rounded-lg border border-gray-200 dark:border-white/10 text-xs font-bold text-gray-500 hover:border-[#07c160] hover:text-[#07c160] disabled:opacity-50 flex items-center gap-1"
            >
              {testing === 'embedding-document' ? <Loader2 size={12} className="animate-spin" /> : <AlertCircle size={12} />}
              测试文档向量
            </button>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <TextField label="Provider" value={values.EMBEDDING_PROVIDER} onChange={v => setField('EMBEDDING_PROVIDER', v)} placeholder="ollama / openai / local" />
            <TextField label="Model" value={values.EMBEDDING_MODEL} onChange={v => setField('EMBEDDING_MODEL', v)} placeholder="nomic-embed-text" />
            <TextField label="Base URL" value={values.EMBEDDING_BASE_URL} onChange={v => setField('EMBEDDING_BASE_URL', v)} placeholder="http://localhost:11434/v1" />
            <SecretField
              label="API Key"
              value={values.EMBEDDING_API_KEY}
              hasKey={values.EMBEDDING_API_KEY === SECRET_PLACEHOLDER}
              onChange={v => setField('EMBEDDING_API_KEY', v)}
              onClear={() => clearSecret('EMBEDDING_API_KEY')}
              disabled={embeddingUsesOllama}
              helpText={embeddingUsesOllama ? 'Ollama 本地向量模型不需要 API Key，保存时会自动留空。' : undefined}
            />
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <TextField
              label="RAG 前缀模式"
              value={values.EMBEDDING_USE_SEARCH_PREFIX}
              onChange={v => setField('EMBEDDING_USE_SEARCH_PREFIX', v)}
              placeholder="auto / true / false"
            />
            <TextField
              label="文档前缀"
              value={values.EMBEDDING_DOCUMENT_PREFIX}
              onChange={v => setField('EMBEDDING_DOCUMENT_PREFIX', v)}
              placeholder="search_document: "
            />
            <TextField
              label="查询前缀"
              value={values.EMBEDDING_QUERY_PREFIX}
              onChange={v => setField('EMBEDDING_QUERY_PREFIX', v)}
              placeholder="search_query: "
            />
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
            <TextField label="固定批量" value={values.EMBEDDING_BATCH_SIZE} onChange={v => setField('EMBEDDING_BATCH_SIZE', v)} placeholder="8" />
            <TextField label="请求超时秒" value={values.EMBEDDING_TIMEOUT} onChange={v => setField('EMBEDDING_TIMEOUT', v)} placeholder="180" />
            <TextField label="CPU worker" value={values.EMBEDDING_PREPROCESS_WORKERS} onChange={v => setField('EMBEDDING_PREPROCESS_WORKERS', v)} placeholder="8" />
            <TextField label="最大 batch" value={values.EMBEDDING_MAX_BATCH_SIZE} onChange={v => setField('EMBEDDING_MAX_BATCH_SIZE', v)} placeholder="32" />
            <TextField label="最大 token batch" value={values.EMBEDDING_MAX_BATCH_TOKENS} onChange={v => setField('EMBEDDING_MAX_BATCH_TOKENS', v)} placeholder="2048" />
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
            <TextField label="HTTP 重试" value={values.EMBEDDING_HTTP_RETRIES} onChange={v => setField('EMBEDDING_HTTP_RETRIES', v)} placeholder="2" />
            <TextField label="重试退避秒" value={values.EMBEDDING_RETRY_BACKOFF} onChange={v => setField('EMBEDDING_RETRY_BACKOFF', v)} placeholder="1.5" />
            <TextField label="失败上限" value={values.EMBEDDING_MAX_FAILED_ITEMS} onChange={v => setField('EMBEDDING_MAX_FAILED_ITEMS', v)} placeholder="0=自动" />
            <TextField label="失败比例" value={values.EMBEDDING_MAX_FAILED_RATIO} onChange={v => setField('EMBEDDING_MAX_FAILED_RATIO', v)} placeholder="0.02" />
          </div>
          <p className="text-xs text-gray-500">
            建立向量数据库时会先用 CPU worker 并行预处理文本，再按估算 token 数动态组合 batch；少量 HTTPError 会重试并跳过，避免单条异常消息打断整次索引。
          </p>
          <p className="text-xs text-gray-500">
            Nomic 类 embedding 在 auto 模式下会自动把入库文本作为 <code className="font-mono">search_document</code>，把检索查询作为 <code className="font-mono">search_query</code>。
          </p>
        </div>
      </div>
    </section>
  );
};

const TextField: React.FC<{
  label: string;
  value: string;
  placeholder?: string;
  icon?: React.ReactNode;
  onChange: (value: string) => void;
}> = ({ label, value, placeholder, icon, onChange }) => (
  <label className="block">
    <span className="flex items-center gap-1.5 text-xs font-bold text-gray-500 mb-1 uppercase tracking-wide">
      {icon}
      {label}
    </span>
    <input
      type="text"
      value={value}
      onChange={e => onChange(e.target.value)}
      placeholder={placeholder}
      className="w-full px-3 py-2 rounded-xl bg-gray-50 dark:bg-white/5 border border-gray-200 dark:border-white/10 text-sm font-mono dk-text outline-none focus:border-[#07c160]"
    />
  </label>
);

const ToggleField: React.FC<{
  label: string;
  checked: boolean;
  description?: string;
  onChange: (checked: boolean) => void;
}> = ({ label, checked, description, onChange }) => (
  <label className="flex items-start justify-between gap-3 rounded-xl border border-gray-200 dark:border-white/10 bg-white dark:bg-black/10 px-3 py-3 cursor-pointer">
    <span className="min-w-0">
      <span className="block text-sm font-bold text-[#1d1d1f] dark:text-white">{label}</span>
      {description && <span className="mt-1 block text-xs leading-relaxed text-gray-500">{description}</span>}
    </span>
    <input
      type="checkbox"
      checked={checked}
      onChange={event => onChange(event.target.checked)}
      className="mt-1 h-4 w-4 accent-[#07c160]"
    />
  </label>
);

const SecretField: React.FC<{
  label: string;
  value: string;
  hasKey: boolean;
  onChange: (value: string) => void;
  onClear: () => void;
  disabled?: boolean;
  helpText?: string;
}> = ({ label, value, hasKey, onChange, onClear, disabled = false, helpText }) => (
  <label className="block">
    <span className="flex items-center justify-between gap-2 text-xs font-bold text-gray-500 mb-1 uppercase tracking-wide">
      <span>{label}</span>
      {hasKey && !disabled && (
        <button
          type="button"
          onClick={onClear}
          className="text-[10px] font-semibold text-red-500 hover:underline normal-case tracking-normal"
        >
          清空已保存 Key
        </button>
      )}
    </span>
    <input
      type="password"
      value={disabled || value === SECRET_PLACEHOLDER ? '' : value}
      onChange={e => onChange(e.target.value)}
      placeholder={disabled ? 'Ollama 不需要 API Key' : hasKey ? '已保存，输入新值可覆盖' : 'sk-...'}
      disabled={disabled}
      className="w-full px-3 py-2 rounded-xl bg-gray-50 dark:bg-white/5 border border-gray-200 dark:border-white/10 text-sm font-mono dk-text outline-none focus:border-[#07c160] disabled:opacity-70 disabled:cursor-not-allowed"
    />
    {helpText && (
      <span className="mt-1 block text-xs text-gray-500 normal-case tracking-normal">{helpText}</span>
    )}
  </label>
);
