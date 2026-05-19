import { useEffect, useMemo, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Cpu,
  Database,
  FolderOpen,
  Loader2,
  Monitor,
  RefreshCw,
  Search,
  Settings,
  Square,
  Users,
} from 'lucide-react';
import { Header } from '../layout/Header';

interface DatabaseManagePageProps {
  onOpenSettings: () => void;
}

interface ContactItem {
  username: string;
  nickname?: string;
  remark?: string;
  avatar?: string;
  is_group?: boolean;
  target_table?: string;
  message_count?: number;
  first_message_time?: string;
  last_message_time?: string;
}

interface BackendStatus {
  status?: string;
  python_backend?: boolean;
  data_dir_exists?: boolean;
  is_initialized?: boolean;
  is_indexing?: boolean;
  total_cached?: number;
  last_error?: string | null;
  hardware?: HardwareInfo;
}

interface HardwareInfo {
  platform?: string;
  machine?: string;
  cpu_model?: string;
  cpu_count?: number;
  recommended_workers?: number;
  gpu_models?: string[];
  accelerator?: string;
}

interface PythonConfigResponse {
  env_path?: string;
  values?: Partial<Record<'DATA_DIR' | 'AI_DB_PATH' | 'VECTOR_DB_PATH', string>>;
}

type FilterType = 'all' | 'self' | 'contact';
type TimeRangePreset = '3m' | '6m' | '1y' | 'all' | 'custom';
const VECTOR_TASK_STORAGE_KEY = 'mindtrace:vector:last-task';
const FACT_TASK_STORAGE_KEY = 'mindtrace:psych-facts:last-task';

interface VectorBuildResult {
  built?: boolean;
  valid?: boolean;
  contact_key?: string;
  index_key?: string;
  source_message_count?: number;
  msg_count?: number;
  actual_vector_count?: number;
  model?: string;
  expected_model?: string;
  dims?: number;
  invalid_reasons?: string[];
  warnings?: string[];
  embedding_batch_size?: number;
  embedding_max_batch_size?: number;
  embedding_max_batch_tokens?: number;
  embedding_timeout?: number;
  preprocess_workers?: number;
  preprocess_requested_workers?: number;
  dynamic_batch_count?: number;
  preprocess_chunks?: number;
  pipeline?: string;
  embedding_failed_count?: number;
  embedding_failure_limit?: number;
  embedding_failed_ratio?: number;
  last_embedding_error?: string;
  incremental?: boolean;
  force_rebuild?: boolean;
  indexed_count?: number;
  skipped_existing?: number;
  rebuild_reason?: string;
  error?: string;
  time_from?: string | null;
  time_to?: string | null;
}

interface VectorBuildTask {
  task_id?: string;
  status?: 'queued' | 'running' | 'completed' | 'failed' | 'canceled';
  stage?: string;
  message?: string;
  progress?: number;
  processed?: number;
  total?: number;
  error?: string;
  cancel_requested?: boolean;
  result?: VectorBuildResult | null;
}

interface FactBuildResult {
  built?: boolean;
  contact_key?: string;
  source_message_count?: number;
  filtered_message_count?: number;
  chunk_size?: number;
  chunk_count?: number;
  fact_count?: number;
  model?: string;
  only_mine?: boolean;
}

interface FactBuildTask {
  task_id?: string;
  status?: 'queued' | 'running' | 'completed' | 'failed' | 'canceled';
  stage?: string;
  message?: string;
  progress?: number;
  processed?: number;
  total?: number;
  error?: string;
  cancel_requested?: boolean;
  result?: FactBuildResult | null;
}

function contactName(contact: ContactItem): string {
  return contact.remark || contact.nickname || contact.username;
}

function matchesContactQuery(contact: ContactItem, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  return `${contact.username} ${contact.nickname || ''} ${contact.remark || ''}`.toLowerCase().includes(needle);
}

function formatDateInput(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function monthsAgo(months: number): string {
  const date = new Date();
  date.setMonth(date.getMonth() - months);
  return formatDateInput(date);
}

function presetRange(preset: TimeRangePreset, customFrom: string, customTo: string) {
  if (preset === 'all') return { from: '', to: '' };
  if (preset === 'custom') return { from: customFrom, to: customTo };
  const to = formatDateInput(new Date());
  if (preset === '3m') return { from: monthsAgo(3), to };
  if (preset === '6m') return { from: monthsAgo(6), to };
  return { from: monthsAgo(12), to };
}

async function readError(resp: Response): Promise<string> {
  const text = await resp.text();
  try {
    const data = JSON.parse(text) as { detail?: string; error?: string };
    return data.detail || data.error || text;
  } catch {
    return text || `HTTP ${resp.status}`;
  }
}

function readSavedVectorTask(): VectorBuildTask | null {
  try {
    const raw = localStorage.getItem(VECTOR_TASK_STORAGE_KEY);
    return raw ? JSON.parse(raw) as VectorBuildTask : null;
  } catch {
    return null;
  }
}

function saveVectorTask(task: VectorBuildTask | null) {
  try {
    if (task) {
      localStorage.setItem(VECTOR_TASK_STORAGE_KEY, JSON.stringify(task));
    } else {
      localStorage.removeItem(VECTOR_TASK_STORAGE_KEY);
    }
  } catch {
    // localStorage may be unavailable in some embedded webviews.
  }
}

function readSavedFactTask(): FactBuildTask | null {
  try {
    const raw = localStorage.getItem(FACT_TASK_STORAGE_KEY);
    return raw ? JSON.parse(raw) as FactBuildTask : null;
  } catch {
    return null;
  }
}

function saveFactTask(task: FactBuildTask | null) {
  try {
    if (task) {
      localStorage.setItem(FACT_TASK_STORAGE_KEY, JSON.stringify(task));
    } else {
      localStorage.removeItem(FACT_TASK_STORAGE_KEY);
    }
  } catch {
    // localStorage may be unavailable in some embedded webviews.
  }
}

export function DatabaseManagePage({ onOpenSettings }: DatabaseManagePageProps) {
  const [contacts, setContacts] = useState<ContactItem[]>([]);
  const [status, setStatus] = useState<BackendStatus | null>(null);
  const [config, setConfig] = useState<PythonConfigResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState<FilterType>('all');
  const [vectorEnabled, setVectorEnabled] = useState(true);
  const [vectorTargetType, setVectorTargetType] = useState<FilterType>('all');
  const [vectorTargetKey, setVectorTargetKey] = useState('');
  const [vectorTargetQuery, setVectorTargetQuery] = useState('');
  const [vectorTimeRange, setVectorTimeRange] = useState<TimeRangePreset>('3m');
  const [vectorTimeFrom, setVectorTimeFrom] = useState('');
  const [vectorTimeTo, setVectorTimeTo] = useState('');
  const [vectorForceRebuild, setVectorForceRebuild] = useState(false);
  const [vectorBuilding, setVectorBuilding] = useState(false);
  const [vectorResult, setVectorResult] = useState<VectorBuildResult | null>(null);
  const [vectorStatus, setVectorStatus] = useState<VectorBuildResult | null>(null);
  const [vectorTask, setVectorTask] = useState<VectorBuildTask | null>(() => readSavedVectorTask());
  const [vectorError, setVectorError] = useState('');
  const [factTargetType, setFactTargetType] = useState<FilterType>('contact');
  const [factTargetKey, setFactTargetKey] = useState('');
  const [factTargetQuery, setFactTargetQuery] = useState('');
  const [factTimeRange, setFactTimeRange] = useState<TimeRangePreset>('3m');
  const [factTimeFrom, setFactTimeFrom] = useState('');
  const [factTimeTo, setFactTimeTo] = useState('');
  const [factBuilding, setFactBuilding] = useState(false);
  const [factTask, setFactTask] = useState<FactBuildTask | null>(() => readSavedFactTask());
  const [factResult, setFactResult] = useState<FactBuildResult | null>(null);
  const [factError, setFactError] = useState('');

  const loadAll = async () => {
    setLoading(true);
    setError('');
    try {
      const [statusResp, configResp, contactsResp] = await Promise.all([
        fetch('/api/status'),
        fetch('/api/python-config'),
        fetch('/api/messages/targets?target_type=all&limit=10000'),
      ]);
      if (!statusResp.ok) throw new Error(await readError(statusResp));
      if (!configResp.ok) throw new Error(await readError(configResp));
      if (!contactsResp.ok) throw new Error(await readError(contactsResp));

      setStatus(await statusResp.json() as BackendStatus);
      setConfig(await configResp.json() as PythonConfigResponse);
      const data = await contactsResp.json() as ContactItem[];
      setContacts(Array.isArray(data) ? data : []);
    } catch (err) {
      setError(err instanceof Error ? err.message : '数据库信息读取失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadAll();
  }, []);

  const counts = useMemo(() => {
    return {
      total: contacts.length,
      contacts: contacts.length,
      groups: 0,
    };
  }, [contacts]);

  const filteredContacts = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return contacts
      .filter((item) => {
        if (item.is_group) return false;
        if (!needle) return true;
        return `${item.username} ${item.nickname || ''} ${item.remark || ''}`.toLowerCase().includes(needle);
      })
      .sort((a, b) => {
        const byTime = String(b.last_message_time || '').localeCompare(String(a.last_message_time || ''));
        if (byTime !== 0) return byTime;
        return Number(b.message_count || 0) - Number(a.message_count || 0);
      });
  }, [contacts, filter, query]);

  const vectorTargetOptions = useMemo(() => {
    if (vectorTargetType === 'all' || vectorTargetType === 'self') return [];
    return contacts.filter((item) => !item.is_group && matchesContactQuery(item, vectorTargetQuery));
  }, [contacts, vectorTargetQuery, vectorTargetType]);

  const factTargetOptions = useMemo(() => {
    if (factTargetType === 'all' || factTargetType === 'self') return [];
    return contacts.filter((item) => !item.is_group && matchesContactQuery(item, factTargetQuery));
  }, [contacts, factTargetQuery, factTargetType]);

  useEffect(() => {
    setVectorTargetKey('');
    setVectorTargetQuery('');
  }, [vectorTargetType]);

  useEffect(() => {
    setFactTargetKey('');
    setFactTargetQuery('');
  }, [factTargetType]);

  const selectedVectorRange = useMemo(
    () => presetRange(vectorTimeRange, vectorTimeFrom, vectorTimeTo),
    [vectorTimeRange, vectorTimeFrom, vectorTimeTo],
  );
  const selectedFactRange = useMemo(
    () => presetRange(factTimeRange, factTimeFrom, factTimeTo),
    [factTimeRange, factTimeFrom, factTimeTo],
  );

  const canBuildVector = vectorEnabled && !vectorBuilding && (vectorTargetType === 'all' || vectorTargetType === 'self' || !!vectorTargetKey);
  const vectorProgress = Math.max(0, Math.min(100, Math.round(Number(vectorTask?.progress || 0))));
  const canBuildFacts = !factBuilding && (factTargetType === 'all' || factTargetType === 'self' || !!factTargetKey);
  const factProgress = Math.max(0, Math.min(100, Math.round(Number(factTask?.progress || 0))));

  useEffect(() => {
    if (!vectorTask) return;
    saveVectorTask(vectorTask);
    if (vectorTask.status === 'queued' || vectorTask.status === 'running') {
      setVectorBuilding(true);
    }
    if (vectorTask.status === 'completed' && vectorTask.result) {
      setVectorResult(vectorTask.result);
    }
    if (vectorTask.status === 'failed' && vectorTask.error) {
      setVectorError(vectorTask.error);
    }
    if (vectorTask.status === 'canceled') {
      setVectorBuilding(false);
    }
  }, [vectorTask]);

  useEffect(() => {
    if (!factTask) return;
    saveFactTask(factTask);
    if (factTask.status === 'queued' || factTask.status === 'running') {
      setFactBuilding(true);
    }
    if (factTask.status === 'completed' && factTask.result) {
      setFactResult(factTask.result);
    }
    if (factTask.status === 'failed' && factTask.error) {
      setFactError(factTask.error);
    }
    if (factTask.status === 'canceled') {
      setFactBuilding(false);
    }
  }, [factTask]);

  const handleBuildVector = async () => {
    setVectorBuilding(true);
    setVectorError('');
    setVectorResult(null);
    setVectorStatus(null);
    setVectorTask(null);
    try {
      const body = {
        target_key: vectorTargetType === 'all' || vectorTargetType === 'self' ? '' : vectorTargetKey,
        target_type: vectorTargetType,
        time_from: selectedVectorRange.from || null,
        time_to: selectedVectorRange.to || null,
        options: {
          embedding_config: {
            force_rebuild: vectorForceRebuild,
          },
        },
      };
      const resp = await fetch('/api/ai/vec/build-index', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) throw new Error(await readError(resp));
      const task = await resp.json() as VectorBuildTask;
      setVectorTask(task);
      saveVectorTask(task);
    } catch (err) {
      setVectorError(err instanceof Error ? err.message : '向量数据库建立失败');
      setVectorBuilding(false);
    }
  };

  const handleCheckVectorStatus = async () => {
    setVectorError('');
    const key = vectorTargetType === 'all' || vectorTargetType === 'self' ? vectorTargetType : vectorTargetKey;
    if (!key) {
      setVectorError('请先选择要检测的联系人。');
      return;
    }
    try {
      const resp = await fetch(`/api/ai/vec/status?key=${encodeURIComponent(key)}`);
      if (!resp.ok) throw new Error(await readError(resp));
      setVectorStatus(await resp.json() as VectorBuildResult);
    } catch (err) {
      setVectorError(err instanceof Error ? err.message : '向量索引状态检测失败');
    }
  };

  const handleCancelVector = async () => {
    const taskId = vectorTask?.task_id;
    if (!taskId) return;
    setVectorError('');
    try {
      const resp = await fetch(`/api/ai/vec/build-index/${taskId}/cancel`, { method: 'POST' });
      if (!resp.ok) throw new Error(await readError(resp));
      const task = await resp.json() as VectorBuildTask;
      setVectorTask(task);
      saveVectorTask(task);
    } catch (err) {
      setVectorError(err instanceof Error ? err.message : '停止向量任务失败');
    }
  };

  const handleBuildFacts = async () => {
    setFactBuilding(true);
    setFactError('');
    setFactResult(null);
    setFactTask(null);
    try {
      const body = {
        target_key: factTargetType === 'all' || factTargetType === 'self' ? '' : factTargetKey,
        target_type: factTargetType,
        time_from: selectedFactRange.from || null,
        time_to: selectedFactRange.to || null,
        options: {
          only_mine: true,
        },
      };
      const resp = await fetch('/api/ai/psych-facts/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) throw new Error(await readError(resp));
      const task = await resp.json() as FactBuildTask;
      setFactTask(task);
      saveFactTask(task);
    } catch (err) {
      setFactError(err instanceof Error ? err.message : '心理事实库构建失败');
      setFactBuilding(false);
    }
  };

  const handleCancelFacts = async () => {
    const taskId = factTask?.task_id;
    if (!taskId) return;
    setFactError('');
    try {
      const resp = await fetch(`/api/ai/psych-facts/build/${taskId}/cancel`, { method: 'POST' });
      if (!resp.ok) throw new Error(await readError(resp));
      const task = await resp.json() as FactBuildTask;
      setFactTask(task);
      saveFactTask(task);
    } catch (err) {
      setFactError(err instanceof Error ? err.message : '停止心理事实库任务失败');
    }
  };

  useEffect(() => {
    const taskId = vectorTask?.task_id;
    if (!taskId || !vectorBuilding) return;
    const poll = async () => {
      try {
        const resp = await fetch(`/api/ai/vec/build-index/${taskId}/progress`);
        if (!resp.ok) {
          if (resp.status === 404) {
            throw new Error(await readError(resp));
          }
          setVectorError(`进度读取暂时失败，会继续重试：${await readError(resp)}`);
          return;
        }
        const task = await resp.json() as VectorBuildTask;
        setVectorError('');
        setVectorTask(task);
        saveVectorTask(task);
        if (task.status === 'completed') {
          setVectorResult(task.result || null);
          setVectorBuilding(false);
        } else if (task.status === 'canceled') {
          setVectorBuilding(false);
        } else if (task.status === 'failed') {
          setVectorError(task.error || '向量数据库建立失败');
          setVectorBuilding(false);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : '向量进度读取失败';
        if (String(message).includes('Failed to fetch') || String(message).includes('NetworkError')) {
          setVectorError(`进度读取暂时失败，会继续重试：${message}`);
          return;
        }
        setVectorError(message);
        setVectorTask((current) => {
          if (!current) return current;
          const next = {
            ...current,
            status: 'failed' as const,
            stage: 'failed',
            message,
            error: message,
          };
          saveVectorTask(next);
          return next;
        });
        setVectorBuilding(false);
      }
    };
    void poll();
    const timer = window.setInterval(() => {
      void poll();
    }, 5000);
    return () => window.clearInterval(timer);
  }, [vectorTask?.task_id, vectorBuilding]);

  useEffect(() => {
    const taskId = factTask?.task_id;
    if (!taskId || !factBuilding) return;
    const poll = async () => {
      try {
        const resp = await fetch(`/api/ai/psych-facts/build/${taskId}/progress`);
        if (!resp.ok) {
          setFactError(`进度读取暂时失败，会继续重试：${await readError(resp)}`);
          return;
        }
        const task = await resp.json() as FactBuildTask;
        setFactError('');
        setFactTask(task);
        saveFactTask(task);
        if (task.status === 'completed') {
          setFactResult(task.result || null);
          setFactBuilding(false);
        } else if (task.status === 'canceled') {
          setFactBuilding(false);
        } else if (task.status === 'failed') {
          setFactError(task.error || '心理事实库构建失败');
          setFactBuilding(false);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : '心理事实库进度读取失败';
        setFactError(`进度读取暂时失败，会继续重试：${message}`);
      }
    };
    void poll();
    const timer = window.setInterval(() => {
      void poll();
    }, 5000);
    return () => window.clearInterval(timer);
  }, [factTask?.task_id, factBuilding]);

  const dataDir = config?.values?.DATA_DIR || '';
  const aiDbPath = config?.values?.AI_DB_PATH || '';
  const vectorDbPath = config?.values?.VECTOR_DB_PATH || '';
  const dataDirReady = status?.data_dir_exists !== false;
  const hardware = status?.hardware;
  const gpuText = hardware?.gpu_models?.length ? hardware.gpu_models.join(' / ') : '未检测到独立 GPU';

  return (
    <div>
      <Header
        title="数据库管理"
        subtitle="查看当前微信数据库目录、Python 分析库和已读取联系人。"
      />

      <div className="space-y-5">
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-3">
          <StatusCard
            icon={<Database size={20} />}
            label="Python 后端"
            value={status?.python_backend === false ? '未确认' : '已连接'}
            ok={status?.python_backend !== false}
          />
          <StatusCard
            icon={<FolderOpen size={20} />}
            label="微信数据目录"
            value={dataDirReady ? '可访问' : '未找到'}
            ok={dataDirReady}
          />
          <StatusCard
            icon={<Users size={20} />}
            label="联系人"
            value={`${counts.contacts}`}
            ok
          />
          <StatusCard
            icon={<Cpu size={20} />}
            label="CPU"
            value={`${hardware?.cpu_count || 0} 核`}
            ok={Boolean(hardware?.cpu_model)}
          />
          <StatusCard
            icon={<Monitor size={20} />}
            label="加速"
            value={hardware?.accelerator || 'CPU'}
            ok
          />
        </div>

        <section className="dk-card bg-white border dk-border rounded-2xl p-5 shadow-sm">
          <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4">
            <div className="min-w-0">
              <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white">数据源</h2>
              <div className="mt-3 space-y-2 text-sm">
                <PathLine label="DATA_DIR" value={dataDir || '未配置'} />
                <PathLine label="AI_DB_PATH" value={aiDbPath || '未配置'} />
                <PathLine label="VECTOR_DB_PATH" value={vectorDbPath || '未配置'} />
                <PathLine label=".env" value={config?.env_path || '未读取'} />
                <PathLine
                  label="CPU"
                  value={`${hardware?.cpu_model || '未识别'} · 推荐 worker=${hardware?.recommended_workers || 1}`}
                />
                <PathLine label="GPU" value={gpuText} />
                <PathLine label="加速模式" value={hardware?.accelerator || 'cpu'} />
              </div>
              {status?.last_error && (
                <div className="mt-4 rounded-xl border border-amber-100 bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:bg-amber-500/10 dark:border-amber-500/20 dark:text-amber-100">
                  {status.last_error}
                </div>
              )}
            </div>

            <div className="flex flex-col sm:flex-row gap-2">
              <button
                onClick={loadAll}
                disabled={loading}
                className="inline-flex items-center justify-center gap-2 rounded-xl border dk-border px-3 py-2 text-sm font-semibold text-gray-600 hover:bg-gray-50 dark:text-gray-200 dark:hover:bg-white/10 disabled:opacity-60"
              >
                {loading ? <Loader2 size={16} className="animate-spin" /> : <RefreshCw size={16} />}
                刷新
              </button>
              <button
                onClick={onOpenSettings}
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-[#07c160] px-3 py-2 text-sm font-black text-white hover:bg-[#06ad56]"
              >
                <Settings size={16} />
                修改配置
              </button>
            </div>
          </div>

          {!dataDirReady && (
            <div className="mt-4 rounded-xl border border-red-100 bg-red-50 px-3 py-2 text-sm text-red-700 dark:bg-red-500/10 dark:border-red-500/20 dark:text-red-200">
              当前 DATA_DIR 不可访问，请在设置页把它指向解密后的微信数据库目录。
            </div>
          )}
        </section>

        <section className="dk-card bg-white border dk-border rounded-2xl p-5 shadow-sm">
          <div className="flex flex-col xl:flex-row xl:items-start xl:justify-between gap-4">
            <div>
              <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white">心理事实库</h2>
              <p className="text-sm text-gray-500 mt-1">
                将聊天记录按 80 条一批交给单独配置的大模型抽取 JSON 事实，再为每条事实做 embedding 并写入 mem_facts。
              </p>
            </div>
          </div>

          <div className="mt-5 grid grid-cols-1 lg:grid-cols-2 gap-4">
            <label className="block">
              <span className="text-sm font-semibold text-gray-700 dark:text-gray-200">事实库对象</span>
              <select
                value={factTargetType}
                onChange={(event) => setFactTargetType(event.target.value as FilterType)}
                disabled={factBuilding}
                className="dk-input mt-2 w-full rounded-xl border dk-border bg-white px-3 py-2.5 text-sm disabled:opacity-60"
              >
                <option value="self">本人</option>
                <option value="contact">联系人</option>
                <option value="all">全部联系人消息</option>
              </select>
            </label>

            {factTargetType !== 'all' && factTargetType !== 'self' && (
              <div className="block">
                <span className="text-sm font-semibold text-gray-700 dark:text-gray-200">选择目标</span>
                <div className="relative mt-2">
                  <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={16} />
                  <input
                    value={factTargetQuery}
                    onChange={(event) => setFactTargetQuery(event.target.value)}
                    placeholder="搜索昵称、备注或账号"
                    disabled={factBuilding}
                    className="dk-input w-full rounded-xl border dk-border bg-white py-2.5 pl-9 pr-3 text-sm disabled:opacity-60"
                  />
                </div>
                <select
                  value={factTargetKey}
                  onChange={(event) => setFactTargetKey(event.target.value)}
                  disabled={factBuilding}
                  className="dk-input mt-2 w-full rounded-xl border dk-border bg-white px-3 py-2.5 text-sm disabled:opacity-60"
                >
                  <option value="">请选择</option>
                  {factTargetOptions.length === 0 ? (
                    <option value="" disabled>没有匹配的联系人</option>
                  ) : (
                    factTargetOptions.map((contact) => (
                      <option key={contact.username} value={contact.username}>
                        {contactName(contact)}{typeof contact.message_count === 'number' ? ` · ${contact.message_count} 条` : ''}{contact.last_message_time ? ` · ${contact.last_message_time.slice(0, 10)}` : ''}
                      </option>
                    ))
                  )}
                </select>
              </div>
            )}
          </div>

          <div className="mt-4">
            <span className="text-sm font-semibold text-gray-700 dark:text-gray-200">时间范围</span>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 mt-2">
              {[
                ['3m', '近三个月'],
                ['6m', '近六个月'],
                ['1y', '近一年'],
                ['all', '全部'],
                ['custom', '自定义'],
              ].map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setFactTimeRange(value as TimeRangePreset)}
                  disabled={factBuilding}
                  className={`rounded-xl border px-3 py-2.5 text-sm font-black transition-colors disabled:opacity-60 ${
                    factTimeRange === value
                      ? 'border-[#07c160] bg-[#e7f8f0] text-[#07c160] dark:bg-[#07c160]/20'
                      : 'dk-border text-gray-600 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-white/10'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            {factTimeRange !== 'all' && factTimeRange !== 'custom' && (
              <p className="text-xs text-gray-500 mt-2">
                当前将抽取 {selectedFactRange.from} 至 {selectedFactRange.to} 的本人消息。
              </p>
            )}
          </div>

          {factTimeRange === 'custom' && (
            <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
              <label className="block">
                <span className="text-sm font-semibold text-gray-700 dark:text-gray-200">开始日期</span>
                <input
                  type="date"
                  value={factTimeFrom}
                  onChange={(event) => setFactTimeFrom(event.target.value)}
                  disabled={factBuilding}
                  className="dk-input mt-2 w-full rounded-xl border dk-border bg-white px-3 py-2.5 text-sm disabled:opacity-60"
                />
              </label>
              <label className="block">
                <span className="text-sm font-semibold text-gray-700 dark:text-gray-200">结束日期</span>
                <input
                  type="date"
                  value={factTimeTo}
                  onChange={(event) => setFactTimeTo(event.target.value)}
                  disabled={factBuilding}
                  className="dk-input mt-2 w-full rounded-xl border dk-border bg-white px-3 py-2.5 text-sm disabled:opacity-60"
                />
              </label>
            </div>
          )}

          <div className="mt-5 flex flex-col sm:flex-row sm:items-center gap-3">
            <button
              onClick={handleBuildFacts}
              disabled={!canBuildFacts}
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-[#07c160] px-4 py-3 text-sm font-black text-white hover:bg-[#06ad56] disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {factBuilding ? <Loader2 size={18} className="animate-spin" /> : <Database size={18} />}
              构建心理事实库
            </button>
            {factBuilding && factTask?.task_id && (
              <button
                type="button"
                onClick={handleCancelFacts}
                disabled={Boolean(factTask.cancel_requested)}
                className="inline-flex items-center justify-center gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-black text-red-700 hover:bg-red-100 disabled:opacity-60 disabled:cursor-not-allowed dark:bg-red-500/10 dark:border-red-500/30 dark:text-red-100"
              >
                <Square size={16} />
                {factTask.cancel_requested ? '正在停止' : '停止'}
              </button>
            )}
            <p className="text-xs text-gray-500">分析页面会直接检索这里构建好的事实库，不再临时调用事实抽取模型。</p>
          </div>

          {factTask && (factBuilding || factTask.status === 'completed' || factTask.status === 'failed') && (
            <div className="mt-4 rounded-xl border dk-border bg-gray-50 dark:bg-white/5 p-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-sm">
                <div className="font-semibold text-gray-700 dark:text-gray-200">
                  {factTask.message || '心理事实库任务运行中'}
                </div>
                <div className="text-xs font-black text-gray-500">
                  {factTask.status || 'running'} 路 {factProgress}%
                </div>
              </div>
              <div className="mt-3 h-2 rounded-full bg-gray-200 dark:bg-white/10 overflow-hidden">
                <div className="h-full bg-[#07c160] transition-all duration-500" style={{ width: `${factProgress}%` }} />
              </div>
              <div className="mt-2 flex flex-wrap gap-2 text-xs text-gray-500">
                <span>阶段：{factTask.stage || '-'}</span>
                <span>已处理：{factTask.processed ?? 0}/{factTask.total ?? 0}</span>
                {factTask.task_id && <span>任务：{factTask.task_id.slice(0, 8)}</span>}
              </div>
            </div>
          )}

          {factError && (
            <div className="mt-4 rounded-xl border border-red-100 bg-red-50 px-3 py-2 text-sm text-red-700 dark:bg-red-500/10 dark:border-red-500/20 dark:text-red-200">
              {factError}
            </div>
          )}

          {factResult && (
            <div className="mt-4 rounded-xl border border-[#07c160]/20 bg-[#e7f8f0] px-3 py-3 text-sm text-[#067a3d] dark:bg-[#07c160]/10 dark:text-emerald-100">
              <div className="font-black">
                心理事实库已构建：{factResult.contact_key || '-'}，写入 {factResult.fact_count ?? 0} 条事实。
              </div>
              <div className="mt-2 flex flex-wrap gap-2 text-xs font-semibold">
                <span>原始消息 {factResult.source_message_count ?? 0}</span>
                <span>有效消息 {factResult.filtered_message_count ?? 0}</span>
                <span>批大小 {factResult.chunk_size ?? '-'}</span>
                <span>批次数 {factResult.chunk_count ?? '-'}</span>
                <span>模型 {factResult.model || '-'}</span>
                <span>仅本人 {String(factResult.only_mine ?? true)}</span>
              </div>
            </div>
          )}
        </section>

        <section className="dk-card bg-white border dk-border rounded-2xl p-5 shadow-sm">
          <div className="flex flex-col xl:flex-row xl:items-start xl:justify-between gap-4">
            <div>
              <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white">向量数据库</h2>
              <p className="text-sm text-gray-500 mt-1">单独为选定联系人、本人或全部联系人消息建立向量索引；消息范围只按时间筛选。</p>
            </div>
            <label className="flex items-start gap-3 rounded-xl border dk-border p-3 cursor-pointer min-w-[220px]">
              <input
                type="checkbox"
                checked={vectorEnabled}
                onChange={(event) => setVectorEnabled(event.target.checked)}
                className="mt-1"
              />
              <span>
                <span className="block text-sm font-semibold text-gray-700 dark:text-gray-200">启用向量索引</span>
                <span className="block text-xs text-gray-500 mt-1">关闭后不会调用 Embedding 模型。</span>
              </span>
            </label>
          </div>

          <div className="mt-5 grid grid-cols-1 lg:grid-cols-2 gap-4">
            <label className="block">
              <span className="text-sm font-semibold text-gray-700 dark:text-gray-200">索引对象</span>
              <select
                value={vectorTargetType}
                onChange={(event) => setVectorTargetType(event.target.value as FilterType)}
                disabled={!vectorEnabled || vectorBuilding}
                className="dk-input mt-2 w-full rounded-xl border dk-border bg-white px-3 py-2.5 text-sm disabled:opacity-60"
              >
                <option value="all">全部联系人消息</option>
                <option value="self">本人</option>
                <option value="contact">联系人</option>
              </select>
            </label>

            {vectorTargetType !== 'all' && vectorTargetType !== 'self' && (
              <div className="block">
                <span className="text-sm font-semibold text-gray-700 dark:text-gray-200">选择目标</span>
                <div className="relative mt-2">
                  <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={16} />
                  <input
                    value={vectorTargetQuery}
                    onChange={(event) => setVectorTargetQuery(event.target.value)}
                    placeholder="搜索昵称、备注或账号"
                    disabled={!vectorEnabled || vectorBuilding}
                    className="dk-input w-full rounded-xl border dk-border bg-white py-2.5 pl-9 pr-3 text-sm disabled:opacity-60"
                  />
                </div>
                <select
                  value={vectorTargetKey}
                  onChange={(event) => setVectorTargetKey(event.target.value)}
                  disabled={!vectorEnabled || vectorBuilding}
                  className="dk-input mt-2 w-full rounded-xl border dk-border bg-white px-3 py-2.5 text-sm disabled:opacity-60"
                >
                  <option value="">请选择</option>
                  {vectorTargetOptions.length === 0 ? (
                    <option value="" disabled>没有匹配的联系人</option>
                  ) : (
                    vectorTargetOptions.map((contact) => (
                      <option key={contact.username} value={contact.username}>
                        {contactName(contact)}{typeof contact.message_count === 'number' ? ` · ${contact.message_count} 条` : ''}{contact.last_message_time ? ` · ${contact.last_message_time.slice(0, 10)}` : ''}
                      </option>
                    ))
                  )}
                </select>
              </div>
            )}
          </div>

          <div className="mt-4">
            <span className="text-sm font-semibold text-gray-700 dark:text-gray-200">时间范围</span>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 mt-2">
              {[
                ['3m', '近三个月'],
                ['6m', '近六个月'],
                ['1y', '近一年'],
                ['all', '全部'],
                ['custom', '自定义'],
              ].map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setVectorTimeRange(value as TimeRangePreset)}
                  disabled={!vectorEnabled || vectorBuilding}
                  className={`rounded-xl border px-3 py-2.5 text-sm font-black transition-colors disabled:opacity-60 ${
                    vectorTimeRange === value
                      ? 'border-[#07c160] bg-[#e7f8f0] text-[#07c160] dark:bg-[#07c160]/20'
                      : 'dk-border text-gray-600 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-white/10'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            {vectorTimeRange !== 'all' && vectorTimeRange !== 'custom' && (
              <p className="text-xs text-gray-500 mt-2">
                当前将索引 {selectedVectorRange.from} 至 {selectedVectorRange.to} 的消息。
              </p>
            )}
          </div>

          {vectorTimeRange === 'custom' && (
            <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
              <label className="block">
                <span className="text-sm font-semibold text-gray-700 dark:text-gray-200">开始日期</span>
                <input
                  type="date"
                  value={vectorTimeFrom}
                  onChange={(event) => setVectorTimeFrom(event.target.value)}
                  disabled={!vectorEnabled || vectorBuilding}
                  className="dk-input mt-2 w-full rounded-xl border dk-border bg-white px-3 py-2.5 text-sm disabled:opacity-60"
                />
              </label>
              <label className="block">
                <span className="text-sm font-semibold text-gray-700 dark:text-gray-200">结束日期</span>
                <input
                  type="date"
                  value={vectorTimeTo}
                  onChange={(event) => setVectorTimeTo(event.target.value)}
                  disabled={!vectorEnabled || vectorBuilding}
                  className="dk-input mt-2 w-full rounded-xl border dk-border bg-white px-3 py-2.5 text-sm disabled:opacity-60"
                />
              </label>
            </div>
          )}

          <div className="mt-5 flex flex-col sm:flex-row sm:items-center gap-3">
            <button
              onClick={handleBuildVector}
              disabled={!canBuildVector}
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-[#07c160] px-4 py-3 text-sm font-black text-white hover:bg-[#06ad56] disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {vectorBuilding ? <Loader2 size={18} className="animate-spin" /> : <Database size={18} />}
              建立向量数据库
            </button>
            <button
              type="button"
              onClick={handleCheckVectorStatus}
              disabled={vectorTargetType !== 'all' && !vectorTargetKey}
              className="inline-flex items-center justify-center gap-2 rounded-xl border dk-border px-4 py-3 text-sm font-black text-gray-600 hover:bg-gray-50 disabled:opacity-60 disabled:cursor-not-allowed dark:text-gray-200 dark:hover:bg-white/10"
            >
              <Search size={18} />
              检测当前索引
            </button>
            {vectorBuilding && vectorTask?.task_id && (
              <button
                type="button"
                onClick={handleCancelVector}
                disabled={Boolean(vectorTask.cancel_requested)}
                className="inline-flex items-center justify-center gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-black text-red-700 hover:bg-red-100 disabled:opacity-60 disabled:cursor-not-allowed dark:bg-red-500/10 dark:border-red-500/30 dark:text-red-100"
              >
                <Square size={16} />
                {vectorTask.cancel_requested ? '正在停止' : '停止'}
              </button>
            )}
            <p className="text-xs text-gray-500">大范围索引会调用较多 Embedding 请求，请先确认本地模型或兼容接口可用。</p>
          </div>

          {vectorTask && (vectorBuilding || vectorTask.status === 'completed' || vectorTask.status === 'failed') && (
            <div className="mt-4 rounded-xl border dk-border bg-gray-50 dark:bg-white/5 p-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-sm">
                <div className="font-semibold text-gray-700 dark:text-gray-200">
                  {vectorTask.message || '向量任务运行中'}
                </div>
                <div className="text-xs font-black text-gray-500">
                  {vectorTask.status || 'running'} · {vectorProgress}%
                </div>
              </div>
              <div className="mt-3 h-2 rounded-full bg-gray-200 dark:bg-white/10 overflow-hidden">
                <div
                  className="h-full bg-[#07c160] transition-all duration-500"
                  style={{ width: `${vectorProgress}%` }}
                />
              </div>
              <div className="mt-2 flex flex-wrap gap-2 text-xs text-gray-500">
                <span>阶段：{vectorTask.stage || '-'}</span>
                <span>已处理：{vectorTask.processed ?? 0}/{vectorTask.total ?? 0}</span>
                {vectorTask.task_id && <span>任务：{vectorTask.task_id.slice(0, 8)}</span>}
              </div>
            </div>
          )}

          {vectorError && (
            <div className="mt-4 rounded-xl border border-red-100 bg-red-50 px-3 py-2 text-sm text-red-700 dark:bg-red-500/10 dark:border-red-500/20 dark:text-red-200">
              {vectorError}
            </div>
          )}

          {vectorStatus && (
            <div className={`mt-4 rounded-xl border px-3 py-3 text-sm ${
              vectorStatus.valid
                ? 'border-[#07c160]/20 bg-[#e7f8f0] text-[#067a3d] dark:bg-[#07c160]/10 dark:text-emerald-100'
                : 'border-amber-200 bg-amber-50 text-amber-800 dark:bg-amber-500/10 dark:border-amber-500/30 dark:text-amber-100'
            }`}>
              <div className="font-black">
                {vectorStatus.valid ? '当前向量索引有效' : '当前向量索引不可用或需要重建'}：
                索引键 {vectorStatus.index_key || vectorStatus.contact_key || '-'}
              </div>
              <div className="mt-2 flex flex-wrap gap-2 text-xs font-semibold">
                <span>状态行数 {vectorStatus.msg_count ?? 0}</span>
                <span>实际向量 {vectorStatus.actual_vector_count ?? 0}</span>
                <span>维度 {vectorStatus.dims ?? 0}</span>
                <span>当前模型 {vectorStatus.model || '-'}</span>
                <span>期望模型 {vectorStatus.expected_model || '-'}</span>
                {vectorStatus.invalid_reasons?.length ? <span>失效原因 {vectorStatus.invalid_reasons.join(', ')}</span> : null}
                {vectorStatus.warnings?.length ? <span>提示 {vectorStatus.warnings.join(', ')}</span> : null}
              </div>
            </div>
          )}

          {vectorResult && (
            <div className="mt-4 rounded-xl border border-[#07c160]/20 bg-[#e7f8f0] px-3 py-3 text-sm text-[#067a3d] dark:bg-[#07c160]/10 dark:text-emerald-100">
              <div className="font-black">
                向量数据库已建立：索引键 {vectorResult.contact_key || vectorResult.index_key || '-'}，入库向量 {vectorResult.msg_count ?? 0} 条。
              </div>
              <div className="mt-2 flex flex-wrap gap-2 text-xs font-semibold">
                <span>{vectorResult.valid === false ? '索引需重建' : '索引有效'}</span>
                <span>{vectorResult.incremental ? '增量构建' : '全量重建'}</span>
                <span>本次新增/更新 {vectorResult.indexed_count ?? '-'}</span>
                <span>跳过已存在 {vectorResult.skipped_existing ?? '-'}</span>
                <span>原始消息 {vectorResult.source_message_count ?? 0}</span>
                <span>实际向量 {vectorResult.actual_vector_count ?? vectorResult.msg_count ?? 0}</span>
                <span>模型 {vectorResult.model || '-'}</span>
                <span>期望模型 {vectorResult.expected_model || '-'}</span>
                <span>维度 {vectorResult.dims ?? 0}</span>
                <span>最大 batch {vectorResult.embedding_max_batch_size ?? vectorResult.embedding_batch_size ?? '-'}</span>
                <span>token batch {vectorResult.embedding_max_batch_tokens ?? '-'}</span>
                <span>CPU worker {vectorResult.preprocess_workers ?? '-'}</span>
                <span>请求 worker {vectorResult.preprocess_requested_workers ?? '-'}</span>
                <span>动态批次 {vectorResult.dynamic_batch_count ?? '-'}</span>
                <span>预处理分片 {vectorResult.preprocess_chunks ?? '-'}</span>
                <span>失败跳过 {vectorResult.embedding_failed_count ?? 0}/{vectorResult.embedding_failure_limit ?? '-'}</span>
                <span>失败比例 {typeof vectorResult.embedding_failed_ratio === 'number' ? `${(vectorResult.embedding_failed_ratio * 100).toFixed(2)}%` : '-'}</span>
                {vectorResult.last_embedding_error && <span>末次错误 {vectorResult.last_embedding_error}</span>}
              </div>
            </div>
          )}
        </section>

        <section className="dk-card bg-white border dk-border rounded-2xl p-5 shadow-sm">
          <div className="flex flex-col xl:flex-row xl:items-center xl:justify-between gap-4">
            <div>
              <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white">联系人</h2>
              <p className="text-sm text-gray-500 mt-1">
                共读取 {counts.contacts} 个可分析联系人。
              </p>
            </div>

            <div className="flex flex-col sm:flex-row gap-2">
              <div className="relative">
                <Search size={17} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="搜索昵称、备注或 username"
                  className="dk-input w-full sm:w-72 rounded-xl border dk-border bg-white pl-9 pr-3 py-2.5 text-sm"
                />
              </div>
              <select
                value={filter}
                onChange={(event) => setFilter(event.target.value as FilterType)}
                className="dk-input rounded-xl border dk-border bg-white px-3 py-2.5 text-sm"
              >
                <option value="all">全部</option>
                <option value="contact">联系人</option>
              </select>
            </div>
          </div>

          {error && (
            <div className="mt-4 rounded-xl border border-red-100 bg-red-50 px-3 py-2 text-sm text-red-700 dark:bg-red-500/10 dark:border-red-500/20 dark:text-red-200">
              {error}
            </div>
          )}

          <div className="mt-5 overflow-hidden rounded-xl border dk-border">
            <div className="hidden md:grid grid-cols-[minmax(0,1.2fr)_minmax(0,1.2fr)_minmax(0,1.6fr)] gap-3 bg-gray-50 dark:bg-white/5 px-4 py-3 text-xs font-black text-gray-500">
              <div>显示名称</div>
              <div>备注 / 昵称</div>
              <div>username</div>
            </div>

            {loading && contacts.length === 0 ? (
              <div className="flex items-center justify-center gap-2 px-4 py-10 text-sm text-gray-500">
                <Loader2 size={18} className="animate-spin" />
                正在读取联系人
              </div>
            ) : filteredContacts.length === 0 ? (
              <div className="px-4 py-10 text-center text-sm text-gray-500">
                暂无联系人。请确认 DATA_DIR 下存在 contact/contact.db。
              </div>
            ) : (
              <div className="divide-y dk-divide">
                {filteredContacts.map((contact) => (
                  <ContactRow key={contact.username} contact={contact} />
                ))}
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

function StatusCard({
  icon,
  label,
  value,
  ok,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  ok: boolean;
}) {
  return (
    <div className="dk-card bg-white border dk-border rounded-2xl p-4 shadow-sm">
      <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
        ok ? 'bg-[#e7f8f0] text-[#07c160]' : 'bg-red-50 text-red-600 dark:bg-red-500/10 dark:text-red-200'
      }`}>
        {icon}
      </div>
      <div className="mt-4 text-sm text-gray-500 font-semibold">{label}</div>
      <div className="mt-1 flex items-center gap-2 text-xl font-black text-[#1d1d1f] dark:text-white">
        {ok ? <CheckCircle2 size={18} className="text-[#07c160]" /> : <AlertCircle size={18} className="text-red-500" />}
        {value}
      </div>
    </div>
  );
}

function PathLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-[120px_1fr] gap-1 md:gap-3">
      <span className="font-black text-gray-500">{label}</span>
      <span className="break-all text-gray-700 dark:text-gray-200">{value}</span>
    </div>
  );
}

function ContactRow({ contact }: { contact: ContactItem }) {
  const name = contactName(contact);
  const secondary = contact.remark && contact.nickname ? `${contact.remark} / ${contact.nickname}` : contact.remark || contact.nickname || '-';
  const meta = [
    typeof contact.message_count === 'number' ? `${contact.message_count} 条消息` : '',
    contact.last_message_time ? `最近 ${contact.last_message_time}` : '',
  ].filter(Boolean).join(' · ');

  return (
    <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1.2fr)_minmax(0,1.2fr)_minmax(0,1.6fr)] gap-2 md:gap-3 px-4 py-3 text-sm">
      <div className="flex items-center gap-3 min-w-0">
        {contact.avatar ? (
          <img src={contact.avatar} alt="" className="w-9 h-9 rounded-xl object-cover bg-gray-100" />
        ) : (
          <div className="w-9 h-9 rounded-xl bg-[#e7f8f0] text-[#07c160] flex items-center justify-center font-black">
            {name.slice(0, 1).toUpperCase()}
          </div>
        )}
        <span className="font-black text-[#1d1d1f] dark:text-white truncate">{name}</span>
      </div>
      <div className="text-gray-600 dark:text-gray-300 truncate md:self-center">
        <div className="truncate">{secondary}</div>
        {meta && <div className="mt-1 text-xs text-gray-500 truncate">{meta}</div>}
      </div>
      <div className="text-gray-500 break-all md:self-center">{contact.username}</div>
    </div>
  );
}
