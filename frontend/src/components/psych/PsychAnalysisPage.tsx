import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AlertCircle,
  BrainCircuit,
  CalendarDays,
  CheckCircle2,
  Clock3,
  Loader2,
  Play,
  RefreshCw,
  Search,
  ShieldAlert,
} from 'lucide-react';
import { Header } from '../layout/Header';
import {
  LAST_PSYCH_RESULT_KEY,
  type PsychAnalyzeResponse,
  type PsychChatMessage,
  type PsychContact,
  type PsychProcessStep,
  type TargetType,
} from './types';

const ACTIVE_TASK_KEY = 'mindtrace:psych:active-task';

type TimeRangePreset = '3m' | '6m' | '1y' | 'all' | 'custom';

interface AutoReviewStatus {
  enabled?: boolean;
  status?: 'queued' | 'running' | 'completed' | 'failed' | 'disabled' | string;
  stage?: string;
  message?: string;
  sample_id?: string;
  proposal_id?: string;
  proposal_status?: string;
  suggestion_count?: number;
  llm_used?: boolean;
  llm_error?: string;
  error?: string;
  review_summary?: string;
  requires_human_confirmation?: boolean;
  duration_seconds?: number;
}

interface PsychTaskProgress {
  task_id?: string;
  status?: 'queued' | 'running' | 'completed' | 'failed';
  stage?: string;
  message?: string;
  progress?: number;
  error?: string;
  process_steps?: PsychProcessStep[];
  result?: PsychAnalyzeResponse | null;
  auto_review?: AutoReviewStatus | null;
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

function rangeFromPreset(preset: TimeRangePreset, customFrom: string, customTo: string) {
  if (preset === 'all') return { from: '', to: '' };
  if (preset === 'custom') return { from: customFrom, to: customTo };
  const to = formatDateInput(new Date());
  if (preset === '3m') return { from: monthsAgo(3), to };
  if (preset === '6m') return { from: monthsAgo(6), to };
  return { from: monthsAgo(12), to };
}

function contactName(contact: PsychContact): string {
  return contact.remark || contact.nickname || contact.username;
}

function contactSearchText(contact: PsychContact): string {
  return `${contact.username} ${contact.nickname || ''} ${contact.remark || ''}`.toLowerCase();
}

function targetTypeLabel(type: TargetType): string {
  if (type === 'self') return '本人';
  return '联系人';
}

function scopeForTarget(type: TargetType, onlyMine: boolean): 'mine' | 'other' | 'all' {
  if (type === 'self') return 'mine';
  return 'other';
}

async function readError(resp: Response): Promise<string> {
  const text = await resp.text();
  try {
    const data = JSON.parse(text) as { detail?: string; error?: string };
    return data.detail || data.error || text || `HTTP ${resp.status}`;
  } catch {
    return text || `HTTP ${resp.status}`;
  }
}

function loadLastResult(): PsychAnalyzeResponse | null {
  try {
    const raw = localStorage.getItem(LAST_PSYCH_RESULT_KEY);
    return raw ? JSON.parse(raw) as PsychAnalyzeResponse : null;
  } catch {
    return null;
  }
}

function saveLastResult(result: PsychAnalyzeResponse | null) {
  try {
    if (result) localStorage.setItem(LAST_PSYCH_RESULT_KEY, JSON.stringify(result));
    else localStorage.removeItem(LAST_PSYCH_RESULT_KEY);
  } catch {
    // localStorage may be disabled in embedded browsers.
  }
}

function loadActiveTaskId(): string {
  try {
    return localStorage.getItem(ACTIVE_TASK_KEY) || '';
  } catch {
    return '';
  }
}

function saveActiveTaskId(taskId: string) {
  try {
    if (taskId) localStorage.setItem(ACTIVE_TASK_KEY, taskId);
    else localStorage.removeItem(ACTIVE_TASK_KEY);
  } catch {
    // Ignore storage failures.
  }
}

function isAutoReviewActive(review?: AutoReviewStatus | null): boolean {
  if (!review?.enabled) return false;
  return review.status === 'queued' || review.status === 'running';
}

function parseManualMessages(raw: string): PsychChatMessage[] | undefined {
  const lines = raw.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (lines.length === 0) return undefined;
  return lines.map((content, index) => ({
    seq: index + 1,
    datetime: '',
    sender: 'me',
    content,
    is_mine: true,
    contact_key: 'manual',
  }));
}

function formatMetricValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return '';
  if (typeof value === 'boolean') return value ? '是' : '否';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (Array.isArray(value)) return `${value.length} 项`;
  if (typeof value === 'object') return 'object';
  return String(value);
}

function stepLabel(key: string): string {
  const labels: Record<string, string> = {
    wait_vector_index: '等待向量任务',
    load_messages: '读取聊天消息',
    preprocess: '预处理与隐私过滤',
    keyword_search: '关键词检索',
    keyword_llm_screen: '大模型筛选（关键词）',
    vector_semantic_search: '向量语义检索',
    vector_llm_screen: '大模型筛选（向量）',
    fact_memory_search: '心理事实库检索',
    scoring: '综合评分',
    report: '生成报告',
  };
  return labels[key] || key;
}

export function PsychAnalysisPage() {
  const [targets, setTargets] = useState<PsychContact[]>([]);
  const [targetType, setTargetType] = useState<TargetType>('self');
  const [targetKey, setTargetKey] = useState('');
  const [timeRange, setTimeRange] = useState<TimeRangePreset>('3m');
  const [timeFrom, setTimeFrom] = useState('');
  const [timeTo, setTimeTo] = useState('');
  const [onlyMine, setOnlyMine] = useState(true);
  const [includeContext, setIncludeContext] = useState(true);
  const [manualMessages, setManualMessages] = useState('');
  const [loadingTargets, setLoadingTargets] = useState(false);
  const [targetError, setTargetError] = useState('');
  const [targetQuery, setTargetQuery] = useState('');
  const [task, setTask] = useState<PsychTaskProgress | null>(null);
  const [result, setResult] = useState<PsychAnalyzeResponse | null>(() => loadLastResult());
  const [error, setError] = useState('');

  const contactTargets = useMemo(() => {
    if (targetType === 'self') return [];
    return targets.filter((item) => !item.is_group);
  }, [targetType, targets]);

  const filteredTargets = useMemo(() => {
    const needle = targetQuery.trim().toLowerCase();
    if (!needle) return contactTargets;
    return contactTargets.filter((item) => contactSearchText(item).includes(needle));
  }, [contactTargets, targetQuery]);

  const selectedTarget = useMemo(
    () => contactTargets.find((item) => item.username === targetKey),
    [contactTargets, targetKey],
  );

  const activeSteps = task?.process_steps && task.process_steps.length > 0 ? task.process_steps : result?.process_steps || [];
  const progressValue = Math.max(0, Math.min(100, Number(task?.progress ?? (result ? 100 : 0))));
  const isRunning = task?.status === 'queued' || task?.status === 'running';
  const autoReviewRunning = isAutoReviewActive(task?.auto_review);
  const range = rangeFromPreset(timeRange, timeFrom, timeTo);
  const messageScope = scopeForTarget(targetType, onlyMine);

  const loadTargets = useCallback(async () => {
    setLoadingTargets(true);
    setTargetError('');
    try {
      const resp = await fetch('/api/messages/targets?target_type=all&limit=10000');
      if (!resp.ok) throw new Error(await readError(resp));
      const data = await resp.json() as PsychContact[];
      setTargets(Array.isArray(data) ? data : []);
    } catch (err) {
      setTargetError(err instanceof Error ? err.message : '联系人列表读取失败');
    } finally {
      setLoadingTargets(false);
    }
  }, []);

  const pollTask = useCallback(async (taskId: string) => {
    const resp = await fetch(`/api/psych/tasks/${encodeURIComponent(taskId)}/progress`);
    if (!resp.ok) {
      if (resp.status === 404) saveActiveTaskId('');
      throw new Error(await readError(resp));
    }
    const data = await resp.json() as PsychTaskProgress;
    setTask(data);
    if (data.status === 'completed' && data.result) {
      setResult(data.result);
      saveLastResult(data.result);
      if (isAutoReviewActive(data.auto_review)) {
        saveActiveTaskId(data.task_id || taskId);
        return false;
      }
      saveActiveTaskId('');
      return true;
    }
    if (data.status === 'failed') {
      saveActiveTaskId('');
      setError(data.error || data.message || '分析失败');
      return true;
    }
    return false;
  }, []);

  useEffect(() => {
    loadTargets();
  }, [loadTargets]);

  useEffect(() => {
    if (targetType === 'self') {
      setTargetKey('');
      setTargetQuery('');
      return;
    }
    if (!targetKey && contactTargets.length > 0) {
      setTargetKey(contactTargets[0].username);
    }
  }, [contactTargets, targetKey, targetType]);

  useEffect(() => {
    const taskId = loadActiveTaskId();
    if (!taskId) return;
    setTask({ task_id: taskId, status: 'running', message: '正在恢复分析任务', progress: 1 });
    pollTask(taskId).catch((err) => {
      setError(err instanceof Error ? err.message : '任务恢复失败');
      saveActiveTaskId('');
    });
  }, [pollTask]);

  useEffect(() => {
    if (!task?.task_id || (!isRunning && !autoReviewRunning)) return undefined;
    const timer = window.setInterval(() => {
      pollTask(task.task_id || '').catch((err) => {
        setError(err instanceof Error ? err.message : '进度读取失败');
      });
    }, 1000);
    return () => window.clearInterval(timer);
  }, [autoReviewRunning, isRunning, pollTask, task?.task_id]);

  const startAnalysis = async () => {
    setError('');
    const messages = parseManualMessages(manualMessages);
    if (!messages && targetType !== 'self' && !targetKey) {
      setError('请先选择一个联系人，或者切换为“本人”。');
      return;
    }

    const body = {
      target_key: messages || targetType === 'self' ? '' : targetKey,
      target_type: messages ? 'self' : targetType,
      time_from: range.from || null,
      time_to: range.to || null,
      only_mine: messageScope === 'mine',
      include_context: includeContext,
      messages,
      options: {
        use_vector: true,
        llm_screening: true,
        llm_screen_include_context: includeContext,
        llm_screen_context_window: 2,
        message_scope: messages ? 'mine' : messageScope,
      },
    };

    try {
      setTask({ status: 'queued', message: '分析任务已提交', progress: 1, process_steps: [] });
      const resp = await fetch('/api/psych/analyze/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) throw new Error(await readError(resp));
      const data = await resp.json() as PsychTaskProgress;
      setTask(data);
      if (data.task_id) {
        saveActiveTaskId(data.task_id);
        await pollTask(data.task_id);
      }
    } catch (err) {
      saveActiveTaskId('');
      setTask(null);
      setError(err instanceof Error ? err.message : '分析任务启动失败');
    }
  };

  return (
    <div>
      <Header
        title="心理风险辅助分析"
        subtitle="联系人模式分析对方消息；本人模式分析你自己发出的消息。本报告不构成医学诊断。"
      />

      <div className="grid grid-cols-1 xl:grid-cols-[420px,1fr] gap-5">
        <section className="rounded-2xl border dk-border bg-white dark:bg-white/5 p-5 shadow-sm">
          <div className="mb-5 flex items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white">分析范围</h2>
              <p className="mt-1 text-sm text-gray-500">只分析你明确选择的对象或时间段。</p>
            </div>
            <button
              type="button"
              onClick={loadTargets}
              className="inline-flex items-center gap-2 rounded-xl border dk-border px-3 py-2 text-sm font-bold text-gray-600 hover:border-[#07c160] hover:text-[#07c160] dark:text-gray-200"
            >
              <RefreshCw size={15} className={loadingTargets ? 'animate-spin' : ''} />
              刷新
            </button>
          </div>

          {targetError && (
            <div className="mb-4 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-200">
              {targetError}
            </div>
          )}

          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm font-bold text-gray-600 dark:text-gray-200">目标类型</label>
              <div className="grid grid-cols-2 gap-2">
                {(['self', 'contact'] as TargetType[]).map((type) => (
                  <button
                    key={type}
                    type="button"
                    onClick={() => setTargetType(type)}
                    className={`rounded-xl border px-3 py-2 text-sm font-bold transition ${
                      targetType === type
                        ? 'border-[#07c160] bg-[#e7f8f0] text-[#07c160] dark:bg-[#07c160]/20'
                        : 'dk-border text-gray-500 hover:border-[#07c160]'
                    }`}
                  >
                    {targetTypeLabel(type)}
                  </button>
                ))}
              </div>
              <p className="mt-2 text-xs text-gray-500">
                {targetType === 'contact'
                  ? '联系人模式会分析所选联系人的消息，不分析本人发出的消息。'
                  : '本人模式会跨聊天读取你自己发出的消息。'}
              </p>
            </div>

            {targetType !== 'self' && (
              <div>
                <label className="mb-2 block text-sm font-bold text-gray-600 dark:text-gray-200">选择对象</label>
                <div className="relative mb-2">
                  <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={16} />
                  <input
                    value={targetQuery}
                    onChange={(event) => setTargetQuery(event.target.value)}
                    placeholder="搜索昵称、备注或账号"
                    disabled={loadingTargets}
                    className="dk-input w-full rounded-xl border dk-border bg-white py-2.5 pl-9 pr-3 text-sm dark:bg-white/5 disabled:opacity-60"
                  />
                </div>
                <select
                  value={targetKey}
                  onChange={(event) => setTargetKey(event.target.value)}
                  className="dk-input w-full rounded-xl border dk-border bg-white px-3 py-2.5 text-sm dark:bg-white/5"
                >
                  {filteredTargets.length === 0 ? (
                    <option value="">没有匹配的联系人</option>
                  ) : (
                    filteredTargets.map((item) => (
                      <option key={item.username} value={item.username}>
                        {contactName(item)} · {item.message_count || 0} 条 · {item.last_message_time || '无时间'}
                      </option>
                    ))
                  )}
                </select>
                <p className="mt-2 text-xs text-gray-400">
                  显示 {filteredTargets.length} / {contactTargets.length} 个联系人
                </p>
                {selectedTarget && <p className="mt-2 text-xs text-gray-400">当前目标：{selectedTarget.username}</p>}
              </div>
            )}

            <div>
              <label className="mb-2 flex items-center gap-2 text-sm font-bold text-gray-600 dark:text-gray-200">
                <CalendarDays size={15} />
                时间范围
              </label>
              <div className="grid grid-cols-5 gap-2">
                {([
                  ['3m', '近三月'],
                  ['6m', '近六月'],
                  ['1y', '近一年'],
                  ['all', '全部'],
                  ['custom', '自定义'],
                ] as [TimeRangePreset, string][]).map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setTimeRange(value)}
                    className={`rounded-xl border px-2 py-2 text-xs font-bold transition ${
                      timeRange === value
                        ? 'border-[#07c160] bg-[#e7f8f0] text-[#07c160] dark:bg-[#07c160]/20'
                        : 'dk-border text-gray-500 hover:border-[#07c160]'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              {timeRange === 'custom' && (
                <div className="mt-3 grid grid-cols-2 gap-2">
                  <input type="date" value={timeFrom} onChange={(event) => setTimeFrom(event.target.value)} className="dk-input rounded-xl border dk-border bg-white px-3 py-2 text-sm dark:bg-white/5" />
                  <input type="date" value={timeTo} onChange={(event) => setTimeTo(event.target.value)} className="dk-input rounded-xl border dk-border bg-white px-3 py-2 text-sm dark:bg-white/5" />
                </div>
              )}
              <p className="mt-2 text-xs text-gray-400">当前：{range.from || '最早'} 到 {range.to || '最新'}</p>
            </div>

            <div className="grid grid-cols-1 gap-2">
              <label className="flex items-center justify-between rounded-xl border dk-border px-3 py-2.5 text-sm font-bold text-gray-600 dark:text-gray-200">
                大模型筛选携带上下文
                <input type="checkbox" checked={includeContext} onChange={(event) => setIncludeContext(event.target.checked)} />
              </label>
            </div>

            <div>
              <label className="mb-2 block text-sm font-bold text-gray-600 dark:text-gray-200">手工测试消息</label>
              <textarea
                value={manualMessages}
                onChange={(event) => setManualMessages(event.target.value)}
                rows={4}
                placeholder="可选。每行一条本人消息，用于不读取微信库时快速测试。"
                className="dk-input w-full resize-none rounded-xl border dk-border bg-white px-3 py-2 text-sm dark:bg-white/5"
              />
            </div>

            <button
              type="button"
              onClick={startAnalysis}
              disabled={isRunning}
              className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-[#07c160] px-4 py-3 text-sm font-black text-white shadow-sm transition hover:bg-[#05a955] disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isRunning ? <Loader2 size={17} className="animate-spin" /> : <Play size={17} />}
              {isRunning ? '分析进行中' : '开始心理风险辅助分析'}
            </button>
          </div>
        </section>

        <div className="space-y-5">
          <RuntimePanel task={task} steps={activeSteps} progress={progressValue} autoReview={task?.auto_review || null} />

          {error && (
            <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-200">
              <div className="flex items-start gap-2">
                <AlertCircle size={18} className="mt-0.5 shrink-0" />
                <span>{error}</span>
              </div>
            </div>
          )}

          {result ? <ResultPanel result={result} /> : (
            <section className="rounded-2xl border dk-border bg-white dark:bg-white/5 p-8 text-center shadow-sm">
              <BrainCircuit className="mx-auto mb-3 text-gray-300" size={42} />
              <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white">等待分析结果</h2>
              <p className="mt-2 text-sm text-gray-500">开始分析后，这里会显示风险评分、主要信号和报告摘要。</p>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}

function RuntimePanel({ task, steps, progress, autoReview }: { task: PsychTaskProgress | null; steps: PsychProcessStep[]; progress: number; autoReview?: AutoReviewStatus | null }) {
  return (
    <section className="rounded-2xl border dk-border bg-white dark:bg-white/5 p-5 shadow-sm">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-black text-[#1d1d1f] dark:text-white">
            <Clock3 size={18} />
            运行过程
          </h2>
          <p className="mt-1 text-sm text-gray-500">{task?.message || (steps.length ? '显示最近一次真实后端步骤' : '任务尚未开始')}</p>
        </div>
        {task?.task_id && <span className="font-mono text-xs text-gray-400">{task.task_id}</span>}
      </div>

      <div className="mb-4 h-2 overflow-hidden rounded-full bg-gray-100 dark:bg-white/10">
        <div className="h-full rounded-full bg-[#07c160] transition-all" style={{ width: `${progress}%` }} />
      </div>

      <div className="space-y-2">
        {steps.length === 0 ? <p className="text-sm text-gray-500">暂无步骤数据。</p> : steps.map((step) => <StepRow key={`${step.key}-${step.status}`} step={step} />)}
      </div>

      {autoReview?.enabled && (
        <div className="mt-4 rounded-xl border border-[#07c160]/20 bg-[#e7f8f0] p-3 text-sm dark:bg-[#07c160]/10">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2 font-black text-[#1d1d1f] dark:text-white">
              {isAutoReviewActive(autoReview) ? <Loader2 size={16} className="animate-spin text-[#07c160]" /> : autoReview.status === 'failed' ? <AlertCircle size={16} className="text-red-500" /> : <CheckCircle2 size={16} className="text-[#07c160]" />}
              <span>自动审核进程</span>
            </div>
            <span className="rounded-full bg-white px-2 py-0.5 text-xs font-bold text-[#07c160] dark:bg-white/10">
              {autoReviewStatusLabel(autoReview.status)}
            </span>
          </div>
          <p className="mt-1 text-xs text-gray-600 dark:text-gray-300">{autoReview.message || '分析结果已完成，自动审核独立运行。'}</p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {autoReview.sample_id && <AutoReviewPill label="样本" value={autoReview.sample_id} />}
            {autoReview.proposal_id && <AutoReviewPill label="草案" value={autoReview.proposal_id} />}
            {typeof autoReview.suggestion_count === 'number' && <AutoReviewPill label="建议数" value={String(autoReview.suggestion_count)} />}
            {typeof autoReview.llm_used === 'boolean' && <AutoReviewPill label="大模型" value={autoReview.llm_used ? '已使用' : '未使用'} />}
            {typeof autoReview.duration_seconds === 'number' && <AutoReviewPill label="耗时" value={`${autoReview.duration_seconds}s`} />}
          </div>
          {(autoReview.error || autoReview.llm_error) && <p className="mt-2 text-xs font-semibold text-red-600 dark:text-red-300">{autoReview.error || autoReview.llm_error}</p>}
        </div>
      )}
    </section>
  );
}

function autoReviewStatusLabel(status?: string): string {
  if (status === 'queued') return '已排队';
  if (status === 'running') return '运行中';
  if (status === 'completed') return '已完成';
  if (status === 'failed') return '失败';
  if (status === 'disabled') return '未启用';
  return status || '未知';
}

function AutoReviewPill({ label, value }: { label: string; value: string }) {
  return (
    <span className="rounded-full bg-white px-2 py-0.5 text-[11px] font-semibold text-gray-600 dark:bg-white/10 dark:text-gray-200">
      {label}: {value}
    </span>
  );
}

function StepRow({ step }: { step: PsychProcessStep }) {
  const entries = Object.entries(step.metrics || {}).filter(([, value]) => {
    const formatted = formatMetricValue(value);
    return formatted !== '' && formatted !== 'object';
  });
  return (
    <div className="rounded-xl border dk-border p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          {step.status === 'running' ? <Loader2 size={16} className="animate-spin text-[#07c160]" /> : step.status === 'failed' ? <AlertCircle size={16} className="text-red-500" /> : <CheckCircle2 size={16} className="text-[#07c160]" />}
          <span className="text-sm font-black text-[#1d1d1f] dark:text-white">{step.name || stepLabel(step.key)}</span>
        </div>
        <span className="text-xs font-semibold text-gray-400">{step.status === 'running' ? '运行中' : `${step.duration_ms} ms`}</span>
      </div>
      {step.detail && <p className="mt-1 text-xs text-gray-500">{step.detail}</p>}
      {entries.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {entries.slice(0, 24).map(([key, value]) => (
            <span key={key} className="rounded-full bg-gray-100 px-2 py-0.5 text-[11px] font-semibold text-gray-600 dark:bg-white/10 dark:text-gray-200">
              {key}: {formatMetricValue(value)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function ResultPanel({ result }: { result: PsychAnalyzeResponse }) {
  const score = result.score;
  return (
    <section className="rounded-2xl border dk-border bg-white dark:bg-white/5 p-5 shadow-sm">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-black text-[#1d1d1f] dark:text-white">
            <ShieldAlert size={18} />
            分析结果
          </h2>
          <p className="mt-1 text-sm text-gray-500">任务状态：{result.status}</p>
        </div>
        <div className="rounded-2xl bg-[#e7f8f0] px-4 py-2 text-center text-[#07c160] dark:bg-[#07c160]/20">
          <div className="text-2xl font-black">{score.depression_signal_score}</div>
          <div className="text-xs font-bold">信号分</div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <MetricCard label="综合风险" value={score.overall_risk} />
        <MetricCard label="多标签等级" value={`${score.risk_level ?? 0} ${score.risk_level_label || ''}`} />
        <MetricCard label="置信度" value={`${Math.round(score.confidence * 100)}%`} />
      </div>

      <div className="mt-4 rounded-xl bg-gray-50 p-4 text-sm text-gray-700 dark:bg-white/5 dark:text-gray-200">{score.summary}</div>

      {score.main_signals.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {score.main_signals.map((item) => (
            <span key={item} className="rounded-full bg-[#e7f8f0] px-3 py-1 text-xs font-bold text-[#07c160] dark:bg-[#07c160]/20">{item}</span>
          ))}
        </div>
      )}

      {score.symptom_labels && score.symptom_labels.length > 0 && (
        <div className="mt-4">
          <div className="text-sm font-black text-gray-600 dark:text-gray-200">多标签分类</div>
          <div className="mt-2 flex flex-wrap gap-2">
            {score.symptom_labels.slice(0, 18).map((item) => {
              const key = String(item.key || item.label || '');
              const label = String(item.label || item.key || '');
              const weight = String(item.weight_label || item.weight || '');
              const level = Number(item.risk_level || 0);
              const protective = Boolean(item.protective);
              const modifier = Boolean(item.modifier);
              const messageCount = Number(item.message_count || 0);
              const className = protective
                ? 'bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-100'
                : modifier
                  ? 'bg-sky-50 text-sky-700 dark:bg-sky-500/10 dark:text-sky-100'
                  : level >= 4
                    ? 'bg-red-50 text-red-700 dark:bg-red-500/10 dark:text-red-100'
                    : 'bg-[#e7f8f0] text-[#067a3d] dark:bg-[#07c160]/20 dark:text-emerald-100';
              return (
                <span key={key} className={`rounded-full px-3 py-1 text-xs font-bold ${className}`}>
                  {label} · {weight || `L${level}`} · {messageCount}条
                </span>
              );
            })}
          </div>
        </div>
      )}

      {score.dimension_scores && score.dimension_scores.length > 0 && (
        <div className="mt-4 space-y-2">
          <div className="text-sm font-black text-gray-600 dark:text-gray-200">评分明细</div>
          {score.dimension_scores.slice(0, 10).map((item) => {
            const strength = item.evidence_strength as Record<string, unknown> | undefined;
            const time = item.time_adjustment as Record<string, unknown> | undefined;
            const label = String(item.label || item.key || '');
            const scoreValue = Number(item.score || 0);
            const maxPoints = Number(item.max_points || 0);
            const baseScore = Number(item.base_score || 0);
            const strengthLabel = String(strength?.label || '-');
            const strengthCoef = Number(strength?.coefficient || 1);
            const timeLabel = String(time?.label || '-');
            const timeCoef = Number(time?.coefficient || 1);
            return (
              <div key={String(item.key || label)} className="rounded-xl border dk-border p-3 text-sm">
                <div className="flex items-center justify-between gap-3">
                  <div className="font-black text-[#1d1d1f] dark:text-white">{label}</div>
                  <div className="font-black text-[#07c160]">{scoreValue}/{maxPoints}</div>
                </div>
                <div className="mt-2 flex flex-wrap gap-2 text-xs text-gray-500 dark:text-gray-300">
                  <span>基础 {baseScore}</span>
                  <span>证据 {strengthLabel} ×{strengthCoef.toFixed(1)}</span>
                  <span>时间 {timeLabel} ×{timeCoef.toFixed(2)}</span>
                  {item.worsening_trend ? <span>趋势加重</span> : null}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {score.scoring_adjustments && (
        <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
          <MetricCard label="维度原始合计" value={String(score.scoring_adjustments.raw_dimension_score ?? score.depression_signal_score)} />
          <MetricCard label="趋势加分" value={String(score.scoring_adjustments.worsening_bonus ?? 0)} />
          <MetricCard label="缓解修正" value={String(score.scoring_adjustments.relief_delta ?? 0)} />
        </div>
      )}

      <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
        <MetricCard label="证据数" value={String(result.evidences.length)} />
        <MetricCard label="事实数" value={String(result.facts.length)} />
        <MetricCard label="步骤数" value={String(result.process_steps?.length || 0)} />
      </div>
    </section>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border dk-border p-3">
      <div className="text-xs font-bold text-gray-400">{label}</div>
      <div className="mt-1 text-base font-black text-[#1d1d1f] dark:text-white">{value}</div>
    </div>
  );
}
