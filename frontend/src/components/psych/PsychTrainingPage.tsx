import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Database,
  GraduationCap,
  Loader2,
  RefreshCw,
  Save,
  Search,
  Sparkles,
  Trash2,
  Users,
} from 'lucide-react';
import { Header } from '../layout/Header';
import { LAST_PSYCH_RESULT_KEY, type PsychAnalyzeResponse } from './types';

interface TrainingReview {
  accurate: boolean;
  human_risk_level: number;
  human_score?: number | null;
  missed_labels: string[];
  false_positive_labels: string[];
  suggested_keywords: string[];
  suggested_negative_keywords: string[];
  notes: string;
}

interface TrainingSample {
  sample_id: string;
  name: string;
  target_key: string;
  target_type: string;
  analysis_task_id: string;
  status: string;
  analysis_json: Record<string, unknown>;
  human_review: TrainingReview;
  notes: string;
  created_at: number;
  updated_at: number;
}

interface TrainingProposal {
  proposal_id: string;
  status: string;
  sample_count: number;
  model_provider: string;
  model_name: string;
  summary: string;
  suggestions: Array<Record<string, unknown>>;
  proposed_config: Record<string, unknown>;
  diagnostics: Record<string, unknown>;
  created_at: number;
  applied_at: number;
}

interface ProposalPreview {
  proposal_id: string;
  selected_suggestion_indexes: number[];
  selected_suggestions: Array<Record<string, unknown>>;
  effective_suggestions?: Array<Record<string, unknown>>;
  ignored_suggestions?: Array<Record<string, unknown>>;
  preview_diff: Record<string, unknown>;
  changed: boolean;
}

interface TrainingProposalTask {
  task_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | string;
  stage?: string;
  message?: string;
  progress?: number;
  error?: string;
  psych_task_id?: string;
  current_psych_task_id?: string;
  psych_progress?: number;
  current_step?: string;
  result?: TrainingProposal | TrainingRepeatResult | null;
}

interface TrainingRepeatResult {
  source_proposal_id?: string;
  reanalyzed?: number;
  auto_reviewed?: number;
  skipped?: number;
  new_sample_ids?: string[];
  proposal?: TrainingProposal;
}

interface TrainingTarget {
  username: string;
  nickname?: string;
  remark?: string;
  is_group?: boolean;
  message_count?: number;
  last_message_time?: string;
}

interface TrainingPromptConfig {
  version?: number;
  allow_ai_prompt_suggestions: boolean;
  training_system_prompt: string;
  auto_review_system_prompt: string;
  manual_notes?: string;
}

type TrainingScopeType = 'all' | 'self' | 'contact';

const ACTIVE_PSYCH_TASK_KEY = 'mindtrace:psych:active-task';

const EMPTY_REVIEW: TrainingReview = {
  accurate: false,
  human_risk_level: 0,
  human_score: null,
  missed_labels: [],
  false_positive_labels: [],
  suggested_keywords: [],
  suggested_negative_keywords: [],
  notes: '',
};

function splitLines(value: string): string[] {
  return value
    .split(/[\n,，;；]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function joinLines(value?: string[]): string {
  return Array.isArray(value) ? value.join('\n') : '';
}

function formatTime(seconds?: number): string {
  if (!seconds) return '-';
  return new Date(seconds * 1000).toLocaleString();
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function readLastResult(): PsychAnalyzeResponse | null {
  try {
    const raw = localStorage.getItem(LAST_PSYCH_RESULT_KEY);
    return raw ? (JSON.parse(raw) as PsychAnalyzeResponse) : null;
  } catch {
    return null;
  }
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

function modelRisk(result: PsychAnalyzeResponse | null): string {
  if (!result) return '-';
  const level = result.score.risk_level ?? 0;
  const score = result.score.depression_signal_score ?? 0;
  return `L${level} / ${score}分`;
}

function sampleScore(sample: TrainingSample): { level: number; score: number } {
  const analysis = sample.analysis_json || {};
  const score = analysis.score as Record<string, unknown> | undefined;
  return {
    level: Number(score?.risk_level || 0),
    score: Number(score?.depression_signal_score || 0),
  };
}

function targetDisplayName(target?: TrainingTarget): string {
  if (!target) return '';
  return target.remark || target.nickname || target.username;
}

function scopeLabel(scope: TrainingScopeType, target?: TrainingTarget): string {
  if (scope === 'self') return '本人样本';
  if (scope === 'contact') return target ? `联系人：${targetDisplayName(target)}` : '特定联系人';
  return '全部人审样本';
}

function proposalFromTaskResult(result: TrainingProposalTask['result']): TrainingProposal | null {
  if (!result || typeof result !== 'object') return null;
  const direct = result as TrainingProposal;
  if (direct.proposal_id) return direct;
  const repeat = result as TrainingRepeatResult;
  return repeat.proposal || null;
}

function saveActivePsychTaskId(taskId: string) {
  try {
    if (taskId) localStorage.setItem(ACTIVE_PSYCH_TASK_KEY, taskId);
    else localStorage.removeItem(ACTIVE_PSYCH_TASK_KEY);
  } catch {
    // Ignore localStorage failures in embedded browsers.
  }
}

export function PsychTrainingPage({ onOpenPsych }: { onOpenPsych?: () => void }) {
  const [samples, setSamples] = useState<TrainingSample[]>([]);
  const [proposals, setProposals] = useState<TrainingProposal[]>([]);
  const [lastResult, setLastResult] = useState<PsychAnalyzeResponse | null>(() => readLastResult());
  const [name, setName] = useState('');
  const [review, setReview] = useState<TrainingReview>(EMPTY_REVIEW);
  const [missedText, setMissedText] = useState('');
  const [falsePositiveText, setFalsePositiveText] = useState('');
  const [keywordText, setKeywordText] = useState('');
  const [negativeKeywordText, setNegativeKeywordText] = useState('');
  const [useLLM, setUseLLM] = useState(true);
  const [maxSamples, setMaxSamples] = useState('30');
  const [editingSampleId, setEditingSampleId] = useState('');
  const [editName, setEditName] = useState('');
  const [editReview, setEditReview] = useState<TrainingReview>(EMPTY_REVIEW);
  const [editMissedText, setEditMissedText] = useState('');
  const [editFalsePositiveText, setEditFalsePositiveText] = useState('');
  const [editKeywordText, setEditKeywordText] = useState('');
  const [editNegativeKeywordText, setEditNegativeKeywordText] = useState('');
  const [editNotes, setEditNotes] = useState('');
  const [loading, setLoading] = useState(false);
  const [proposalLoading, setProposalLoading] = useState(false);
  const [proposalTask, setProposalTask] = useState<TrainingProposalTask | null>(null);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [lastAppliedDiff, setLastAppliedDiff] = useState<Record<string, unknown> | null>(null);
  const [trainingScope, setTrainingScope] = useState<TrainingScopeType>('all');
  const [trainingTargetKey, setTrainingTargetKey] = useState('');
  const [targets, setTargets] = useState<TrainingTarget[]>([]);
  const [targetQuery, setTargetQuery] = useState('');
  const [targetLoading, setTargetLoading] = useState(false);
  const [promptConfig, setPromptConfig] = useState<TrainingPromptConfig | null>(null);
  const [promptLoading, setPromptLoading] = useState(false);
  const openedRetestPsychTaskRef = useRef('');

  const reviewedCount = useMemo(() => samples.filter((item) => item.human_review).length, [samples]);
  const latestProposal = proposals[0];
  const selectedTrainingTarget = useMemo(
    () => targets.find((item) => item.username === trainingTargetKey),
    [targets, trainingTargetKey],
  );
  const currentScopeLabel = scopeLabel(trainingScope, selectedTrainingTarget);
  const activeRetestPsychTaskId = proposalTask
    ? String(proposalTask.current_psych_task_id || proposalTask.psych_task_id || '')
    : '';
  const filteredTargets = useMemo(() => {
    const query = targetQuery.trim().toLowerCase();
    return targets
      .filter((item) => !item.is_group)
      .filter((item) => {
        if (!query) return true;
        const haystack = `${item.username} ${item.nickname || ''} ${item.remark || ''}`.toLowerCase();
        return haystack.includes(query);
      })
      .slice(0, 80);
  }, [targets, targetQuery]);

  const loadData = useCallback(async () => {
    setError('');
    try {
      const params = new URLSearchParams({ limit: '100' });
      const proposalParams = new URLSearchParams({ limit: '200' });
      if (trainingScope !== 'all') {
        params.set('target_type', trainingScope);
        proposalParams.set('target_type', trainingScope);
      }
      if (trainingScope === 'contact' && trainingTargetKey) {
        params.set('target_key', trainingTargetKey);
        proposalParams.set('target_key', trainingTargetKey);
      }
      const [sampleResp, proposalResp] = await Promise.all([
        fetch(`/api/psych/training/samples?${params.toString()}`),
        fetch(`/api/psych/training/proposals?${proposalParams.toString()}`),
      ]);
      if (!sampleResp.ok) throw new Error(await readError(sampleResp));
      if (!proposalResp.ok) throw new Error(await readError(proposalResp));
      setSamples((await sampleResp.json()) as TrainingSample[]);
      setProposals((await proposalResp.json()) as TrainingProposal[]);
    } catch (err) {
      setError(err instanceof Error ? err.message : '训练数据读取失败');
    }
  }, [trainingScope, trainingTargetKey]);

  const openRetestPsychTask = useCallback((task: TrainingProposalTask, force = false) => {
    const psychTaskId = String(task.current_psych_task_id || task.psych_task_id || '');
    if (!psychTaskId) return;
    saveActivePsychTaskId(psychTaskId);
    if (force || openedRetestPsychTaskRef.current !== psychTaskId) {
      openedRetestPsychTaskRef.current = psychTaskId;
      onOpenPsych?.();
    }
  }, [onOpenPsych]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const loadPromptConfig = useCallback(async () => {
    setPromptLoading(true);
    try {
      const resp = await fetch('/api/psych/training/prompts');
      if (!resp.ok) throw new Error(await readError(resp));
      setPromptConfig((await resp.json()) as TrainingPromptConfig);
    } catch (err) {
      setError(err instanceof Error ? err.message : '训练 Prompt 读取失败');
    } finally {
      setPromptLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadPromptConfig();
  }, [loadPromptConfig]);

  useEffect(() => {
    let alive = true;
    setTargetLoading(true);
    fetch('/api/messages/targets?target_type=contact&limit=10000')
      .then(async (resp) => {
        if (!resp.ok) throw new Error(await readError(resp));
        return resp.json() as Promise<TrainingTarget[]>;
      })
      .then((items) => {
        if (!alive) return;
        setTargets(items.filter((item) => !item.is_group));
      })
      .catch(() => {
        if (!alive) return;
        setTargets([]);
      })
      .finally(() => {
        if (alive) setTargetLoading(false);
      });
    return () => {
      alive = false;
    };
  }, []);

  const refreshLastResult = () => {
    const result = readLastResult();
    setLastResult(result);
    if (result) {
      setName(`人审样本 L${result.score.risk_level ?? 0} ${new Date().toLocaleString()}`);
      setReview((current) => ({
        ...current,
        human_risk_level: Number(result.score.risk_level ?? 0),
        human_score: Number(result.score.depression_signal_score ?? 0),
      }));
    }
  };

  const saveSample = async () => {
    if (!lastResult) {
      setError('没有最近一次分析结果，请先在心理分析页面完成一次分析。');
      return;
    }
    setLoading(true);
    setError('');
    setMessage('');
    const nextReview: TrainingReview = {
      ...review,
      missed_labels: splitLines(missedText),
      false_positive_labels: splitLines(falsePositiveText),
      suggested_keywords: splitLines(keywordText),
      suggested_negative_keywords: splitLines(negativeKeywordText),
    };
    try {
      const resp = await fetch('/api/psych/training/samples', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name || `人审样本 ${new Date().toLocaleString()}`,
          target_key: String(lastResult.report_json?.target_key || ''),
          target_type: String(lastResult.report_json?.target_type || ''),
          analysis_task_id: lastResult.task_id,
          analysis_result: lastResult,
          human_review: nextReview,
          notes: nextReview.notes,
        }),
      });
      if (!resp.ok) throw new Error(await readError(resp));
      setMessage('训练样本已保存。');
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : '训练样本保存失败');
    } finally {
      setLoading(false);
    }
  };

  const pollProposalTask = useCallback(async (taskId: string) => {
    for (;;) {
      const resp = await fetch(`/api/psych/training/proposals/generate/tasks/${encodeURIComponent(taskId)}/progress`);
      if (!resp.ok) throw new Error(await readError(resp));
      const task = (await resp.json()) as TrainingProposalTask;
      setProposalTask(task);
      openRetestPsychTask(task);
      if (task.status === 'completed') {
        const proposal = proposalFromTaskResult(task.result);
        if (proposal?.proposal_id) {
          setProposals((current) => [proposal, ...current.filter((item) => item.proposal_id !== proposal.proposal_id)]);
          const repeat = task.result && typeof task.result === 'object' && 'auto_reviewed' in task.result
            ? task.result as TrainingRepeatResult
            : null;
          setMessage(repeat
            ? `自动复测闭环完成：复测 ${repeat.reanalyzed || 0} 个样本，自动审核 ${repeat.auto_reviewed || 0} 个结果，并生成下一轮待审核草案。`
            : '优化草案已生成，确认后可应用到评分标准。');
          await loadData();
        } else {
          setMessage('任务已完成，但没有返回可用草案。');
        }
        setProposalLoading(false);
        return;
      }
      if (task.status === 'failed') {
        throw new Error(task.error || task.message || '优化草案生成失败');
      }
      await delay(1500);
    }
  }, [loadData]);

  const generateProposal = async () => {
    if (trainingScope === 'contact' && !trainingTargetKey) {
      setError('请先选择一个联系人，再生成特定联系人训练草案。');
      return;
    }
    setProposalLoading(true);
    setError('');
    setMessage('');
    setProposalTask(null);
    try {
      const resp = await fetch('/api/psych/training/proposals/generate/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          max_samples: maxSamples.trim().toLowerCase() === 'max' ? 'max' : Number(maxSamples || 30),
          use_llm: useLLM,
          target_type: trainingScope,
          target_key: trainingScope === 'contact' ? trainingTargetKey : '',
        }),
      });
      if (!resp.ok) throw new Error(await readError(resp));
      const task = (await resp.json()) as TrainingProposalTask;
      setProposalTask(task);
      setMessage(useLLM ? '草案生成任务已启动，正在等待大模型返回。' : '草案生成任务已启动，正在使用本地规则生成。');
      await pollProposalTask(task.task_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : '优化草案生成失败');
      setProposalLoading(false);
    }
  };

  const savePromptConfig = async () => {
    if (!promptConfig) return;
    setPromptLoading(true);
    setError('');
    setMessage('');
    try {
      const resp = await fetch('/api/psych/training/prompts', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(promptConfig),
      });
      if (!resp.ok) throw new Error(await readError(resp));
      setPromptConfig((await resp.json()) as TrainingPromptConfig);
      setMessage('训练 Prompt 已保存。下一次生成草案会使用新 Prompt。');
    } catch (err) {
      setError(err instanceof Error ? err.message : '训练 Prompt 保存失败');
    } finally {
      setPromptLoading(false);
    }
  };

  const resetPromptConfig = async () => {
    setPromptLoading(true);
    setError('');
    setMessage('');
    try {
      const resp = await fetch('/api/psych/training/prompts/reset', { method: 'POST' });
      if (!resp.ok) throw new Error(await readError(resp));
      setPromptConfig((await resp.json()) as TrainingPromptConfig);
      setMessage('训练 Prompt 已恢复默认。');
    } catch (err) {
      setError(err instanceof Error ? err.message : '训练 Prompt 恢复失败');
    } finally {
      setPromptLoading(false);
    }
  };

  const startEditSample = (sample: TrainingSample) => {
    const nextReview = sample.human_review || EMPTY_REVIEW;
    setEditingSampleId(sample.sample_id);
    setEditName(sample.name || '');
    setEditReview({
      ...EMPTY_REVIEW,
      ...nextReview,
      human_score: nextReview.human_score ?? null,
    });
    setEditMissedText(joinLines(nextReview.missed_labels));
    setEditFalsePositiveText(joinLines(nextReview.false_positive_labels));
    setEditKeywordText(joinLines(nextReview.suggested_keywords));
    setEditNegativeKeywordText(joinLines(nextReview.suggested_negative_keywords));
    setEditNotes(sample.notes || nextReview.notes || '');
  };

  const cancelEditSample = () => {
    setEditingSampleId('');
    setEditName('');
    setEditReview(EMPTY_REVIEW);
    setEditMissedText('');
    setEditFalsePositiveText('');
    setEditKeywordText('');
    setEditNegativeKeywordText('');
    setEditNotes('');
  };

  const saveSampleEdit = async (sampleId: string) => {
    setLoading(true);
    setError('');
    setMessage('');
    const nextReview: TrainingReview = {
      ...editReview,
      missed_labels: splitLines(editMissedText),
      false_positive_labels: splitLines(editFalsePositiveText),
      suggested_keywords: splitLines(editKeywordText),
      suggested_negative_keywords: splitLines(editNegativeKeywordText),
      notes: editReview.notes || editNotes,
    };
    try {
      const resp = await fetch(`/api/psych/training/samples/${encodeURIComponent(sampleId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: editName,
          human_review: nextReview,
          notes: editNotes,
        }),
      });
      if (!resp.ok) throw new Error(await readError(resp));
      setMessage('训练样本审核已更新。重新生成草案时会使用新评语和评分。');
      cancelEditSample();
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : '训练样本更新失败');
    } finally {
      setLoading(false);
    }
  };

  const applyProposal = async (
    proposalId: string,
    selectedSuggestionIndexes?: number[],
    humanReviewConfirmed = false,
    autoRepeatAfterApply = false,
  ) => {
    setProposalLoading(true);
    setError('');
    setMessage('');
    try {
      const resp = await fetch('/api/psych/training/proposals/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          proposal_id: proposalId,
          selected_suggestion_indexes: selectedSuggestionIndexes,
          human_review_confirmed: humanReviewConfirmed,
          auto_repeat_after_apply: autoRepeatAfterApply,
          repeat_max_samples: maxSamples.trim().toLowerCase() === 'max' ? 'max' : Number(maxSamples || 30),
          repeat_use_llm: useLLM,
        }),
      });
      if (!resp.ok) throw new Error(await readError(resp));
      const applied = await resp.json() as Record<string, unknown>;
      const diff = applied.applied_diff as Record<string, unknown> | undefined;
      setLastAppliedDiff(diff || null);
      const changed = Boolean(diff?.changed);
      const effectiveCount = Array.isArray(applied.effective_suggestions) ? applied.effective_suggestions.length : 0;
      const ignoredCount = Array.isArray(applied.ignored_suggestions) ? applied.ignored_suggestions.length : 0;
      const repeatTask = applied.repeat_task as TrainingProposalTask | undefined;
      if (repeatTask?.task_id) {
        setProposalTask(repeatTask);
        openRetestPsychTask(repeatTask);
        setMessage(changed
          ? `评分标准已应用（生效 ${effectiveCount} 条，屏蔽重复 ${ignoredCount} 条），正在按同一批参考样本自动复测并生成下一轮待审核草案。`
          : `草案未产生新的配置差异（屏蔽重复 ${ignoredCount} 条）；仍将按设置执行自动复测闭环。`);
        await pollProposalTask(repeatTask.task_id);
      } else {
        setMessage(changed
          ? `评分标准已应用（生效 ${effectiveCount} 条，屏蔽重复 ${ignoredCount} 条）。建议去“评分标准”页面复核后再继续分析。`
          : `没有检测到新的配置差异；该草案可能已应用过，或建议已被规则覆盖。屏蔽重复 ${ignoredCount} 条。`);
        await loadData();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '应用草案失败');
    } finally {
      setProposalLoading(false);
    }
  };

  const deleteSample = async (sampleId: string) => {
    setError('');
    setMessage('');
    try {
      const resp = await fetch(`/api/psych/training/samples/${encodeURIComponent(sampleId)}`, {
        method: 'DELETE',
      });
      if (!resp.ok) throw new Error(await readError(resp));
      setMessage('训练样本已删除。');
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : '删除样本失败');
    }
  };

  return (
    <div>
      <Header
        title="训练优化"
        subtitle="用样本分析结果进行人工审核，再让大模型生成评分标准优化草案；应用前必须人工确认。"
      />

      <section className="mb-5 rounded-2xl border dk-border bg-white dark:bg-white/5 p-5 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="flex items-center gap-2 text-lg font-black text-[#1d1d1f] dark:text-white">
              <Users size={18} />
              训练范围
            </h2>
            <p className="mt-1 text-sm text-gray-500">
              当前范围：{currentScopeLabel}。重复训练会只读取该范围内的人审样本，避免不同联系人语境互相污染。
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {([
              ['all', '全部样本'],
              ['self', '本人样本'],
              ['contact', '特定联系人'],
            ] as Array<[TrainingScopeType, string]>).map(([key, label]) => (
              <button
                key={key}
                type="button"
                onClick={() => setTrainingScope(key)}
                className={`rounded-xl border px-3 py-2 text-sm font-black transition ${
                  trainingScope === key
                    ? 'border-[#07c160] bg-emerald-50 text-[#067a3d] dark:bg-emerald-500/10 dark:text-emerald-100'
                    : 'dk-border text-gray-500 hover:border-[#07c160] hover:text-[#07c160] dark:text-gray-200'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {trainingScope === 'contact' && (
          <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-[280px,1fr]">
            <label className="relative block">
              <Search size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
              <input
                value={targetQuery}
                onChange={(event) => setTargetQuery(event.target.value)}
                placeholder="搜索联系人昵称、备注或账号"
                className="dk-input w-full rounded-xl border dk-border bg-white py-2.5 pl-9 pr-3 text-sm dark:bg-white/5"
              />
            </label>
            <select
              value={trainingTargetKey}
              onChange={(event) => setTrainingTargetKey(event.target.value)}
              className="dk-input w-full rounded-xl border dk-border bg-white px-3 py-2.5 text-sm dark:bg-white/5"
            >
              <option value="">{targetLoading ? '正在读取联系人...' : '选择一个联系人训练范围'}</option>
              {filteredTargets.map((target) => (
                <option key={target.username} value={target.username}>
                  {targetDisplayName(target)} · {target.message_count || 0} 条 · {target.last_message_time || target.username}
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-4">
          {[
            '先保存该范围的人审样本，并补齐人工等级、分数和评语。',
            '若结果不符合预期，优先编辑旧样本评语，再重新生成草案。',
            '草案只会进入待审核状态，必须人工查看差异并勾选确认。',
            '应用后重新分析同一范围，继续补样本形成重复训练闭环。',
          ].map((text, index) => (
            <div key={text} className="rounded-xl bg-gray-50 p-3 text-sm text-gray-600 dark:bg-white/5 dark:text-gray-200">
              <div className="mb-1 text-xs font-black text-[#07c160]">步骤 {index + 1}</div>
              {text}
            </div>
          ))}
        </div>
      </section>

      <section className="mb-5 rounded-2xl border dk-border bg-white dark:bg-white/5 p-5 shadow-sm">
        <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="flex items-center gap-2 text-lg font-black text-[#1d1d1f] dark:text-white">
              <Sparkles size={18} />
              规则优化 Prompt
            </h2>
            <p className="mt-1 text-sm text-gray-500">
              这里控制“AI 如何生成规则优化草案”。AI 可以提出 Prompt 修改建议，但仍需你在草案中勾选并人工审核后才会应用。
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => void loadPromptConfig()}
              disabled={promptLoading}
              className="inline-flex items-center gap-2 rounded-xl border dk-border px-3 py-2 text-sm font-bold text-gray-600 hover:border-[#07c160] hover:text-[#07c160] disabled:opacity-60 dark:text-gray-200"
            >
              {promptLoading ? <Loader2 size={15} className="animate-spin" /> : <RefreshCw size={15} />}
              读取
            </button>
            <button
              type="button"
              onClick={() => void resetPromptConfig()}
              disabled={promptLoading}
              className="rounded-xl border dk-border px-3 py-2 text-sm font-bold text-gray-600 hover:border-amber-400 hover:text-amber-600 disabled:opacity-60 dark:text-gray-200"
            >
              恢复默认
            </button>
            <button
              type="button"
              onClick={() => void savePromptConfig()}
              disabled={promptLoading || !promptConfig}
              className="inline-flex items-center gap-2 rounded-xl bg-[#07c160] px-4 py-2 text-sm font-black text-white hover:bg-[#05a955] disabled:opacity-60"
            >
              {promptLoading ? <Loader2 size={15} className="animate-spin" /> : <Save size={15} />}
              保存 Prompt
            </button>
          </div>
        </div>

        {promptConfig ? (
          <div className="space-y-4">
            <label className="flex items-center justify-between rounded-xl border dk-border px-3 py-2.5 text-sm font-bold text-gray-600 dark:text-gray-200">
              允许 AI 在草案中提出 Prompt 修改建议
              <input
                type="checkbox"
                checked={promptConfig.allow_ai_prompt_suggestions}
                onChange={(event) => setPromptConfig((current) => current ? { ...current, allow_ai_prompt_suggestions: event.target.checked } : current)}
              />
            </label>
            <TextAreaField
              label="规则优化 System Prompt"
              value={promptConfig.training_system_prompt}
              onChange={(value) => setPromptConfig((current) => current ? { ...current, training_system_prompt: value } : current)}
              placeholder="用于生成评分标准优化草案的系统提示词。"
              rows={8}
            />
            <TextAreaField
              label="自动审核 System Prompt"
              value={promptConfig.auto_review_system_prompt}
              onChange={(value) => setPromptConfig((current) => current ? { ...current, auto_review_system_prompt: value } : current)}
              placeholder="用于分析后自动审核结果的系统提示词。"
              rows={7}
            />
            <TextAreaField
              label="人工备注"
              value={promptConfig.manual_notes || ''}
              onChange={(value) => setPromptConfig((current) => current ? { ...current, manual_notes: value } : current)}
              placeholder="记录你为什么这样调整 Prompt，方便后续复盘。"
            />
          </div>
        ) : (
          <div className="rounded-xl bg-gray-50 p-4 text-sm text-gray-500 dark:bg-white/5">
            尚未读取 Prompt 配置。
          </div>
        )}
      </section>

      <div className="grid grid-cols-1 2xl:grid-cols-[420px,1fr] gap-5">
        <section className="rounded-2xl border dk-border bg-white dark:bg-white/5 p-5 shadow-sm">
          <div className="mb-5 flex items-center justify-between gap-3">
            <div>
              <h2 className="flex items-center gap-2 text-lg font-black text-[#1d1d1f] dark:text-white">
                <GraduationCap size={19} />
                人工审核样本
              </h2>
              <p className="mt-1 text-sm text-gray-500">从最近一次心理分析结果导入，审核模型是否误判或漏判。</p>
            </div>
            <button
              type="button"
              onClick={refreshLastResult}
              className="inline-flex items-center gap-2 rounded-xl border dk-border px-3 py-2 text-sm font-bold text-gray-600 hover:border-[#07c160] hover:text-[#07c160] dark:text-gray-200"
            >
              <RefreshCw size={15} />
              读取最近结果
            </button>
          </div>

          <div className="rounded-xl bg-gray-50 p-4 text-sm text-gray-600 dark:bg-white/5 dark:text-gray-200">
            <div className="flex items-center justify-between gap-3">
              <span className="font-bold">最近分析</span>
              <span className="font-mono text-xs text-gray-400">{lastResult?.task_id || '无'}</span>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <MiniMetric label="模型评分" value={modelRisk(lastResult)} />
              <MiniMetric label="证据数" value={String(lastResult?.evidences?.length || 0)} />
            </div>
          </div>

          <div className="mt-4 space-y-4">
            <label className="block">
              <span className="mb-2 block text-sm font-bold text-gray-600 dark:text-gray-200">样本名称</span>
              <input
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="例如：低风险误判为中风险样本"
                className="dk-input w-full rounded-xl border dk-border bg-white px-3 py-2.5 text-sm dark:bg-white/5"
              />
            </label>

            <div className="grid grid-cols-2 gap-3">
              <label className="block">
                <span className="mb-2 block text-sm font-bold text-gray-600 dark:text-gray-200">人工风险等级</span>
                <select
                  value={review.human_risk_level}
                  onChange={(event) => setReview((current) => ({ ...current, human_risk_level: Number(event.target.value) }))}
                  className="dk-input w-full rounded-xl border dk-border bg-white px-3 py-2.5 text-sm dark:bg-white/5"
                >
                  {[0, 1, 2, 3, 4, 5].map((level) => (
                    <option key={level} value={level}>L{level}</option>
                  ))}
                </select>
              </label>
              <label className="block">
                <span className="mb-2 block text-sm font-bold text-gray-600 dark:text-gray-200">人工分数</span>
                <input
                  type="number"
                  min={0}
                  max={100}
                  value={review.human_score ?? ''}
                  onChange={(event) => setReview((current) => ({ ...current, human_score: event.target.value === '' ? null : Number(event.target.value) }))}
                  className="dk-input w-full rounded-xl border dk-border bg-white px-3 py-2.5 text-sm dark:bg-white/5"
                />
              </label>
            </div>

            <label className="flex items-center justify-between rounded-xl border dk-border px-3 py-2.5 text-sm font-bold text-gray-600 dark:text-gray-200">
              模型判断基本准确
              <input
                type="checkbox"
                checked={review.accurate}
                onChange={(event) => setReview((current) => ({ ...current, accurate: event.target.checked }))}
              />
            </label>

            <TextAreaField label="漏判标签或维度" value={missedText} onChange={setMissedText} placeholder="每行一个，例如：兴趣下降、正面情绪、社交退缩" />
            <TextAreaField label="误判标签或维度" value={falsePositiveText} onChange={setFalsePositiveText} placeholder="每行一个，例如：自我否定、存在/死亡型表达" />
            <TextAreaField label="建议加入关键词" value={keywordText} onChange={setKeywordText} placeholder="每行一个，被漏判时使用" />
            <TextAreaField label="建议降低触发的词" value={negativeKeywordText} onChange={setNegativeKeywordText} placeholder="每行一个，被误判时使用" />
            <TextAreaField
              label="审核说明"
              value={review.notes}
              onChange={(value) => setReview((current) => ({ ...current, notes: value }))}
              placeholder="说明为什么要调整，不要写医学诊断结论。"
            />

            <button
              type="button"
              onClick={saveSample}
              disabled={loading}
              className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-[#07c160] px-4 py-3 text-sm font-black text-white shadow-sm transition hover:bg-[#05a955] disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? <Loader2 size={17} className="animate-spin" /> : <Save size={17} />}
              保存人审样本
            </button>
          </div>
        </section>

        <div className="space-y-5">
          {error && <Notice tone="error" text={error} />}
          {message && <Notice tone="success" text={message} />}
          {lastAppliedDiff && (
            <section className="rounded-2xl border dk-border bg-white dark:bg-white/5 p-5 shadow-sm">
              <h2 className="text-base font-black text-[#1d1d1f] dark:text-white">最近应用差异</h2>
              <p className="mt-1 text-sm text-gray-500">用于确认草案是否真的改动了评分标准。重新分析前请确认这里不是空差异。</p>
              {lastAppliedDiff.changed ? (
                <RuleDiffView diff={lastAppliedDiff} />
              ) : (
                <div className="mt-3 rounded-xl bg-amber-50 p-3 text-sm text-amber-700 dark:bg-amber-500/10 dark:text-amber-100">
                  本次应用没有检测到评分标准差异。
                </div>
              )}
            </section>
          )}

          <section className="rounded-2xl border dk-border bg-white dark:bg-white/5 p-5 shadow-sm">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="flex items-center gap-2 text-lg font-black text-[#1d1d1f] dark:text-white">
                  <Database size={18} />
                  训练样本库
                </h2>
                <p className="mt-1 text-sm text-gray-500">已保存 {samples.length} 个样本，其中 {reviewedCount} 个有人审结果。</p>
              </div>
              <button
                type="button"
                onClick={loadData}
                className="inline-flex items-center gap-2 rounded-xl border dk-border px-3 py-2 text-sm font-bold text-gray-600 hover:border-[#07c160] hover:text-[#07c160] dark:text-gray-200"
              >
                <RefreshCw size={15} />
                刷新
              </button>
            </div>

            <div className="space-y-2">
              {samples.length === 0 ? (
                <p className="rounded-xl bg-gray-50 p-4 text-sm text-gray-500 dark:bg-white/5">暂无训练样本。先完成一次心理分析，再在左侧导入并人工审核。</p>
              ) : (
                samples.map((sample) => {
                  const model = sampleScore(sample);
                  const human = sample.human_review || EMPTY_REVIEW;
                  return (
                    <div key={sample.sample_id} className="rounded-xl border dk-border p-4">
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div>
                          <div className="font-black text-[#1d1d1f] dark:text-white">{sample.name || sample.sample_id}</div>
                          <div className="mt-1 text-xs text-gray-400">更新于 {formatTime(sample.updated_at)}</div>
                        </div>
                        <div className="flex items-center gap-1">
                          <button
                            type="button"
                            onClick={() => startEditSample(sample)}
                            className="rounded-lg px-2.5 py-2 text-xs font-bold text-gray-500 hover:bg-emerald-50 hover:text-[#07c160] dark:hover:bg-emerald-500/10"
                          >
                            编辑审核
                          </button>
                          <button
                            type="button"
                            onClick={() => deleteSample(sample.sample_id)}
                            className="rounded-lg p-2 text-gray-400 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-500/10"
                            title="删除样本"
                          >
                            <Trash2 size={16} />
                          </button>
                        </div>
                      </div>
                      <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2">
                        <MiniMetric label="模型等级" value={`L${model.level}`} />
                        <MiniMetric label="模型分数" value={String(model.score)} />
                        <MiniMetric label="人工等级" value={`L${human.human_risk_level ?? 0}`} />
                        <MiniMetric label="人工分数" value={human.human_score == null ? '-' : String(human.human_score)} />
                      </div>
                      {human.notes && <p className="mt-3 text-sm text-gray-600 dark:text-gray-200">{human.notes}</p>}
                      {editingSampleId === sample.sample_id && (
                        <div className="mt-4 rounded-xl bg-gray-50 p-4 dark:bg-white/5">
                          <div className="mb-3 text-sm font-black text-[#1d1d1f] dark:text-white">编辑参考样本审核</div>
                          <div className="space-y-3">
                            <label className="block">
                              <span className="mb-1 block text-xs font-bold text-gray-500">样本名称</span>
                              <input
                                value={editName}
                                onChange={(event) => setEditName(event.target.value)}
                                className="dk-input w-full rounded-xl border dk-border bg-white px-3 py-2 text-sm dark:bg-white/5"
                              />
                            </label>
                            <div className="grid grid-cols-2 gap-3">
                              <label className="block">
                                <span className="mb-1 block text-xs font-bold text-gray-500">人工风险等级</span>
                                <select
                                  value={editReview.human_risk_level}
                                  onChange={(event) => setEditReview((current) => ({ ...current, human_risk_level: Number(event.target.value) }))}
                                  className="dk-input w-full rounded-xl border dk-border bg-white px-3 py-2 text-sm dark:bg-white/5"
                                >
                                  {[0, 1, 2, 3, 4, 5].map((level) => (
                                    <option key={level} value={level}>L{level}</option>
                                  ))}
                                </select>
                              </label>
                              <label className="block">
                                <span className="mb-1 block text-xs font-bold text-gray-500">人工分数</span>
                                <input
                                  type="number"
                                  min={0}
                                  max={100}
                                  value={editReview.human_score ?? ''}
                                  onChange={(event) => setEditReview((current) => ({ ...current, human_score: event.target.value === '' ? null : Number(event.target.value) }))}
                                  className="dk-input w-full rounded-xl border dk-border bg-white px-3 py-2 text-sm dark:bg-white/5"
                                />
                              </label>
                            </div>
                            <label className="flex items-center justify-between rounded-xl border dk-border bg-white px-3 py-2.5 text-sm font-bold text-gray-600 dark:bg-white/5 dark:text-gray-200">
                              模型判断基本准确
                              <input
                                type="checkbox"
                                checked={editReview.accurate}
                                onChange={(event) => setEditReview((current) => ({ ...current, accurate: event.target.checked }))}
                              />
                            </label>
                            <TextAreaField label="漏判标签或维度" value={editMissedText} onChange={setEditMissedText} />
                            <TextAreaField label="误判标签或维度" value={editFalsePositiveText} onChange={setEditFalsePositiveText} />
                            <TextAreaField label="建议加入关键词" value={editKeywordText} onChange={setEditKeywordText} />
                            <TextAreaField label="建议降低触发的词" value={editNegativeKeywordText} onChange={setEditNegativeKeywordText} />
                            <TextAreaField
                              label="审核评语"
                              value={editReview.notes}
                              onChange={(value) => setEditReview((current) => ({ ...current, notes: value }))}
                              placeholder="补充人工评语、误判原因或需要修改的评分点。"
                            />
                            <TextAreaField
                              label="样本备注"
                              value={editNotes}
                              onChange={setEditNotes}
                              placeholder="样本管理备注，不直接作为人工审核结论。"
                            />
                            <div className="flex flex-wrap gap-2">
                              <button
                                type="button"
                                onClick={() => saveSampleEdit(sample.sample_id)}
                                disabled={loading}
                                className="inline-flex items-center gap-2 rounded-xl bg-[#07c160] px-4 py-2 text-sm font-black text-white hover:bg-[#05a955] disabled:opacity-60"
                              >
                                {loading ? <Loader2 size={15} className="animate-spin" /> : <Save size={15} />}
                                保存修改
                              </button>
                              <button
                                type="button"
                                onClick={cancelEditSample}
                                className="rounded-xl border dk-border px-4 py-2 text-sm font-bold text-gray-600 hover:border-gray-400 dark:text-gray-200"
                              >
                                取消
                              </button>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </section>

          <section className="rounded-2xl border dk-border bg-white dark:bg-white/5 p-5 shadow-sm">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="flex items-center gap-2 text-lg font-black text-[#1d1d1f] dark:text-white">
                  <Sparkles size={18} />
                  评分标准迭代草案
                </h2>
                <p className="mt-1 text-sm text-gray-500">大模型只生成建议，是否应用由你确认。</p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <label className="inline-flex items-center gap-2 rounded-xl border dk-border px-3 py-2 text-sm font-bold text-gray-600 dark:text-gray-200">
                  使用大模型
                  <input type="checkbox" checked={useLLM} onChange={(event) => setUseLLM(event.target.checked)} />
                </label>
                <input
                  type="text"
                  value={maxSamples}
                  onChange={(event) => setMaxSamples(event.target.value)}
                  className="dk-input w-24 rounded-xl border dk-border bg-white px-3 py-2 text-sm dark:bg-white/5"
                  title="最多使用样本数，可填 max"
                />
                <button
                  type="button"
                  onClick={() => setMaxSamples('max')}
                  className="rounded-xl border dk-border px-3 py-2 text-sm font-bold text-gray-600 hover:border-[#07c160] hover:text-[#07c160] dark:text-gray-200"
                >
                  max
                </button>
                <button
                  type="button"
                  onClick={generateProposal}
                  disabled={proposalLoading || samples.length === 0 || (trainingScope === 'contact' && !trainingTargetKey)}
                  className="inline-flex items-center gap-2 rounded-xl bg-[#07c160] px-4 py-2.5 text-sm font-black text-white hover:bg-[#05a955] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {proposalLoading ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
                  生成草案
                </button>
              </div>
            </div>

            {proposalTask && (
              <div className="mb-4 rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-800 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-100">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="font-black">{proposalTask.message || '正在生成优化草案'}</div>
                  <div className="font-mono text-xs opacity-70">{proposalTask.task_id.slice(0, 8)}</div>
                </div>
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-white/80 dark:bg-black/20">
                  <div
                    className="h-full rounded-full bg-[#07c160] transition-all"
                    style={{ width: `${Math.max(0, Math.min(100, Number(proposalTask.progress || 0)))}%` }}
                  />
                </div>
                <div className="mt-2 flex flex-wrap justify-between gap-2 text-xs opacity-80">
                  <span>阶段：{proposalTask.stage || '-'}</span>
                  <span>{Math.round(Number(proposalTask.progress || 0))}%</span>
                </div>
                {activeRetestPsychTaskId && (
                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-xl bg-white/70 p-3 dark:bg-black/20">
                    <div className="text-xs">
                      <div className="font-black">当前复测样本细化流程</div>
                      <div className="mt-0.5 font-mono opacity-70">{activeRetestPsychTaskId.slice(0, 8)}</div>
                    </div>
                    <button
                      type="button"
                      onClick={() => openRetestPsychTask(proposalTask, true)}
                      className="inline-flex items-center gap-2 rounded-xl bg-[#07c160] px-3 py-2 text-xs font-black text-white hover:bg-[#05a955]"
                    >
                      <RefreshCw size={14} />
                      查看心理分析流程
                    </button>
                  </div>
                )}
              </div>
            )}

            {latestProposal ? (
              <ProposalCard
                proposal={latestProposal}
                onApply={(indexes, confirmed, autoRepeat) => applyProposal(latestProposal.proposal_id, indexes, confirmed, autoRepeat)}
                applying={proposalLoading}
              />
            ) : (
              <p className="rounded-xl bg-gray-50 p-4 text-sm text-gray-500 dark:bg-white/5">暂无优化草案。保存至少一个人审样本后生成。</p>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}

function TextAreaField({
  label,
  value,
  onChange,
  placeholder,
  rows = 3,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  rows?: number;
}) {
  return (
    <label className="block">
      <span className="mb-2 block text-sm font-bold text-gray-600 dark:text-gray-200">{label}</span>
      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        rows={rows}
        placeholder={placeholder}
        className="dk-input w-full resize-none rounded-xl border dk-border bg-white px-3 py-2 text-sm dark:bg-white/5"
      />
    </label>
  );
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border dk-border bg-white px-3 py-2 dark:bg-white/5">
      <div className="text-xs font-bold text-gray-400">{label}</div>
      <div className="mt-1 text-sm font-black text-[#1d1d1f] dark:text-white">{value}</div>
    </div>
  );
}

function Notice({ tone, text }: { tone: 'success' | 'error'; text: string }) {
  const isError = tone === 'error';
  return (
    <div className={`rounded-2xl border p-4 text-sm ${isError ? 'border-red-200 bg-red-50 text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-200' : 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-100'}`}>
      <div className="flex items-start gap-2">
        {isError ? <AlertCircle size={18} className="mt-0.5 shrink-0" /> : <CheckCircle2 size={18} className="mt-0.5 shrink-0" />}
        <span>{text}</span>
      </div>
    </div>
  );
}

function ProposalCard({
  proposal,
  onApply,
  applying,
}: {
  proposal: TrainingProposal;
  onApply: (selectedIndexes: number[], humanReviewConfirmed: boolean, autoRepeatAfterApply: boolean) => void;
  applying: boolean;
}) {
  const diagnostics = proposal.diagnostics || {};
  const allIndexes = useMemo(
    () => (proposal.suggestions || []).map((_, index) => index),
    [proposal.suggestions],
  );
  const [selectedIndexes, setSelectedIndexes] = useState<number[]>(allIndexes);
  const [preview, setPreview] = useState<ProposalPreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState('');
  const [reviewConfirmed, setReviewConfirmed] = useState(false);

  useEffect(() => {
    setSelectedIndexes(allIndexes);
    setReviewConfirmed(false);
  }, [allIndexes, proposal.proposal_id]);

  const refreshPreview = useCallback(async () => {
    setPreviewLoading(true);
    setPreviewError('');
    try {
      const resp = await fetch('/api/psych/training/proposals/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          proposal_id: proposal.proposal_id,
          selected_suggestion_indexes: selectedIndexes,
        }),
      });
      if (!resp.ok) throw new Error(await readError(resp));
      setPreview((await resp.json()) as ProposalPreview);
    } catch (err) {
      setPreviewError(err instanceof Error ? err.message : '草案差异预览失败');
    } finally {
      setPreviewLoading(false);
    }
  }, [proposal.proposal_id, selectedIndexes]);

  useEffect(() => {
    void refreshPreview();
  }, [refreshPreview]);

  const toggleIndex = (index: number) => {
    setSelectedIndexes((current) => (
      current.includes(index)
        ? current.filter((item) => item !== index)
        : [...current, index].sort((a, b) => a - b)
    ));
  };

  const selectAll = () => setSelectedIndexes(allIndexes);
  const clearAll = () => setSelectedIndexes([]);
  const proposalScope = asRecord(diagnostics.training_scope);
  const ignoredSuggestionCount = Number(diagnostics.ignored_suggestion_count || 0);
  const ignoredSuggestions = Array.isArray(diagnostics.ignored_suggestions)
    ? diagnostics.ignored_suggestions as Array<Record<string, unknown>>
    : [];
  const guidance = Array.isArray(diagnostics.repeat_training_guidance)
    ? diagnostics.repeat_training_guidance.map((item) => String(item))
    : [];

  return (
    <div className="rounded-xl border dk-border p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-base font-black text-[#1d1d1f] dark:text-white">{proposal.summary || '评分标准优化草案'}</div>
          <div className="mt-1 text-xs text-gray-400">
            {proposal.model_provider || 'local'} / {proposal.model_name || 'rule'} · {proposal.sample_count} 个样本 · {formatTime(proposal.created_at)}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => onApply(selectedIndexes, reviewConfirmed, false)}
            disabled={applying || selectedIndexes.length === 0 || !reviewConfirmed}
            className="inline-flex items-center gap-2 rounded-xl border border-[#07c160] px-4 py-2 text-sm font-black text-[#067a3d] hover:bg-emerald-50 disabled:cursor-not-allowed disabled:opacity-60 dark:text-emerald-100 dark:hover:bg-emerald-500/10"
          >
            {applying ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
            仅应用所选
          </button>
          <button
            type="button"
            onClick={() => onApply(selectedIndexes, reviewConfirmed, true)}
            disabled={applying || selectedIndexes.length === 0 || !reviewConfirmed}
            className="inline-flex items-center gap-2 rounded-xl bg-[#07c160] px-4 py-2 text-sm font-black text-white hover:bg-[#05a955] disabled:cursor-not-allowed disabled:opacity-60"
          >
            {applying ? <Loader2 size={16} className="animate-spin" /> : <RefreshCw size={16} />}
            应用并自动复测
          </button>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2">
        <MiniMetric label="建议数" value={String(proposal.suggestions?.length || 0)} />
        <MiniMetric label="已选择" value={String(selectedIndexes.length)} />
        <MiniMetric label="屏蔽重复" value={String(ignoredSuggestionCount)} />
        <MiniMetric label="LLM" value={diagnostics.llm_used ? '已使用' : '未使用/失败'} />
        <MiniMetric label="草案样本" value={String(proposal.sample_count || 0)} />
      </div>

      <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-100">
        <label className="flex items-start gap-3 font-bold">
          <input
            type="checkbox"
            checked={reviewConfirmed}
            onChange={(event) => setReviewConfirmed(event.target.checked)}
            className="mt-1"
          />
          <span>
            我已人工审核本草案的参考样本、规则建议和差异预览，确认只应用当前勾选的变更。
          </span>
        </label>
      </div>

      {diagnostics.llm_error ? (
        <div className="mt-3 rounded-xl bg-amber-50 p-3 text-sm text-amber-700 dark:bg-amber-500/10 dark:text-amber-100">
          大模型调用未完成，已使用本地规则兜底：{String(diagnostics.llm_error)}
        </div>
      ) : null}

      {ignoredSuggestions.length > 0 ? (
        <div className="mt-3 rounded-xl bg-gray-50 p-3 text-sm text-gray-600 dark:bg-white/5 dark:text-gray-200">
          <div className="font-bold text-[#1d1d1f] dark:text-white">已屏蔽的重复/无效建议</div>
          <p className="mt-1 text-xs text-gray-500">这些建议已经被当前规则覆盖，或应用后不会产生真实差异，因此不会进入可应用列表。</p>
          <div className="mt-2 space-y-2">
            {ignoredSuggestions.slice(0, 5).map((item, index) => (
              <div key={`ignored-${proposal.proposal_id}-${index}`} className="rounded-lg border dk-border bg-white p-2 text-xs dark:bg-black/10">
                <div className="font-black text-gray-500">
                  #{index + 1} {suggestionTypeLabel(String(item.type || 'suggestion'))}
                  {item.dimension_key ? ` · ${String(item.dimension_key)}` : ''}
                  {item.label_key ? ` · ${String(item.label_key)}` : ''}
                </div>
                <SuggestionBody item={item} />
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {Object.keys(proposalScope).length > 0 ? (
        <div className="mt-3 rounded-xl bg-gray-50 p-3 text-sm text-gray-600 dark:bg-white/5 dark:text-gray-200">
          <div className="font-bold text-[#1d1d1f] dark:text-white">草案训练范围</div>
          <div className="mt-1 text-xs">
            {String(proposalScope.label || proposalScope.target_type || '全部人审样本')}
            {proposalScope.target_key ? ` · ${String(proposalScope.target_key).slice(0, 18)}` : ''}
          </div>
          {guidance.length > 0 && (
            <ol className="mt-2 list-decimal space-y-1 pl-4 text-xs text-gray-500 dark:text-gray-300">
              {guidance.map((item) => <li key={item}>{item}</li>)}
            </ol>
          )}
        </div>
      ) : null}

      {Array.isArray(diagnostics.sample_ids) && diagnostics.sample_ids.length > 0 ? (
        <div className="mt-3 rounded-xl bg-gray-50 p-3 text-sm text-gray-600 dark:bg-white/5 dark:text-gray-200">
          <div className="mb-2 font-bold text-[#1d1d1f] dark:text-white">
            参考样本：{diagnostics.sample_ids.length} 个
          </div>
          <div className="flex flex-wrap gap-1.5">
            {diagnostics.sample_ids.slice(0, 24).map((sampleId) => (
              <span key={String(sampleId)} className="rounded-full bg-white px-2 py-0.5 font-mono text-[11px] text-gray-500 dark:bg-black/20">
                {String(sampleId).slice(0, 8)}
              </span>
            ))}
            {diagnostics.sample_ids.length > 24 && (
              <span className="rounded-full bg-white px-2 py-0.5 text-[11px] text-gray-500 dark:bg-black/20">
                +{diagnostics.sample_ids.length - 24}
              </span>
            )}
          </div>
          <p className="mt-2 text-xs text-gray-500">需要补充评分或评语时，可在上方样本库中点击“编辑审核”，然后重新生成草案。</p>
        </div>
      ) : null}

      <div className="mt-4 rounded-xl border dk-border p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <div className="text-sm font-black text-[#1d1d1f] dark:text-white">选择要应用的规则变更</div>
            <p className="mt-1 text-xs text-gray-500">只会应用已勾选的建议；未勾选建议保留在草案中，不写入评分标准。</p>
          </div>
          <div className="flex gap-2">
            <button type="button" onClick={selectAll} className="rounded-lg border dk-border px-2.5 py-1.5 text-xs font-bold text-gray-500 hover:text-[#07c160]">
              全选
            </button>
            <button type="button" onClick={clearAll} className="rounded-lg border dk-border px-2.5 py-1.5 text-xs font-bold text-gray-500 hover:text-red-600">
              清空
            </button>
          </div>
        </div>

        <div className="mt-3 space-y-2">
          {(proposal.suggestions || []).map((item, index) => (
            <label
              key={`${proposal.proposal_id}-${index}`}
              className={`block rounded-xl border p-3 text-sm transition ${selectedIndexes.includes(index) ? 'border-[#07c160] bg-emerald-50/70 dark:bg-emerald-500/10' : 'dk-border bg-gray-50 dark:bg-white/5'}`}
            >
              <div className="flex items-start gap-3">
                <input
                  type="checkbox"
                  checked={selectedIndexes.includes(index)}
                  onChange={() => toggleIndex(index)}
                  className="mt-1"
                />
                <div className="min-w-0 flex-1">
                  <div className="font-black text-[#1d1d1f] dark:text-white">
                    #{index + 1} {suggestionTypeLabel(String(item.type || 'suggestion'))}
                    {item.dimension_key ? ` · ${String(item.dimension_key)}` : ''}
                    {item.label_key ? ` · ${String(item.label_key)}` : ''}
                  </div>
                  <SuggestionBody item={item} />
                </div>
              </div>
            </label>
          ))}
        </div>
      </div>

      <div className="mt-4 rounded-xl border dk-border p-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-sm font-black text-[#1d1d1f] dark:text-white">所选规则变更预览</div>
            <p className="mt-1 text-xs text-gray-500">这里展示应用后评分标准会发生的真实差异。</p>
          </div>
          <button
            type="button"
            onClick={() => void refreshPreview()}
            disabled={previewLoading}
            className="inline-flex items-center gap-2 rounded-lg border dk-border px-2.5 py-1.5 text-xs font-bold text-gray-500 hover:text-[#07c160] disabled:opacity-60"
          >
            {previewLoading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
            刷新预览
          </button>
        </div>
        {previewError ? (
          <div className="mt-3 rounded-lg bg-red-50 p-3 text-sm text-red-700 dark:bg-red-500/10 dark:text-red-100">{previewError}</div>
        ) : previewLoading && !preview ? (
          <div className="mt-3 rounded-lg bg-gray-50 p-3 text-sm text-gray-500 dark:bg-white/5">正在计算差异...</div>
        ) : preview?.changed ? (
          <div>
            <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">
              <MiniMetric label="真实生效" value={String(preview.effective_suggestions?.length || 0)} />
              <MiniMetric label="已覆盖/无效" value={String(preview.ignored_suggestions?.length || 0)} />
              <MiniMetric label="选择数量" value={String(selectedIndexes.length)} />
            </div>
            <RuleDiffView diff={preview.preview_diff} />
          </div>
        ) : (
          <div className="mt-3 rounded-lg bg-amber-50 p-3 text-sm text-amber-700 dark:bg-amber-500/10 dark:text-amber-100">
            当前选择不会改变评分标准。可能原因：这些建议已经应用过、未命中可修改字段，或已被你清空选择。
            {preview?.ignored_suggestions?.length ? ` 已识别重复/无效 ${preview.ignored_suggestions.length} 条。` : ''}
          </div>
        )}
      </div>

      <div className="mt-4 rounded-xl border border-[#07c160]/40 bg-emerald-50/70 p-4 dark:bg-emerald-500/10">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-sm font-black text-[#1d1d1f] dark:text-white">审核完成后的操作</div>
            <p className="mt-1 text-xs text-gray-600 dark:text-gray-200">
              先勾选上方“我已人工审核”，再选择仅应用，或应用后自动复测并生成下一轮待审核草案。
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => onApply(selectedIndexes, reviewConfirmed, false)}
              disabled={applying || selectedIndexes.length === 0 || !reviewConfirmed}
              className="inline-flex items-center gap-2 rounded-xl border border-[#07c160] bg-white px-4 py-2.5 text-sm font-black text-[#067a3d] hover:bg-emerald-50 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-black/20 dark:text-emerald-100 dark:hover:bg-emerald-500/10"
            >
              {applying ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
              仅应用所选
            </button>
            <button
              type="button"
              onClick={() => onApply(selectedIndexes, reviewConfirmed, true)}
              disabled={applying || selectedIndexes.length === 0 || !reviewConfirmed}
              className="inline-flex items-center gap-2 rounded-xl bg-[#07c160] px-4 py-2.5 text-sm font-black text-white hover:bg-[#05a955] disabled:cursor-not-allowed disabled:opacity-60"
            >
              {applying ? <Loader2 size={16} className="animate-spin" /> : <RefreshCw size={16} />}
              应用并自动复测
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function suggestionTypeLabel(type: string): string {
  const labels: Record<string, string> = {
    add_keywords: '新增关键词',
    add_keyword: '新增关键词',
    remove_keywords: '降低触发/移入排除词',
    remove_keyword: '降低触发/移入排除词',
    exclude_keywords: '新增排除词',
    add_exclude_keywords: '新增排除词',
    lower_trigger: '降低触发',
    add_vector_query: '新增向量查询',
    add_vector_queries: '新增向量查询',
    adjust_threshold: '调整阈值',
    set_threshold: '调整阈值',
    adjust_dimension_points: '调整维度分值',
    adjust_dimension_weight: '调整维度分值',
    set_max_points: '调整维度分值',
    update_description: '更新规则说明',
    set_description: '更新规则说明',
    update_rule_description: '更新规则说明',
    append_reason: '应用优化理由',
    append_training_prompt_instruction: '追加规则优化 Prompt 指令',
    update_training_prompt: '替换规则优化 Prompt',
    append_auto_review_prompt_instruction: '追加自动审核 Prompt 指令',
    update_auto_review_prompt: '替换自动审核 Prompt',
  };
  return labels[type] || type;
}

function SuggestionBody({ item }: { item: Record<string, unknown> }) {
  const keywords = Array.isArray(item.keywords) ? item.keywords : [];
  return (
    <div>
      {keywords.length > 0 ? (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {keywords.map((word) => (
            <span key={String(word)} className="rounded-full bg-white px-2 py-0.5 text-xs font-bold text-[#067a3d] dark:bg-black/20 dark:text-emerald-100">
              {String(word)}
            </span>
          ))}
        </div>
      ) : null}
      {item.query ? <p className="mt-2 text-gray-600 dark:text-gray-200">{String(item.query)}</p> : null}
      {item.threshold ? <p className="mt-2 text-xs text-gray-500">阈值：{String(item.threshold)} → {String(item.value ?? '')}</p> : null}
      {item.description ? <p className="mt-2 text-xs text-gray-600 dark:text-gray-200">规则说明：{String(item.description)}</p> : null}
      {item.instruction ? <p className="mt-2 text-xs text-gray-600 dark:text-gray-200">Prompt 指令：{String(item.instruction)}</p> : null}
      {item.prompt_text ? <pre className="mt-2 max-h-40 overflow-auto rounded-lg bg-white p-2 text-xs leading-5 text-gray-600 dark:bg-black/20 dark:text-gray-200">{String(item.prompt_text)}</pre> : null}
      {item.reason ? <p className="mt-2 text-xs text-gray-500">{String(item.reason)}</p> : null}
    </div>
  );
}

function RuleDiffView({ diff }: { diff: Record<string, unknown> }) {
  const thresholds = asRecord(diff.thresholds);
  const dimensions = asRecord(diff.dimensions);
  const labels = asRecord(diff.labels);
  const prompts = asRecord(diff.prompts);
  const groups = [
    { title: '风险阈值', data: thresholds },
    { title: '评分维度', data: dimensions },
    { title: '多标签规则', data: labels },
    { title: '训练 Prompt', data: prompts },
  ].filter((item) => Object.keys(item.data).length > 0);

  if (groups.length === 0) {
    return <div className="mt-3 rounded-lg bg-amber-50 p-3 text-sm text-amber-700 dark:bg-amber-500/10 dark:text-amber-100">没有检测到配置差异。</div>;
  }

  return (
    <div className="mt-3 space-y-3">
      {groups.map((group) => (
        <div key={group.title} className="rounded-xl bg-gray-50 p-3 dark:bg-white/5">
          <div className="mb-2 text-sm font-black text-[#1d1d1f] dark:text-white">{group.title}</div>
          <div className="space-y-2">
            {Object.entries(group.data).map(([key, value]) => (
              <RuleDiffItem key={`${group.title}-${key}`} itemKey={key} value={asRecord(value)} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function RuleDiffItem({ itemKey, value }: { itemKey: string; value: Record<string, unknown> }) {
  return (
    <div className="rounded-lg border dk-border bg-white p-3 text-sm dark:bg-black/10">
      <div className="font-mono text-xs font-black text-gray-500">{itemKey}</div>
      <div className="mt-2 space-y-2">
        {Object.entries(value).map(([field, raw]) => (
          <FieldDiff key={`${itemKey}-${field}`} field={field} value={asRecord(raw)} />
        ))}
      </div>
    </div>
  );
}

function FieldDiff({ field, value }: { field: string; value: Record<string, unknown> }) {
  const added = Array.isArray(value.added) ? value.added : [];
  const removed = Array.isArray(value.removed) ? value.removed : [];
  if (field === 'description' || field.includes('prompt') || field === 'manual_notes') {
    return (
      <div>
        <div className="mb-1 text-xs font-bold text-gray-500">{fieldLabel(field)}</div>
        <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
          <div className="rounded-lg bg-red-50 p-2 text-xs leading-5 text-red-700 dark:bg-red-500/20 dark:text-red-100">
            <div className="mb-1 font-bold">应用前</div>
            <pre className="whitespace-pre-wrap break-words font-sans">{String(value.before ?? '-')}</pre>
          </div>
          <div className="rounded-lg bg-emerald-50 p-2 text-xs leading-5 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-100">
            <div className="mb-1 font-bold">应用后</div>
            <pre className="whitespace-pre-wrap break-words font-sans">{String(value.after ?? '-')}</pre>
          </div>
        </div>
      </div>
    );
  }
  if (added.length || removed.length) {
    return (
      <div>
        <div className="mb-1 text-xs font-bold text-gray-500">{fieldLabel(field)}</div>
        {added.length > 0 && (
          <div className="mb-1 flex flex-wrap gap-1.5">
            {added.map((item) => (
              <span key={`add-${field}-${String(item)}`} className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-bold text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-100">
                + {String(item)}
              </span>
            ))}
          </div>
        )}
        {removed.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {removed.map((item) => (
              <span key={`remove-${field}-${String(item)}`} className="rounded-full bg-red-50 px-2 py-0.5 text-xs font-bold text-red-700 dark:bg-red-500/20 dark:text-red-100">
                - {String(item)}
              </span>
            ))}
          </div>
        )}
      </div>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="font-bold text-gray-500">{fieldLabel(field)}</span>
      <span className="rounded bg-red-50 px-2 py-0.5 text-red-700 dark:bg-red-500/20 dark:text-red-100">{String(value.before ?? '-')}</span>
      <span className="text-gray-400">→</span>
      <span className="rounded bg-emerald-50 px-2 py-0.5 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-100">{String(value.after ?? '-')}</span>
    </div>
  );
}

function fieldLabel(field: string): string {
  const labels: Record<string, string> = {
    keywords: '关键词',
    exclude_keywords: '排除词',
    strong_keywords: '强信号词',
    vector_queries: '向量查询',
    max_points: '最高分',
    enabled: '启用状态',
    redline: '红线项',
    risk_level: '风险等级',
    description: '规则说明/优化理由',
    training_system_prompt: '规则优化 System Prompt',
    auto_review_system_prompt: '自动审核 System Prompt',
    allow_ai_prompt_suggestions: '允许 AI 提出 Prompt 修改',
    manual_notes: '人工备注',
  };
  return labels[field] || field;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}
