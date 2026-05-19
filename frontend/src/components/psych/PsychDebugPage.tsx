import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { BrainCircuit, ClipboardList, FileText, ListFilter, RefreshCw, Search, Sparkles } from 'lucide-react';
import { Header } from '../layout/Header';
import { LAST_PSYCH_RESULT_KEY, type PsychAnalyzeResponse } from './types';

type DebugMessage = {
  id?: number;
  seq?: number;
  datetime?: string;
  sender?: string;
  is_mine?: boolean;
  content?: string;
  candidate?: string;
  text?: string;
  fact?: string;
  score?: number;
  hit_groups?: string[];
  hit_words?: string[];
  labels?: string[];
  label_names?: string[];
  risk_level?: number;
  context_before?: DebugMessage[];
  context_after?: DebugMessage[];
};

type DebugStep = Record<string, unknown> & {
  step?: Record<string, unknown>;
  diagnostics?: Record<string, unknown>;
  events?: Array<Record<string, unknown>>;
  candidates?: DebugMessage[];
  messages?: DebugMessage[];
  input_messages?: DebugMessage[];
  output_messages?: DebugMessage[];
  context_messages?: DebugMessage[];
  input_candidates?: DebugMessage[];
  output_candidates?: DebugMessage[];
  selected?: DebugMessage[];
  payload?: DebugMessage[];
  raw_hits?: DebugMessage[];
  scoped_hits?: DebugMessage[];
  system_prompt?: string;
  user_prompt?: string;
  reply?: string;
  selected_ids?: number[];
  metrics?: Record<string, unknown>;
  dimension_scores?: Array<Record<string, unknown>>;
  symptom_labels?: Array<Record<string, unknown>>;
  scoring_adjustments?: Record<string, unknown>;
  fact_hits?: DebugMessage[];
  facts?: DebugMessage[];
  report_preview?: string;
  auto_review?: Record<string, unknown>;
};

type StepConfig = {
  key: string;
  icon: ReactNode;
  title: string;
  subtitle: string;
};

const STEP_META: Record<string, Omit<StepConfig, 'key'>> = {
  load_messages: {
    icon: <ClipboardList size={18} />,
    title: '1. 读取聊天消息',
    subtitle: '消息来源、目标范围、读取诊断和脱敏样例。',
  },
  preprocess: {
    icon: <ListFilter size={18} />,
    title: '2. 预处理与隐私过滤',
    subtitle: '过滤前后消息、分析对象范围和上下文窗口。',
  },
  keyword_search: {
    icon: <Search size={18} />,
    title: '3. 关键词检索',
    subtitle: '规则词典命中后进入候选池的消息。',
  },
  keyword_llm_screen: {
    icon: <BrainCircuit size={18} />,
    title: '4. 大模型筛选（关键词候选）',
    subtitle: '关键词候选、上下文 payload、发送给 AI 的 prompt 和筛选结果。',
  },
  vector_semantic_search: {
    icon: <Sparkles size={18} />,
    title: '5. 向量语义检索',
    subtitle: '向量查询、原始语义命中、过滤到本次范围后的命中。',
  },
  vector_llm_screen: {
    icon: <ListFilter size={18} />,
    title: '6. 大模型筛选（向量候选）',
    subtitle: '向量召回候选、上下文 payload、发送给 AI 的 prompt 和筛选结果。',
  },
  fact_memory_search: {
    icon: <FileText size={18} />,
    title: '7. 心理事实库检索',
    subtitle: '从 mem_facts 召回的心理事实和相似度结果。',
  },
  scoring: {
    icon: <ClipboardList size={18} />,
    title: '8. 综合评分',
    subtitle: '维度分、标签、证据强度、时间修正和保护性修正。',
  },
  report: {
    icon: <FileText size={18} />,
    title: '9. 生成报告',
    subtitle: '报告结构、免责声明和 Markdown 预览。',
  },
  training_auto_review: {
    icon: <Sparkles size={18} />,
    title: '10. 自动审核与优化草案',
    subtitle: '分析后自动生成训练样本和优化草案的结果。',
  },
};

const FALLBACK_STEP_KEYS = [
  'load_messages',
  'preprocess',
  'keyword_search',
  'keyword_llm_screen',
  'vector_semantic_search',
  'vector_llm_screen',
  'fact_memory_search',
  'scoring',
  'report',
  'training_auto_review',
];

function loadLastResult(): PsychAnalyzeResponse | null {
  try {
    const raw = localStorage.getItem(LAST_PSYCH_RESULT_KEY);
    return raw ? JSON.parse(raw) as PsychAnalyzeResponse : null;
  } catch {
    return null;
  }
}

function debugSteps(result: PsychAnalyzeResponse | null): Record<string, DebugStep> {
  const value = result?.report_json?.debug_steps;
  return value && typeof value === 'object' ? value as Record<string, DebugStep> : {};
}

function stepConfigs(result: PsychAnalyzeResponse | null, steps: Record<string, DebugStep>): StepConfig[] {
  const seen = new Set<string>();
  const configs: StepConfig[] = [];
  for (const step of result?.process_steps || []) {
    const meta = STEP_META[step.key] || {
      icon: <ClipboardList size={18} />,
      title: step.name || step.key,
      subtitle: step.detail || '该步骤的运行指标和调试信息。',
    };
    configs.push({
      key: step.key,
      icon: meta.icon,
      title: meta.title,
      subtitle: meta.subtitle,
    });
    seen.add(step.key);
  }
  for (const key of FALLBACK_STEP_KEYS) {
    if (!seen.has(key) && steps[key] && Object.keys(steps[key]).length > 0) {
      const meta = STEP_META[key] || {
        icon: <ClipboardList size={18} />,
        title: key,
        subtitle: '该步骤的运行指标和调试信息。',
      };
      configs.push({ key, ...meta });
      seen.add(key);
    }
  }
  return configs;
}

function metricEntries(value: unknown): [string, unknown][] {
  if (!value || typeof value !== 'object') return [];
  return Object.entries(value as Record<string, unknown>).filter(([key, item]) => {
    if (item === null || item === undefined || item === '') return false;
    return ![
      'payload',
      'messages',
      'input_messages',
      'output_messages',
      'context_messages',
      'candidates',
      'input_candidates',
      'output_candidates',
      'raw_hits',
      'scoped_hits',
      'system_prompt',
      'user_prompt',
      'reply',
      'events',
      'selected',
      'dimension_scores',
      'symptom_labels',
      'scoring_adjustments',
      'fact_hits',
      'facts',
      'report_preview',
      'auto_review',
      'step',
      'diagnostics',
    ].includes(key);
  });
}

function label(key: string): string {
  const labels: Record<string, string> = {
    input_message_count: '输入消息',
    keyword_limit: '关键词上限',
    dictionary_groups: '词典组',
    top_k: 'topK',
    vector_key: '向量键',
    context_enabled: '上下文',
    context_window: '上下文窗口',
    context_items: '上下文片段',
    prompt_chars: 'Prompt 字符',
  };
  return labels[key] || key;
}

export function PsychDebugPage() {
  const [result, setResult] = useState<PsychAnalyzeResponse | null>(() => loadLastResult());
  const steps = useMemo(() => debugSteps(result), [result]);
  const configs = useMemo(() => stepConfigs(result, steps), [result, steps]);

  useEffect(() => {
    const onStorage = () => setResult(loadLastResult());
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  return (
    <div>
      <Header
        title="分析细节可视化"
        subtitle="查看全部分析步骤的输入输出、运行指标、候选数据、AI prompt 与筛选结果。"
      />

      <div className="mb-5 flex flex-wrap items-center justify-between gap-3 rounded-2xl border dk-border bg-white dark:bg-white/5 p-4">
        <div className="text-sm text-gray-600 dark:text-gray-300">
          {result ? (
            <>
              最近任务：<span className="font-mono">{result.task_id}</span>
              <span className="mx-2 text-gray-300">|</span>
              状态：{result.status}
            </>
          ) : (
            '还没有可查看的分析结果。请先完成一次心理风险辅助分析。'
          )}
        </div>
        <button
          type="button"
          onClick={() => setResult(loadLastResult())}
          className="inline-flex items-center gap-2 rounded-xl border dk-border px-3 py-2 text-sm font-bold text-gray-600 hover:border-[#07c160] hover:text-[#07c160] dark:text-gray-200"
        >
          <RefreshCw size={15} />
          刷新
        </button>
      </div>

      {!result ? (
        <div className="rounded-2xl border dk-border bg-white dark:bg-white/5 p-8 text-center">
          <ClipboardList className="mx-auto mb-3 text-gray-300" size={40} />
          <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white">暂无调试数据</h2>
          <p className="mt-2 text-sm text-gray-500">完成一次新的分析后，这里会列出全部步骤的细节。</p>
        </div>
      ) : (
        <div className="space-y-5">
          {configs.map((config) => (
            <DebugStepSection
              key={config.key}
              config={config}
              data={{
                step: (result.process_steps?.find(step => step.key === config.key) || {}) as unknown as Record<string, unknown>,
                metrics: result.process_steps?.find(step => step.key === config.key)?.metrics || {},
                ...(steps[config.key] || {}),
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function DebugStepSection({
  config,
  data,
}: {
  config: StepConfig;
  data: DebugStep;
}) {
  const stepMetrics = data.step && typeof data.step === 'object' ? {
    status: data.step.status,
    duration_ms: data.step.duration_ms,
  } : {};
  const metrics = metricEntries({ ...stepMetrics, ...(data.metrics || data) });
  const inputItems = data.messages || data.input_messages || data.candidates || data.input_candidates || data.scoped_hits || data.fact_hits || data.facts || [];
  const outputItems = data.output_messages || data.output_candidates || data.selected || data.raw_hits || data.context_messages || [];
  return (
    <section className="rounded-2xl border dk-border bg-white dark:bg-white/5 p-5 shadow-sm">
      <div className="mb-4 flex items-start gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[#e7f8f0] text-[#07c160] dark:bg-[#07c160]/20">
          {config.icon}
        </div>
        <div>
          <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white">{config.title}</h2>
          <p className="mt-1 text-sm text-gray-500">{config.subtitle}</p>
        </div>
      </div>

      {metrics.length > 0 && (
        <div className="mb-4 flex flex-wrap gap-2">
          {metrics.slice(0, 24).map(([key, value]) => (
            <span key={key} className="rounded-full bg-gray-100 px-2.5 py-1 text-xs font-semibold text-gray-600 dark:bg-white/10 dark:text-gray-200">
              {label(key)}: {formatValue(value)}
            </span>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <MessagePanel title="输入/候选数据" items={inputItems} />
        <MessagePanel title="输出/命中数据" items={outputItems} />
      </div>

      {Array.isArray(data.payload) && data.payload.length > 0 && (
        <div className="mt-4">
          <h3 className="mb-2 flex items-center gap-2 text-sm font-black text-[#1d1d1f] dark:text-white">
            <FileText size={15} />
            入模 Payload
          </h3>
          <PayloadList items={data.payload} />
        </div>
      )}

      {(data.system_prompt || data.user_prompt || data.reply) && (
        <div className="mt-4 grid grid-cols-1 xl:grid-cols-2 gap-4">
          <PromptBox title="System Prompt" text={String(data.system_prompt || '')} />
          <PromptBox title="User Prompt" text={String(data.user_prompt || '')} />
          <PromptBox title="AI Reply" text={String(data.reply || '')} />
        </div>
      )}

      <div className="mt-4 grid grid-cols-1 xl:grid-cols-2 gap-4">
        <JsonPanel title="读取诊断" value={data.diagnostics} />
        <JsonPanel title="运行事件" value={data.events} />
        <JsonPanel title="维度评分" value={data.dimension_scores} />
        <JsonPanel title="多标签结果" value={data.symptom_labels} />
        <JsonPanel title="评分修正" value={data.scoring_adjustments} />
        <JsonPanel title="自动复盘" value={data.auto_review} />
      </div>

      {data.report_preview && (
        <div className="mt-4">
          <PromptBox title="报告预览" text={String(data.report_preview)} />
        </div>
      )}
    </section>
  );
}

function MessagePanel({ title, items }: { title: string; items: DebugMessage[] }) {
  return (
    <div className="rounded-2xl border dk-border p-4">
      <h3 className="text-sm font-black text-[#1d1d1f] dark:text-white">{title}</h3>
      {items.length === 0 ? (
        <p className="mt-3 text-sm text-gray-500">无数据</p>
      ) : (
        <div className="mt-3 max-h-[420px] overflow-auto space-y-2 pr-1">
          {items.slice(0, 80).map((item, index) => (
            <MessageItem key={`${item.seq || item.id || index}-${index}`} item={item} index={index} />
          ))}
        </div>
      )}
    </div>
  );
}

function MessageItem({ item, index }: { item: DebugMessage; index: number }) {
  const text = item.content || item.candidate || item.text || item.fact || '';
  const displayLabels = item.label_names && item.label_names.length > 0 ? item.label_names : item.labels || [];
  return (
    <div className="rounded-xl bg-gray-50 p-3 text-sm dark:bg-black/20">
      <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-gray-500">
        <span>#{item.id ?? index}</span>
        {typeof item.score === 'number' && <span>score {item.score}</span>}
        {item.seq !== undefined && <span>seq {item.seq}</span>}
        {item.datetime && <span>{item.datetime}</span>}
        {item.sender && <span>{item.sender}</span>}
        <span>{item.is_mine ? '本人' : '上下文'}</span>
      </div>
      <p className="whitespace-pre-wrap break-words leading-6 text-gray-700 dark:text-gray-200">{text || '空内容'}</p>
      {Array.isArray(item.hit_words) && item.hit_words.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {item.hit_words.map((word) => (
            <span key={word} className="rounded-full bg-[#e7f8f0] px-2 py-0.5 text-[11px] font-bold text-[#067a3d]">
              {word}
            </span>
          ))}
        </div>
      )}
      {displayLabels.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {displayLabels.map((name) => (
            <span key={name} className="rounded-full bg-sky-50 px-2 py-0.5 text-[11px] font-bold text-sky-700">
              {name}
            </span>
          ))}
          {typeof item.risk_level === 'number' && (
            <span className="rounded-full bg-gray-200 px-2 py-0.5 text-[11px] font-bold text-gray-700">
              风险等级 {item.risk_level}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function JsonPanel({ title, value }: { title: string; value: unknown }) {
  if (value === null || value === undefined) return null;
  if (Array.isArray(value) && value.length === 0) return null;
  if (typeof value === 'object' && !Array.isArray(value) && Object.keys(value as Record<string, unknown>).length === 0) return null;
  return (
    <div>
      <h3 className="mb-2 text-sm font-black text-[#1d1d1f] dark:text-white">{title}</h3>
      <pre className="max-h-[420px] overflow-auto whitespace-pre-wrap break-words rounded-2xl bg-gray-50 p-4 text-xs leading-6 text-gray-700 dark:bg-black/30 dark:text-gray-100">
        {JSON.stringify(value, null, 2)}
      </pre>
    </div>
  );
}

function PayloadList({ items }: { items: DebugMessage[] }) {
  return (
    <div className="max-h-[460px] overflow-auto rounded-2xl border dk-border p-3 space-y-3">
      {items.slice(0, 60).map((item, index) => (
        <div key={`${item.id ?? index}`} className="rounded-xl bg-gray-50 p-3 dark:bg-black/20">
          <MessageItem item={item} index={index} />
          <ContextList title="前置上下文" items={item.context_before || []} />
          <ContextList title="后置上下文" items={item.context_after || []} />
        </div>
      ))}
    </div>
  );
}

function ContextList({ title, items }: { title: string; items: DebugMessage[] }) {
  if (!items.length) return null;
  return (
    <div className="mt-2 border-t dk-border pt-2">
      <div className="mb-1 text-xs font-bold text-gray-500">{title}</div>
      <div className="space-y-1.5">
        {items.map((item, index) => (
          <div key={`${item.datetime || ''}-${index}`} className="rounded-lg bg-white px-2 py-1.5 text-xs text-gray-600 dark:bg-white/5 dark:text-gray-300">
            <span className="mr-2 text-gray-400">{item.sender}</span>
            {item.text || item.content}
          </div>
        ))}
      </div>
    </div>
  );
}

function PromptBox({ title, text }: { title: string; text: string }) {
  if (!text) return null;
  return (
    <div>
      <h3 className="mb-2 text-sm font-black text-[#1d1d1f] dark:text-white">{title}</h3>
      <pre className="max-h-[520px] overflow-auto whitespace-pre-wrap break-words rounded-2xl bg-gray-950 p-4 text-xs leading-6 text-gray-100">
        {text}
      </pre>
    </div>
  );
}

function formatValue(value: unknown): string {
  if (Array.isArray(value)) return `${value.length}`;
  if (typeof value === 'object' && value !== null) return JSON.stringify(value);
  return String(value);
}
