import { useEffect, useMemo, useState } from 'react';
import type React from 'react';
import {
  Activity,
  BarChart3,
  Gauge,
  PieChart as PieChartIcon,
  RefreshCw,
  ShieldCheck,
  Tags,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Header } from '../layout/Header';
import { LAST_PSYCH_RESULT_KEY, type PsychAnalyzeResponse, type PsychEvidence } from './types';

type ChartDatum = Record<string, string | number>;

const COLORS = ['#07c160', '#2563eb', '#f59e0b', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#64748b'];

const EVIDENCE_TYPE_LABELS: Record<string, string> = {
  depression_mood: '情绪低落/绝望',
  low_mood_hopelessness: '情绪低落/绝望',
  mood: '情绪低落',
  negative: '负面情绪',
  negative_emotion: '负面情绪',
  interest_loss_anhedonia: '兴趣下降',
  interest_loss: '兴趣下降',
  sleep_disturbance: '睡眠异常',
  sleep_problem: '睡眠异常',
  fatigue_low_energy: '精力下降/疲乏',
  fatigue: '精力下降/疲乏',
  appetite_weight_change: '食欲/体重变化',
  appetite: '食欲/体重变化',
  self_blame_worthlessness: '自责/无价值感',
  self_negation: '自我否定',
  self_negation_worthlessness: '自我否定',
  concentration_decision_difficulty: '注意力/决策困难',
  concentration: '注意力下降',
  psychomotor_change: '行为迟滞/激越',
  social_withdrawal_function_impairment: '社交退缩/功能受损',
  social_withdrawal: '社交退缩',
  function_impairment: '功能受损',
  self_harm_suicide: '自伤/轻生红线',
  crisis: '红线风险',
  redline: '红线风险',
  protective: '保护性信号',
  positive_emotion: '正面情绪',
  future_hope: '希望感/未来感',
  social_connection: '社会连接',
  function_maintained: '功能保持',
  humor_context: '幽默/语境缓冲',
  keyword: '关键词证据',
  semantic: '语义检索证据',
  llm_screened: '大模型筛选证据',
  text: '文本证据',
  evidence: '文本证据',
  low: '低强度证据',
  medium: '中强度证据',
  high: '高强度证据',
  crisis_level: '红线证据',
};

function humanizeEvidenceType(value: unknown): string {
  const raw = String(value || '').trim();
  if (!raw) return '文本证据';
  const key = raw.toLowerCase();
  if (EVIDENCE_TYPE_LABELS[key]) return EVIDENCE_TYPE_LABELS[key];
  if (/[\u4e00-\u9fff]/.test(raw)) return raw;
  return raw
    .replace(/^label_/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function loadLastResult(): PsychAnalyzeResponse | null {
  try {
    const raw = localStorage.getItem(LAST_PSYCH_RESULT_KEY);
    return raw ? JSON.parse(raw) as PsychAnalyzeResponse : null;
  } catch {
    return null;
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? value as Record<string, unknown> : {};
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function num(value: unknown, fallback = 0): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function shortName(value: unknown, max = 12): string {
  const text = String(value || '');
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

function percent(value: number): string {
  if (!Number.isFinite(value)) return '0%';
  return `${Math.round(value)}%`;
}

function datePart(value: string): string {
  return value ? value.slice(0, 10) : '未知日期';
}

function resultDebugSteps(result: PsychAnalyzeResponse | null): Record<string, unknown> {
  return asRecord(result?.report_json?.debug_steps);
}

function collectKeywordWords(result: PsychAnalyzeResponse | null): ChartDatum[] {
  const counts = new Map<string, number>();
  const steps = resultDebugSteps(result);
  const keywordStep = asRecord(steps.keyword_search);
  const candidates = [
    ...asArray(keywordStep.candidates),
    ...asArray(keywordStep.output_candidates),
    ...asArray(keywordStep.selected),
  ];
  for (const item of candidates) {
    const record = asRecord(item);
    for (const word of asArray(record.hit_words)) {
      const text = String(word || '').trim();
      if (!text) continue;
      counts.set(text, (counts.get(text) || 0) + 1);
    }
    for (const label of asArray(record.label_names)) {
      const text = String(label || '').trim();
      if (!text) continue;
      counts.set(text, (counts.get(text) || 0) + 1);
    }
  }

  if (!counts.size) {
    for (const evidence of result?.evidences || []) {
      const key = evidence.evidence_type || evidence.reason || '文本证据';
      counts.set(key, (counts.get(key) || 0) + 1);
    }
  }

  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 16)
    .map(([name, count]) => ({ name: shortName(name), fullName: name, count }));
}

function dimensionData(result: PsychAnalyzeResponse | null): ChartDatum[] {
  return asArray(result?.score.dimension_scores)
    .map((item) => {
      const record = asRecord(item);
      const max = Math.max(1, num(record.max_points, 1));
      const score = num(record.score);
      return {
        key: String(record.key || ''),
        name: shortName(record.label || record.key || '评分维度', 10),
        fullName: String(record.label || record.key || ''),
        score,
        max,
        rate: Math.round((score / max) * 100),
        keyword: num(record.keyword_message_count),
        semantic: num(record.semantic_hit_count),
      };
    })
    .filter((item) => Number(item.score) > 0)
    .slice(0, 10);
}

function labelData(result: PsychAnalyzeResponse | null): ChartDatum[] {
  return asArray(result?.score.symptom_labels)
    .map((item) => {
      const record = asRecord(item);
      return {
        key: String(record.key || ''),
        name: shortName(record.label || record.key || '标签', 10),
        fullName: String(record.label || record.key || ''),
        count: num(record.message_count),
        risk: num(record.risk_level),
        activeDays: num(record.active_days),
        protective: record.protective ? 1 : 0,
        modifier: record.modifier ? 1 : 0,
      };
    })
    .filter((item) => Number(item.count) > 0)
    .slice(0, 14);
}

function evidenceTypeData(evidences: PsychEvidence[]): ChartDatum[] {
  const counts = new Map<string, number>();
  const rawNames = new Map<string, Set<string>>();
  for (const item of evidences) {
    const key = item.evidence_type || item.severity || '文本证据';
    const label = humanizeEvidenceType(key);
    counts.set(label, (counts.get(label) || 0) + 1);
    const rawSet = rawNames.get(label) || new Set<string>();
    rawSet.add(String(key));
    rawNames.set(label, rawSet);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([name, count]) => ({
      name: shortName(name, 10),
      fullName: name,
      rawName: [...(rawNames.get(name) || [])].join(', '),
      count,
    }));
}

function evidenceTimeline(evidences: PsychEvidence[]): ChartDatum[] {
  const counts = new Map<string, number>();
  for (const item of evidences) {
    const day = datePart(item.datetime);
    counts.set(day, (counts.get(day) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .slice(-30)
    .map(([date, count]) => ({ date: date.slice(5), count }));
}

function protectiveData(result: PsychAnalyzeResponse | null): ChartDatum[] {
  const adjustments = asRecord(result?.score.scoring_adjustments);
  return asArray(adjustments.protective_factors)
    .map((item) => {
      const record = asRecord(item);
      return {
        name: shortName(record.label || record.key || '保护因子', 10),
        fullName: String(record.label || record.key || ''),
        delta: num(record.delta),
        messages: num(record.message_count),
      };
    })
    .filter((item) => Number(item.delta) !== 0 || Number(item.messages) > 0);
}

function scoreCards(result: PsychAnalyzeResponse | null) {
  const score = result?.score;
  const adjustments = asRecord(score?.scoring_adjustments);
  return [
    {
      icon: <Gauge size={18} />,
      label: '抑郁相关信号',
      value: String(score?.depression_signal_score ?? '-'),
      hint: `主表 ${adjustments.raw_dimension_score ?? '-'}，修正后 ${adjustments.final_score ?? '-'}`,
      tone: 'green',
    },
    {
      icon: <Activity size={18} />,
      label: '多标签等级',
      value: String(score?.risk_level ?? '-'),
      hint: score?.risk_level_label || '未生成',
      tone: 'blue',
    },
    {
      icon: <ShieldCheck size={18} />,
      label: '保护性修正',
      value: String(adjustments.protective_delta ?? 0),
      hint: adjustments.redline_blocks_protective_reduction ? '红线已阻止自动降分' : '未触发红线阻断',
      tone: 'amber',
    },
    {
      icon: <Tags size={18} />,
      label: '文本证据',
      value: String(result?.evidences.length ?? 0),
      hint: `置信度 ${score?.confidence ?? '-'}`,
      tone: 'violet',
    },
  ];
}

export function PsychVisualizationPage() {
  const [result, setResult] = useState<PsychAnalyzeResponse | null>(() => loadLastResult());
  const cards = useMemo(() => scoreCards(result), [result]);
  const dimensions = useMemo(() => dimensionData(result), [result]);
  const keywords = useMemo(() => collectKeywordWords(result), [result]);
  const labels = useMemo(() => labelData(result), [result]);
  const evidenceTypes = useMemo(() => evidenceTypeData(result?.evidences || []), [result]);
  const timeline = useMemo(() => evidenceTimeline(result?.evidences || []), [result]);
  const protective = useMemo(() => protectiveData(result), [result]);

  useEffect(() => {
    const onStorage = () => setResult(loadLastResult());
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  return (
    <div>
      <Header
        title="可视化分析"
        subtitle="将最近一次心理风险辅助筛查结果转成关键词、标签、评分和修正因子的图表视图。"
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
            '暂无可视化数据。请先完成一次心理风险辅助分析。'
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
        <section className="rounded-2xl border border-amber-100 bg-amber-50 px-4 py-4 text-sm text-amber-800 dark:border-amber-500/20 dark:bg-amber-500/10 dark:text-amber-100">
          完成一次心理分析后，这里会显示关键词统计、标签分布、评分维度和保护性修正图表。
        </section>
      ) : (
        <div className="space-y-5">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
            {cards.map((item) => (
              <StatCard key={item.label} {...item} />
            ))}
          </div>

          <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1.1fr,0.9fr]">
            <ChartPanel
              icon={<BarChart3 size={18} />}
              title="关键词命中排行"
              subtitle="来自关键词检索候选、标签命中和文本证据类型。"
              empty={!keywords.length}
            >
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={keywords} layout="vertical" margin={{ left: 8, right: 20, top: 8, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" allowDecimals={false} />
                  <YAxis type="category" dataKey="name" width={94} tickLine={false} axisLine={false} />
                  <Tooltip formatter={(value, name, props) => [value, `${props.payload.fullName || name}${props.payload.rawName ? ` (${props.payload.rawName})` : ''}`]} />
                  <Bar dataKey="count" name="命中次数" radius={[0, 6, 6, 0]}>
                    {keywords.map((_, index) => <Cell key={index} fill={COLORS[index % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartPanel>

            <ChartPanel
              icon={<PieChartIcon size={18} />}
              title="证据类型分布"
              subtitle="按证据类型或严重程度聚合。"
              empty={!evidenceTypes.length}
            >
              <ResponsiveContainer width="100%" height={320}>
                <PieChart>
                  <Pie data={evidenceTypes} dataKey="count" nameKey="name" innerRadius={58} outerRadius={108} paddingAngle={3}>
                    {evidenceTypes.map((_, index) => <Cell key={index} fill={COLORS[index % COLORS.length]} />)}
                  </Pie>
                  <Tooltip formatter={(value, name, props) => [value, props.payload.fullName || name]} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </ChartPanel>
          </div>

          <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
            <ChartPanel
              icon={<Gauge size={18} />}
              title="评分维度雷达"
              subtitle="按各维度得分占该维度满分比例显示。"
              empty={!dimensions.length}
            >
              <ResponsiveContainer width="100%" height={330}>
                <RadarChart data={dimensions}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="name" />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
                  <Radar name="得分占比" dataKey="rate" stroke="#07c160" fill="#07c160" fillOpacity={0.25} />
                  <Tooltip formatter={(value, name, props) => [`${value}%`, props.payload.fullName || name]} />
                </RadarChart>
              </ResponsiveContainer>
            </ChartPanel>

            <ChartPanel
              icon={<Tags size={18} />}
              title="多标签消息数"
              subtitle="普通抱怨、压力、情绪、认知、保护性等标签的命中消息数。"
              empty={!labels.length}
            >
              <ResponsiveContainer width="100%" height={330}>
                <BarChart data={labels} margin={{ left: 0, right: 12, top: 8, bottom: 50 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" angle={-35} textAnchor="end" height={64} interval={0} />
                  <YAxis allowDecimals={false} />
                  <Tooltip formatter={(value, name, props) => [value, props.payload.fullName || name]} />
                  <Bar dataKey="count" name="消息数" radius={[6, 6, 0, 0]}>
                    {labels.map((item, index) => (
                      <Cell
                        key={index}
                        fill={Number(item.protective) ? '#07c160' : Number(item.risk) >= 4 ? '#ef4444' : COLORS[(index + 1) % COLORS.length]}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartPanel>
          </div>

          <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1fr,0.85fr]">
            <ChartPanel
              icon={<Activity size={18} />}
              title="证据时间趋势"
              subtitle="按日期统计文本证据数量，最多显示最近 30 个有证据日期。"
              empty={!timeline.length}
            >
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={timeline} margin={{ left: 0, right: 18, top: 10, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="date" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Line type="monotone" dataKey="count" name="证据数" stroke="#2563eb" strokeWidth={3} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </ChartPanel>

            <ChartPanel
              icon={<ShieldCheck size={18} />}
              title="保护性修正"
              subtitle="正面情绪、未来感、社会连接、功能保持和语境缓冲的修正分。"
              empty={!protective.length}
            >
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={protective} layout="vertical" margin={{ left: 8, right: 18, top: 8, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" domain={[-20, 10]} />
                  <YAxis type="category" dataKey="name" width={92} tickLine={false} axisLine={false} />
                  <Tooltip formatter={(value, name, props) => [value, props.payload.fullName || name]} />
                  <Bar dataKey="delta" name="修正分" radius={[6, 6, 6, 6]}>
                    {protective.map((item, index) => (
                      <Cell key={index} fill={Number(item.delta) < 0 ? '#07c160' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartPanel>
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  hint,
  tone,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  hint: string;
  tone: string;
}) {
  const toneClass = {
    green: 'bg-[#e7f8f0] text-[#07c160]',
    blue: 'bg-blue-50 text-blue-600 dark:bg-blue-500/10 dark:text-blue-200',
    amber: 'bg-amber-50 text-amber-600 dark:bg-amber-500/10 dark:text-amber-200',
    violet: 'bg-violet-50 text-violet-600 dark:bg-violet-500/10 dark:text-violet-200',
  }[tone] || 'bg-gray-100 text-gray-600';
  return (
    <section className="rounded-2xl border dk-border bg-white p-4 shadow-sm dark:bg-white/5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-bold text-gray-500 dark:text-gray-300">{label}</div>
          <div className="mt-2 text-3xl font-black text-[#1d1d1f] dark:text-white">{value}</div>
          <div className="mt-1 text-xs text-gray-400">{hint}</div>
        </div>
        <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${toneClass}`}>
          {icon}
        </div>
      </div>
    </section>
  );
}

function ChartPanel({
  icon,
  title,
  subtitle,
  empty,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  empty: boolean;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border dk-border bg-white p-5 shadow-sm dark:bg-white/5">
      <div className="mb-4 flex items-start gap-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gray-100 text-gray-600 dark:bg-white/10 dark:text-gray-200">
          {icon}
        </div>
        <div>
          <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white">{title}</h2>
          <p className="mt-0.5 text-xs leading-5 text-gray-500">{subtitle}</p>
        </div>
      </div>
      {empty ? (
        <div className="flex h-[260px] items-center justify-center rounded-xl border border-dashed dk-border text-sm text-gray-400">
          暂无可视化数据
        </div>
      ) : (
        children
      )}
    </section>
  );
}
