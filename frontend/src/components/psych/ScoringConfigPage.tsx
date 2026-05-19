import { useEffect, useMemo, useState } from 'react';
import type React from 'react';
import {
  AlertCircle,
  BadgeCheck,
  BookOpenText,
  Clock3,
  Filter,
  Gauge,
  Layers3,
  RefreshCw,
  RotateCcw,
  Save,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Tags,
} from 'lucide-react';
import { Header } from '../layout/Header';

interface ScoringLevel {
  score: number;
  description: string;
  rule?: Record<string, unknown>;
}

interface ScoringDimension {
  key: string;
  label: string;
  description: string;
  max_points: number;
  enabled: boolean;
  redline: boolean;
  keywords: string[];
  exclude_keywords?: string[];
  strong_keywords?: string[];
  vector_queries: string[];
  subgroups?: Record<string, string[]>;
  levels: ScoringLevel[];
}

interface EvidenceStrengthLevel {
  key: string;
  label: string;
  coefficient: number;
  description: string;
  rule?: Record<string, unknown>;
}

interface TimeAdjustmentLevel {
  key: string;
  label: string;
  min_days: number;
  max_days: number;
  level_shift: number;
  coefficient: number;
  score_delta: number;
  description: string;
}

interface ProtectiveAdjustmentLevel {
  delta: number;
  description: string;
  rule?: Record<string, unknown>;
}

interface ProtectiveAdjustmentFactor {
  key: string;
  label: string;
  max_delta: number;
  label_keys: string[];
  description: string;
  levels: ProtectiveAdjustmentLevel[];
}

interface SymptomLabelRule {
  key: string;
  label: string;
  category: string;
  weight: string;
  weight_label: string;
  risk_level: number;
  enabled: boolean;
  protective?: boolean;
  modifier?: boolean;
  description: string;
  keywords: string[];
  exclude_keywords?: string[];
  dimension_keys?: string[];
  vector_queries?: string[];
}

interface ScoringConfig {
  version: number;
  max_score: number;
  thresholds: {
    medium: number;
    high: number;
    crisis: number;
  };
  evidence_strength?: {
    enabled: boolean;
    levels: EvidenceStrengthLevel[];
  };
  time_adjustment?: {
    enabled: boolean;
    levels: TimeAdjustmentLevel[];
    worsening_bonus?: {
      enabled: boolean;
      min_bonus: number;
      max_bonus: number;
      description?: string;
    };
    relief_reduction?: {
      enabled: boolean;
      score_delta: number;
      keywords: string[];
      description?: string;
    };
  };
  protective_adjustment?: {
    enabled: boolean;
    min_delta: number;
    max_delta: number;
    redline_blocks_reduction: boolean;
    redline_bonus?: {
      enabled: boolean;
      delta: number;
      description?: string;
    };
    factors: ProtectiveAdjustmentFactor[];
  };
  symptom_labels?: {
    enabled: boolean;
    labels: SymptomLabelRule[];
    risk_levels?: Array<{ level: number; label: string; description: string }>;
  };
  dimensions: ScoringDimension[];
  confidence?: Record<string, unknown>;
}

type SectionKey = 'overview' | 'dimensions' | 'labels' | 'adjustments';

const EMPTY_CONFIG: ScoringConfig = {
  version: 5,
  max_score: 100,
  thresholds: { medium: 35, high: 65, crisis: 85 },
  dimensions: [],
  evidence_strength: { enabled: true, levels: [] },
  time_adjustment: { enabled: true, levels: [] },
  protective_adjustment: { enabled: true, min_delta: -20, max_delta: 10, redline_blocks_reduction: true, factors: [] },
  symptom_labels: { enabled: true, labels: [] },
};

const SECTIONS: Array<{ key: SectionKey; label: string; desc: string; icon: React.ReactNode }> = [
  { key: 'overview', label: '总览', desc: '阈值与流程', icon: <Gauge size={16} /> },
  { key: 'dimensions', label: '评分维度', desc: '100 分主表', icon: <Layers3 size={16} /> },
  { key: 'labels', label: '多标签', desc: '表达类型与症状', icon: <Tags size={16} /> },
  { key: 'adjustments', label: '修正因子', desc: '证据/时间/保护', icon: <ShieldCheck size={16} /> },
];

async function readError(resp: Response): Promise<string> {
  const text = await resp.text();
  try {
    const data = JSON.parse(text) as { detail?: string; error?: string };
    return data.detail || data.error || text || `HTTP ${resp.status}`;
  } catch {
    return text || `HTTP ${resp.status}`;
  }
}

function splitLines(value: string): string[] {
  return value
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function joinLines(value?: string[]): string {
  return (value || []).join('\n');
}

function clampNumber(value: number, min: number, max: number): number {
  const parsed = Number.isFinite(value) ? value : min;
  return Math.max(min, Math.min(max, parsed));
}

function normalizeThresholds(config: ScoringConfig): ScoringConfig {
  const maxScore = Math.max(1, Number(config.max_score) || 100);
  const medium = clampNumber(Number(config.thresholds.medium) || 35, 1, maxScore);
  const high = clampNumber(Number(config.thresholds.high) || 65, medium, maxScore);
  const crisis = clampNumber(Number(config.thresholds.crisis) || 85, high, maxScore);
  return { ...config, max_score: maxScore, thresholds: { medium, high, crisis } };
}

function labelTone(item: SymptomLabelRule): string {
  if (item.protective) return 'border-emerald-100 bg-emerald-50 text-emerald-700 dark:border-emerald-500/20 dark:bg-emerald-500/10 dark:text-emerald-100';
  if (item.modifier) return 'border-blue-100 bg-blue-50 text-blue-700 dark:border-blue-500/20 dark:bg-blue-500/10 dark:text-blue-100';
  if (item.risk_level >= 5) return 'border-red-100 bg-red-50 text-red-700 dark:border-red-500/20 dark:bg-red-500/10 dark:text-red-100';
  if (item.risk_level >= 4) return 'border-orange-100 bg-orange-50 text-orange-700 dark:border-orange-500/20 dark:bg-orange-500/10 dark:text-orange-100';
  return 'border-gray-100 bg-gray-50 text-gray-600 dark:border-white/10 dark:bg-white/5 dark:text-gray-200';
}

export function ScoringConfigPage() {
  const [config, setConfig] = useState<ScoringConfig>(EMPTY_CONFIG);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [section, setSection] = useState<SectionKey>('overview');
  const [dimensionQuery, setDimensionQuery] = useState('');
  const [labelQuery, setLabelQuery] = useState('');

  const evidenceStrength = config.evidence_strength || { enabled: true, levels: [] };
  const timeAdjustment = config.time_adjustment || { enabled: true, levels: [] };
  const protectiveAdjustment = config.protective_adjustment || {
    enabled: true,
    min_delta: -20,
    max_delta: 10,
    redline_blocks_reduction: true,
    factors: [],
  };
  const symptomLabels = config.symptom_labels || { enabled: true, labels: [] };

  const totalEnabledMax = useMemo(
    () => config.dimensions.filter((item) => item.enabled).reduce((sum, item) => sum + Number(item.max_points || 0), 0),
    [config.dimensions],
  );

  const filteredDimensions = useMemo(() => {
    const query = dimensionQuery.trim().toLowerCase();
    if (!query) return config.dimensions;
    return config.dimensions.filter((item) => {
      const haystack = [
        item.label,
        item.key,
        item.description,
        ...(item.keywords || []),
        ...(item.vector_queries || []),
      ].join(' ').toLowerCase();
      return haystack.includes(query);
    });
  }, [config.dimensions, dimensionQuery]);

  const labelGroups = useMemo(() => {
    const query = labelQuery.trim().toLowerCase();
    const labels = symptomLabels.labels.filter((item) => {
      if (!query) return true;
      const haystack = [
        item.label,
        item.key,
        item.category,
        item.description,
        ...(item.keywords || []),
        ...(item.vector_queries || []),
      ].join(' ').toLowerCase();
      return haystack.includes(query);
    });
    return labels.reduce<Record<string, SymptomLabelRule[]>>((groups, item) => {
      const key = item.category || '未分类';
      groups[key] = groups[key] || [];
      groups[key].push(item);
      return groups;
    }, {});
  }, [symptomLabels.labels, labelQuery]);

  const loadConfig = async () => {
    setLoading(true);
    setError('');
    setMessage('');
    try {
      const resp = await fetch('/api/psych/scoring-config');
      if (!resp.ok) throw new Error(await readError(resp));
      setConfig(await resp.json() as ScoringConfig);
    } catch (err) {
      setError(err instanceof Error ? err.message : '评分标准读取失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadConfig();
  }, []);

  const updateConfig = (patch: Partial<ScoringConfig>) => setConfig((prev) => ({ ...prev, ...patch }));

  const updateEvidenceStrength = (patch: Partial<NonNullable<ScoringConfig['evidence_strength']>>) => {
    setConfig((prev) => ({
      ...prev,
      evidence_strength: { enabled: true, levels: [], ...(prev.evidence_strength || {}), ...patch },
    }));
  };

  const updateEvidenceLevel = (key: string, patch: Partial<EvidenceStrengthLevel>) => {
    updateEvidenceStrength({
      levels: evidenceStrength.levels.map((level) => (level.key === key ? { ...level, ...patch } : level)),
    });
  };

  const updateTimeAdjustment = (patch: Partial<NonNullable<ScoringConfig['time_adjustment']>>) => {
    setConfig((prev) => ({
      ...prev,
      time_adjustment: { enabled: true, levels: [], ...(prev.time_adjustment || {}), ...patch },
    }));
  };

  const updateTimeLevel = (key: string, patch: Partial<TimeAdjustmentLevel>) => {
    updateTimeAdjustment({
      levels: timeAdjustment.levels.map((level) => (level.key === key ? { ...level, ...patch } : level)),
    });
  };

  const updateProtectiveAdjustment = (patch: Partial<NonNullable<ScoringConfig['protective_adjustment']>>) => {
    setConfig((prev) => ({
      ...prev,
      protective_adjustment: {
        enabled: true,
        min_delta: -20,
        max_delta: 10,
        redline_blocks_reduction: true,
        factors: [],
        ...(prev.protective_adjustment || {}),
        ...patch,
      },
    }));
  };

  const updateProtectiveFactor = (key: string, patch: Partial<ProtectiveAdjustmentFactor>) => {
    updateProtectiveAdjustment({
      factors: protectiveAdjustment.factors.map((item) => (item.key === key ? { ...item, ...patch } : item)),
    });
  };

  const updateProtectiveLevel = (factorKey: string, index: number, patch: Partial<ProtectiveAdjustmentLevel>) => {
    updateProtectiveAdjustment({
      factors: protectiveAdjustment.factors.map((factor) => {
        if (factor.key !== factorKey) return factor;
        return {
          ...factor,
          levels: factor.levels.map((level, levelIndex) => (
            levelIndex === index ? { ...level, ...patch } : level
          )),
        };
      }),
    });
  };

  const updateDimension = (key: string, patch: Partial<ScoringDimension>) => {
    setConfig((prev) => ({
      ...prev,
      dimensions: prev.dimensions.map((item) => (item.key === key ? { ...item, ...patch } : item)),
    }));
  };

  const updateSymptomLabels = (patch: Partial<NonNullable<ScoringConfig['symptom_labels']>>) => {
    setConfig((prev) => ({
      ...prev,
      symptom_labels: { enabled: true, labels: [], ...(prev.symptom_labels || {}), ...patch },
    }));
  };

  const updateSymptomLabel = (key: string, patch: Partial<SymptomLabelRule>) => {
    updateSymptomLabels({
      labels: symptomLabels.labels.map((item) => (item.key === key ? { ...item, ...patch } : item)),
    });
  };

  const updateRiskLevel = (level: number, patch: Partial<{ label: string; description: string }>) => {
    updateSymptomLabels({
      risk_levels: (symptomLabels.risk_levels || []).map((item) => (
        item.level === level ? { ...item, ...patch } : item
      )),
    });
  };

  const updateLevel = (dimensionKey: string, index: number, patch: Partial<ScoringLevel>) => {
    setConfig((prev) => ({
      ...prev,
      dimensions: prev.dimensions.map((dimension) => {
        if (dimension.key !== dimensionKey) return dimension;
        return {
          ...dimension,
          levels: dimension.levels.map((level, levelIndex) => (
            levelIndex === index ? { ...level, ...patch } : level
          )),
        };
      }),
    }));
  };

  const saveConfig = async () => {
    setSaving(true);
    setError('');
    setMessage('');
    const payload = normalizeThresholds({
      ...config,
      dimensions: config.dimensions.map((dimension) => ({
        ...dimension,
        max_points: Math.max(0, Number(dimension.max_points) || 0),
        levels: dimension.levels.map((level) => ({
          ...level,
          score: Math.max(0, Number(level.score) || 0),
        })),
      })),
    });
    try {
      const resp = await fetch('/api/psych/scoring-config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) throw new Error(await readError(resp));
      setConfig(await resp.json() as ScoringConfig);
      setMessage('评分标准已保存，下一次心理分析会使用新标准。');
    } catch (err) {
      setError(err instanceof Error ? err.message : '评分标准保存失败');
    } finally {
      setSaving(false);
    }
  };

  const resetConfig = async () => {
    setSaving(true);
    setError('');
    setMessage('');
    try {
      const resp = await fetch('/api/psych/scoring-config/reset', { method: 'POST' });
      if (!resp.ok) throw new Error(await readError(resp));
      setConfig(await resp.json() as ScoringConfig);
      setMessage('已恢复默认评分标准。');
    } catch (err) {
      setError(err instanceof Error ? err.message : '恢复默认失败');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div>
      <Header
        title="评分标准"
        subtitle="配置 100 分主表、多标签分类、证据强度、时间修正和保护性修正因子。"
      />

      <div className="space-y-5">
        <section className="rounded-2xl border dk-border bg-white p-4 shadow-sm dark:bg-white/5">
          <div className="flex flex-col gap-4 2xl:flex-row 2xl:items-center 2xl:justify-between">
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <Metric icon={<Gauge size={18} />} label="最高分" value={config.max_score} hint={`启用维度 ${totalEnabledMax}`} />
              <Metric icon={<Layers3 size={18} />} label="评分维度" value={config.dimensions.length} hint={`${config.dimensions.filter((item) => item.enabled).length} 项启用`} />
              <Metric icon={<Tags size={18} />} label="标签规则" value={symptomLabels.labels.length} hint={`${symptomLabels.labels.filter((item) => item.enabled).length} 项启用`} />
              <Metric icon={<ShieldCheck size={18} />} label="保护修正" value={`${protectiveAdjustment.min_delta}~+${protectiveAdjustment.max_delta}`} hint={protectiveAdjustment.redline_blocks_reduction ? '红线阻断降分' : '允许保护性降分'} />
            </div>

            <div className="flex flex-wrap gap-2">
              <ActionButton onClick={saveConfig} disabled={saving} tone="primary" icon={saving ? <RefreshCw size={16} className="animate-spin" /> : <Save size={16} />}>
                保存
              </ActionButton>
              <ActionButton onClick={loadConfig} disabled={loading} icon={<RefreshCw size={16} className={loading ? 'animate-spin' : ''} />}>
                重新读取
              </ActionButton>
              <ActionButton onClick={resetConfig} disabled={saving} icon={<RotateCcw size={16} />}>
                恢复默认
              </ActionButton>
            </div>
          </div>

          {(error || message) && (
            <div className={`mt-4 rounded-xl border p-3 text-sm ${
              error
                ? 'border-red-200 bg-red-50 text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-100'
                : 'border-[#07c160]/20 bg-[#e7f8f0] text-[#067a3d] dark:bg-[#07c160]/10 dark:text-emerald-100'
            }`}
            >
              <div className="flex items-start gap-2">
                {error ? <AlertCircle size={16} className="mt-0.5 shrink-0" /> : <BadgeCheck size={16} className="mt-0.5 shrink-0" />}
                <span>{error || message}</span>
              </div>
            </div>
          )}
        </section>

        <nav className="grid grid-cols-2 gap-2 lg:grid-cols-4">
          {SECTIONS.map((item) => (
            <button
              key={item.key}
              type="button"
              onClick={() => setSection(item.key)}
              className={`flex items-center gap-3 rounded-2xl border px-4 py-3 text-left transition-all ${
                section === item.key
                  ? 'border-[#07c160]/30 bg-[#e7f8f0] text-[#067a3d] shadow-sm dark:bg-[#07c160]/15 dark:text-emerald-100'
                  : 'dk-border bg-white text-gray-500 hover:border-[#07c160]/30 hover:text-[#067a3d] dark:bg-white/5 dark:text-gray-300'
              }`}
            >
              <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white/70 dark:bg-white/10">
                {item.icon}
              </span>
              <span className="min-w-0">
                <span className="block text-sm font-black">{item.label}</span>
                <span className="block truncate text-xs opacity-70">{item.desc}</span>
              </span>
            </button>
          ))}
        </nav>

        {section === 'overview' && (
          <OverviewSection
            config={config}
            symptomLabels={symptomLabels}
            totalEnabledMax={totalEnabledMax}
            updateConfig={updateConfig}
            updateRiskLevel={updateRiskLevel}
          />
        )}

        {section === 'dimensions' && (
          <section className="space-y-4">
            <SectionHeader
              icon={<Layers3 size={18} />}
              title="评分维度"
              subtitle="这是 100 分主表。每个维度都同时用于关键词检索和向量语义搜索。"
              action={(
                <SearchBox
                  value={dimensionQuery}
                  onChange={setDimensionQuery}
                  placeholder="搜索维度、关键词或语义句"
                />
              )}
            />
            <div className="space-y-4">
              {filteredDimensions.map((dimension, index) => (
                <DimensionEditor
                  key={dimension.key}
                  dimension={dimension}
                  index={index}
                  updateDimension={updateDimension}
                  updateLevel={updateLevel}
                />
              ))}
              {!filteredDimensions.length && <EmptyState text="没有匹配的评分维度。" />}
            </div>
          </section>
        )}

        {section === 'labels' && (
          <section className="space-y-4">
            <SectionHeader
              icon={<Tags size={18} />}
              title="多标签分类"
              subtitle="用于区分事件型抱怨、压力表达、核心症状、保护性信号和语境缓冲。"
              action={(
                <div className="flex flex-wrap items-center gap-3">
                  <label className="flex items-center gap-2 text-sm font-bold text-gray-600 dark:text-gray-200">
                    启用
                    <input type="checkbox" checked={symptomLabels.enabled} onChange={(event) => updateSymptomLabels({ enabled: event.target.checked })} />
                  </label>
                  <SearchBox value={labelQuery} onChange={setLabelQuery} placeholder="搜索标签或关键词" />
                </div>
              )}
            />

            {(symptomLabels.risk_levels || []).length > 0 && (
              <SectionCard>
                <div className="mb-3 flex items-center gap-2 text-sm font-black text-gray-700 dark:text-gray-100">
                  <Gauge size={16} />
                  风险等级 0-5
                </div>
                <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
                  {(symptomLabels.risk_levels || []).map((item) => (
                    <div key={item.level} className="grid grid-cols-[44px,1fr] gap-2 rounded-xl border dk-border p-2">
                      <div className="flex items-center justify-center rounded-xl bg-gray-100 text-sm font-black text-gray-600 dark:bg-white/10 dark:text-gray-200">
                        {item.level}
                      </div>
                      <div className="space-y-2">
                        <input
                          type="text"
                          value={item.label}
                          onChange={(event) => updateRiskLevel(item.level, { label: event.target.value })}
                          className="dk-input w-full rounded-xl border dk-border bg-white px-3 py-2 text-sm dark:bg-white/5"
                        />
                        <input
                          type="text"
                          value={item.description}
                          onChange={(event) => updateRiskLevel(item.level, { description: event.target.value })}
                          className="dk-input w-full rounded-xl border dk-border bg-white px-3 py-2 text-xs dark:bg-white/5"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </SectionCard>
            )}

            {Object.entries(labelGroups).map(([category, labels]) => (
              <div key={category} className="space-y-3">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white">{category}</h2>
                  <span className="rounded-full bg-gray-100 px-3 py-1 text-xs font-black text-gray-500 dark:bg-white/10 dark:text-gray-200">
                    {labels.length} 项
                  </span>
                </div>
                <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                  {labels.map((item) => (
                    <SymptomLabelCard key={item.key} item={item} updateSymptomLabel={updateSymptomLabel} />
                  ))}
                </div>
              </div>
            ))}
          </section>
        )}

        {section === 'adjustments' && (
          <AdjustmentsSection
            evidenceStrength={evidenceStrength}
            timeAdjustment={timeAdjustment}
            protectiveAdjustment={protectiveAdjustment}
            updateEvidenceStrength={updateEvidenceStrength}
            updateEvidenceLevel={updateEvidenceLevel}
            updateTimeAdjustment={updateTimeAdjustment}
            updateTimeLevel={updateTimeLevel}
            updateProtectiveAdjustment={updateProtectiveAdjustment}
            updateProtectiveFactor={updateProtectiveFactor}
            updateProtectiveLevel={updateProtectiveLevel}
          />
        )}
      </div>
    </div>
  );
}

function OverviewSection({
  config,
  symptomLabels,
  totalEnabledMax,
  updateConfig,
  updateRiskLevel,
}: {
  config: ScoringConfig;
  symptomLabels: NonNullable<ScoringConfig['symptom_labels']>;
  totalEnabledMax: number;
  updateConfig: (patch: Partial<ScoringConfig>) => void;
  updateRiskLevel: (level: number, patch: Partial<{ label: string; description: string }>) => void;
}) {
  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-[0.95fr,1.05fr]">
      <SectionCard>
        <div className="mb-4 flex items-center gap-2">
          <Gauge size={18} className="text-[#07c160]" />
          <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white">总分与阈值</h2>
        </div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <NumberField label="最高分" min={1} max={300} value={config.max_score} onChange={(value) => updateConfig({ max_score: value })} />
          <ReadOnlyField label="启用维度满分" value={`${totalEnabledMax}`} />
          <NumberField label="中风险阈值" min={1} max={config.max_score} value={config.thresholds.medium} onChange={(value) => updateConfig({ thresholds: { ...config.thresholds, medium: value } })} />
          <NumberField label="高风险阈值" min={config.thresholds.medium} max={config.max_score} value={config.thresholds.high} onChange={(value) => updateConfig({ thresholds: { ...config.thresholds, high: value } })} />
          <NumberField label="红线/危机阈值" min={config.thresholds.high} max={config.max_score} value={config.thresholds.crisis} onChange={(value) => updateConfig({ thresholds: { ...config.thresholds, crisis: value } })} />
        </div>
        <div className="mt-4 rounded-xl bg-gray-50 p-4 text-sm leading-6 text-gray-600 dark:bg-white/5 dark:text-gray-200">
          分数先由 10 个症状维度组成主表，再叠加证据强度、时间持续、趋势变化、缓解和保护性修正。红线项不会被保护性表达自动抵消。
        </div>
      </SectionCard>

      <SectionCard>
        <div className="mb-4 flex items-center gap-2">
          <BookOpenText size={18} className="text-[#07c160]" />
          <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white">阅读顺序</h2>
        </div>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <FlowStep title="1. 评分维度" text="设置 100 分主表、关键词、向量语义句和分档描述。" />
          <FlowStep title="2. 多标签分类" text="区分事件抱怨、压力、情绪、认知、保护性信号。" />
          <FlowStep title="3. 修正因子" text="配置证据强度、持续时间、缓解、保护性降分和红线加分。" />
          <FlowStep title="4. 保存生效" text="保存后下一次心理分析会使用新的规则。" />
        </div>
      </SectionCard>

      <SectionCard className="xl:col-span-2">
        <div className="mb-4 flex items-center gap-2">
          <Tags size={18} className="text-[#07c160]" />
          <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white">风险等级速览</h2>
        </div>
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
          {(symptomLabels.risk_levels || []).map((item) => (
            <div key={item.level} className="rounded-2xl border dk-border p-4">
              <div className="mb-2 flex items-center gap-2">
                <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-gray-100 text-sm font-black text-gray-600 dark:bg-white/10 dark:text-gray-200">
                  {item.level}
                </span>
                <input
                  type="text"
                  value={item.label}
                  onChange={(event) => updateRiskLevel(item.level, { label: event.target.value })}
                  className="dk-input min-w-0 flex-1 rounded-xl border dk-border bg-white px-3 py-2 text-sm font-bold dark:bg-white/5"
                />
              </div>
              <textarea
                rows={2}
                value={item.description}
                onChange={(event) => updateRiskLevel(item.level, { description: event.target.value })}
                className="dk-input w-full resize-y rounded-xl border dk-border bg-white px-3 py-2 text-xs leading-5 dark:bg-white/5"
              />
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function DimensionEditor({
  dimension,
  index,
  updateDimension,
  updateLevel,
}: {
  dimension: ScoringDimension;
  index: number;
  updateDimension: (key: string, patch: Partial<ScoringDimension>) => void;
  updateLevel: (dimensionKey: string, index: number, patch: Partial<ScoringLevel>) => void;
}) {
  return (
    <details className="rounded-2xl border dk-border bg-white p-5 shadow-sm open:shadow-md dark:bg-white/5" open={index < 2}>
      <summary className="cursor-pointer list-none">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-black text-gray-500 dark:bg-white/10">
                {index + 1}
              </span>
              <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white">{dimension.label}</h2>
              {dimension.redline && (
                <span className="rounded-full bg-red-50 px-2 py-0.5 text-xs font-black text-red-600 dark:bg-red-500/10 dark:text-red-200">
                  红线项
                </span>
              )}
              {!dimension.enabled && (
                <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-black text-gray-500 dark:bg-white/10">
                  已停用
                </span>
              )}
            </div>
            <p className="mt-1 text-sm leading-6 text-gray-500">{dimension.description}</p>
            <p className="mt-1 font-mono text-[11px] text-gray-400">{dimension.key}</p>
          </div>
          <div className="flex shrink-0 items-center gap-4">
            <NumberField label="满分" min={0} max={100} value={dimension.max_points} onChange={(value) => updateDimension(dimension.key, { max_points: value })} compact />
            <label className="flex items-center gap-2 text-sm font-bold text-gray-600 dark:text-gray-200">
              启用
              <input type="checkbox" checked={dimension.enabled} onChange={(event) => updateDimension(dimension.key, { enabled: event.target.checked })} />
            </label>
          </div>
        </div>
      </summary>

      <div className="mt-5 grid grid-cols-1 gap-5 2xl:grid-cols-[1fr,0.9fr]">
        <div className="space-y-4">
          <SubTitle icon={<Filter size={15} />} title="检索入口" />
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <TextAreaField label="关键词检索" value={joinLines(dimension.keywords)} rows={6} onChange={(value) => updateDimension(dimension.key, { keywords: splitLines(value) })} />
            <TextAreaField label="向量搜索语义句" value={joinLines(dimension.vector_queries)} rows={6} onChange={(value) => updateDimension(dimension.key, { vector_queries: splitLines(value) })} />
          </div>
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <TextAreaField label="强信号词" value={joinLines(dimension.strong_keywords)} rows={3} onChange={(value) => updateDimension(dimension.key, { strong_keywords: splitLines(value) })} />
            <TextAreaField label="排除词" value={joinLines(dimension.exclude_keywords)} rows={3} onChange={(value) => updateDimension(dimension.key, { exclude_keywords: splitLines(value) })} />
          </div>
        </div>

        <div>
          <SubTitle icon={<SlidersHorizontal size={15} />} title="评分分档" />
          <div className="mt-3 space-y-2">
            {dimension.levels.map((level, levelIndex) => (
              <div key={`${dimension.key}-${levelIndex}`} className="grid grid-cols-[72px,1fr] gap-2 rounded-xl border dk-border p-2">
                <input
                  type="number"
                  min={0}
                  max={dimension.max_points}
                  value={level.score}
                  onChange={(event) => updateLevel(dimension.key, levelIndex, { score: Number(event.target.value) })}
                  className="dk-input rounded-xl border dk-border bg-white px-3 py-2 text-sm font-black dark:bg-white/5"
                />
                <input
                  type="text"
                  value={level.description}
                  onChange={(event) => updateLevel(dimension.key, levelIndex, { description: event.target.value })}
                  className="dk-input rounded-xl border dk-border bg-white px-3 py-2 text-sm dark:bg-white/5"
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </details>
  );
}

function SymptomLabelCard({
  item,
  updateSymptomLabel,
}: {
  item: SymptomLabelRule;
  updateSymptomLabel: (key: string, patch: Partial<SymptomLabelRule>) => void;
}) {
  return (
    <details className={`rounded-2xl border p-4 ${labelTone(item)}`} open={item.risk_level >= 4 || item.protective}>
      <summary className="cursor-pointer list-none">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-base font-black">{item.label}</span>
              <span className="rounded-full bg-white/70 px-2 py-0.5 text-[11px] font-black dark:bg-white/10">
                等级 {item.risk_level}
              </span>
              <span className="rounded-full bg-white/70 px-2 py-0.5 text-[11px] font-black dark:bg-white/10">
                {item.weight_label || item.weight}
              </span>
            </div>
            <p className="mt-1 text-xs leading-5 opacity-80">{item.description}</p>
            <p className="mt-1 font-mono text-[11px] opacity-50">{item.key}</p>
          </div>
          <label className="flex shrink-0 items-center gap-2 text-xs font-bold">
            启用
            <input type="checkbox" checked={item.enabled} onChange={(event) => updateSymptomLabel(item.key, { enabled: event.target.checked })} />
          </label>
        </div>
      </summary>

      <div className="mt-4 rounded-xl bg-white/70 p-3 dark:bg-black/10">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <TextField label="标签名" value={item.label} onChange={(value) => updateSymptomLabel(item.key, { label: value })} />
          <TextField label="权重" value={item.weight_label || item.weight} onChange={(value) => updateSymptomLabel(item.key, { weight_label: value })} />
          <NumberField label="风险等级" min={0} max={5} value={item.risk_level} onChange={(value) => updateSymptomLabel(item.key, { risk_level: value })} />
          <TextField label="分类" value={item.category} onChange={(value) => updateSymptomLabel(item.key, { category: value })} />
        </div>
        <TextAreaField label="说明" value={item.description} rows={2} onChange={(value) => updateSymptomLabel(item.key, { description: value })} />
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
          <TextAreaField label="标签关键词" value={joinLines(item.keywords)} rows={4} onChange={(value) => updateSymptomLabel(item.key, { keywords: splitLines(value) })} />
          <TextAreaField label="标签向量语义句" value={joinLines(item.vector_queries)} rows={4} onChange={(value) => updateSymptomLabel(item.key, { vector_queries: splitLines(value) })} />
        </div>
      </div>
    </details>
  );
}

function AdjustmentsSection({
  evidenceStrength,
  timeAdjustment,
  protectiveAdjustment,
  updateEvidenceStrength,
  updateEvidenceLevel,
  updateTimeAdjustment,
  updateTimeLevel,
  updateProtectiveAdjustment,
  updateProtectiveFactor,
  updateProtectiveLevel,
}: {
  evidenceStrength: NonNullable<ScoringConfig['evidence_strength']>;
  timeAdjustment: NonNullable<ScoringConfig['time_adjustment']>;
  protectiveAdjustment: NonNullable<ScoringConfig['protective_adjustment']>;
  updateEvidenceStrength: (patch: Partial<NonNullable<ScoringConfig['evidence_strength']>>) => void;
  updateEvidenceLevel: (key: string, patch: Partial<EvidenceStrengthLevel>) => void;
  updateTimeAdjustment: (patch: Partial<NonNullable<ScoringConfig['time_adjustment']>>) => void;
  updateTimeLevel: (key: string, patch: Partial<TimeAdjustmentLevel>) => void;
  updateProtectiveAdjustment: (patch: Partial<NonNullable<ScoringConfig['protective_adjustment']>>) => void;
  updateProtectiveFactor: (key: string, patch: Partial<ProtectiveAdjustmentFactor>) => void;
  updateProtectiveLevel: (factorKey: string, index: number, patch: Partial<ProtectiveAdjustmentLevel>) => void;
}) {
  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
      <SectionCard>
        <ToggleTitle
          icon={<ShieldCheck size={18} />}
          title="保护性修正因子"
          checked={protectiveAdjustment.enabled}
          onChange={(enabled) => updateProtectiveAdjustment({ enabled })}
        />
        <p className="mt-2 text-sm leading-6 text-gray-500">
          在 100 分主表之后独立修正。正面情绪会按消息数、活跃天数和命中词数分档；明确红线信号不会被正面词、幽默或“我没事”自动抵消。
        </p>
        <div className="mt-4 grid grid-cols-2 gap-3">
          <NumberField label="最低修正" min={-50} max={0} value={protectiveAdjustment.min_delta} onChange={(value) => updateProtectiveAdjustment({ min_delta: value })} />
          <NumberField label="最高修正" min={0} max={50} value={protectiveAdjustment.max_delta} onChange={(value) => updateProtectiveAdjustment({ max_delta: value })} />
        </div>
        <label className="mt-4 flex items-center gap-2 text-sm font-bold text-gray-600 dark:text-gray-200">
          <input
            type="checkbox"
            checked={protectiveAdjustment.redline_blocks_reduction}
            onChange={(event) => updateProtectiveAdjustment({ redline_blocks_reduction: event.target.checked })}
          />
          红线项阻止保护性降分
        </label>
        <div className="mt-4 rounded-xl border dk-border p-3">
          <div className="grid grid-cols-[1fr,100px] gap-3">
            <div>
              <div className="text-sm font-black text-gray-700 dark:text-gray-100">红线人工复核加分</div>
              <p className="mt-1 text-xs leading-5 text-gray-500">用于确保红线表达不会被语境缓冲误降级。</p>
            </div>
            <NumberField
              label="加分"
              min={0}
              max={50}
              value={protectiveAdjustment.redline_bonus?.delta || 0}
              onChange={(value) => updateProtectiveAdjustment({
                redline_bonus: {
                  enabled: protectiveAdjustment.redline_bonus?.enabled ?? true,
                  delta: value,
                  description: protectiveAdjustment.redline_bonus?.description || '',
                },
              })}
            />
          </div>
          <label className="mt-2 flex items-center gap-2 text-xs font-bold text-gray-500">
            <input
              type="checkbox"
              checked={protectiveAdjustment.redline_bonus?.enabled ?? true}
              onChange={(event) => updateProtectiveAdjustment({
                redline_bonus: {
                  enabled: event.target.checked,
                  delta: protectiveAdjustment.redline_bonus?.delta || 10,
                  description: protectiveAdjustment.redline_bonus?.description || '',
                },
              })}
            />
            启用红线加分
          </label>
        </div>
        <div className="mt-4 space-y-3">
          {protectiveAdjustment.factors.map((factor) => (
            <details key={factor.key} className="rounded-xl border dk-border p-3" open={factor.key === 'positive_emotion'}>
              <summary className="cursor-pointer list-none">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="font-black text-gray-700 dark:text-gray-100">{factor.label}</div>
                    <div className="text-xs text-gray-400">{factor.max_delta} 分 · {factor.key}</div>
                  </div>
                  <span className="rounded-full bg-gray-100 px-2 py-1 text-xs font-black text-gray-500 dark:bg-white/10">
                    {factor.levels.length} 档
                  </span>
                </div>
              </summary>
              <div className="mt-3 space-y-3">
                <div className="grid grid-cols-[1fr,100px] gap-3">
                  <TextField label="名称" value={factor.label} onChange={(value) => updateProtectiveFactor(factor.key, { label: value })} />
                  <NumberField label="最大修正" min={-20} max={0} value={factor.max_delta} onChange={(value) => updateProtectiveFactor(factor.key, { max_delta: value })} />
                </div>
                <TextAreaField label="说明" value={factor.description} rows={2} onChange={(value) => updateProtectiveFactor(factor.key, { description: value })} />
                <TextAreaField label="关联标签 key" value={joinLines(factor.label_keys)} rows={2} onChange={(value) => updateProtectiveFactor(factor.key, { label_keys: splitLines(value) })} />
                {factor.key === 'positive_emotion' && (
                  <div className="rounded-xl bg-emerald-50 px-3 py-2 text-xs leading-5 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-100">
                    正面情绪修正量现在会参考命中词数：单次自然积极体验通常 -1，短期明确正面体验 -2，多次出现 -3，跨多天稳定自然出现可到 -5。
                  </div>
                )}
                {factor.levels.map((level, index) => (
                  <div key={`${factor.key}-${index}`} className="grid grid-cols-[76px,1fr] gap-2">
                    <input
                      type="number"
                      min={-20}
                      max={10}
                      value={level.delta}
                      onChange={(event) => updateProtectiveLevel(factor.key, index, { delta: Number(event.target.value) })}
                      className="dk-input rounded-xl border dk-border bg-white px-3 py-2 text-sm dark:bg-white/5"
                    />
                    <input
                      type="text"
                      value={level.description}
                      onChange={(event) => updateProtectiveLevel(factor.key, index, { description: event.target.value })}
                      className="dk-input rounded-xl border dk-border bg-white px-3 py-2 text-sm dark:bg-white/5"
                    />
                  </div>
                ))}
              </div>
            </details>
          ))}
        </div>
      </SectionCard>

      <div className="space-y-5">
        <SectionCard>
          <ToggleTitle
            icon={<BadgeCheck size={18} />}
            title="证据强度系数"
            checked={evidenceStrength.enabled}
            onChange={(enabled) => updateEvidenceStrength({ enabled })}
          />
          <div className="mt-4 grid grid-cols-1 gap-3">
            {evidenceStrength.levels.map((level) => (
              <div key={level.key} className="rounded-xl border dk-border p-3">
                <div className="grid grid-cols-[1fr,96px] gap-3">
                  <TextField label="名称" value={level.label} onChange={(value) => updateEvidenceLevel(level.key, { label: value })} />
                  <NumberField label="系数" min={0} max={5} step={0.1} value={level.coefficient} onChange={(value) => updateEvidenceLevel(level.key, { coefficient: value })} />
                </div>
                <TextAreaField label="判断标准" value={level.description} rows={2} onChange={(value) => updateEvidenceLevel(level.key, { description: value })} />
              </div>
            ))}
          </div>
        </SectionCard>

        <SectionCard>
          <ToggleTitle
            icon={<Clock3 size={18} />}
            title="时间维度修正"
            checked={timeAdjustment.enabled}
            onChange={(enabled) => updateTimeAdjustment({ enabled })}
          />
          <div className="mt-4 space-y-3">
            {timeAdjustment.levels.map((level) => (
              <div key={level.key} className="rounded-xl border dk-border p-3">
                <TextField label="持续时间" value={level.label} onChange={(value) => updateTimeLevel(level.key, { label: value })} />
                <div className="mt-2 grid grid-cols-2 gap-2">
                  <NumberField label="最少天数" min={0} max={9999} value={level.min_days} onChange={(value) => updateTimeLevel(level.key, { min_days: value })} />
                  <NumberField label="最多天数" min={0} max={9999} value={level.max_days} onChange={(value) => updateTimeLevel(level.key, { max_days: value })} />
                  <NumberField label="级别位移" min={-5} max={5} value={level.level_shift} onChange={(value) => updateTimeLevel(level.key, { level_shift: value })} />
                  <NumberField label="时间系数" min={0} max={5} step={0.05} value={level.coefficient} onChange={(value) => updateTimeLevel(level.key, { coefficient: value })} />
                </div>
                <TextAreaField label="说明" value={level.description} rows={2} onChange={(value) => updateTimeLevel(level.key, { description: value })} />
              </div>
            ))}
          </div>

          <div className="mt-4 rounded-xl border dk-border p-3">
            <div className="text-sm font-black text-gray-700 dark:text-gray-100">越来越重</div>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <NumberField
                label="最低加分"
                min={0}
                max={50}
                value={timeAdjustment.worsening_bonus?.min_bonus || 0}
                onChange={(value) => updateTimeAdjustment({ worsening_bonus: { enabled: true, max_bonus: timeAdjustment.worsening_bonus?.max_bonus || 10, min_bonus: value, description: timeAdjustment.worsening_bonus?.description || '' } })}
              />
              <NumberField
                label="最高加分"
                min={0}
                max={50}
                value={timeAdjustment.worsening_bonus?.max_bonus || 0}
                onChange={(value) => updateTimeAdjustment({ worsening_bonus: { enabled: true, min_bonus: timeAdjustment.worsening_bonus?.min_bonus || 5, max_bonus: value, description: timeAdjustment.worsening_bonus?.description || '' } })}
              />
            </div>
          </div>

          <div className="mt-4 rounded-xl border dk-border p-3">
            <div className="text-sm font-black text-gray-700 dark:text-gray-100">短期缓解</div>
            <NumberField
              label="修正分"
              min={-50}
              max={50}
              value={timeAdjustment.relief_reduction?.score_delta || 0}
              onChange={(value) => updateTimeAdjustment({ relief_reduction: { enabled: true, score_delta: value, keywords: timeAdjustment.relief_reduction?.keywords || [] } })}
            />
            <TextAreaField
              label="缓解关键词"
              value={joinLines(timeAdjustment.relief_reduction?.keywords)}
              rows={3}
              onChange={(value) => updateTimeAdjustment({ relief_reduction: { enabled: true, score_delta: timeAdjustment.relief_reduction?.score_delta || -5, keywords: splitLines(value) } })}
            />
          </div>
        </SectionCard>
      </div>
    </div>
  );
}

function Metric({ icon, label, value, hint }: { icon: React.ReactNode; label: string; value: React.ReactNode; hint: string }) {
  return (
    <div className="rounded-2xl border dk-border bg-gray-50 p-3 dark:bg-white/5">
      <div className="mb-2 flex items-center gap-2 text-xs font-black text-gray-500 dark:text-gray-300">
        {icon}
        {label}
      </div>
      <div className="text-2xl font-black text-[#1d1d1f] dark:text-white">{value}</div>
      <div className="mt-1 text-xs text-gray-400">{hint}</div>
    </div>
  );
}

function SectionHeader({ icon, title, subtitle, action }: { icon: React.ReactNode; title: string; subtitle: string; action?: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-3 rounded-2xl border dk-border bg-white p-4 shadow-sm dark:bg-white/5 lg:flex-row lg:items-center lg:justify-between">
      <div className="flex items-start gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[#e7f8f0] text-[#07c160] dark:bg-[#07c160]/20">
          {icon}
        </div>
        <div>
          <h2 className="text-xl font-black text-[#1d1d1f] dark:text-white">{title}</h2>
          <p className="mt-1 text-sm leading-6 text-gray-500">{subtitle}</p>
        </div>
      </div>
      {action}
    </div>
  );
}

function SectionCard({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <section className={`rounded-2xl border dk-border bg-white p-5 shadow-sm dark:bg-white/5 ${className}`}>
      {children}
    </section>
  );
}

function FlowStep({ title, text }: { title: string; text: string }) {
  return (
    <div className="rounded-2xl border dk-border p-4">
      <div className="font-black text-gray-700 dark:text-gray-100">{title}</div>
      <p className="mt-1 text-sm leading-6 text-gray-500">{text}</p>
    </div>
  );
}

function ToggleTitle({ icon, title, checked, onChange }: { icon: React.ReactNode; title: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="flex items-center gap-2">
        <span className="text-[#07c160]">{icon}</span>
        <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white">{title}</h2>
      </div>
      <label className="flex items-center gap-2 text-sm font-bold text-gray-600 dark:text-gray-200">
        启用
        <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      </label>
    </div>
  );
}

function SubTitle({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="flex items-center gap-2 text-sm font-black text-gray-600 dark:text-gray-200">
      {icon}
      {title}
    </div>
  );
}

function SearchBox({ value, onChange, placeholder }: { value: string; onChange: (value: string) => void; placeholder: string }) {
  return (
    <label className="relative block w-full lg:w-80">
      <Search size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
      <input
        type="text"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        className="dk-input w-full rounded-xl border dk-border bg-white py-2 pl-9 pr-3 text-sm dark:bg-white/5"
      />
    </label>
  );
}

function ActionButton({ children, icon, tone, disabled, onClick }: { children: React.ReactNode; icon: React.ReactNode; tone?: 'primary'; disabled?: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-black transition disabled:opacity-60 ${
        tone === 'primary'
          ? 'bg-[#07c160] text-white hover:bg-[#05a955]'
          : 'border dk-border text-gray-600 hover:border-[#07c160] hover:text-[#07c160] dark:text-gray-200'
      }`}
    >
      {icon}
      {children}
    </button>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="rounded-2xl border border-dashed dk-border bg-white p-8 text-center text-sm text-gray-400 dark:bg-white/5">
      {text}
    </div>
  );
}

function ReadOnlyField({ label, value }: { label: string; value: string }) {
  return (
    <label className="block">
      <span className="text-sm font-bold text-gray-600 dark:text-gray-200">{label}</span>
      <div className="mt-2 rounded-xl border dk-border bg-gray-50 px-3 py-2 text-sm font-black text-gray-600 dark:bg-white/5 dark:text-gray-200">
        {value}
      </div>
    </label>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  compact = false,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  compact?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <label className={compact ? 'block w-24' : 'block'}>
      <span className="text-sm font-bold text-gray-600 dark:text-gray-200">{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="dk-input mt-2 w-full rounded-xl border dk-border bg-white px-3 py-2 text-sm dark:bg-white/5"
      />
    </label>
  );
}

function TextField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="block">
      <span className="text-sm font-bold text-gray-600 dark:text-gray-200">{label}</span>
      <input
        type="text"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="dk-input mt-2 w-full rounded-xl border dk-border bg-white px-3 py-2 text-sm dark:bg-white/5"
      />
    </label>
  );
}

function TextAreaField({
  label,
  value,
  rows,
  onChange,
}: {
  label: string;
  value: string;
  rows: number;
  onChange: (value: string) => void;
}) {
  return (
    <label className="block">
      <span className="text-sm font-black text-gray-600 dark:text-gray-200">{label}</span>
      <textarea
        rows={rows}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="dk-input mt-2 w-full resize-y rounded-xl border dk-border bg-white px-3 py-2 text-sm leading-6 dark:bg-white/5"
      />
    </label>
  );
}
