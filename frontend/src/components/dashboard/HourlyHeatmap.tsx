/**
 * 24 小时活跃度分布（私聊 + 群聊，支持切换视图）
 */

import React, { useMemo, useState } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { GlobalStats } from '../../types';

type ViewMode = 'all' | 'private' | 'group';

interface HourlyHeatmapProps {
  data: GlobalStats | null;
}

const ModeButton: React.FC<{ active: boolean; onClick: () => void; children: React.ReactNode }> = ({ active, onClick, children }) => (
  <button
    onClick={onClick}
    className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
      active ? 'bg-[#1d1d1f] text-white' : 'text-gray-400 hover:text-gray-600'
    }`}
  >
    {children}
  </button>
);

const CustomTooltip = ({ active, payload, label, mode }: any) => {
  if (!active || !payload?.length) return null;
  const priv = payload.find((p: any) => p.dataKey === 'private')?.value ?? 0;
  const grp  = payload.find((p: any) => p.dataKey === 'group')?.value  ?? 0;
  return (
    <div style={{ background: '#fff', border: '1px solid #e5e5e5', borderRadius: 12, padding: '10px 14px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)', minWidth: 148 }}>
      <p style={{ fontWeight: 700, fontSize: 13, marginBottom: 6, color: '#1d1d1f' }}>时段：{label}</p>
      {mode !== 'group'   && <p style={{ fontSize: 12, color: '#07c160', margin: '2px 0' }}>私聊：{priv.toLocaleString()} 条</p>}
      {mode !== 'private' && grp > 0 && <p style={{ fontSize: 12, color: '#10aeff', margin: '2px 0' }}>群聊：{grp.toLocaleString()} 条</p>}
      {mode === 'all' && grp > 0 && (
        <p style={{ fontSize: 12, color: '#999', margin: '4px 0 0', borderTop: '1px solid #f0f0f0', paddingTop: 4 }}>
          合计：{(priv + grp).toLocaleString()} 条
        </p>
      )}
    </div>
  );
};

export const HourlyHeatmap: React.FC<HourlyHeatmapProps> = ({ data }) => {
  const [mode, setMode] = useState<ViewMode>('all');

  const hasGroups = useMemo(() =>
    (data?.group_hourly_heatmap ?? []).some((v) => v > 0),
  [data]);

  const chartData = useMemo(() => {
    if (!data?.hourly_heatmap) return [];
    return data.hourly_heatmap.map((v, h) => ({
      h: `${h.toString().padStart(2, '0')}:00`,
      private: v,
      group: data.group_hourly_heatmap?.[h] ?? 0,
    }));
  }, [data]);

  if (!chartData.length) {
    return (
      <div className="dk-card bg-white dk-border p-8 rounded-3xl border border-gray-100 h-96 flex items-center justify-center">
        <p className="text-gray-300 font-semibold">暂无数据</p>
      </div>
    );
  }

  // 当前模式下每个时段的展示值（用于 Y 轴 domain）
  const visibleData = chartData.map((d) => {
    if (mode === 'private') return d.private;
    if (mode === 'group')   return d.group;
    return d.private + d.group;
  });
  const yMax = Math.max(...visibleData, 1);

  const showPrivate = mode !== 'group';
  const showGroup   = mode !== 'private' && hasGroups;

  return (
    <div className="dk-card bg-white dk-border p-8 rounded-3xl border border-gray-100">
      <div className="flex items-center justify-between mb-6">
        <h3 className="dk-text text-xl font-black text-[#1d1d1f]">24 小时活跃度分布</h3>
        {hasGroups && (
          <div className="flex items-center gap-1 bg-gray-100 rounded-xl p-1">
            <ModeButton active={mode === 'all'}     onClick={() => setMode('all')}>全部</ModeButton>
            <ModeButton active={mode === 'private'} onClick={() => setMode('private')}>私聊</ModeButton>
            <ModeButton active={mode === 'group'}   onClick={() => setMode('group')}>群聊</ModeButton>
          </div>
        )}
      </div>
      <div className="h-[200px] sm:h-[300px]">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="hhPrivateGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#07c160" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#07c160" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="hhGroupGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10aeff" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#10aeff" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="h" tick={{ fontSize: 11, fill: '#999' }} tickLine={false} interval={2} />
          <YAxis
            tick={{ fontSize: 12, fill: '#999' }} tickLine={false} axisLine={false}
            tickFormatter={(v) => v >= 10000 ? `${(v / 1000).toFixed(0)}k` : v}
            domain={[0, yMax]}
          />
          <Tooltip content={<CustomTooltip mode={mode} />} />
          {showPrivate && (
            <Area type="monotone" dataKey="private"
              stroke="#07c160" strokeWidth={3} fill="url(#hhPrivateGrad)" />
          )}
          {showGroup && (
            <Area type="monotone" dataKey="group"
              stroke="#10aeff" strokeWidth={3} fill="url(#hhGroupGrad)" />
          )}
        </AreaChart>
      </ResponsiveContainer>
      </div>
    </div>
  );
};
