/**
 * 情感分析图表
 */

import React, { useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { X, Loader2 } from 'lucide-react';
import type { SentimentResult, ChatMessage } from '../../types';
import { contactsApi } from '../../services/api';

interface Props {
  data: SentimentResult;
  username: string;
  contactName: string;
  includeMine?: boolean;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const score = payload[0].value as number;
  const count = payload[0].payload.count as number;
  const label2 = score >= 0.6 ? '😊 积极' : score <= 0.4 ? '😔 消极' : '😐 中性';
  const color = score >= 0.6 ? '#07c160' : score <= 0.4 ? '#f56c6c' : '#909399';
  return (
    <div className="bg-white border border-gray-100 rounded-2xl shadow-lg px-4 py-3 text-sm">
      <p className="font-black text-[#1d1d1f] mb-1">{label}</p>
      <p style={{ color }} className="font-bold">{label2} · {Math.round(score * 100)}分</p>
      <p className="text-gray-400 text-xs mt-0.5">参与统计 {count} 条消息 · 点击查看</p>
    </div>
  );
};

interface MonthPanelProps {
  username: string;
  month: string;
  contactName: string;
  includeMine: boolean;
  onClose: () => void;
}

const MonthMessagesPanel: React.FC<MonthPanelProps> = ({ username, month, contactName, includeMine, onClose }) => {
  const [messages, setMessages] = React.useState<ChatMessage[]>([]);
  const [loading, setLoading] = React.useState(true);
  const bottomRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    setLoading(true);
    contactsApi.getMonthMessages(username, month, includeMine)
      .then(data => setMessages(data ?? []))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [username, month, includeMine]);

  React.useEffect(() => {
    if (!loading) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [loading]);

  const [y, m] = month.split('-');
  const title = `${y}年${parseInt(m)}月`;

  return (
    <div
      className="fixed inset-0 z-[60] flex items-end sm:items-center justify-center sm:p-8 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-t-[32px] sm:rounded-[32px] w-full sm:max-w-lg flex flex-col max-h-[85vh] shadow-2xl animate-in slide-in-from-bottom sm:zoom-in duration-300"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 pt-5 pb-3 border-b border-gray-100 flex-shrink-0">
          <div>
            <div className="font-black text-[#1d1d1f] text-base">{title}</div>
            <div className="text-xs text-gray-400 mt-0.5">
              与 {contactName} 的{includeMine ? '双方' : '对方'}文本消息 · {messages.length} 条
            </div>
          </div>
          <button onClick={onClose} className="text-gray-300 hover:text-gray-600 transition-colors">
            <X size={22} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-2">
          {loading ? (
            <div className="flex items-center justify-center h-40">
              <Loader2 size={28} className="text-[#07c160] animate-spin" />
            </div>
          ) : messages.length === 0 ? (
            <div className="text-center text-gray-300 py-12 text-sm">暂无文字记录</div>
          ) : (
            messages.map((msg, i) => (
              <div key={i} className={`flex items-end gap-2 ${msg.is_mine ? 'flex-row-reverse' : 'flex-row'}`}>
                <div className={`w-6 h-6 rounded-full flex-shrink-0 flex items-center justify-center text-white text-[9px] font-black
                  ${msg.is_mine ? 'bg-[#07c160]' : 'bg-[#576b95]'}`}>
                  {msg.is_mine ? '我' : contactName.charAt(0)}
                </div>
                <div className={`flex flex-col gap-0.5 max-w-[72%] ${msg.is_mine ? 'items-end' : 'items-start'}`}>
                  <div className={`px-3 py-2 rounded-2xl text-sm leading-relaxed break-words whitespace-pre-wrap
                    ${msg.is_mine
                      ? 'bg-[#07c160] text-white rounded-br-sm'
                      : 'bg-[#f0f0f0] text-[#1d1d1f] rounded-bl-sm'
                    }`}
                  >
                    {msg.content}
                  </div>
                  <span className="text-[10px] text-gray-300 px-1">{msg.time}</span>
                </div>
              </div>
            ))
          )}
          <div ref={bottomRef} />
        </div>
      </div>
    </div>
  );
};

export const SentimentChart: React.FC<Props> = ({ data, username, contactName, includeMine = false }) => {
  const { monthly, overall, positive, negative, neutral } = data;
  const total = positive + negative + neutral;
  const [selectedMonth, setSelectedMonth] = useState<string | null>(null);

  const overallLabel = overall >= 0.6 ? '整体积极' : overall <= 0.4 ? '整体消极' : '整体中性';
  const overallColor = overall >= 0.6 ? '#07c160' : overall <= 0.4 ? '#f56c6c' : '#909399';
  const overallEmoji = overall >= 0.6 ? '😊' : overall <= 0.4 ? '😔' : '😐';

  const tickFormatter = (month: string) => {
    if (month.endsWith('-01')) return month.slice(0, 4);
    if (monthly.indexOf(monthly.find(m => m.month === month)!) === 0) return month.slice(0, 7);
    return month.slice(5);
  };

  const handleDotClick = (payload: any) => {
    if (payload?.activePayload?.[0]?.payload?.month) {
      setSelectedMonth(payload.activePayload[0].payload.month);
    }
  };

  return (
    <div>
      <p className="text-xs text-gray-400 mb-6">
        基于关键词对文本消息逐条情感打分，按月聚合均值；0.5 为中性基线，越高越积极
      </p>

      {/* 整体指标卡片 */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-8">
        <div className="bg-white border border-gray-100 rounded-2xl p-4 text-center">
          <p className="text-2xl mb-1">{overallEmoji}</p>
          <p className="text-lg font-black" style={{ color: overallColor }}>
            {Math.round(overall * 100)}
          </p>
          <p className="text-xs text-gray-400 mt-0.5">{overallLabel}</p>
        </div>
        <div className="bg-[#f0fdf4] rounded-2xl p-4 text-center">
          <p className="text-2xl mb-1">😊</p>
          <p className="text-lg font-black text-[#07c160]">
            {total > 0 ? Math.round(positive / total * 100) : 0}%
          </p>
          <p className="text-xs text-gray-400 mt-0.5">积极消息</p>
        </div>
        <div className="bg-[#fff7f7] rounded-2xl p-4 text-center">
          <p className="text-2xl mb-1">😔</p>
          <p className="text-lg font-black text-[#f56c6c]">
            {total > 0 ? Math.round(negative / total * 100) : 0}%
          </p>
          <p className="text-xs text-gray-400 mt-0.5">消极消息</p>
        </div>
        <div className="bg-gray-50 rounded-2xl p-4 text-center">
          <p className="text-2xl mb-1">😐</p>
          <p className="text-lg font-black text-gray-500">
            {total > 0 ? Math.round(neutral / total * 100) : 0}%
          </p>
          <p className="text-xs text-gray-400 mt-0.5">中性消息</p>
        </div>
      </div>

      {/* 月度折线图 */}
      {monthly.length > 1 ? (
        <div>
          <p className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3">
            情感波动曲线 · 月度趋势
          </p>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={monthly} margin={{ top: 8, right: 8, bottom: 0, left: -20 }} onClick={handleDotClick} style={{ cursor: 'pointer' }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="month"
                tickFormatter={tickFormatter}
                tick={{ fontSize: 11, fill: '#9ca3af' }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                domain={[0, 1]}
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                tickFormatter={(v) => `${Math.round(v * 100)}`}
                tick={{ fontSize: 11, fill: '#9ca3af' }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0.5} stroke="#e5e7eb" strokeDasharray="4 4" />
              <Line
                type="monotone"
                dataKey="score"
                stroke="#07c160"
                strokeWidth={2.5}
                dot={(props: any) => {
                  const { cx, cy, payload } = props;
                  const color = payload.score >= 0.6 ? '#07c160' : payload.score <= 0.4 ? '#f56c6c' : '#d1d5db';
                  return <circle key={`dot-${payload.month}`} cx={cx} cy={cy} r={3} fill={color} stroke="white" strokeWidth={1.5} />;
                }}
                activeDot={{ r: 6, stroke: '#07c160', strokeWidth: 2, fill: 'white' }}
              />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-4 mt-3 justify-center text-xs text-gray-400">
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[#07c160] inline-block" />积极（&gt;60）</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-gray-200 inline-block" />中性（40–60）</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[#f56c6c] inline-block" />消极（&lt;40）</span>
          </div>
        </div>
      ) : (
        <div className="text-center text-gray-300 py-8 text-sm">
          消息数量不足，无法生成月度趋势
        </div>
      )}

      {selectedMonth && (
        <MonthMessagesPanel
          username={username}
          month={selectedMonth}
          contactName={contactName}
          includeMine={includeMine}
          onClose={() => setSelectedMonth(null)}
        />
      )}
    </div>
  );
};
