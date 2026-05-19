import { useEffect, useState } from 'react';
import type React from 'react';
import { Copy, Download, FileJson, FileText, RefreshCw } from 'lucide-react';
import { Header } from '../layout/Header';
import { LAST_PSYCH_RESULT_KEY, type PsychAnalyzeResponse } from './types';

function downloadText(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function makeName(result: PsychAnalyzeResponse, suffix: string) {
  const id = result.task_id.slice(0, 8) || 'report';
  return `mindtrace-risk-${id}.${suffix}`;
}

export function PsychExportPage() {
  const [result, setResult] = useState<PsychAnalyzeResponse | null>(null);
  const [message, setMessage] = useState('');

  const loadResult = () => {
    setMessage('');
    try {
      const saved = localStorage.getItem(LAST_PSYCH_RESULT_KEY);
      setResult(saved ? JSON.parse(saved) as PsychAnalyzeResponse : null);
    } catch {
      setResult(null);
      setMessage('本地分析结果读取失败，请重新完成一次分析。');
    }
  };

  useEffect(() => {
    loadResult();
  }, []);

  const exportMarkdown = () => {
    if (!result) return;
    downloadText(makeName(result, 'md'), result.report_md, 'text/markdown;charset=utf-8');
  };

  const exportJson = () => {
    if (!result) return;
    downloadText(makeName(result, 'json'), JSON.stringify(result, null, 2), 'application/json;charset=utf-8');
  };

  const copyMarkdown = async () => {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(result.report_md);
      setMessage('报告内容已复制。');
    } catch {
      setMessage('复制失败，可以使用 Markdown 下载。');
    }
  };

  return (
    <div>
      <Header
        title="导出"
        subtitle="导出最近一次心理风险辅助筛查报告，默认只在本地浏览器生成文件。"
      />

      <div className="max-w-4xl space-y-5">
        <div className="rounded-2xl border border-amber-100 bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:bg-amber-500/10 dark:border-amber-500/20 dark:text-amber-100">
          导出内容可能包含敏感聊天摘录。请只保存在你信任的位置，不要上传到无关平台。
        </div>

        <section className="dk-card bg-white border dk-border rounded-2xl p-5 shadow-sm">
          <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
            <div>
              <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white">最近一次报告</h2>
              <p className="text-sm text-gray-500 mt-1">
                {result ? `任务 ID：${result.task_id}` : '暂无可导出的分析结果。'}
              </p>
            </div>
            <button
              onClick={loadResult}
              className="inline-flex items-center justify-center gap-2 rounded-xl border dk-border px-3 py-2 text-sm font-semibold text-gray-600 hover:bg-gray-50 dark:text-gray-200 dark:hover:bg-white/10"
            >
              <RefreshCw size={16} />
              刷新
            </button>
          </div>

          {result ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-5">
              <ExportAction
                icon={<FileText size={20} />}
                title="Markdown"
                desc="适合归档、打印或二次编辑。"
                onClick={exportMarkdown}
              />
              <ExportAction
                icon={<FileJson size={20} />}
                title="JSON"
                desc="包含分数、特征、证据和报告。"
                onClick={exportJson}
              />
              <ExportAction
                icon={<Copy size={20} />}
                title="复制报告"
                desc="复制 Markdown 报告到剪贴板。"
                onClick={copyMarkdown}
              />
            </div>
          ) : (
            <div className="mt-5 rounded-xl bg-gray-50 dark:bg-white/5 p-6 text-center text-sm text-gray-500">
              请先进入“心理分析”完成一次辅助筛查。
            </div>
          )}

          {message && (
            <div className="mt-4 rounded-xl bg-gray-50 dark:bg-white/5 px-3 py-2 text-sm text-gray-600 dark:text-gray-200">
              {message}
            </div>
          )}
        </section>

        {result && (
          <section className="dk-card bg-white border dk-border rounded-2xl p-5 shadow-sm">
            <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white mb-3">导出预览</h2>
            <pre className="whitespace-pre-wrap break-words rounded-xl bg-gray-50 dark:bg-black/20 p-4 text-sm leading-7 text-gray-700 dark:text-gray-200 max-h-[560px] overflow-auto">
              {result.report_md}
            </pre>
          </section>
        )}
      </div>
    </div>
  );
}

function ExportAction({
  icon,
  title,
  desc,
  onClick,
}: {
  icon: React.ReactNode;
  title: string;
  desc: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="text-left rounded-2xl border dk-border p-4 hover:border-[#07c160] hover:bg-[#f6fffa] dark:hover:bg-[#07c160]/10 transition-colors"
    >
      <div className="w-10 h-10 rounded-xl bg-[#e7f8f0] text-[#07c160] flex items-center justify-center">
        {icon}
      </div>
      <div className="mt-4 text-sm font-black text-[#1d1d1f] dark:text-white">{title}</div>
      <div className="mt-1 text-xs leading-5 text-gray-500">{desc}</div>
      <div className="mt-4 inline-flex items-center gap-2 text-xs font-black text-[#07c160]">
        <Download size={14} />
        执行
      </div>
    </button>
  );
}
