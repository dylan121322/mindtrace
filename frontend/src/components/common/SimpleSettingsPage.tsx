import { Moon, Sun, Type } from 'lucide-react';
import { Header } from '../layout/Header';
import { PythonBackendSettingsPanel } from './PythonBackendSettingsPanel';

interface SimpleSettingsPageProps {
  dark: boolean;
  onToggleDark: () => void;
  fontSize: number;
  onFontSizeChange: (size: number) => void;
}

export function SimpleSettingsPage({
  dark,
  onToggleDark,
  fontSize,
  onFontSizeChange,
}: SimpleSettingsPageProps) {
  return (
    <div>
      <Header
        title="设置"
        subtitle="配置微信数据库目录、AI 模型、后端地址与基础显示偏好。"
      />

      <div className="max-w-5xl space-y-5">
        <section className="dk-card bg-white border dk-border rounded-2xl p-5 shadow-sm">
          <h2 className="text-lg font-black text-[#1d1d1f] dark:text-white">界面</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            <button
              onClick={onToggleDark}
              className="flex items-center gap-3 rounded-xl border dk-border px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-white/10"
            >
              <span className="w-10 h-10 rounded-xl bg-[#e7f8f0] text-[#07c160] flex items-center justify-center">
                {dark ? <Sun size={20} /> : <Moon size={20} />}
              </span>
              <span>
                <span className="block text-sm font-black text-[#1d1d1f] dark:text-white">
                  {dark ? '切换到浅色模式' : '切换到深色模式'}
                </span>
                <span className="block text-xs text-gray-500 mt-1">只影响当前浏览器显示。</span>
              </span>
            </button>

            <label className="rounded-xl border dk-border px-4 py-3">
              <span className="flex items-center gap-2 text-sm font-black text-[#1d1d1f] dark:text-white">
                <Type size={18} />
                字号
              </span>
              <input
                type="range"
                min={14}
                max={20}
                value={fontSize}
                onChange={(event) => onFontSizeChange(Number(event.target.value))}
                className="w-full mt-3"
              />
              <span className="text-xs text-gray-500">{fontSize}px</span>
            </label>
          </div>
        </section>

        <PythonBackendSettingsPanel />
      </div>
    </div>
  );
}

