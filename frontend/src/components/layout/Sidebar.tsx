import { BarChart3, Brain, Database, Download, FileSearch, GraduationCap, Moon, Settings, SlidersHorizontal, Sun } from 'lucide-react';
import type React from 'react';
import type { TabType } from '../../types';

interface SidebarProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
  dark: boolean;
  onToggleDark: () => void;
}

const navItems: { tab: TabType; icon: React.ReactNode; label: string }[] = [
  { tab: 'psych', icon: <Brain size={20} strokeWidth={2} />, label: '心理分析' },
  { tab: 'db', icon: <Database size={20} strokeWidth={2} />, label: '数据库' },
  { tab: 'visual', icon: <BarChart3 size={20} strokeWidth={2} />, label: '可视化' },
  { tab: 'debug', icon: <FileSearch size={20} strokeWidth={2} />, label: '细节调试' },
  { tab: 'scoring', icon: <SlidersHorizontal size={20} strokeWidth={2} />, label: '评分标准' },
  { tab: 'training', icon: <GraduationCap size={20} strokeWidth={2} />, label: '训练优化' },
  { tab: 'export', icon: <Download size={20} strokeWidth={2} />, label: '导出' },
  { tab: 'settings', icon: <Settings size={20} strokeWidth={2} />, label: '设置' },
];

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onTabChange, dark, onToggleDark }) => {
  return (
    <>
      <aside className="hidden sm:flex w-56 shrink-0 dk-card bg-white dk-border border-r flex-col px-3 py-5 shadow-sm z-10">
        <div className="flex items-center gap-3 px-2 pb-5 border-b dk-border">
          <div className="w-10 h-10 rounded-xl bg-[#e7f8f0] text-[#07c160] flex items-center justify-center">
            <Brain size={22} />
          </div>
          <div className="min-w-0">
            <div className="text-base font-black text-[#1d1d1f] dark:text-white leading-tight">MindTrace</div>
            <div className="text-xs text-gray-400 font-medium">心理风险辅助筛查</div>
          </div>
        </div>

        <nav className="flex flex-col gap-1 mt-5 flex-1">
          {navItems.map(({ tab, icon, label }) => (
            <button
              key={tab}
              onClick={() => onTabChange(tab)}
              className={`flex items-center gap-3 rounded-xl px-3 py-3 text-left transition-all ${
                activeTab === tab
                  ? 'bg-[#e7f8f0] text-[#07c160] dark:bg-[#07c160]/20'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50 dark:text-gray-400 dark:hover:text-gray-100 dark:hover:bg-white/5'
              }`}
            >
              <span className="flex-shrink-0">{icon}</span>
              <span className="text-sm font-semibold">{label}</span>
            </button>
          ))}
        </nav>

        <button
          onClick={onToggleDark}
          className="flex items-center gap-3 rounded-xl px-3 py-3 text-gray-500 hover:text-gray-700 hover:bg-gray-50 dark:text-gray-400 dark:hover:text-gray-100 dark:hover:bg-white/5 transition-all"
        >
          {dark ? <Sun size={20} /> : <Moon size={20} />}
          <span className="text-sm font-semibold">{dark ? '浅色模式' : '深色模式'}</span>
        </button>
      </aside>

      <nav className="sm:hidden fixed bottom-0 left-0 right-0 z-50 dk-card bg-white dk-border border-t flex">
        {navItems.map(({ tab, icon, label }) => (
          <button
            key={tab}
            onClick={() => onTabChange(tab)}
            className={`flex-1 flex flex-col items-center justify-center py-2.5 gap-1 transition-colors min-w-0 ${
              activeTab === tab ? 'text-[#07c160]' : 'text-gray-400'
            }`}
          >
            <span className="flex-shrink-0">{icon}</span>
            <span className="text-[11px] font-semibold truncate w-full text-center px-1">{label}</span>
          </button>
        ))}
      </nav>
    </>
  );
};
