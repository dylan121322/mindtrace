import { useCallback, useEffect, useState } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { SimpleSettingsPage } from './components/common/SimpleSettingsPage';
import { DatabaseManagePage } from './components/database/DatabaseManagePage';
import { PsychAnalysisPage } from './components/psych/PsychAnalysisPage';
import { PsychDebugPage } from './components/psych/PsychDebugPage';
import { PsychExportPage } from './components/psych/PsychExportPage';
import { PsychVisualizationPage } from './components/psych/PsychVisualizationPage';
import { ScoringConfigPage } from './components/psych/ScoringConfigPage';
import { PsychTrainingPage } from './components/psych/PsychTrainingPage';
import { useDarkMode } from './hooks/useDarkMode';
import type { TabType } from './types';

const APP_TABS: TabType[] = ['psych', 'db', 'visual', 'debug', 'scoring', 'training', 'export', 'settings'];

function parseHashTab(): TabType {
  const raw = window.location.hash.replace('#/', '').replace('#', '');
  const tab = raw.split('/')[0] as TabType;
  return APP_TABS.includes(tab) ? tab : 'psych';
}

function App() {
  const { dark, toggle: toggleDark } = useDarkMode();
  const [activeTab, setActiveTabRaw] = useState<TabType>(() => parseHashTab());
  const [fontSize, setFontSize] = useState(() => Number(localStorage.getItem('mindtrace_fontSize')) || 16);

  useEffect(() => {
    document.documentElement.style.fontSize = `${fontSize}px`;
    localStorage.setItem('mindtrace_fontSize', String(fontSize));
  }, [fontSize]);

  const setActiveTab = useCallback((tab: TabType) => {
    const next = APP_TABS.includes(tab) ? tab : 'psych';
    setActiveTabRaw(next);
    window.history.pushState(null, '', `#/${next}`);
  }, []);

  useEffect(() => {
    const onPop = () => setActiveTabRaw(parseHashTab());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  return (
    <div className="flex h-screen dk-page bg-[#f8f9fb] dk-text text-[#1d1d1f] font-sans overflow-hidden">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} dark={dark} onToggleDark={toggleDark} />

      <main className="flex-1 overflow-y-auto dk-page p-4 sm:p-8 lg:p-10 pb-24 sm:pb-10">
        {activeTab === 'export' ? (
          <PsychExportPage />
        ) : activeTab === 'visual' ? (
          <PsychVisualizationPage />
        ) : activeTab === 'scoring' ? (
          <ScoringConfigPage />
        ) : activeTab === 'training' ? (
          <PsychTrainingPage onOpenPsych={() => setActiveTab('psych')} />
        ) : activeTab === 'debug' ? (
          <PsychDebugPage />
        ) : activeTab === 'db' ? (
          <DatabaseManagePage onOpenSettings={() => setActiveTab('settings')} />
        ) : activeTab === 'settings' ? (
          <SimpleSettingsPage
            dark={dark}
            onToggleDark={toggleDark}
            fontSize={fontSize}
            onFontSizeChange={setFontSize}
          />
        ) : (
          <PsychAnalysisPage />
        )}
      </main>
    </div>
  );
}

export default App;
