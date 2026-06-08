import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar.jsx';
import { Topbar } from './components/Topbar.jsx';
import { DashboardPage } from './pages/Dashboard/index.jsx';
import { ChatbotPage } from './pages/Chatbot/index.jsx';
import { RagPage } from './pages/RagManage/index.jsx';
import { withAlpha } from './components/ui/helpers.js';

const ACCENT_HOVER = {
  '#2993D1': '#1F7AB0',
  '#214290': '#1A3576',
  '#1F8A7A': '#176A5E',
  '#7A5AE0': '#6344C4',
};

export function App() {
  const [route, setRoute] = useState('dashboard');
  const [ragSection, setRagSection] = useState('docs');
  const [collapsed, setCollapsed] = useState(false);
  const [accent] = useState('#2993D1');

  function navTo(id, section) {
    setRoute(id);
    if (id === 'rag' && section) setRagSection(section);
  }

  const accentStyle = {
    '--il-blue': accent,
    '--il-blue-hover': ACCENT_HOVER[accent] || accent,
    '--il-blue-soft': withAlpha(accent, 0.14),
    '--il-focus-ring': accent,
  };

  const pad = 24;

  return (
    <div className="il-root" style={{ display: 'flex', height: '100vh', overflow: 'hidden', ...accentStyle }}>
      <Sidebar route={route} onNav={navTo} collapsed={collapsed} labels="both" accent={accent} />
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <Topbar route={route} collapsed={collapsed} onToggle={() => setCollapsed(c => !c)} />
        <main style={{
          flex: 1, overflowY: 'auto', padding: pad, minHeight: 0, background: 'var(--il-bg-base)'
        }}>
          <div style={{
            maxWidth: 1320, margin: '0 auto',
            height: route === 'chat' ? '100%' : 'auto',
            minHeight: route === 'chat' ? 0 : 'auto',
            display: route === 'chat' ? 'flex' : 'block',
            flexDirection: 'column',
          }}>
            {route === 'dashboard' && <DashboardPage onNav={navTo} />}
            {route === 'chat'      && <ChatbotPage />}
            {route === 'rag'       && <RagPage section={ragSection} setSection={setRagSection} />}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
