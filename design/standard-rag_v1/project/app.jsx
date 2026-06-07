// app.jsx — Standard RAG shell: sidebar nav + topbar + routing + tweaks
const { useState: useStateApp } = React;
const {
  useTweaks, TweaksPanel, TweakSection, TweakRadio, TweakToggle, TweakColor
} = window;

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "sidebar": "expanded",
  "showHeader": true,
  "labels": "both",
  "accent": "#2993D1",
  "density": "regular"
} /*EDITMODE-END*/;

const NAV = [
{ id: 'dashboard', glyph: 'space_dashboard', kr: '메인', en: 'Dashboard' },
{ id: 'chat', glyph: 'forum', kr: '챗봇', en: 'Chatbot' },
{ id: 'rag', glyph: 'database', kr: 'RAG 관리', en: 'RAG Management' }];


const PAGE_META = {
  dashboard: { kr: '메인', en: 'Dashboard', desc: '전체 현황을 한눈에' },
  chat: { kr: '챗봇', en: 'Chatbot', desc: '문서 기반 질의응답' },
  rag: { kr: 'RAG 관리', en: 'RAG Management', desc: '문서 · 컬렉션 · 설정 · 평가' }
};

function Sidebar({ route, onNav, collapsed, labels, accent }) {
  const { Ico, StatusDot } = window;
  const w = collapsed ? 72 : 244;
  return (
    <aside style={{
      width: w, flexShrink: 0, background: 'var(--il-surface)', borderRight: '1px solid var(--il-overlay)',
      display: 'flex', flexDirection: 'column', transition: 'width .2s cubic-bezier(.4,0,.2,1)', overflow: 'hidden'
    }}>
      {/* Brand */}
      <div style={{ height: 64, display: 'flex', alignItems: 'center', gap: 11, padding: collapsed ? '0 16px' : '0 20px', borderBottom: '1px solid var(--il-overlay)' }}>
        <img src="assets/ilive-logo-en.png" alt="iLive" style={{ height: 22, flexShrink: 0 }} />
        {!collapsed &&
        <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.15, borderLeft: '1px solid var(--il-overlay)', paddingLeft: 11 }}>
            <span style={{ fontSize: 13.5, fontWeight: 700, color: 'var(--il-text-primary)' }}>Standard RAG</span>
            <span style={{ fontSize: 10.5, color: 'var(--il-text-hint)' }}>RAG 파이프라인 콘솔</span>
          </div>
        }
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: collapsed ? '12px 12px' : '14px 12px', display: 'flex', flexDirection: 'column', gap: 4 }}>
        {!collapsed && <div style={{ fontSize: 10.5, fontWeight: 600, color: 'var(--il-text-hint)', textTransform: 'uppercase', letterSpacing: '.5px', padding: '4px 12px 6px' }}></div>}
        {NAV.map((n) => {
          const active = route === n.id;
          return (
            <button key={n.id} onClick={() => onNav(n.id)} title={collapsed ? n.kr : ''} style={{
              position: 'relative', display: 'flex', alignItems: 'center', gap: 12, padding: collapsed ? '11px 0' : '10px 12px',
              justifyContent: collapsed ? 'center' : 'flex-start', borderRadius: 10, border: 'none', cursor: 'pointer', width: '100%',
              background: active ? 'var(--il-blue-soft)' : 'transparent', color: active ? 'var(--il-blue)' : 'var(--il-text-sec)', transition: 'background .12s, color .12s'
            }}
            onMouseEnter={(e) => {if (!active) e.currentTarget.style.background = 'var(--il-hover)';}}
            onMouseLeave={(e) => {if (!active) e.currentTarget.style.background = 'transparent';}}>
              {active && <span style={{ position: 'absolute', left: 0, top: '50%', transform: 'translateY(-50%)', width: 3, height: 20, borderRadius: 3, background: 'var(--il-blue)' }} />}
              <Ico name={n.glyph} size={22} fill={active ? 1 : 0} style={{ flexShrink: 0 }} />
              {!collapsed &&
              <span style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.2, alignItems: 'flex-start' }}>
                  <span style={{ fontSize: 14, fontWeight: active ? 600 : 500, color: active ? 'var(--il-text-primary)' : 'var(--il-text-sec)' }}>{n.kr}</span>
                  {labels === 'both' && <span style={{ fontSize: 10.5, color: 'var(--il-text-hint)' }}>{n.en}</span>}
                </span>
              }
            </button>);

        })}
      </nav>

      {/* Footer: env + user */}
      <div style={{ padding: collapsed ? 12 : 14, borderTop: '1px solid var(--il-overlay)', display: 'flex', flexDirection: 'column', gap: 10 }}>
        {!collapsed &&
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 11px', background: 'var(--il-bg-base)', borderRadius: 9 }}>
            <StatusDot status="ready" pulse />
            <div style={{ flex: 1, lineHeight: 1.2 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--il-text-primary)' }}>development</div>
              <div style={{ fontSize: 10.5, color: 'var(--il-text-hint)' }}>ChromaDB · :8000</div>
            </div>
          </div>
        }
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, justifyContent: collapsed ? 'center' : 'flex-start' }}>
          <span style={{ width: 32, height: 32, borderRadius: '50%', background: 'var(--il-grad-brand)', color: '#fff', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: 13, fontWeight: 700, flexShrink: 0 }}>J</span>
          {!collapsed &&
          <div style={{ flex: 1, lineHeight: 1.2 }}>
              <div style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--il-text-primary)' }}>JKH</div>
              <div style={{ fontSize: 10.5, color: 'var(--il-text-hint)' }}>kihyun@ilivesoft.com</div>
            </div>
          }
        </div>
      </div>
    </aside>);

}

function Topbar({ route, onToggle, collapsed }) {
  const { Ico } = window;
  const meta = PAGE_META[route];
  return (
    <header style={{ height: 64, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 14, padding: '0 24px', borderBottom: '1px solid var(--il-overlay)', background: 'rgba(26,28,33,0.65)', backdropFilter: 'blur(8px)' }}>
      <button onClick={onToggle} className="il-btn-icon" style={{ marginLeft: -8 }}><Ico name={collapsed ? 'menu' : 'menu_open'} size={22} /></button>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 9 }}>
        <h1 style={{ margin: 0, fontSize: 19, fontWeight: 700, color: 'var(--il-text-primary)' }}>{meta.kr}</h1>
        <span style={{ fontSize: 13, color: 'var(--il-text-hint)' }}>{"\n"}</span>
      </div>
      <span style={{ fontSize: 12.5, color: 'var(--il-text-hint)', borderLeft: '1px solid var(--il-overlay)', paddingLeft: 14 }}>{meta.desc}</span>
      <div style={{ flex: 1 }} />
      <div style={{ display: 'flex', alignItems: 'center', gap: 7, background: 'var(--il-bg-base)', border: '1px solid var(--il-overlay)', borderRadius: 20, padding: '0 14px', width: 240 }}>
        <Ico name="search" size={18} style={{ color: 'var(--il-icon)' }} />
        <input placeholder="문서 · 대화 검색" style={{ flex: 1, border: 'none', outline: 'none', background: 'transparent', color: 'var(--il-text-primary)', fontSize: 13, padding: '9px 0', fontFamily: 'var(--il-font)' }} />
      </div>
      <button className="il-btn-icon"><Ico name="notifications" size={22} /></button>
      <button className="il-btn-icon"><Ico name="help" size={22} /></button>
    </header>);

}

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const [route, setRoute] = useStateApp('dashboard');
  const [ragSection, setRagSection] = useStateApp('docs');
  const [manualCollapse, setManualCollapse] = useStateApp(null); // override topbar toggle

  const collapsed = manualCollapse != null ? manualCollapse : t.sidebar === 'icons';
  const pad = t.density === 'compact' ? 18 : 24;

  function navTo(id, section) {
    setRoute(id);
    if (id === 'rag' && section) setRagSection(section);
  }

  // accent override (concrete values — screenshot-safe)
  const ACCENT_HOVER = { '#2993D1': '#1F7AB0', '#214290': '#1A3576', '#1F8A7A': '#176A5E', '#7A5AE0': '#6344C4' };
  const accentStyle = {
    '--il-blue': t.accent,
    '--il-blue-hover': ACCENT_HOVER[t.accent] || t.accent,
    '--il-blue-soft': window.withAlpha(t.accent, 0.14),
    '--il-focus-ring': t.accent
  };

  return (
    <div className="il-root" style={{ display: 'flex', height: '100vh', overflow: 'hidden', ...accentStyle }}>
      <Sidebar route={route} onNav={navTo} collapsed={collapsed} labels={t.labels} accent={t.accent} />
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {t.showHeader && <Topbar route={route} collapsed={collapsed} onToggle={() => setManualCollapse(!collapsed)} />}
        <main style={{ flex: 1, overflowY: 'auto', padding: route === 'chat' ? pad : pad, minHeight: 0, background: 'var(--il-bg-base)' }}>
          <div style={{ maxWidth: 1320, margin: '0 auto', height: route === 'chat' ? '100%' : 'auto', minHeight: route === 'chat' ? 0 : 'auto', display: route === 'chat' ? 'flex' : 'block', flexDirection: 'column' }}>
            {route === 'dashboard' && <window.DashboardPage onNav={navTo} />}
            {route === 'chat' && <window.ChatbotPage />}
            {route === 'rag' && <window.RagPage section={ragSection} setSection={setRagSection} />}
          </div>
        </main>
      </div>

      <TweaksPanel>
        <TweakSection label="레이아웃 · Layout" />
        <TweakRadio label="사이드바" value={t.sidebar} options={['expanded', 'icons']}
        onChange={(v) => {setTweak('sidebar', v);setManualCollapse(null);}} />
        <TweakToggle label="상단 헤더" value={t.showHeader} onChange={(v) => setTweak('showHeader', v)} />
        <TweakRadio label="밀도 Density" value={t.density} options={['compact', 'regular']} onChange={(v) => setTweak('density', v)} />
        <TweakSection label="라벨 · Labels" />
        <TweakRadio label="메뉴 라벨" value={t.labels} options={['both', 'ko']} onChange={(v) => setTweak('labels', v)} />
        <TweakSection label="브랜드 · Brand" />
        <TweakColor label="강조색 Accent" value={t.accent}
        options={['#2993D1', '#214290', '#1F8A7A', '#7A5AE0']} onChange={(v) => setTweak('accent', v)} />
      </TweaksPanel>
    </div>);

}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);