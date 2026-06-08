import React from 'react';
import { Ico } from './ui/Ico.jsx';
import { StatusDot } from './ui/StatusDot.jsx';

const NAV = [
  { id: 'dashboard', glyph: 'space_dashboard', kr: '메인', en: 'Dashboard' },
  { id: 'chat', glyph: 'forum', kr: '챗봇', en: 'Chatbot' },
  { id: 'rag', glyph: 'database', kr: 'RAG 관리', en: 'RAG Management' },
];

export function Sidebar({ route, onNav, collapsed, labels }) {
  const w = collapsed ? 72 : 244;
  return (
    <aside style={{
      width: w, flexShrink: 0, background: 'var(--il-surface)', borderRight: '1px solid var(--il-overlay)',
      display: 'flex', flexDirection: 'column', transition: 'width .2s cubic-bezier(.4,0,.2,1)', overflow: 'hidden'
    }}>
      {/* Brand */}
      <div style={{ height: 64, display: 'flex', alignItems: 'center', gap: 11, padding: collapsed ? '0 16px' : '0 20px', borderBottom: '1px solid var(--il-overlay)' }}>
        <img src="/assets/ilive-logo-en.png" alt="iLive" style={{ height: 22, flexShrink: 0 }} />
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
            onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = 'var(--il-hover)'; }}
            onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = 'transparent'; }}>
              {active && <span style={{ position: 'absolute', left: 0, top: '50%', transform: 'translateY(-50%)', width: 3, height: 20, borderRadius: 3, background: 'var(--il-blue)' }} />}
              <Ico name={n.glyph} size={22} fill={active ? 1 : 0} style={{ flexShrink: 0 }} />
              {!collapsed &&
                <span style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.2, alignItems: 'flex-start' }}>
                  <span style={{ fontSize: 14, fontWeight: active ? 600 : 500, color: active ? 'var(--il-text-primary)' : 'var(--il-text-sec)' }}>{n.kr}</span>
                  {labels === 'both' && <span style={{ fontSize: 10.5, color: 'var(--il-text-hint)' }}>{n.en}</span>}
                </span>
              }
            </button>
          );
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
    </aside>
  );
}

export default Sidebar;
