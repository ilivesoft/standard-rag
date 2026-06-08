import React from 'react';
import { Ico } from './ui/Ico.jsx';

const PAGE_META = {
  dashboard: { kr: '메인', en: 'Dashboard', desc: '전체 현황을 한눈에' },
  chat: { kr: '챗봇', en: 'Chatbot', desc: '문서 기반 질의응답' },
  rag: { kr: 'RAG 관리', en: 'RAG Management', desc: '문서 · 컬렉션 · 설정 · 평가' }
};

export function Topbar({ route, onToggle, collapsed }) {
  const meta = PAGE_META[route];
  return (
    <header style={{ height: 64, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 14, padding: '0 24px', borderBottom: '1px solid var(--il-overlay)', background: 'rgba(26,28,33,0.65)', backdropFilter: 'blur(8px)' }}>
      <button onClick={onToggle} className="il-btn-icon" style={{ marginLeft: -8 }}>
        <Ico name={collapsed ? 'menu' : 'menu_open'} size={22} />
      </button>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 9 }}>
        <h1 style={{ margin: 0, fontSize: 19, fontWeight: 700, color: 'var(--il-text-primary)' }}>{meta.kr}</h1>
      </div>
      <span style={{ fontSize: 12.5, color: 'var(--il-text-hint)', borderLeft: '1px solid var(--il-overlay)', paddingLeft: 14 }}>{meta.desc}</span>
      <div style={{ flex: 1 }} />
      <div style={{ display: 'flex', alignItems: 'center', gap: 7, background: 'var(--il-bg-base)', border: '1px solid var(--il-overlay)', borderRadius: 20, padding: '0 14px', width: 240 }}>
        <Ico name="search" size={18} style={{ color: 'var(--il-icon)' }} />
        <input placeholder="문서 · 대화 검색" style={{ flex: 1, border: 'none', outline: 'none', background: 'transparent', color: 'var(--il-text-primary)', fontSize: 13, padding: '9px 0', fontFamily: 'var(--il-font)' }} />
      </div>
      <button className="il-btn-icon"><Ico name="notifications" size={22} /></button>
      <button className="il-btn-icon"><Ico name="help" size={22} /></button>
    </header>
  );
}

export default Topbar;
