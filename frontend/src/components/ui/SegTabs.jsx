import React from 'react';
import { Ico } from './Ico.jsx';

export function SegTabs({ tabs, value, onChange }) {
  return (
    <div style={{ display: 'inline-flex', gap: 2, padding: 4, background: 'var(--il-bg-base)', borderRadius: 11, border: '1px solid var(--il-overlay)' }}>
      {tabs.map(t => {
        const active = t.id === value;
        return (
          <button key={t.id} onClick={() => onChange(t.id)} style={{
            display: 'inline-flex', alignItems: 'center', gap: 7, padding: '8px 15px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: active ? 'var(--il-surface-el)' : 'transparent',
            color: active ? 'var(--il-text-primary)' : 'var(--il-text-sec)',
            fontSize: 13.5, fontWeight: active ? 600 : 500, fontFamily: 'var(--il-font)',
            boxShadow: active ? '0 1px 2px rgba(0,0,0,.3)' : 'none', transition: 'all .15s',
          }}>
            {t.glyph && <Ico name={t.glyph} size={18} fill={active ? 1 : 0} />}
            {t.label}
            {t.count != null && <span style={{ fontSize: 11, fontWeight: 700, color: active ? 'var(--il-blue)' : 'var(--il-text-hint)' }}>{t.count}</span>}
          </button>
        );
      })}
    </div>
  );
}

export default SegTabs;
