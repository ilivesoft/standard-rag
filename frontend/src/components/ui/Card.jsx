import React from 'react';

export function Card({ title, titleEn, right, children, style, bodyStyle, pad = true }) {
  return (
    <section style={{
      background: 'var(--il-surface)', borderRadius: 14,
      border: '1px solid rgba(56,59,67,0.55)',
      display: 'flex', flexDirection: 'column', minWidth: 0, ...style,
    }}>
      {title && (
        <header style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 12, padding: '16px 18px 0' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, minWidth: 0 }}>
            <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: 'var(--il-text-primary)' }}>{title}</h3>
            {titleEn && <span style={{ fontSize: 12, color: 'var(--il-text-hint)' }}>{titleEn}</span>}
          </div>
          {right}
        </header>
      )}
      <div style={{ padding: pad ? 18 : 0, flex: 1, minHeight: 0, ...bodyStyle }}>{children}</div>
    </section>
  );
}

export default Card;
