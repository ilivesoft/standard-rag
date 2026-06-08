import React from 'react';

export function GaugeBar({ label, labelEn, value, color = 'var(--il-blue)' }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <span style={{ fontSize: 13, color: 'var(--il-text-primary)', fontWeight: 500 }}>
          {label} <span style={{ color: 'var(--il-text-hint)', fontSize: 11.5, fontWeight: 400 }}>{labelEn}</span>
        </span>
        <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--il-text-primary)', fontFamily: 'var(--il-font-mono)' }}>{value.toFixed(2)}</span>
      </div>
      <div style={{ height: 7, borderRadius: 4, background: 'var(--il-overlay)', overflow: 'hidden' }}>
        <div style={{ width: (value * 100) + '%', height: '100%', borderRadius: 4, background: color, transition: 'width .6s cubic-bezier(.4,0,.2,1)' }} />
      </div>
    </div>
  );
}

export default GaugeBar;
