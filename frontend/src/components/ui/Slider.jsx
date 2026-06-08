import React from 'react';

export function Slider({ label, labelEn, value, min, max, step, onChange, fmt, hintLeft, hintRight }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <span style={{ fontSize: 13, color: 'var(--il-text-primary)', fontWeight: 500 }}>
          {label} {labelEn && <span style={{ color: 'var(--il-text-hint)', fontSize: 11.5, fontWeight: 400 }}>{labelEn}</span>}
        </span>
        <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--il-blue)', fontFamily: 'var(--il-font-mono)' }}>{fmt ? fmt(value) : value}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: '100%', height: 5, borderRadius: 3, appearance: 'none', WebkitAppearance: 'none', cursor: 'pointer', margin: 0,
          background: `linear-gradient(to right, var(--il-blue) ${pct}%, var(--il-overlay) ${pct}%)` }} />
      {(hintLeft || hintRight) && (
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--il-text-hint)' }}>
          <span>{hintLeft}</span><span>{hintRight}</span>
        </div>
      )}
    </div>
  );
}

export default Slider;
