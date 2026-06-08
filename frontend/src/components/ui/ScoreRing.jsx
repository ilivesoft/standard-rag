import React from 'react';

export function ScoreRing({ value, size = 116, color = 'var(--il-blue)' }) {
  const sw = 10, r = (size - sw) / 2, c = 2 * Math.PI * r;
  return (
    <div style={{ position: 'relative', width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--il-overlay)" strokeWidth={sw} />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={sw}
          strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - value)}
          style={{ transition: 'stroke-dashoffset .8s cubic-bezier(.4,0,.2,1)' }} />
      </svg>
      <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ fontSize: 28, fontWeight: 800, color: 'var(--il-text-primary)', letterSpacing: '-0.5px' }}>{value.toFixed(2)}</span>
        <span style={{ fontSize: 11, color: 'var(--il-text-hint)' }}>overall</span>
      </div>
    </div>
  );
}

export default ScoreRing;
