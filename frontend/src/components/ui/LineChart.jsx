import React from 'react';

export function LineChart({ data, w = 620, h = 200, color = 'var(--il-blue)' }) {
  const padL = 4, padB = 22, padT = 10;
  const max = Math.max(...data.map(d => d.v)) * 1.12;
  const iw = w - padL * 2, ih = h - padB - padT;
  const x = i => padL + (i / (data.length - 1)) * iw;
  const y = v => padT + ih - (v / max) * ih;
  const line = data.map((d, i) => (i ? 'L' : 'M') + x(i).toFixed(1) + ' ' + y(d.v).toFixed(1)).join(' ');
  const area = line + ` L${x(data.length - 1)} ${padT + ih} L${x(0)} ${padT + ih} Z`;
  const grid = [0.25, 0.5, 0.75, 1].map(f => padT + ih - f * ih);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h} preserveAspectRatio="none" style={{ display: 'block' }}>
      <defs>
        <linearGradient id="qaFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.28" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {grid.map((gy, i) => <line key={i} x1={padL} y1={gy} x2={w - padL} y2={gy} stroke="var(--il-overlay)" strokeWidth="1" strokeDasharray="2 4" opacity="0.5" />)}
      <path d={area} fill="url(#qaFill)" />
      <path d={line} fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
      {data.map((d, i) => (
        <g key={i}>
          {i === data.length - 1 && <circle cx={x(i)} cy={y(d.v)} r="4" fill={color} stroke="var(--il-surface)" strokeWidth="2.5" />}
          {i % 2 === 0 && <text x={x(i)} y={h - 6} fontSize="10.5" fill="var(--il-text-hint)" textAnchor="middle">{d.d}</text>}
        </g>
      ))}
    </svg>
  );
}

export default LineChart;
