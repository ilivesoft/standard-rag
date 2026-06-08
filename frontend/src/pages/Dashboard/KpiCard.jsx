import React from 'react';
import { Ico } from '../../components/ui/Ico.jsx';
import { Sparkline } from '../../components/ui/Sparkline.jsx';
import { DeltaChip } from '../../components/ui/DeltaChip.jsx';
import { fmt } from './mockData.js';

export function KpiCard({ k }) {
  const display = k.id === 'lat' ? k.value.toFixed(2) : fmt(k.value);
  return (
    <div style={{
      background: 'var(--il-surface)', borderRadius: 14, padding: 18,
      border: '1px solid rgba(56,59,67,0.55)',
      display: 'flex', flexDirection: 'column', gap: 14, position: 'relative', overflow: 'hidden',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
          <span style={{ width: 34, height: 34, borderRadius: 9, background: 'var(--il-blue-soft)', color: 'var(--il-blue)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}>
            <Ico name={k.glyph} size={19} />
          </span>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--il-text-primary)' }}>{k.kr}</span>
            <span style={{ fontSize: 11, color: 'var(--il-text-hint)' }}>{k.en}</span>
          </div>
        </div>
        <DeltaChip delta={k.delta} invert={k.invert} />
      </div>
      <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', gap: 8 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
          <span style={{ fontSize: 32, fontWeight: 800, letterSpacing: '-1px', color: 'var(--il-text-primary)', fontFamily: k.id === 'lat' ? 'var(--il-font-mono)' : 'inherit' }}>{display}</span>
          <span style={{ fontSize: 13, color: 'var(--il-text-sec)' }}>{k.unit}</span>
        </div>
        <Sparkline data={k.spark} color={k.invert ? 'var(--il-success)' : 'var(--il-blue)'} />
      </div>
    </div>
  );
}

export default KpiCard;
