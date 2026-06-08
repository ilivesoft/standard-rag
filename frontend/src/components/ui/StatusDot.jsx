import React from 'react';

export const STATUS = {
  ready:    { label: '정상',   en: 'Ready',    color: '#29B473' },
  indexed:  { label: '인덱싱', en: 'Indexed',  color: '#29B473' },
  degraded: { label: '지연',   en: 'Degraded', color: '#E0A516' },
  idle:     { label: '대기',   en: 'Idle',     color: '#6F757E' },
  indexing: { label: '처리중', en: 'Indexing', color: '#2993D1' },
  failed:   { label: '실패',   en: 'Failed',   color: '#FF2D2D' },
};

export function StatusDot({ status, pulse }) {
  const c = (STATUS[status] || STATUS.idle).color;
  return (
    <span style={{ position: 'relative', display: 'inline-flex', width: 9, height: 9 }}>
      <span style={{ width: 9, height: 9, borderRadius: '50%', background: c, boxShadow: `0 0 0 3px ${c}22` }} />
      {pulse && <span style={{ position: 'absolute', inset: 0, borderRadius: '50%', background: c, animation: 'srPulse 1.6s ease-out infinite' }} />}
    </span>
  );
}

export default StatusDot;
