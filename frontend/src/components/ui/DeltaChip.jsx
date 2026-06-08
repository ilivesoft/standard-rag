import React from 'react';
import { Ico } from './Ico.jsx';

export function DeltaChip({ delta, invert }) {
  const up = delta >= 0;
  const good = invert ? !up : up;
  const color = good ? 'var(--il-success)' : 'var(--il-live)';
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 2, fontSize: 12.5, fontWeight: 600, color }}>
      <Ico name={up ? 'arrow_drop_up' : 'arrow_drop_down'} size={18} />
      {Math.abs(delta)}%
    </span>
  );
}

export default DeltaChip;
