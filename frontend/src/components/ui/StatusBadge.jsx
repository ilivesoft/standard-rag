import React from 'react';
import { STATUS } from './StatusDot.jsx';
import { withAlpha } from './helpers.js';

export function StatusBadge({ status }) {
  const s = STATUS[status] || STATUS.idle;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '4px 9px 4px 7px', borderRadius: 20, fontSize: 12, fontWeight: 600,
      color: s.color, background: withAlpha(s.color, 0.14),
    }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: s.color }} />
      {s.label}
    </span>
  );
}

export default StatusBadge;
