import React from 'react';

export function ProgressBar({ pct, color = 'var(--il-blue)' }) {
  return (
    <div style={{ height: 6, borderRadius: 3, background: 'var(--il-overlay)', overflow: 'hidden' }}>
      <div style={{ width: pct + '%', height: '100%', borderRadius: 3, background: color, transition: 'width .4s ease' }} />
    </div>
  );
}

export default ProgressBar;
