import React from 'react';
import { withAlpha } from './helpers.js';

export const TYPE_COLORS = { PDF: '#E0524D', DOCX: '#2993D1', MD: '#8C6FE0', HTML: '#E0A516', TXT: '#8C9199' };

export function TypeTag({ type }) {
  const c = TYPE_COLORS[type] || '#8C9199';
  return (
    <span style={{ display: 'inline-flex', padding: '2px 7px', borderRadius: 5, fontSize: 10.5, fontWeight: 700, letterSpacing: '.3px',
      color: c, background: withAlpha(c, 0.16) }}>{type}</span>
  );
}

export default TypeTag;
