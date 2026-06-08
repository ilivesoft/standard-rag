import React from 'react';

export function Ico({ name, size = 20, fill = 0, style }) {
  return (
    <span
      className="material-symbols-outlined"
      style={{
        fontSize: size,
        lineHeight: 1,
        width: size,
        height: size,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        flexShrink: 0,
        fontVariationSettings: `'FILL' ${fill}, 'wght' 400, 'GRAD' 0, 'opsz' ${size}`,
        ...style,
      }}
    >
      {name}
    </span>
  );
}

export default Ico;
