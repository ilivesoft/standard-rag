// ui.jsx — shared presentational components for Standard RAG
// Dark theme, iLive tokens. Exported to window.

// ── color helper: hex/var → rgba with alpha (screenshot-safe) ─
function withAlpha(hex, a) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || '');
  if (!m) return hex;
  return `rgba(${parseInt(m[1],16)},${parseInt(m[2],16)},${parseInt(m[3],16)},${a})`;
}

// ── Material Symbols icon ───────────────────────────────────
function Ico({ name, size = 20, fill = 0, style }) {
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

// ── Status config shared by badges / dots ───────────────────
const STATUS = {
  ready:    { label: '정상',   en: 'Ready',    color: '#29B473' },
  indexed:  { label: '인덱싱', en: 'Indexed',  color: '#29B473' },
  degraded: { label: '지연',   en: 'Degraded', color: '#E0A516' },
  idle:     { label: '대기',   en: 'Idle',     color: '#6F757E' },
  indexing: { label: '처리중', en: 'Indexing', color: '#2993D1' },
  failed:   { label: '실패',   en: 'Failed',   color: '#FF2D2D' },
};

function StatusDot({ status, pulse }) {
  const c = (STATUS[status] || STATUS.idle).color;
  return (
    <span style={{ position: 'relative', display: 'inline-flex', width: 9, height: 9 }}>
      <span style={{ width: 9, height: 9, borderRadius: '50%', background: c, boxShadow: `0 0 0 3px ${c}22` }} />
      {pulse && <span style={{ position: 'absolute', inset: 0, borderRadius: '50%', background: c, animation: 'srPulse 1.6s ease-out infinite' }} />}
    </span>
  );
}

function StatusBadge({ status }) {
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

// ── Delta chip (▲ +8.2%) ────────────────────────────────────
function DeltaChip({ delta, invert }) {
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

// ── Sparkline ───────────────────────────────────────────────
function Sparkline({ data, color = 'var(--il-blue)', w = 110, h = 36 }) {
  const max = Math.max(...data), min = Math.min(...data);
  const span = max - min || 1;
  const pts = data.map((v, i) => [ (i / (data.length - 1)) * w, h - ((v - min) / span) * (h - 6) - 3 ]);
  const d = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
  const area = d + ` L${w} ${h} L0 ${h} Z`;
  const gid = 'sg' + Math.random().toString(36).slice(2, 7);
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: 'block', overflow: 'visible' }}>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.30" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${gid})`} />
      <path d={d} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ── Section card ────────────────────────────────────────────
function Card({ title, titleEn, right, children, style, bodyStyle, pad = true }) {
  return (
    <section style={{
      background: 'var(--il-surface)', borderRadius: 14,
      border: '1px solid rgba(56,59,67,0.55)',
      display: 'flex', flexDirection: 'column', minWidth: 0, ...style,
    }}>
      {title && (
        <header style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 12, padding: '16px 18px 0' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, minWidth: 0 }}>
            <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: 'var(--il-text-primary)' }}>{title}</h3>
            {titleEn && <span style={{ fontSize: 12, color: 'var(--il-text-hint)' }}>{titleEn}</span>}
          </div>
          {right}
        </header>
      )}
      <div style={{ padding: pad ? 18 : 0, flex: 1, minHeight: 0, ...bodyStyle }}>{children}</div>
    </section>
  );
}

// ── Line/area chart (daily Q&A) ─────────────────────────────
function LineChart({ data, w = 620, h = 200, color = 'var(--il-blue)' }) {
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

// ── Horizontal gauge bar (RAGAS metric) ─────────────────────
function GaugeBar({ label, labelEn, value, color = 'var(--il-blue)' }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <span style={{ fontSize: 13, color: 'var(--il-text-primary)', fontWeight: 500 }}>
          {label} <span style={{ color: 'var(--il-text-hint)', fontSize: 11.5, fontWeight: 400 }}>{labelEn}</span>
        </span>
        <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--il-text-primary)', fontFamily: 'var(--il-font-mono)' }}>{value.toFixed(2)}</span>
      </div>
      <div style={{ height: 7, borderRadius: 4, background: 'var(--il-overlay)', overflow: 'hidden' }}>
        <div style={{ width: (value * 100) + '%', height: '100%', borderRadius: 4, background: color, transition: 'width .6s cubic-bezier(.4,0,.2,1)' }} />
      </div>
    </div>
  );
}

// ── Radial score ring ───────────────────────────────────────
function ScoreRing({ value, size = 116, color = 'var(--il-blue)' }) {
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

// ── Progress bar (jobs) ─────────────────────────────────────
function ProgressBar({ pct, color = 'var(--il-blue)' }) {
  return (
    <div style={{ height: 6, borderRadius: 3, background: 'var(--il-overlay)', overflow: 'hidden' }}>
      <div style={{ width: pct + '%', height: '100%', borderRadius: 3, background: color, transition: 'width .4s ease' }} />
    </div>
  );
}

// ── Segmented tabs ──────────────────────────────────────────
function SegTabs({ tabs, value, onChange }) {
  return (
    <div style={{ display: 'inline-flex', gap: 2, padding: 4, background: 'var(--il-bg-base)', borderRadius: 11, border: '1px solid var(--il-overlay)' }}>
      {tabs.map(t => {
        const active = t.id === value;
        return (
          <button key={t.id} onClick={() => onChange(t.id)} style={{
            display: 'inline-flex', alignItems: 'center', gap: 7, padding: '8px 15px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: active ? 'var(--il-surface-el)' : 'transparent',
            color: active ? 'var(--il-text-primary)' : 'var(--il-text-sec)',
            fontSize: 13.5, fontWeight: active ? 600 : 500, fontFamily: 'var(--il-font)',
            boxShadow: active ? '0 1px 2px rgba(0,0,0,.3)' : 'none', transition: 'all .15s',
          }}>
            {t.glyph && <Ico name={t.glyph} size={18} fill={active ? 1 : 0} />}
            {t.label}
            {t.count != null && <span style={{ fontSize: 11, fontWeight: 700, color: active ? 'var(--il-blue)' : 'var(--il-text-hint)' }}>{t.count}</span>}
          </button>
        );
      })}
    </div>
  );
}

// ── File-type tag ───────────────────────────────────────────
const TYPE_COLORS = { PDF: '#E0524D', DOCX: '#2993D1', MD: '#8C6FE0', HTML: '#E0A516', TXT: '#8C9199' };
function TypeTag({ type }) {
  const c = TYPE_COLORS[type] || '#8C9199';
  return (
    <span style={{ display: 'inline-flex', padding: '2px 7px', borderRadius: 5, fontSize: 10.5, fontWeight: 700, letterSpacing: '.3px',
      color: c, background: withAlpha(c, 0.16) }}>{type}</span>
  );
}

// ── Range slider with value bubble ──────────────────────────
function Slider({ label, labelEn, value, min, max, step, onChange, fmt, hintLeft, hintRight }) {
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

Object.assign(window, {
  withAlpha, Ico, STATUS, StatusDot, StatusBadge, DeltaChip, Sparkline, Card,
  LineChart, GaugeBar, ScoreRing, ProgressBar, SegTabs, TypeTag, Slider,
});
