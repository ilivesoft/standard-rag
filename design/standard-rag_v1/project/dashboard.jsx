// dashboard.jsx — 메인 dashboard for Standard RAG
const { useState: useStateDash } = React;

function KpiCard({ k }) {
  const Ico = window.Ico, Sparkline = window.Sparkline, DeltaChip = window.DeltaChip;
  const display = k.id === 'lat' ? k.value.toFixed(2) : window.fmt(k.value);
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

function DashboardPage({ onNav }) {
  const { Card, LineChart, GaugeBar, ScoreRing, ProgressBar, StatusDot, Ico } = window;
  const totalDocs = window.COLLECTIONS.reduce((a, c) => a + c.docs, 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {/* KPI row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: 14 }}>
        {window.KPIS.map(k => <KpiCard key={k.id} k={k} />)}
      </div>

      {/* Trend + RAGAS */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.7fr 1fr', gap: 18 }}>
        <Card title="질의 응답 추이" titleEn="Daily Q&A · last 14 days"
          right={<div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--il-text-sec)' }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: 'var(--il-blue)' }} />질의 건수</div>}>
          <LineChart data={window.QA_TREND} />
          <div style={{ display: 'flex', gap: 28, marginTop: 12, paddingTop: 14, borderTop: '1px solid var(--il-overlay)' }}>
            {[['오늘','388건','var(--il-blue)'],['주간 평균','296건','var(--il-text-primary)'],['최고치','388건','var(--il-success)']].map(([a,b,c]) => (
              <div key={a} style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <span style={{ fontSize: 11.5, color: 'var(--il-text-hint)' }}>{a}</span>
                <span style={{ fontSize: 16, fontWeight: 700, color: c }}>{b}</span>
              </div>
            ))}
          </div>
        </Card>

        <Card title="RAGAS 평가" titleEn="Quality score"
          right={<button onClick={() => onNav('rag', 'eval')} className="il-btn-ghost" style={{ padding: '6px 11px', fontSize: 12 }}>평가 실행</button>}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 18, marginBottom: 16 }}>
            <ScoreRing value={window.RAGAS_OVERALL} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: 12.5, color: 'var(--il-text-sec)' }}>최근 배치 · 24건</span>
              <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--il-success)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                <Ico name="trending_up" size={16} /> 양호 (Good)
              </span>
              <span style={{ fontSize: 11.5, color: 'var(--il-text-hint)', marginTop: 2 }}>2시간 전 실행</span>
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 13 }}>
            {window.RAGAS.map((m, i) => (
              <GaugeBar key={m.en} label={m.kr} labelEn={m.en} value={m.v}
                color={['var(--il-blue)','var(--il-success)','var(--il-warning)','#8C6FE0'][i]} />
            ))}
          </div>
        </Card>
      </div>

      {/* Collections + Jobs + System */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr 1fr', gap: 18 }}>
        {/* Collections */}
        <Card title="컬렉션 현황" titleEn="Collections"
          right={<button onClick={() => onNav('rag', 'collections')} className="il-btn-ghost" style={{ padding: '6px 11px', fontSize: 12 }}>관리</button>}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div style={{ display: 'flex', height: 9, borderRadius: 5, overflow: 'hidden', background: 'var(--il-overlay)' }}>
              {window.COLLECTIONS.map(c => <div key={c.id} title={c.name} style={{ width: (c.docs / totalDocs * 100) + '%', background: c.color }} />)}
            </div>
            {window.COLLECTIONS.map(c => (
              <div key={c.id} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ width: 9, height: 9, borderRadius: 3, background: c.color, flexShrink: 0 }} />
                <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--il-text-primary)', flex: 1, fontFamily: 'var(--il-font-mono)' }}>{c.name}</span>
                <span style={{ fontSize: 12.5, color: 'var(--il-text-sec)' }}>{c.docs} docs</span>
                <span style={{ fontSize: 12.5, color: 'var(--il-text-hint)', width: 64, textAlign: 'right' }}>{window.fmt(c.chunks)} chunks</span>
              </div>
            ))}
          </div>
        </Card>

        {/* Jobs */}
        <Card title="인덱싱 진행률" titleEn="Active jobs"
          right={<span style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--il-blue)', background: 'var(--il-blue-soft)', padding: '3px 8px', borderRadius: 20 }}>{window.JOBS.length} 처리 중</span>}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 15 }}>
            {window.JOBS.map(j => (
              <div key={j.id} style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Ico name="sync" size={15} style={{ color: 'var(--il-blue)', animation: 'srSpin 1.8s linear infinite' }} />
                  <span style={{ fontSize: 12.5, fontWeight: 500, color: 'var(--il-text-primary)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{j.name}</span>
                </div>
                <ProgressBar pct={j.pct} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--il-text-hint)' }}>
                  <span>{j.stage} · {j.stageEn}</span><span>{j.pct}% · ETA {j.eta}</span>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* System */}
        <Card title="시스템 상태" titleEn="Components">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {window.SYSTEM.map(s => (
              <div key={s.en} style={{ display: 'flex', alignItems: 'center', gap: 11, padding: '8px 0' }}>
                <span style={{ width: 30, height: 30, borderRadius: 8, background: 'var(--il-surface-el)', color: 'var(--il-icon)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <Ico name={s.glyph} size={17} />
                </span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--il-text-primary)' }}>{s.kr} <span style={{ fontSize: 11, color: 'var(--il-text-hint)', fontWeight: 400 }}>{s.en}</span></div>
                  {s.note && <div style={{ fontSize: 11, color: 'var(--il-text-hint)', fontFamily: 'var(--il-font-mono)' }}>{s.note}</div>}
                </div>
                <StatusDot status={s.status} pulse={s.status === 'ready'} />
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Activity */}
      <Card title="최근 활동" titleEn="Recent activity"
        right={<button className="il-btn-ghost" style={{ padding: '6px 11px', fontSize: 12 }}>전체 보기</button>}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2px 32px' }}>
          {window.ACTIVITY.map((a, i) => {
            const tone = { blue: '#2993D1', green: '#29B473', amber: '#E0A516', red: '#FF2D2D' }[a.tone];
            return (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '11px 0', borderBottom: i < window.ACTIVITY.length - (window.ACTIVITY.length % 2 === 0 ? 2 : 1) ? '1px solid rgba(56,59,67,0.5)' : 'none' }}>
                <span style={{ width: 32, height: 32, borderRadius: '50%', background: window.withAlpha(tone, 0.15), color: tone, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <Ico name={a.glyph} size={17} />
                </span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--il-text-primary)' }}>{a.kr}</span>
                  <span style={{ fontSize: 12, color: 'var(--il-text-hint)', marginLeft: 7 }}>{a.en}</span>
                  <div style={{ fontSize: 12, color: 'var(--il-text-sec)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{a.sub}</div>
                </div>
                <span style={{ fontSize: 11.5, color: 'var(--il-text-hint)', flexShrink: 0 }}>{a.t}</span>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}

Object.assign(window, { DashboardPage });
