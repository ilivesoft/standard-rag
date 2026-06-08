import React, { useState } from 'react';
import { Ico } from '../../components/ui/Ico.jsx';
import { ScoreRing } from '../../components/ui/ScoreRing.jsx';
import { GaugeBar } from '../../components/ui/GaugeBar.jsx';
import { runEvaluate } from '../../api/client.js';

const RAGAS = [
  { kr: '충실도',     en: 'Faithfulness',     v: 0.91 },
  { kr: '답변 관련성', en: 'Answer relevancy', v: 0.88 },
  { kr: '문맥 정밀도', en: 'Context precision', v: 0.85 },
  { kr: '문맥 재현율', en: 'Context recall',   v: 0.93 },
];
const RAGAS_OVERALL = 0.89;

const EVAL_ROWS = [
  { q: '환불 정책의 예외 조항은?',    f: 0.94, r: 0.91, p: 0.88, c: 0.95 },
  { q: 'bge-m3 임베딩 차원 수는?',    f: 0.88, r: 0.90, p: 0.82, c: 0.91 },
  { q: '연차 산정 기준을 설명해줘',   f: 0.92, r: 0.86, p: 0.84, c: 0.93 },
  { q: 'NDA 위반 시 책임 범위는?',    f: 0.90, r: 0.85, p: 0.87, c: 0.89 },
  { q: '청크 크기 기본값과 겹침은?',  f: 0.96, r: 0.93, p: 0.90, c: 0.97 },
];

const cols = [
  ['질문 Question', 'q', 'left'],
  ['충실도', 'f'],
  ['관련성', 'r'],
  ['정밀도', 'p'],
  ['재현율', 'c'],
];

export function EvalSection() {
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState(EVAL_ROWS);
  const [overall, setOverall] = useState(RAGAS_OVERALL);
  const [metrics, setMetrics] = useState(RAGAS);

  async function handleRunEval() {
    setRunning(true);
    try {
      const questions = EVAL_ROWS.map(r => r.q);
      const data = await runEvaluate(questions);
      if (data && data.results) {
        setResults(data.results);
        if (data.overall) setOverall(data.overall);
        if (data.metrics) setMetrics(data.metrics);
      }
    } catch {
      // Keep mock data on failure
    }
    setRunning(false);
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 16 }}>
        <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', padding: 20, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
          <ScoreRing value={overall} size={140} color="var(--il-blue)" />
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 13, color: 'var(--il-text-sec)' }}>최근 배치 평가 · 24건</div>
            <div style={{ fontSize: 12, color: 'var(--il-text-hint)', marginTop: 2 }}>2시간 전 · RAGAS</div>
          </div>
          <button
            className="il-btn-follow"
            onClick={handleRunEval}
            disabled={running}
            style={{ width: '100%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 7 }}>
            {running
              ? <><Ico name="sync" size={18} style={{ animation: 'srSpin 1s linear infinite' }} /> 평가 중...</>
              : <><Ico name="play_arrow" size={19} fill={1} /> 평가 실행 Run eval</>
            }
          </button>
        </div>
        <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', padding: 20 }}>
          <div style={{ fontSize: 14.5, fontWeight: 700, marginBottom: 18 }}>메트릭 평균 <span style={{ fontSize: 12, color: 'var(--il-text-hint)', fontWeight: 400 }}>Metric averages</span></div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {metrics.map((m, i) => (
              <GaugeBar key={m.en} label={m.kr} labelEn={m.en} value={m.v}
                color={['var(--il-blue)', 'var(--il-success)', 'var(--il-warning)', '#8C6FE0'][i]} />
            ))}
          </div>
        </div>
      </div>
      <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', overflow: 'hidden' }}>
        <div style={{ padding: '15px 18px', borderBottom: '1px solid var(--il-overlay)', fontSize: 14.5, fontWeight: 700 }}>
          평가 결과 <span style={{ fontSize: 12, color: 'var(--il-text-hint)', fontWeight: 400 }}>Results · {results.length}</span>
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ fontSize: 11.5, color: 'var(--il-text-hint)' }}>
              {cols.map(([h,, align], i) => (
                <th key={i} style={{ padding: '11px 18px', fontWeight: 600, textAlign: align === 'left' ? 'left' : 'right' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {results.map((r, i) => (
              <tr key={i} style={{ borderTop: '1px solid rgba(56,59,67,0.5)' }}>
                <td style={{ padding: '12px 18px', fontSize: 13, color: 'var(--il-text-primary)' }}>{r.q}</td>
                {['f', 'r', 'p', 'c'].map(k => (
                  <td key={k} style={{ padding: '12px 18px', textAlign: 'right' }}>
                    <span style={{
                      fontSize: 12.5, fontWeight: 600, fontFamily: 'var(--il-font-mono)',
                      color: r[k] >= 0.9 ? 'var(--il-success)' : r[k] >= 0.85 ? 'var(--il-text-primary)' : 'var(--il-warning)'
                    }}>{r[k].toFixed(2)}</span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default EvalSection;
