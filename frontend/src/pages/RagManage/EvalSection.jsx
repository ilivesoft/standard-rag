import React, { useState } from 'react';
import { Ico } from '../../components/ui/Ico.jsx';
import { ScoreRing } from '../../components/ui/ScoreRing.jsx';
import { GaugeBar } from '../../components/ui/GaugeBar.jsx';
import { queryFull, runEvaluateBatch } from '../../api/client.js';
import { useToast } from '../../components/ui/Toast.jsx';

const DEFAULT_QUESTIONS = [
  '환불 정책의 예외 조항은?',
  'bge-m3 임베딩 차원 수는?',
  '연차 산정 기준을 설명해줘',
  'NDA 위반 시 책임 범위는?',
  '청크 크기 기본값과 겹침은?',
];

const MOCK_ROWS = DEFAULT_QUESTIONS.map(q => ({ q, f: 0, r: 0, p: 0, c: 0, evaluated: false }));
const MOCK_RAGAS = [
  { kr: '충실도',      en: 'Faithfulness',     v: 0 },
  { kr: '답변 관련성', en: 'Answer relevancy',  v: 0 },
  { kr: '문맥 정밀도', en: 'Context precision', v: 0 },
  { kr: '문맥 재현율', en: 'Context recall',    v: 0 },
];

const cols = [
  ['질문 Question', 'q', 'left'],
  ['충실도', 'f'],
  ['관련성', 'r'],
  ['정밀도', 'p'],
  ['재현율', 'c'],
];

export function EvalSection() {
  const toast = useToast();
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState(MOCK_ROWS);
  const [overall, setOverall] = useState(0);
  const [metrics, setMetrics] = useState(MOCK_RAGAS);
  const [newQ, setNewQ] = useState('');
  const [hasRun, setHasRun] = useState(false);

  function addQuestion() {
    const q = newQ.trim();
    if (!q) return;
    setResults(prev => [...prev, { q, f: 0, r: 0, p: 0, c: 0, evaluated: false }]);
    setNewQ('');
  }

  function removeQuestion(idx) {
    setResults(prev => prev.filter((_, i) => i !== idx));
  }

  async function handleRunEval() {
    if (results.length === 0) { toast('평가할 질문이 없습니다.', 'info'); return; }
    setRunning(true);
    try {
      // 1단계: 각 질문을 RAG 파이프라인으로 실행
      const items = [];
      for (const row of results) {
        try {
          const res = await queryFull(row.q, { top_k: 10, top_n: 3 });
          const answer = res?.answer || '';
          const contexts = (res?.sources || []).map(s => s.content || s.text || s.page_content || '').filter(Boolean);
          items.push({ question: row.q, answer, contexts });
        } catch {
          items.push({ question: row.q, answer: '', contexts: [] });
        }
      }

      // 2단계: RAGAS 평가 실행
      const data = await runEvaluateBatch(items);

      if (data?.results) {
        const newResults = data.results.map((r, i) => ({
          q: results[i]?.q || r.question || '',
          f: r.faithfulness ?? 0,
          r: r.answer_relevancy ?? 0,
          p: r.context_precision ?? 0,
          c: r.context_recall ?? 0,
          evaluated: true,
        }));
        setResults(newResults);

        const avg = arr => arr.reduce((a, b) => a + b, 0) / (arr.length || 1);
        const newOverall = avg(data.results.map(r =>
          ((r.faithfulness ?? 0) + (r.answer_relevancy ?? 0) + (r.context_precision ?? 0)) / 3
        ));
        setOverall(newOverall);
        setMetrics([
          { kr: '충실도',      en: 'Faithfulness',     v: data.faithfulness ?? avg(data.results.map(r => r.faithfulness ?? 0)) },
          { kr: '답변 관련성', en: 'Answer relevancy',  v: data.answer_relevancy ?? avg(data.results.map(r => r.answer_relevancy ?? 0)) },
          { kr: '문맥 정밀도', en: 'Context precision', v: data.context_precision ?? avg(data.results.map(r => r.context_precision ?? 0)) },
          { kr: '문맥 재현율', en: 'Context recall',    v: data.context_recall ?? avg(data.results.map(r => r.context_recall ?? 0)) },
        ]);
        setHasRun(true);
        toast(`${newResults.length}건 평가 완료`, 'success');
      }
    } catch (e) {
      toast('평가 실행 중 오류가 발생했습니다.', 'error');
    }
    setRunning(false);
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 16 }}>
        <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', padding: 20, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
          <ScoreRing value={overall} size={140} color="var(--il-blue)" />
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 13, color: 'var(--il-text-sec)' }}>
              {hasRun ? `최근 배치 평가 · ${results.length}건` : '평가를 실행하면 결과가 표시됩니다'}
            </div>
            <div style={{ fontSize: 12, color: 'var(--il-text-hint)', marginTop: 2 }}>RAGAS</div>
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

      {/* 커스텀 질문 추가 */}
      <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', padding: '15px 18px' }}>
        <div style={{ fontSize: 13.5, fontWeight: 600, marginBottom: 12 }}>질문 추가 <span style={{ fontSize: 12, color: 'var(--il-text-hint)', fontWeight: 400 }}>Add question</span></div>
        <div style={{ display: 'flex', gap: 10 }}>
          <input
            value={newQ}
            onChange={e => setNewQ(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && addQuestion()}
            placeholder="평가할 질문을 입력하세요..."
            style={{
              flex: 1, background: 'var(--il-bg-base)', border: '1px solid var(--il-overlay)',
              borderRadius: 8, padding: '9px 14px', fontSize: 13.5, color: 'var(--il-text-primary)',
              fontFamily: 'var(--il-font)', outline: 'none',
            }}
          />
          <button className="il-btn-follow" onClick={addQuestion} style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}>
            <Ico name="add" size={18} /> 추가
          </button>
        </div>
      </div>

      {/* 평가 결과 테이블 */}
      <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', overflow: 'hidden' }}>
        <div style={{ padding: '15px 18px', borderBottom: '1px solid var(--il-overlay)', fontSize: 14.5, fontWeight: 700 }}>
          평가 질문 목록 <span style={{ fontSize: 12, color: 'var(--il-text-hint)', fontWeight: 400 }}>
            {hasRun ? `Results · ${results.length}` : `Questions · ${results.length}`}
          </span>
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ fontSize: 11.5, color: 'var(--il-text-hint)' }}>
              {cols.map(([h,, align], i) => (
                <th key={i} style={{ padding: '11px 18px', fontWeight: 600, textAlign: align === 'left' ? 'left' : 'right' }}>{h}</th>
              ))}
              <th style={{ padding: '11px 18px', width: 40 }}></th>
            </tr>
          </thead>
          <tbody>
            {results.map((r, i) => (
              <tr key={i} style={{ borderTop: '1px solid rgba(56,59,67,0.5)' }}>
                <td style={{ padding: '12px 18px', fontSize: 13, color: 'var(--il-text-primary)' }}>{r.q}</td>
                {['f', 'r', 'p', 'c'].map(k => (
                  <td key={k} style={{ padding: '12px 18px', textAlign: 'right' }}>
                    {r.evaluated
                      ? <span style={{
                          fontSize: 12.5, fontWeight: 600, fontFamily: 'var(--il-font-mono)',
                          color: r[k] >= 0.9 ? 'var(--il-success)' : r[k] >= 0.75 ? 'var(--il-text-primary)' : 'var(--il-warning)'
                        }}>{r[k].toFixed(2)}</span>
                      : <span style={{ fontSize: 12, color: 'var(--il-text-hint)' }}>—</span>
                    }
                  </td>
                ))}
                <td style={{ padding: '12px 18px', textAlign: 'right' }}>
                  <button
                    className="il-btn-icon"
                    style={{ width: 28, height: 28 }}
                    onClick={() => removeQuestion(i)}>
                    <Ico name="close" size={16} style={{ color: 'var(--il-text-hint)' }} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default EvalSection;
