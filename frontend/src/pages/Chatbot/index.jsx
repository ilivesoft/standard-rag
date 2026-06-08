import React, { useState, useRef, useEffect } from 'react';
import { Ico } from '../../components/ui/Ico.jsx';
import { Slider } from '../../components/ui/Slider.jsx';
import { StatusDot } from '../../components/ui/StatusDot.jsx';
import { fetchConversations, createConversation, queryStream, queryFull } from '../../api/client.js';

const selStyle = {
  background: 'var(--il-bg-base)', color: 'var(--il-text-primary)', border: '1px solid var(--il-overlay)',
  borderRadius: 8, padding: '6px 10px', fontSize: 12.5, fontFamily: 'var(--il-font-mono)', outline: 'none', cursor: 'pointer',
};

const MOCK_COLLECTIONS = [
  { id: 'default',   name: 'default' },
  { id: 'hr-policy', name: 'hr-policy' },
  { id: 'legal',     name: 'legal-contracts' },
  { id: 'eng-wiki',  name: 'eng-wiki' },
];

const CHAT_SEED = [
  { role: 'user', text: '환불 정책에서 예외가 적용되는 경우를 알려줘.' },
  {
    role: 'assistant',
    text: '환불 정책상 예외가 적용되는 경우는 크게 세 가지입니다. ① 디지털 콘텐츠를 이미 다운로드·이용한 경우 [1], ② 맞춤 제작 상품으로 재판매가 불가능한 경우 [2], ③ 구매 후 14일이 경과한 경우입니다 [1]. 다만 제품 하자가 확인되면 기간과 무관하게 환불이 가능합니다 [3].',
    sources: [
      { n: 1, doc: '환불정책_2026.md', page: 'p.2', chunk: '#28', score: 0.94, text: '디지털 콘텐츠를 다운로드하거나 스트리밍으로 이용을 시작한 경우 청약 철회가 제한됩니다. 또한 구매일로부터 14일이 지난 주문은…' },
      { n: 2, doc: '환불정책_2026.md', page: 'p.3', chunk: '#31', score: 0.89, text: '주문 제작(커스텀) 상품은 재판매가 불가능하므로 단순 변심에 의한 환불 대상에서 제외됩니다…' },
      { n: 3, doc: 'support_faq.txt', page: '—', chunk: '#12', score: 0.81, text: '제품에 하자가 있는 경우 구매 시점과 관계없이 전액 환불 또는 교환이 가능합니다…' },
    ],
  },
];

const MOCK_CONVERSATIONS = [
  { id: 'c1', title: '환불 정책 예외 조항', sub: '6개 메시지 · default', t: '방금 전', active: true },
  { id: 'c2', title: 'bge-m3 임베딩 차원', sub: '4개 메시지 · eng-wiki', t: '2시간 전' },
  { id: 'c3', title: '연차 산정 기준 문의', sub: '8개 메시지 · hr-policy', t: '어제' },
  { id: 'c4', title: 'NDA 위반 시 책임 범위', sub: '5개 메시지 · legal', t: '어제' },
  { id: 'c5', title: '하이브리드 검색 alpha 값', sub: '3개 메시지 · default', t: '6/5' },
  { id: 'c6', title: 'OCR 스캔 PDF 처리', sub: '7개 메시지 · default', t: '6/4' },
];

const STREAM_ANSWER = '하이브리드 검색의 alpha 값은 Vector 검색과 BM25 검색의 가중치를 조절합니다 [1]. alpha=1.0이면 Vector 검색만, alpha=0.0이면 BM25 검색만 사용하며, 기본값 0.5는 두 방식을 균등하게 결합합니다 [2]. 도메인 특화 용어가 많은 문서에는 BM25 비중을 높이고(alpha를 낮추고), 의미 기반 검색이 중요하면 alpha를 높이는 것을 권장합니다 [1].';
const STREAM_SOURCES = [
  { n: 1, doc: 'api_reference.html', page: '—', chunk: '#54', score: 0.92, text: 'HYBRID_ALPHA 파라미터는 0과 1 사이의 값으로 Vector 검색 점수와 BM25 점수를 가중 결합합니다…' },
  { n: 2, doc: 'README.md', page: '—', chunk: '#7', score: 0.86, text: 'alpha 0.0은 BM25 검색만, 0.5는 균등 가중, 1.0은 Vector 검색만 사용합니다…' },
];

function CitationChip({ n, onClick }) {
  return (
    <sup onClick={onClick} style={{
      cursor: 'pointer', fontSize: 10.5, fontWeight: 700, color: 'var(--il-blue)',
      background: 'var(--il-blue-soft)', padding: '1px 5px', borderRadius: 4, margin: '0 1px', verticalAlign: 'top',
    }}>{n}</sup>
  );
}

function renderAnswer(text, onCite) {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((p, i) => {
    const m = p.match(/^\[(\d+)\]$/);
    if (m) return <CitationChip key={i} n={m[1]} onClick={() => onCite(parseInt(m[1]))} />;
    return <span key={i}>{p}</span>;
  });
}

export function ChatbotPage() {
  const [msgs, setMsgs] = useState(CHAT_SEED);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [rightTab, setRightTab] = useState('sources');
  const [coll, setColl] = useState('default');
  const [topK, setTopK] = useState(10);
  const [topN, setTopN] = useState(3);
  const [alpha, setAlpha] = useState(0.5);
  const [activeConv, setActiveConv] = useState('c1');
  const [conversations, setConversations] = useState(MOCK_CONVERSATIONS);
  const scrollRef = useRef(null);

  const lastSources = [...msgs].reverse().find(m => m.role === 'assistant' && m.sources)?.sources || CHAT_SEED[1].sources;

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [msgs, streaming]);

  useEffect(() => {
    fetchConversations().then(data => {
      if (data && data.length > 0) {
        const mapped = data.map(c => ({
          id: c.id || c._id,
          title: c.title || c.name || '대화',
          sub: c.collection ? `· ${c.collection}` : '',
          t: c.updated_at ? new Date(c.updated_at).toLocaleDateString('ko-KR') : '',
        }));
        setConversations(mapped);
      }
    }).catch(() => {});
  }, []);

  function send() {
    if (!input.trim() || streaming) return;
    const q = input.trim();
    setInput('');
    setMsgs(m => [...m, { role: 'user', text: q }]);
    setStreaming(true);
    setMsgs(m => [...m, { role: 'assistant', text: '', streaming: true }]);

    const opts = { collection: coll, top_k: topK, top_n: topN, alpha };

    queryStream(q, opts,
      (token) => {
        setMsgs(m => {
          const copy = m.slice();
          const last = copy[copy.length - 1];
          copy[copy.length - 1] = { ...last, text: last.text + token, streaming: true };
          return copy;
        });
      },
      (payload) => {
        setMsgs(m => {
          const copy = m.slice();
          copy[copy.length - 1] = {
            role: 'assistant',
            text: payload.answer || copy[copy.length - 1].text,
            sources: payload.sources || [],
          };
          return copy;
        });
        setStreaming(false);
        setRightTab('sources');
      }
    ).catch(() => {
      // Fallback: simulate streaming with mock data
      const tokens = STREAM_ANSWER.split(/(\s+)/);
      let idx = 0;
      const iv = setInterval(() => {
        idx += 1;
        const partial = tokens.slice(0, idx).join('');
        setMsgs(m => {
          const copy = m.slice();
          copy[copy.length - 1] = { role: 'assistant', text: partial, streaming: idx < tokens.length };
          return copy;
        });
        if (idx >= tokens.length) {
          clearInterval(iv);
          setMsgs(m => {
            const copy = m.slice();
            copy[copy.length - 1] = { role: 'assistant', text: STREAM_ANSWER, sources: STREAM_SOURCES };
            return copy;
          });
          setStreaming(false);
          setRightTab('sources');
        }
      }, 38);
    });
  }

  function startNewChat() {
    setMsgs([]);
    setActiveConv(null);
    setInput('');
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '244px 1fr 320px', gap: 16, height: '100%', minHeight: 0 }}>
      {/* History panel */}
      <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <div style={{ padding: 14 }}>
          <button onClick={startNewChat} className="il-btn-follow" style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7 }}>
            <Ico name="add" size={18} /> 새 대화 <span style={{ opacity: .8, fontWeight: 400 }}>New chat</span>
          </button>
        </div>
        <div style={{ padding: '0 12px 4px', fontSize: 11, fontWeight: 600, color: 'var(--il-text-hint)', textTransform: 'uppercase', letterSpacing: '.4px' }}>대화 이력 · History</div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '4px 8px 12px' }}>
          {conversations.map(c => {
            const active = c.id === activeConv;
            return (
              <button key={c.id} onClick={() => setActiveConv(c.id)} style={{
                width: '100%', textAlign: 'left', border: 'none', cursor: 'pointer', borderRadius: 9, padding: '9px 10px', marginBottom: 2,
                background: active ? 'var(--il-blue-soft)' : 'transparent', display: 'flex', flexDirection: 'column', gap: 2, transition: 'background .12s',
              }}
              onMouseEnter={e => { if (!active) e.currentTarget.style.background = 'var(--il-hover)'; }}
              onMouseLeave={e => { if (!active) e.currentTarget.style.background = 'transparent'; }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Ico name="chat_bubble" size={14} style={{ color: active ? 'var(--il-blue)' : 'var(--il-icon)', flexShrink: 0 }} />
                  <span style={{ fontSize: 13, fontWeight: 500, color: active ? 'var(--il-text-primary)' : 'var(--il-text-sec)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.title}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', paddingLeft: 20 }}>
                  <span style={{ fontSize: 11, color: 'var(--il-text-hint)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.sub}</span>
                  <span style={{ fontSize: 11, color: 'var(--il-text-hint)', flexShrink: 0, marginLeft: 6 }}>{c.t}</span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Chat column */}
      <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '14px 18px', borderBottom: '1px solid var(--il-overlay)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ width: 32, height: 32, borderRadius: '50%', background: 'var(--il-grad-brand)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}>
              <Ico name="smart_toy" size={18} style={{ color: '#fff' }} />
            </span>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--il-text-primary)' }}>RAG 어시스턴트</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11.5, color: 'var(--il-text-hint)' }}>
                <StatusDot status="ready" /> 온라인 · llama3.2
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 11.5, color: 'var(--il-text-hint)' }}>컬렉션</span>
            <select value={coll} onChange={e => setColl(e.target.value)} style={selStyle}>
              {MOCK_COLLECTIONS.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
          </div>
        </header>

        {/* messages */}
        <div ref={scrollRef} style={{ flex: 1, overflowY: 'auto', padding: 24, display: 'flex', flexDirection: 'column', gap: 18 }}>
          {msgs.map((m, i) => m.role === 'user' ? (
            <div key={i} style={{ alignSelf: 'flex-end', maxWidth: '78%', display: 'flex', gap: 10, flexDirection: 'row-reverse' }}>
              <span style={{ width: 30, height: 30, borderRadius: '50%', background: 'var(--il-surface-el)', color: 'var(--il-text-sec)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, fontSize: 12, fontWeight: 700 }}>나</span>
              <div style={{ background: 'var(--il-blue)', color: '#fff', padding: '11px 15px', borderRadius: '14px 14px 4px 14px', fontSize: 14, lineHeight: 1.55 }}>{m.text}</div>
            </div>
          ) : (
            <div key={i} style={{ alignSelf: 'flex-start', maxWidth: '88%', display: 'flex', gap: 10 }}>
              <span style={{ width: 30, height: 30, borderRadius: '50%', background: 'var(--il-grad-brand)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <Ico name="smart_toy" size={16} style={{ color: '#fff' }} />
              </span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 9, minWidth: 0 }}>
                <div style={{ background: 'var(--il-surface-el)', color: 'var(--il-text-primary)', padding: '12px 16px', borderRadius: '14px 14px 14px 4px', fontSize: 14, lineHeight: 1.65 }}>
                  {renderAnswer(m.text, () => setRightTab('sources'))}
                  {m.streaming && <span style={{ display: 'inline-block', width: 7, height: 15, background: 'var(--il-blue)', marginLeft: 2, verticalAlign: 'text-bottom', animation: 'srBlink 1s step-end infinite' }} />}
                </div>
                {m.sources && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {m.sources.map(s => (
                      <button key={s.n} onClick={() => setRightTab('sources')} style={{
                        display: 'inline-flex', alignItems: 'center', gap: 6, padding: '5px 10px', borderRadius: 8, cursor: 'pointer',
                        background: 'var(--il-bg-base)', border: '1px solid var(--il-overlay)', color: 'var(--il-text-sec)', fontSize: 12,
                      }}>
                        <span style={{ fontSize: 10, fontWeight: 700, color: 'var(--il-blue)', background: 'var(--il-blue-soft)', width: 16, height: 16, borderRadius: 4, display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}>{s.n}</span>
                        <span style={{ fontFamily: 'var(--il-font-mono)', fontSize: 11.5 }}>{s.doc}</span>
                        <span style={{ color: 'var(--il-text-hint)' }}>· {(s.score * 100).toFixed(0)}%</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* composer */}
        <div style={{ padding: 16, borderTop: '1px solid var(--il-overlay)' }}>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 10, background: 'var(--il-bg-base)', border: '1px solid var(--il-overlay)', borderRadius: 14, padding: '8px 8px 8px 16px' }}>
            <textarea value={input} rows={1} placeholder="질문을 입력하세요…  Ask anything about your documents"
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
              style={{ flex: 1, resize: 'none', border: 'none', outline: 'none', background: 'transparent', color: 'var(--il-text-primary)', fontSize: 14, fontFamily: 'var(--il-font)', lineHeight: 1.5, padding: '6px 0', maxHeight: 120 }} />
            <button onClick={send} disabled={!input.trim() || streaming} style={{
              width: 40, height: 40, borderRadius: 11, border: 'none', flexShrink: 0, cursor: input.trim() && !streaming ? 'pointer' : 'default',
              background: input.trim() && !streaming ? 'var(--il-blue)' : 'var(--il-overlay)', color: '#fff', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', transition: 'background .15s',
            }}>
              <Ico name={streaming ? 'more_horiz' : 'arrow_upward'} size={20} />
            </button>
          </div>
          <div style={{ display: 'flex', gap: 14, marginTop: 9, fontSize: 11, color: 'var(--il-text-hint)' }}>
            <span><b style={{ color: 'var(--il-text-sec)' }}>Enter</b> 전송</span>
            <span><b style={{ color: 'var(--il-text-sec)' }}>Shift+Enter</b> 줄바꿈</span>
            <span style={{ marginLeft: 'auto', fontFamily: 'var(--il-font-mono)' }}>top_k {topK} · top_n {topN} · α {alpha.toFixed(1)}</span>
          </div>
        </div>
      </div>

      {/* Right panel: sources / settings */}
      <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <div style={{ display: 'flex', padding: 6, gap: 4, borderBottom: '1px solid var(--il-overlay)' }}>
          {[['sources', '출처', 'description'], ['settings', '검색 설정', 'tune']].map(([id, label, glyph]) => {
            const active = rightTab === id;
            return (
              <button key={id} onClick={() => setRightTab(id)} style={{
                flex: 1, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 6, padding: '9px', borderRadius: 9, border: 'none', cursor: 'pointer',
                background: active ? 'var(--il-surface-el)' : 'transparent', color: active ? 'var(--il-text-primary)' : 'var(--il-text-sec)', fontSize: 13, fontWeight: active ? 600 : 500, fontFamily: 'var(--il-font)',
              }}>
                <Ico name={glyph} size={17} fill={active ? 1 : 0} /> {label}
              </button>
            );
          })}
        </div>

        {rightTab === 'sources' ? (
          <div style={{ flex: 1, overflowY: 'auto', padding: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ fontSize: 11.5, color: 'var(--il-text-hint)' }}>{lastSources.length}개 출처 · 재순위 적용됨</div>
            {lastSources.map(s => (
              <div key={s.n} style={{ background: 'var(--il-bg-base)', border: '1px solid var(--il-overlay)', borderRadius: 11, padding: 13 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <span style={{ width: 20, height: 20, borderRadius: 6, background: 'var(--il-blue)', color: '#fff', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 700, flexShrink: 0 }}>{s.n}</span>
                  <span style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--il-text-primary)', fontFamily: 'var(--il-font-mono)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.doc}</span>
                </div>
                <p style={{ margin: '0 0 10px', fontSize: 12.5, lineHeight: 1.6, color: 'var(--il-text-sec)' }}>{s.text}</p>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, color: 'var(--il-text-hint)' }}>
                  {s.page !== '—' && <span style={{ background: 'var(--il-surface-el)', padding: '2px 7px', borderRadius: 5 }}>{s.page}</span>}
                  <span style={{ background: 'var(--il-surface-el)', padding: '2px 7px', borderRadius: 5, fontFamily: 'var(--il-font-mono)' }}>chunk {s.chunk}</span>
                  <span style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 5 }}>
                    score <b style={{ color: 'var(--il-success)', fontFamily: 'var(--il-font-mono)' }}>{s.score.toFixed(2)}</b>
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ flex: 1, overflowY: 'auto', padding: 18, display: 'flex', flexDirection: 'column', gap: 22 }}>
            <div>
              <div style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--il-text-hint)', textTransform: 'uppercase', letterSpacing: '.4px', marginBottom: 12 }}>검색 파라미터 · Retrieval</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                <Slider label="top_k" labelEn="초기 검색" value={topK} min={1} max={50} step={1} onChange={setTopK} hintLeft="1" hintRight="50" />
                <Slider label="top_n" labelEn="재순위 후" value={topN} min={1} max={10} step={1} onChange={setTopN} hintLeft="1" hintRight="10" />
                <Slider label="alpha" labelEn="하이브리드 가중" value={alpha} min={0} max={1} step={0.1} onChange={setAlpha} fmt={v => v.toFixed(1)} hintLeft="BM25 (0.0)" hintRight="Vector (1.0)" />
              </div>
            </div>
            <div style={{ background: 'var(--il-blue-soft)', borderRadius: 11, padding: 13, display: 'flex', gap: 10 }}>
              <Ico name="lightbulb" size={18} style={{ color: 'var(--il-blue)', flexShrink: 0 }} />
              <p style={{ margin: 0, fontSize: 12, lineHeight: 1.6, color: 'var(--il-text-sec)' }}>
                α=0.5는 Vector와 BM25를 균등 결합합니다. 전문 용어가 많으면 α를 낮춰 BM25 비중을 높여 보세요.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ChatbotPage;
