// chatbot.jsx — 챗봇 screen with streaming, sources, history, params
const { useState: useStateChat, useRef: useRefChat, useEffect: useEffectChat } = React;

function CitationChip({ n, onClick }) {
  return (
    <sup onClick={onClick} style={{
      cursor: 'pointer', fontSize: 10.5, fontWeight: 700, color: 'var(--il-blue)',
      background: 'var(--il-blue-soft)', padding: '1px 5px', borderRadius: 4, margin: '0 1px', verticalAlign: 'top',
    }}>{n}</sup>
  );
}

// render text with [n] turned into citation chips
function renderAnswer(text, onCite) {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((p, i) => {
    const m = p.match(/^\[(\d+)\]$/);
    if (m) return <CitationChip key={i} n={m[1]} onClick={() => onCite(parseInt(m[1]))} />;
    return <span key={i}>{p}</span>;
  });
}

function ChatbotPage() {
  const { Ico, Slider, StatusDot } = window;
  const [msgs, setMsgs] = useStateChat(window.CHAT_SEED);
  const [input, setInput] = useStateChat('');
  const [streaming, setStreaming] = useStateChat(false);
  const [rightTab, setRightTab] = useStateChat('sources'); // 'sources' | 'settings'
  const [coll, setColl] = useStateChat('default');
  const [topK, setTopK] = useStateChat(10);
  const [topN, setTopN] = useStateChat(3);
  const [alpha, setAlpha] = useStateChat(0.5);
  const [activeConv, setActiveConv] = useStateChat('c1');
  const scrollRef = useRefChat(null);

  // latest sources shown in right panel
  const lastSources = [...msgs].reverse().find(m => m.role === 'assistant' && m.sources)?.sources || window.CHAT_SEED[1].sources;

  useEffectChat(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [msgs, streaming]);

  function send() {
    if (!input.trim() || streaming) return;
    const q = input.trim();
    setInput('');
    setMsgs(m => [...m, { role: 'user', text: q }]);
    setStreaming(true);
    // simulate streaming answer
    const full = window.STREAM_ANSWER;
    const tokens = full.split(/(\s+)/);
    let idx = 0;
    setMsgs(m => [...m, { role: 'assistant', text: '', streaming: true }]);
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
          copy[copy.length - 1] = { role: 'assistant', text: full, sources: window.STREAM_SOURCES };
          return copy;
        });
        setStreaming(false);
        setRightTab('sources');
      }
    }, 38);
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '244px 1fr 320px', gap: 16, height: '100%', minHeight: 0 }}>
      {/* History panel */}
      <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <div style={{ padding: 14 }}>
          <button className="il-btn-follow" style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7 }}>
            <Ico name="add" size={18} /> 새 대화 <span style={{ opacity: .8, fontWeight: 400 }}>New chat</span>
          </button>
        </div>
        <div style={{ padding: '0 12px 4px', fontSize: 11, fontWeight: 600, color: 'var(--il-text-hint)', textTransform: 'uppercase', letterSpacing: '.4px' }}>대화 이력 · History</div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '4px 8px 12px' }}>
          {window.CONVERSATIONS.map(c => {
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
              {window.COLLECTIONS.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
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
          {[['sources','출처','description'],['settings','검색 설정','tune']].map(([id, label, glyph]) => {
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

const selStyle = {
  background: 'var(--il-bg-base)', color: 'var(--il-text-primary)', border: '1px solid var(--il-overlay)',
  borderRadius: 8, padding: '6px 10px', fontSize: 12.5, fontFamily: 'var(--il-font-mono)', outline: 'none', cursor: 'pointer',
};

Object.assign(window, { ChatbotPage, selStyle });
