// ragmanage.jsx — RAG 관리 screen: documents / collections / settings / eval
const { useState: useStateRag } = React;

function UploadZone() {
  const { Ico } = window;
  const [drag, setDrag] = useStateRag(false);
  return (
    <div
      onDragOver={e => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={e => { e.preventDefault(); setDrag(false); }}
      style={{
        border: '2px dashed ' + (drag ? 'var(--il-blue)' : 'var(--il-overlay)'),
        background: drag ? 'var(--il-blue-soft)' : 'var(--il-bg-base)',
        borderRadius: 14, padding: '28px 24px', display: 'flex', alignItems: 'center', gap: 20, transition: 'all .15s', cursor: 'pointer',
      }}>
      <span style={{ width: 52, height: 52, borderRadius: 13, background: 'var(--il-blue-soft)', color: 'var(--il-blue)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
        <Ico name="cloud_upload" size={28} />
      </span>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--il-text-primary)' }}>파일을 끌어다 놓거나 클릭하여 업로드</div>
        <div style={{ fontSize: 12.5, color: 'var(--il-text-hint)', marginTop: 3 }}>Drag &amp; drop or browse · PDF, DOCX, TXT, MD, HTML · 최대 100MB · OCR 지원</div>
      </div>
      <button className="il-btn-follow" style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}>
        <Ico name="folder_open" size={18} /> 파일 선택
      </button>
    </div>
  );
}

function DocsSection() {
  const { Ico, TypeTag, StatusBadge, ProgressBar } = window;
  const [coll, setColl] = useStateRag('all');
  const docs = coll === 'all' ? window.DOCS : window.DOCS.filter(d => d.coll === coll);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <UploadZone />
      {/* in-flight */}
      <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', padding: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 13 }}>
          <Ico name="sync" size={17} style={{ color: 'var(--il-blue)', animation: 'srSpin 1.8s linear infinite' }} />
          <span style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--il-text-primary)' }}>인덱싱 진행 중</span>
          <span style={{ fontSize: 12, color: 'var(--il-text-hint)' }}>Indexing · {window.JOBS.length}건</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
          {window.JOBS.map(j => (
            <div key={j.id} style={{ background: 'var(--il-bg-base)', borderRadius: 11, padding: 13, display: 'flex', flexDirection: 'column', gap: 8 }}>
              <span style={{ fontSize: 12.5, fontWeight: 500, color: 'var(--il-text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{j.name}</span>
              <ProgressBar pct={j.pct} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--il-text-hint)' }}>
                <span>{j.stage} · {j.stageEn}</span><span style={{ fontFamily: 'var(--il-font-mono)' }}>{j.pct}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* documents table */}
      <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', overflow: 'hidden' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '15px 18px', borderBottom: '1px solid var(--il-overlay)' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700 }}>문서 목록</h3>
            <span style={{ fontSize: 12, color: 'var(--il-text-hint)' }}>Documents · {docs.length}</span>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <select value={coll} onChange={e => setColl(e.target.value)} style={window.selStyle}>
              <option value="all">전체 컬렉션</option>
              {window.COLLECTIONS.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7, background: 'var(--il-bg-base)', border: '1px solid var(--il-overlay)', borderRadius: 8, padding: '0 11px' }}>
              <Ico name="search" size={17} style={{ color: 'var(--il-icon)' }} />
              <input placeholder="파일 검색" style={{ border: 'none', outline: 'none', background: 'transparent', color: 'var(--il-text-primary)', fontSize: 12.5, padding: '7px 0', width: 110, fontFamily: 'var(--il-font)' }} />
            </div>
          </div>
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ fontSize: 11.5, color: 'var(--il-text-hint)', textAlign: 'left' }}>
              {['파일명 File','타입','컬렉션','상태','청크','토큰','업로드','']
                .map((h, i) => <th key={i} style={{ padding: '11px 18px', fontWeight: 600, textAlign: i >= 4 && i <= 5 ? 'right' : 'left' }}>{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {docs.map((d, i) => (
              <tr key={i} style={{ borderTop: '1px solid rgba(56,59,67,0.5)', transition: 'background .12s' }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--il-hover)'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                <td style={{ padding: '12px 18px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                    <Ico name="description" size={17} style={{ color: 'var(--il-icon)' }} />
                    <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--il-text-primary)' }}>{d.name}</span>
                  </div>
                </td>
                <td style={{ padding: '12px 18px' }}><TypeTag type={d.type} /></td>
                <td style={{ padding: '12px 18px', fontSize: 12.5, color: 'var(--il-text-sec)', fontFamily: 'var(--il-font-mono)' }}>{d.coll}</td>
                <td style={{ padding: '12px 18px' }}><StatusBadge status={d.status} /></td>
                <td style={{ padding: '12px 18px', fontSize: 12.5, color: 'var(--il-text-sec)', textAlign: 'right', fontFamily: 'var(--il-font-mono)' }}>{d.chunks || '—'}</td>
                <td style={{ padding: '12px 18px', fontSize: 12.5, color: 'var(--il-text-sec)', textAlign: 'right', fontFamily: 'var(--il-font-mono)' }}>{d.tokens}</td>
                <td style={{ padding: '12px 18px', fontSize: 12, color: 'var(--il-text-hint)', fontFamily: 'var(--il-font-mono)' }}>{d.date}</td>
                <td style={{ padding: '12px 18px', textAlign: 'right' }}>
                  <button className="il-btn-icon" style={{ width: 30, height: 30 }}><Ico name="more_vert" size={18} /></button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CollectionsSection() {
  const { Ico, fmt } = window;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700 }}>컬렉션 <span style={{ fontSize: 13, color: 'var(--il-text-hint)', fontWeight: 400 }}>Collections · {window.COLLECTIONS.length}</span></h3>
        </div>
        <button className="il-btn-follow" style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}>
          <Ico name="create_new_folder" size={18} /> 새 컬렉션
        </button>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>
        {window.COLLECTIONS.map(c => (
          <div key={c.id} style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', padding: 18, display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 11 }}>
              <span style={{ width: 40, height: 40, borderRadius: 11, background: window.withAlpha(c.color, 0.16), color: c.color, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <Ico name="database" size={21} />
              </span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 14.5, fontWeight: 600, color: 'var(--il-text-primary)', fontFamily: 'var(--il-font-mono)', overflow: 'hidden', textOverflow: 'ellipsis' }}>{c.name}</div>
                {c.id === 'default' && <span style={{ fontSize: 10.5, fontWeight: 600, color: 'var(--il-blue)', background: 'var(--il-blue-soft)', padding: '1px 6px', borderRadius: 4 }}>기본</span>}
              </div>
              <button className="il-btn-icon" style={{ width: 30, height: 30 }}><Ico name="more_vert" size={18} /></button>
            </div>
            <div style={{ display: 'flex', gap: 18 }}>
              <div>
                <div style={{ fontSize: 21, fontWeight: 800, color: 'var(--il-text-primary)', letterSpacing: '-.5px' }}>{fmt(c.docs)}</div>
                <div style={{ fontSize: 11.5, color: 'var(--il-text-hint)' }}>문서 docs</div>
              </div>
              <div style={{ width: 1, background: 'var(--il-overlay)' }} />
              <div>
                <div style={{ fontSize: 21, fontWeight: 800, color: 'var(--il-text-primary)', letterSpacing: '-.5px' }}>{fmt(c.chunks)}</div>
                <div style={{ fontSize: 11.5, color: 'var(--il-text-hint)' }}>청크 chunks</div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="il-btn-pill" style={{ flex: 1, justifyContent: 'center', fontSize: 12.5 }}><Ico name="visibility" size={16} /> 보기</button>
              <button className="il-btn-pill" style={{ flex: 1, justifyContent: 'center', fontSize: 12.5 }}><Ico name="delete" size={16} /> 삭제</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SettingsSection() {
  const { Slider, Ico } = window;
  const [chunkSize, setChunkSize] = useStateRag(512);
  const [overlap, setOverlap] = useStateRag(64);
  const [device, setDevice] = useStateRag('cpu');
  const [topK, setTopK] = useStateRag(10);
  const [topN, setTopN] = useStateRag(3);
  const [alpha, setAlpha] = useStateRag(0.5);

  const block = (title, en, glyph, children) => (
    <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', padding: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 18 }}>
        <span style={{ width: 32, height: 32, borderRadius: 9, background: 'var(--il-blue-soft)', color: 'var(--il-blue)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}><Ico name={glyph} size={18} /></span>
        <div><div style={{ fontSize: 14.5, fontWeight: 700 }}>{title}</div><div style={{ fontSize: 11.5, color: 'var(--il-text-hint)' }}>{en}</div></div>
      </div>
      {children}
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {block('청킹', 'Chunking', 'content_cut', (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <Slider label="chunk_size" labelEn="청크 크기(토큰)" value={chunkSize} min={128} max={1024} step={64} onChange={setChunkSize} hintLeft="128" hintRight="1024" />
            <Slider label="chunk_overlap" labelEn="겹침(토큰)" value={overlap} min={0} max={256} step={16} onChange={setOverlap} hintLeft="0" hintRight="256" />
          </div>
        ))}
        {block('임베딩', 'Embedding', 'polyline', (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <Field label="embedding_model" value="BAAI/bge-m3" mono />
            <div>
              <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 8 }}>device <span style={{ fontSize: 11.5, color: 'var(--il-text-hint)' }}>실행 디바이스</span></div>
              <div style={{ display: 'inline-flex', gap: 2, padding: 4, background: 'var(--il-bg-base)', borderRadius: 10, border: '1px solid var(--il-overlay)' }}>
                {['cpu','cuda'].map(d => (
                  <button key={d} onClick={() => setDevice(d)} style={{ padding: '7px 20px', borderRadius: 7, border: 'none', cursor: 'pointer', fontSize: 13, fontFamily: 'var(--il-font-mono)',
                    background: device === d ? 'var(--il-surface-el)' : 'transparent', color: device === d ? 'var(--il-text-primary)' : 'var(--il-text-sec)', fontWeight: device === d ? 600 : 400 }}>{d}</button>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
      {block('하이브리드 검색', 'Retrieval · Vector + BM25 + RRF', 'search', (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 28 }}>
          <Slider label="top_k" labelEn="초기 검색 결과" value={topK} min={1} max={50} step={1} onChange={setTopK} hintLeft="1" hintRight="50" />
          <Slider label="top_n" labelEn="재순위 후 결과" value={topN} min={1} max={10} step={1} onChange={setTopN} hintLeft="1" hintRight="10" />
          <Slider label="alpha" labelEn="하이브리드 가중치" value={alpha} min={0} max={1} step={0.1} onChange={setAlpha} fmt={v => v.toFixed(1)} hintLeft="BM25" hintRight="Vector" />
        </div>
      ))}
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
        <button className="il-btn-ghost">기본값 복원</button>
        <button className="il-btn-follow" style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}><Ico name="save" size={18} /> 설정 저장</button>
      </div>
    </div>
  );
}

function Field({ label, value, mono }) {
  return (
    <div>
      <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 8 }}>{label}</div>
      <div style={{ background: 'var(--il-bg-base)', border: '1px solid var(--il-overlay)', borderRadius: 8, padding: '10px 13px', fontSize: 13, color: 'var(--il-text-primary)', fontFamily: mono ? 'var(--il-font-mono)' : 'var(--il-font)' }}>{value}</div>
    </div>
  );
}

function EvalSection() {
  const { Ico, ScoreRing, GaugeBar } = window;
  const cols = [['질문 Question','q','left'],['충실도','f'],['관련성','r'],['정밀도','p'],['재현율','c']];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 16 }}>
        <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', padding: 20, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
          <ScoreRing value={window.RAGAS_OVERALL} size={140} color="var(--il-blue)" />
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 13, color: 'var(--il-text-sec)' }}>최근 배치 평가 · 24건</div>
            <div style={{ fontSize: 12, color: 'var(--il-text-hint)', marginTop: 2 }}>2시간 전 · RAGAS</div>
          </div>
          <button className="il-btn-follow" style={{ width: '100%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 7 }}>
            <Ico name="play_arrow" size={19} fill={1} /> 평가 실행 Run eval
          </button>
        </div>
        <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', padding: 20 }}>
          <div style={{ fontSize: 14.5, fontWeight: 700, marginBottom: 18 }}>메트릭 평균 <span style={{ fontSize: 12, color: 'var(--il-text-hint)', fontWeight: 400 }}>Metric averages</span></div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {window.RAGAS.map((m, i) => <GaugeBar key={m.en} label={m.kr} labelEn={m.en} value={m.v} color={['var(--il-blue)','var(--il-success)','var(--il-warning)','#8C6FE0'][i]} />)}
          </div>
        </div>
      </div>
      <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', overflow: 'hidden' }}>
        <div style={{ padding: '15px 18px', borderBottom: '1px solid var(--il-overlay)', fontSize: 14.5, fontWeight: 700 }}>평가 결과 <span style={{ fontSize: 12, color: 'var(--il-text-hint)', fontWeight: 400 }}>Results · {window.EVAL_ROWS.length}</span></div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ fontSize: 11.5, color: 'var(--il-text-hint)' }}>
              {cols.map(([h,, align], i) => <th key={i} style={{ padding: '11px 18px', fontWeight: 600, textAlign: align === 'left' ? 'left' : 'right' }}>{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {window.EVAL_ROWS.map((r, i) => (
              <tr key={i} style={{ borderTop: '1px solid rgba(56,59,67,0.5)' }}>
                <td style={{ padding: '12px 18px', fontSize: 13, color: 'var(--il-text-primary)' }}>{r.q}</td>
                {['f','r','p','c'].map(k => (
                  <td key={k} style={{ padding: '12px 18px', textAlign: 'right' }}>
                    <span style={{ fontSize: 12.5, fontWeight: 600, fontFamily: 'var(--il-font-mono)', color: r[k] >= 0.9 ? 'var(--il-success)' : r[k] >= 0.85 ? 'var(--il-text-primary)' : 'var(--il-warning)' }}>{r[k].toFixed(2)}</span>
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

function RagPage({ section, setSection }) {
  const { SegTabs } = window;
  const tabs = [
    { id: 'docs', label: '문서', glyph: 'description', count: window.DOCS.length },
    { id: 'collections', label: '컬렉션', glyph: 'database', count: window.COLLECTIONS.length },
    { id: 'settings', label: '설정', glyph: 'tune' },
    { id: 'eval', label: '평가', glyph: 'fact_check' },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      <SegTabs tabs={tabs} value={section} onChange={setSection} />
      {section === 'docs' && <DocsSection />}
      {section === 'collections' && <CollectionsSection />}
      {section === 'settings' && <SettingsSection />}
      {section === 'eval' && <EvalSection />}
    </div>
  );
}

Object.assign(window, { RagPage });
