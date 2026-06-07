// data.jsx — dummy data + small helpers for Standard RAG admin prototype
// All exported to window for cross-file (Babel) access.

const fmt = (n) => n.toLocaleString('en-US');

// ── KPI stat cards ───────────────────────────────────────────
const KPIS = [
  { id: 'docs',  kr: '총 문서',      en: 'Total documents', value: 1284, unit: '개', delta: +8.2,  glyph: 'description', spark: [12,14,13,18,17,22,24,28,31,30,34,38] },
  { id: 'chunks',kr: '총 청크',      en: 'Total chunks',     value: 57820,unit: '개', delta: +6.4,  glyph: 'dataset',     spark: [40,42,45,44,49,52,55,57,58,60,63,66] },
  { id: 'qa',    kr: '누적 질의응답', en: 'Q&A handled',      value: 9461, unit: '건', delta: +14.7, glyph: 'forum',       spark: [10,12,11,15,19,18,24,27,29,33,38,42] },
  { id: 'lat',   kr: '평균 응답시간', en: 'Avg. latency',     value: 1.82, unit: '초', delta: -5.1,  glyph: 'bolt',        spark: [26,24,25,22,21,20,19,20,18,17,18,16], invert: true },
];

// ── Daily Q&A trend (last 14 days) ───────────────────────────
const QA_TREND = [
  { d: '5/25', v: 210 }, { d: '5/26', v: 188 }, { d: '5/27', v: 246 },
  { d: '5/28', v: 275 }, { d: '5/29', v: 232 }, { d: '5/30', v: 198 },
  { d: '5/31', v: 164 }, { d: '6/1', v: 289 }, { d: '6/2', v: 312 },
  { d: '6/3', v: 298 }, { d: '6/4', v: 341 }, { d: '6/5', v: 366 },
  { d: '6/6', v: 352 }, { d: '6/7', v: 388 },
];

// ── RAGAS evaluation scores ──────────────────────────────────
const RAGAS = [
  { kr: '충실도',     en: 'Faithfulness',       v: 0.91 },
  { kr: '답변 관련성', en: 'Answer relevancy',   v: 0.88 },
  { kr: '문맥 정밀도', en: 'Context precision',  v: 0.85 },
  { kr: '문맥 재현율', en: 'Context recall',     v: 0.93 },
];
const RAGAS_OVERALL = 0.89;

// ── Collections ──────────────────────────────────────────────
const COLLECTIONS = [
  { id: 'default',    name: 'default',         docs: 642, chunks: 28940, color: '#2993D1' },
  { id: 'hr-policy',  name: 'hr-policy',       docs: 188, chunks: 9120,  color: '#29B473' },
  { id: 'legal',      name: 'legal-contracts', docs: 274, chunks: 13860, color: '#E0A516' },
  { id: 'eng-wiki',   name: 'eng-wiki',        docs: 180, chunks: 5900,  color: '#8C6FE0' },
];

// ── In-flight indexing jobs ─────────────────────────────────
const JOBS = [
  { id: 'j1', name: '2026_연간보고서.pdf',     stage: '임베딩',  stageEn: 'Embedding', pct: 72, eta: '00:48' },
  { id: 'j2', name: 'product_specs_v3.docx',   stage: '청킹',    stageEn: 'Chunking',  pct: 38, eta: '01:24' },
  { id: 'j3', name: 'onboarding_guide.md',     stage: '파싱',    stageEn: 'Parsing',   pct: 12, eta: '02:05' },
];

// ── System component health ─────────────────────────────────
const SYSTEM = [
  { kr: '파서',        en: 'Parser',      glyph: 'article',        status: 'ready' },
  { kr: '임베더',      en: 'Embedder',    glyph: 'polyline',       status: 'ready', note: 'BAAI/bge-m3' },
  { kr: '벡터스토어',  en: 'VectorStore', glyph: 'database',       status: 'ready', note: 'ChromaDB' },
  { kr: '생성기',      en: 'Generator',   glyph: 'smart_toy',      status: 'degraded', note: 'Ollama · llama3.2' },
  { kr: '재순위기',    en: 'Reranker',    glyph: 'sort',           status: 'ready' },
  { kr: '평가기',      en: 'Evaluator',   glyph: 'fact_check',     status: 'idle',  note: 'RAGAS' },
];

// ── Recent activity feed ────────────────────────────────────
const ACTIVITY = [
  { kr: '질의응답 완료', en: 'Query answered', glyph: 'forum',       sub: '"환불 정책의 예외 조항은?" · default', t: '방금 전',  tone: 'blue'  },
  { kr: '인덱싱 완료',   en: 'Indexing done',  glyph: 'task_alt',    sub: 'q3_재무제표.pdf · 41 chunks',         t: '3분 전',   tone: 'green' },
  { kr: '컬렉션 생성',   en: 'Collection created', glyph: 'create_new_folder', sub: 'eng-wiki',                t: '1시간 전', tone: 'blue'  },
  { kr: '평가 실행',     en: 'Eval run',       glyph: 'fact_check',  sub: '배치 24건 · overall 0.89',          t: '2시간 전', tone: 'amber' },
  { kr: '인덱싱 실패',   en: 'Indexing failed',glyph: 'error',       sub: 'scan_form.pdf · OCR timeout',         t: '3시간 전', tone: 'red'   },
  { kr: '문서 업로드',   en: 'Document uploaded', glyph: 'upload_file', sub: '12개 파일 · 184 MB',              t: '5시간 전', tone: 'blue'  },
];

// ── Documents table ─────────────────────────────────────────
const DOCS = [
  { name: '2026_연간보고서.pdf',     type: 'PDF',  coll: 'default',   chunks: 312, tokens: '148K', status: 'indexed',  date: '2026-06-07' },
  { name: 'product_specs_v3.docx',   type: 'DOCX', coll: 'eng-wiki',  chunks: 0,   tokens: '—',    status: 'indexing', date: '2026-06-07' },
  { name: '환불정책_2026.md',        type: 'MD',   coll: 'hr-policy', chunks: 28,  tokens: '12K',  status: 'indexed',  date: '2026-06-06' },
  { name: 'nda_template.pdf',        type: 'PDF',  coll: 'legal',     chunks: 64,  tokens: '31K',  status: 'indexed',  date: '2026-06-06' },
  { name: 'scan_form.pdf',           type: 'PDF',  coll: 'default',   chunks: 0,   tokens: '—',    status: 'failed',   date: '2026-06-05' },
  { name: 'onboarding_guide.md',     type: 'MD',   coll: 'hr-policy', chunks: 19,  tokens: '8.2K', status: 'indexing', date: '2026-06-05' },
  { name: 'api_reference.html',      type: 'HTML', coll: 'eng-wiki',  chunks: 96,  tokens: '44K',  status: 'indexed',  date: '2026-06-04' },
  { name: 'q3_재무제표.pdf',         type: 'PDF',  coll: 'default',   chunks: 41,  tokens: '19K',  status: 'indexed',  date: '2026-06-04' },
  { name: 'support_faq.txt',         type: 'TXT',  coll: 'default',   chunks: 33,  tokens: '14K',  status: 'indexed',  date: '2026-06-03' },
];

// ── Conversation history (chatbot) ──────────────────────────
const CONVERSATIONS = [
  { id: 'c1', title: '환불 정책 예외 조항', sub: '6개 메시지 · default', t: '방금 전',  active: true },
  { id: 'c2', title: 'bge-m3 임베딩 차원', sub: '4개 메시지 · eng-wiki', t: '2시간 전' },
  { id: 'c3', title: '연차 산정 기준 문의', sub: '8개 메시지 · hr-policy', t: '어제' },
  { id: 'c4', title: 'NDA 위반 시 책임 범위', sub: '5개 메시지 · legal', t: '어제' },
  { id: 'c5', title: '하이브리드 검색 alpha 값', sub: '3개 메시지 · default', t: '6/5' },
  { id: 'c6', title: 'OCR 스캔 PDF 처리', sub: '7개 메시지 · default', t: '6/4' },
];

// ── Seed chat thread for the active conversation ────────────
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

// ── Streaming answer used for the typing demo ───────────────
const STREAM_ANSWER = '하이브리드 검색의 alpha 값은 Vector 검색과 BM25 검색의 가중치를 조절합니다 [1]. alpha=1.0이면 Vector 검색만, alpha=0.0이면 BM25 검색만 사용하며, 기본값 0.5는 두 방식을 균등하게 결합합니다 [2]. 도메인 특화 용어가 많은 문서에는 BM25 비중을 높이고(alpha를 낮추고), 의미 기반 검색이 중요하면 alpha를 높이는 것을 권장합니다 [1].';
const STREAM_SOURCES = [
  { n: 1, doc: 'api_reference.html', page: '—', chunk: '#54', score: 0.92, text: 'HYBRID_ALPHA 파라미터는 0과 1 사이의 값으로 Vector 검색 점수와 BM25 점수를 가중 결합합니다…' },
  { n: 2, doc: 'README.md', page: '—', chunk: '#7', score: 0.86, text: 'alpha 0.0은 BM25 검색만, 0.5는 균등 가중, 1.0은 Vector 검색만 사용합니다…' },
];

// ── RAGAS batch eval result rows ────────────────────────────
const EVAL_ROWS = [
  { q: '환불 정책의 예외 조항은?',        f: 0.94, r: 0.91, p: 0.88, c: 0.95 },
  { q: 'bge-m3 임베딩 차원 수는?',        f: 0.88, r: 0.90, p: 0.82, c: 0.91 },
  { q: '연차 산정 기준을 설명해줘',       f: 0.92, r: 0.86, p: 0.84, c: 0.93 },
  { q: 'NDA 위반 시 책임 범위는?',        f: 0.90, r: 0.85, p: 0.87, c: 0.89 },
  { q: '청크 크기 기본값과 겹침은?',      f: 0.96, r: 0.93, p: 0.90, c: 0.97 },
];

Object.assign(window, {
  fmt, KPIS, QA_TREND, RAGAS, RAGAS_OVERALL, COLLECTIONS, JOBS, SYSTEM,
  ACTIVITY, DOCS, CONVERSATIONS, CHAT_SEED, STREAM_ANSWER, STREAM_SOURCES, EVAL_ROWS,
});
