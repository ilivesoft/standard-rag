export const fmt = (n) => n.toLocaleString('en-US');

export const KPIS = [
  { id: 'docs',   kr: '총 문서',       en: 'Total documents', value: 1284, unit: '개', delta: +8.2,  glyph: 'description', spark: [12,14,13,18,17,22,24,28,31,30,34,38] },
  { id: 'chunks', kr: '총 청크',       en: 'Total chunks',    value: 57820, unit: '개', delta: +6.4,  glyph: 'dataset',     spark: [40,42,45,44,49,52,55,57,58,60,63,66] },
  { id: 'qa',     kr: '누적 질의응답', en: 'Q&A handled',     value: 9461, unit: '건', delta: +14.7, glyph: 'forum',       spark: [10,12,11,15,19,18,24,27,29,33,38,42] },
  { id: 'lat',    kr: '평균 응답시간', en: 'Avg. latency',    value: 1.82, unit: '초', delta: -5.1,  glyph: 'bolt',        spark: [26,24,25,22,21,20,19,20,18,17,18,16], invert: true },
];

export const QA_TREND = [
  { d: '5/25', v: 210 }, { d: '5/26', v: 188 }, { d: '5/27', v: 246 },
  { d: '5/28', v: 275 }, { d: '5/29', v: 232 }, { d: '5/30', v: 198 },
  { d: '5/31', v: 164 }, { d: '6/1', v: 289 },  { d: '6/2', v: 312 },
  { d: '6/3', v: 298 },  { d: '6/4', v: 341 },  { d: '6/5', v: 366 },
  { d: '6/6', v: 352 },  { d: '6/7', v: 388 },
];

export const RAGAS = [
  { kr: '충실도',     en: 'Faithfulness',     v: 0.91 },
  { kr: '답변 관련성', en: 'Answer relevancy', v: 0.88 },
  { kr: '문맥 정밀도', en: 'Context precision', v: 0.85 },
  { kr: '문맥 재현율', en: 'Context recall',   v: 0.93 },
];

export const RAGAS_OVERALL = 0.89;

export const MOCK_COLLECTIONS = [
  { id: 'default',   name: 'default',         docs: 642, chunks: 28940, color: '#2993D1' },
  { id: 'hr-policy', name: 'hr-policy',       docs: 188, chunks: 9120,  color: '#29B473' },
  { id: 'legal',     name: 'legal-contracts', docs: 274, chunks: 13860, color: '#E0A516' },
  { id: 'eng-wiki',  name: 'eng-wiki',        docs: 180, chunks: 5900,  color: '#8C6FE0' },
];

export const JOBS = [
  { id: 'j1', name: '2026_연간보고서.pdf',   stage: '임베딩', stageEn: 'Embedding', pct: 72, eta: '00:48' },
  { id: 'j2', name: 'product_specs_v3.docx', stage: '청킹',   stageEn: 'Chunking',  pct: 38, eta: '01:24' },
  { id: 'j3', name: 'onboarding_guide.md',   stage: '파싱',   stageEn: 'Parsing',   pct: 12, eta: '02:05' },
];

export const MOCK_SYSTEM = [
  { kr: '파서',       en: 'Parser',      glyph: 'article',    status: 'ready' },
  { kr: '임베더',     en: 'Embedder',    glyph: 'polyline',   status: 'ready',    note: 'BAAI/bge-m3' },
  { kr: '벡터스토어', en: 'VectorStore', glyph: 'database',   status: 'ready',    note: 'ChromaDB' },
  { kr: '생성기',     en: 'Generator',   glyph: 'smart_toy',  status: 'degraded', note: 'Ollama · llama3.2' },
  { kr: '재순위기',   en: 'Reranker',    glyph: 'sort',       status: 'ready' },
  { kr: '평가기',     en: 'Evaluator',   glyph: 'fact_check', status: 'idle',     note: 'RAGAS' },
];

export const ACTIVITY = [
  { kr: '질의응답 완료',  en: 'Query answered',      glyph: 'forum',               sub: '"환불 정책의 예외 조항은?" · default', t: '방금 전',  tone: 'blue'  },
  { kr: '인덱싱 완료',    en: 'Indexing done',        glyph: 'task_alt',            sub: 'q3_재무제표.pdf · 41 chunks',         t: '3분 전',   tone: 'green' },
  { kr: '컬렉션 생성',    en: 'Collection created',   glyph: 'create_new_folder',   sub: 'eng-wiki',                            t: '1시간 전', tone: 'blue'  },
  { kr: '평가 실행',      en: 'Eval run',             glyph: 'fact_check',          sub: '배치 24건 · overall 0.89',            t: '2시간 전', tone: 'amber' },
  { kr: '인덱싱 실패',    en: 'Indexing failed',      glyph: 'error',               sub: 'scan_form.pdf · OCR timeout',         t: '3시간 전', tone: 'red'   },
  { kr: '문서 업로드',    en: 'Document uploaded',    glyph: 'upload_file',         sub: '12개 파일 · 184 MB',                  t: '5시간 전', tone: 'blue'  },
];
