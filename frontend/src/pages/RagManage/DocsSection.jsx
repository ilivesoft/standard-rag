import React, { useState, useEffect, useRef } from 'react';
import { Ico } from '../../components/ui/Ico.jsx';
import { TypeTag } from '../../components/ui/TypeTag.jsx';
import { StatusBadge } from '../../components/ui/StatusBadge.jsx';
import { ProgressBar } from '../../components/ui/ProgressBar.jsx';
import { fetchDocuments, uploadFile } from '../../api/client.js';

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

const MOCK_DOCS = [
  { name: '2026_연간보고서.pdf',   type: 'PDF',  coll: 'default',   chunks: 312, tokens: '148K', status: 'indexed',  date: '2026-06-07' },
  { name: 'product_specs_v3.docx', type: 'DOCX', coll: 'eng-wiki',  chunks: 0,   tokens: '—',    status: 'indexing', date: '2026-06-07' },
  { name: '환불정책_2026.md',      type: 'MD',   coll: 'hr-policy', chunks: 28,  tokens: '12K',  status: 'indexed',  date: '2026-06-06' },
  { name: 'nda_template.pdf',      type: 'PDF',  coll: 'legal',     chunks: 64,  tokens: '31K',  status: 'indexed',  date: '2026-06-06' },
  { name: 'scan_form.pdf',         type: 'PDF',  coll: 'default',   chunks: 0,   tokens: '—',    status: 'failed',   date: '2026-06-05' },
  { name: 'onboarding_guide.md',   type: 'MD',   coll: 'hr-policy', chunks: 19,  tokens: '8.2K', status: 'indexing', date: '2026-06-05' },
  { name: 'api_reference.html',    type: 'HTML', coll: 'eng-wiki',  chunks: 96,  tokens: '44K',  status: 'indexed',  date: '2026-06-04' },
  { name: 'q3_재무제표.pdf',       type: 'PDF',  coll: 'default',   chunks: 41,  tokens: '19K',  status: 'indexed',  date: '2026-06-04' },
  { name: 'support_faq.txt',       type: 'TXT',  coll: 'default',   chunks: 33,  tokens: '14K',  status: 'indexed',  date: '2026-06-03' },
];

const MOCK_JOBS = [
  { id: 'j1', name: '2026_연간보고서.pdf',   stage: '임베딩', stageEn: 'Embedding', pct: 72, eta: '00:48' },
  { id: 'j2', name: 'product_specs_v3.docx', stage: '청킹',   stageEn: 'Chunking',  pct: 38, eta: '01:24' },
  { id: 'j3', name: 'onboarding_guide.md',   stage: '파싱',   stageEn: 'Parsing',   pct: 12, eta: '02:05' },
];

function UploadZone({ onUpload }) {
  const [drag, setDrag] = useState(false);
  const fileRef = useRef(null);

  function handleDrop(e) {
    e.preventDefault();
    setDrag(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0 && onUpload) onUpload(files);
  }

  function handleFileChange(e) {
    const files = Array.from(e.target.files);
    if (files.length > 0 && onUpload) onUpload(files);
  }

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
      onClick={() => fileRef.current?.click()}
      style={{
        border: '2px dashed ' + (drag ? 'var(--il-blue)' : 'var(--il-overlay)'),
        background: drag ? 'var(--il-blue-soft)' : 'var(--il-bg-base)',
        borderRadius: 14, padding: '28px 24px', display: 'flex', alignItems: 'center', gap: 20, transition: 'all .15s', cursor: 'pointer',
      }}>
      <input ref={fileRef} type="file" multiple accept=".pdf,.docx,.txt,.md,.html" style={{ display: 'none' }} onChange={handleFileChange} />
      <span style={{ width: 52, height: 52, borderRadius: 13, background: 'var(--il-blue-soft)', color: 'var(--il-blue)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
        <Ico name="cloud_upload" size={28} />
      </span>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--il-text-primary)' }}>파일을 끌어다 놓거나 클릭하여 업로드</div>
        <div style={{ fontSize: 12.5, color: 'var(--il-text-hint)', marginTop: 3 }}>Drag &amp; drop or browse · PDF, DOCX, TXT, MD, HTML · 최대 100MB · OCR 지원</div>
      </div>
      <button className="il-btn-follow" onClick={e => { e.stopPropagation(); fileRef.current?.click(); }} style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}>
        <Ico name="folder_open" size={18} /> 파일 선택
      </button>
    </div>
  );
}

export function DocsSection() {
  const [coll, setColl] = useState('all');
  const [docs, setDocs] = useState(MOCK_DOCS);
  const [search, setSearch] = useState('');
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    fetchDocuments().then(data => {
      if (data && data.length > 0) {
        const mapped = data.map(d => ({
          name: d.filename || d.name || d.id,
          type: (d.filename || d.name || '').split('.').pop().toUpperCase() || 'TXT',
          coll: d.collection || 'default',
          chunks: d.chunk_count || d.chunks || 0,
          tokens: d.token_count ? `${Math.round(d.token_count / 1000)}K` : '—',
          status: d.status || 'indexed',
          date: d.created_at ? new Date(d.created_at).toISOString().slice(0, 10) : '—',
        }));
        setDocs(mapped);
      }
    }).catch(() => {});
  }, []);

  async function handleUpload(files) {
    setUploading(true);
    for (const file of files) {
      try {
        await uploadFile(file, coll === 'all' ? 'default' : coll);
      } catch (e) {
        console.error('Upload failed:', e);
      }
    }
    setUploading(false);
    fetchDocuments().then(data => {
      if (data && data.length > 0) {
        const mapped = data.map(d => ({
          name: d.filename || d.name || d.id,
          type: (d.filename || d.name || '').split('.').pop().toUpperCase() || 'TXT',
          coll: d.collection || 'default',
          chunks: d.chunk_count || d.chunks || 0,
          tokens: d.token_count ? `${Math.round(d.token_count / 1000)}K` : '—',
          status: d.status || 'indexed',
          date: d.created_at ? new Date(d.created_at).toISOString().slice(0, 10) : '—',
        }));
        setDocs(mapped);
      }
    }).catch(() => {});
  }

  const filtered = docs.filter(d => {
    const collMatch = coll === 'all' || d.coll === coll;
    const searchMatch = !search || d.name.toLowerCase().includes(search.toLowerCase());
    return collMatch && searchMatch;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <UploadZone onUpload={handleUpload} />

      {/* in-flight */}
      <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', padding: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 13 }}>
          <Ico name="sync" size={17} style={{ color: 'var(--il-blue)', animation: 'srSpin 1.8s linear infinite' }} />
          <span style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--il-text-primary)' }}>인덱싱 진행 중</span>
          <span style={{ fontSize: 12, color: 'var(--il-text-hint)' }}>Indexing · {MOCK_JOBS.length}건</span>
          {uploading && <span style={{ fontSize: 12, color: 'var(--il-blue)', marginLeft: 8 }}>업로드 중...</span>}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
          {MOCK_JOBS.map(j => (
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
            <span style={{ fontSize: 12, color: 'var(--il-text-hint)' }}>Documents · {filtered.length}</span>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <select value={coll} onChange={e => setColl(e.target.value)} style={selStyle}>
              <option value="all">전체 컬렉션</option>
              {MOCK_COLLECTIONS.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7, background: 'var(--il-bg-base)', border: '1px solid var(--il-overlay)', borderRadius: 8, padding: '0 11px' }}>
              <Ico name="search" size={17} style={{ color: 'var(--il-icon)' }} />
              <input
                placeholder="파일 검색"
                value={search}
                onChange={e => setSearch(e.target.value)}
                style={{ border: 'none', outline: 'none', background: 'transparent', color: 'var(--il-text-primary)', fontSize: 12.5, padding: '7px 0', width: 110, fontFamily: 'var(--il-font)' }}
              />
            </div>
          </div>
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ fontSize: 11.5, color: 'var(--il-text-hint)', textAlign: 'left' }}>
              {['파일명 File', '타입', '컬렉션', '상태', '청크', '토큰', '업로드', ''].map((h, i) => (
                <th key={i} style={{ padding: '11px 18px', fontWeight: 600, textAlign: i >= 4 && i <= 5 ? 'right' : 'left' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((d, i) => (
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

export default DocsSection;
