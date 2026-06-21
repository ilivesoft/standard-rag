import React, { useState, useEffect, useRef } from 'react';
import { Ico } from '../../components/ui/Ico.jsx';
import { TypeTag } from '../../components/ui/TypeTag.jsx';
import { StatusBadge } from '../../components/ui/StatusBadge.jsx';
import { ProgressBar } from '../../components/ui/ProgressBar.jsx';
import { fetchDocuments, fetchCollections, uploadFile, deleteDocument } from '../../api/client.js';
import { useToast } from '../../components/ui/Toast.jsx';

const selStyle = {
  background: 'var(--il-bg-base)', color: 'var(--il-text-primary)', border: '1px solid var(--il-overlay)',
  borderRadius: 8, padding: '6px 10px', fontSize: 12.5, fontFamily: 'var(--il-font-mono)', outline: 'none', cursor: 'pointer',
};

const MOCK_DOCS = [
  { name: '2026_연간보고서.pdf',   type: 'PDF',  coll: 'default',   chunks: 312, tokens: '148K', status: 'indexed',  date: '2026-06-07' },
  { name: '환불정책_2026.md',      type: 'MD',   coll: 'hr-policy', chunks: 28,  tokens: '12K',  status: 'indexed',  date: '2026-06-06' },
  { name: 'nda_template.pdf',      type: 'PDF',  coll: 'legal',     chunks: 64,  tokens: '31K',  status: 'indexed',  date: '2026-06-06' },
  { name: 'api_reference.html',    type: 'HTML', coll: 'eng-wiki',  chunks: 96,  tokens: '44K',  status: 'indexed',  date: '2026-06-04' },
  { name: 'support_faq.txt',       type: 'TXT',  coll: 'default',   chunks: 33,  tokens: '14K',  status: 'indexed',  date: '2026-06-03' },
];

function mapDoc(d) {
  const src = d.source || d.filename || d.name || d.id || '';
  return {
    name: src,
    type: src.split('.').pop().toUpperCase() || 'TXT',
    coll: d.collection || 'default',
    chunks: d.chunk_count ?? 0,
    tokens: d.token_count ? `${Math.round(d.token_count / 1000)}K` : '—',
    status: d.status || 'indexed',
    date: d.created_at ? new Date(d.created_at).toISOString().slice(0, 10) : '—',
  };
}

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
    e.target.value = '';
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

export function DocsSection({ initialColl = 'all', onCollChange }) {
  const toast = useToast();
  const [coll, setColl] = useState(initialColl);
  const [collList, setCollList] = useState([]);
  const [docs, setDocs] = useState(MOCK_DOCS);
  const [search, setSearch] = useState('');
  const [jobs, setJobs] = useState([]);
  const [activeMenu, setActiveMenu] = useState(null);
  const menuRef = useRef(null);

  useEffect(() => { setColl(initialColl); }, [initialColl]);

  useEffect(() => {
    loadDocs();
    fetchCollections().then(data => {
      if (data?.length) setCollList(data.map(c => ({ id: c.name || c.id || String(c), name: c.name || c.id || String(c) })));
    }).catch(() => {});
  }, []);

  useEffect(() => {
    function handler(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) setActiveMenu(null);
    }
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  function loadDocs() {
    fetchDocuments().then(data => {
      if (data?.length) setDocs(data.map(mapDoc));
    }).catch(() => {});
  }

  function handleCollChange(val) {
    setColl(val);
    if (onCollChange) onCollChange(val);
  }

  async function handleUpload(files) {
    const targetColl = coll === 'all' ? 'default' : coll;
    const newJobs = files.map(f => ({ id: f.name, name: f.name, pct: 50 }));
    setJobs(prev => [...prev, ...newJobs]);

    for (const file of files) {
      try {
        await uploadFile(file, targetColl);
        setJobs(prev => prev.filter(j => j.id !== file.name));
        toast(`${file.name} 업로드 완료`, 'success');
      } catch {
        setJobs(prev => prev.filter(j => j.id !== file.name));
        toast(`${file.name} 업로드 실패`, 'error');
      }
    }
    loadDocs();
  }

  async function handleDelete(docName) {
    setActiveMenu(null);
    if (!window.confirm(`'${docName}' 문서를 삭제하시겠습니까?`)) return;
    const ok = await deleteDocument(docName);
    if (ok) {
      toast('문서가 삭제되었습니다.', 'success');
      setDocs(prev => prev.filter(d => d.name !== docName));
    } else {
      toast('삭제에 실패했습니다.', 'error');
    }
  }

  const filtered = docs.filter(d => {
    const collMatch = coll === 'all' || d.coll === coll;
    const searchMatch = !search || d.name.toLowerCase().includes(search.toLowerCase());
    return collMatch && searchMatch;
  });

  const allColls = [{ id: 'all', name: '전체 컬렉션' }, ...collList];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <UploadZone onUpload={handleUpload} />

      {jobs.length > 0 && (
        <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 13 }}>
            <Ico name="sync" size={17} style={{ color: 'var(--il-blue)', animation: 'srSpin 1.8s linear infinite' }} />
            <span style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--il-text-primary)' }}>인덱싱 진행 중</span>
            <span style={{ fontSize: 12, color: 'var(--il-text-hint)' }}>Indexing · {jobs.length}건</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 16 }}>
            {jobs.map(j => (
              <div key={j.id} style={{ background: 'var(--il-bg-base)', borderRadius: 11, padding: 13, display: 'flex', flexDirection: 'column', gap: 8 }}>
                <span style={{ fontSize: 12.5, fontWeight: 500, color: 'var(--il-text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{j.name}</span>
                <ProgressBar pct={j.pct} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--il-text-hint)' }}>
                  <span>업로드 중...</span>
                  <span style={{ fontFamily: 'var(--il-font-mono)' }}>{j.pct}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', overflow: 'hidden' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '15px 18px', borderBottom: '1px solid var(--il-overlay)' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700 }}>문서 목록</h3>
            <span style={{ fontSize: 12, color: 'var(--il-text-hint)' }}>Documents · {filtered.length}</span>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <select value={coll} onChange={e => handleCollChange(e.target.value)} style={selStyle}>
              {allColls.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
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
                <td style={{ padding: '12px 18px', textAlign: 'right', position: 'relative' }}>
                  <div ref={activeMenu === d.name ? menuRef : null} style={{ display: 'inline-block' }}>
                    <button
                      className="il-btn-icon"
                      style={{ width: 30, height: 30 }}
                      onClick={e => { e.stopPropagation(); setActiveMenu(activeMenu === d.name ? null : d.name); }}>
                      <Ico name="more_vert" size={18} />
                    </button>
                    {activeMenu === d.name && (
                      <div style={{
                        position: 'absolute', right: 0, top: '100%', zIndex: 100,
                        background: 'var(--il-surface-el)', border: '1px solid var(--il-overlay)',
                        borderRadius: 10, overflow: 'hidden', minWidth: 130,
                        boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
                      }}>
                        <button
                          onClick={() => handleDelete(d.name)}
                          style={{
                            width: '100%', padding: '10px 16px', border: 'none', cursor: 'pointer',
                            background: 'transparent', color: 'var(--il-live)', fontSize: 13,
                            display: 'flex', alignItems: 'center', gap: 8, fontFamily: 'var(--il-font)',
                          }}
                          onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,45,45,0.1)'}
                          onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                          <Ico name="delete" size={16} /> 삭제
                        </button>
                      </div>
                    )}
                  </div>
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
