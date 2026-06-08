import React, { useState, useEffect } from 'react';
import { Ico } from '../../components/ui/Ico.jsx';
import { withAlpha, fmt } from '../../components/ui/helpers.js';
import { fetchCollections } from '../../api/client.js';

const MOCK_COLLECTIONS = [
  { id: 'default',   name: 'default',         docs: 642, chunks: 28940, color: '#2993D1' },
  { id: 'hr-policy', name: 'hr-policy',       docs: 188, chunks: 9120,  color: '#29B473' },
  { id: 'legal',     name: 'legal-contracts', docs: 274, chunks: 13860, color: '#E0A516' },
  { id: 'eng-wiki',  name: 'eng-wiki',        docs: 180, chunks: 5900,  color: '#8C6FE0' },
];

export function CollectionsSection() {
  const [collections, setCollections] = useState(MOCK_COLLECTIONS);

  useEffect(() => {
    fetchCollections().then(data => {
      if (data && data.length > 0) {
        const mapped = data.map((c, i) => ({
          id: c.name || c.id || String(i),
          name: c.name || c.id || String(i),
          docs: c.document_count || c.docs || 0,
          chunks: c.chunk_count || c.chunks || 0,
          color: MOCK_COLLECTIONS[i % MOCK_COLLECTIONS.length].color,
        }));
        setCollections(mapped);
      }
    }).catch(() => {});
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700 }}>컬렉션 <span style={{ fontSize: 13, color: 'var(--il-text-hint)', fontWeight: 400 }}>Collections · {collections.length}</span></h3>
        </div>
        <button className="il-btn-follow" style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}>
          <Ico name="create_new_folder" size={18} /> 새 컬렉션
        </button>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>
        {collections.map(c => (
          <div key={c.id} style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', padding: 18, display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 11 }}>
              <span style={{ width: 40, height: 40, borderRadius: 11, background: withAlpha(c.color, 0.16), color: c.color, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
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

export default CollectionsSection;
