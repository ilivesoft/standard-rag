import React, { useState, useEffect } from 'react';
import { Ico } from '../../components/ui/Ico.jsx';
import { Slider } from '../../components/ui/Slider.jsx';
import { useToast } from '../../components/ui/Toast.jsx';

const STORAGE_KEY = 'rag_settings';
const DEFAULTS = { chunkSize: 512, overlap: 64, device: 'cpu', topK: 10, topN: 3, alpha: 0.5 };

function Field({ label, value, mono }) {
  return (
    <div>
      <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 8 }}>{label}</div>
      <div style={{ background: 'var(--il-bg-base)', border: '1px solid var(--il-overlay)', borderRadius: 8, padding: '10px 13px', fontSize: 13, color: 'var(--il-text-primary)', fontFamily: mono ? 'var(--il-font-mono)' : 'var(--il-font)' }}>{value}</div>
    </div>
  );
}

export function SettingsSection() {
  const toast = useToast();
  const [chunkSize, setChunkSize] = useState(DEFAULTS.chunkSize);
  const [overlap, setOverlap] = useState(DEFAULTS.overlap);
  const [device, setDevice] = useState(DEFAULTS.device);
  const [topK, setTopK] = useState(DEFAULTS.topK);
  const [topN, setTopN] = useState(DEFAULTS.topN);
  const [alpha, setAlpha] = useState(DEFAULTS.alpha);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const p = JSON.parse(saved);
        if (p.chunkSize != null) setChunkSize(p.chunkSize);
        if (p.overlap   != null) setOverlap(p.overlap);
        if (p.device    != null) setDevice(p.device);
        if (p.topK      != null) setTopK(p.topK);
        if (p.topN      != null) setTopN(p.topN);
        if (p.alpha     != null) setAlpha(p.alpha);
      }
    } catch {}
  }, []);

  function handleSave() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ chunkSize, overlap, device, topK, topN, alpha }));
      toast('설정이 저장되었습니다.', 'success');
    } catch {
      toast('설정 저장에 실패했습니다.', 'error');
    }
  }

  function handleReset() {
    setChunkSize(DEFAULTS.chunkSize);
    setOverlap(DEFAULTS.overlap);
    setDevice(DEFAULTS.device);
    setTopK(DEFAULTS.topK);
    setTopN(DEFAULTS.topN);
    setAlpha(DEFAULTS.alpha);
    localStorage.removeItem(STORAGE_KEY);
    toast('기본값으로 복원되었습니다.', 'info');
  }

  function block(title, en, glyph, children) {
    return (
      <div style={{ background: 'var(--il-surface)', borderRadius: 14, border: '1px solid rgba(56,59,67,0.55)', padding: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 18 }}>
          <span style={{ width: 32, height: 32, borderRadius: 9, background: 'var(--il-blue-soft)', color: 'var(--il-blue)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}>
            <Ico name={glyph} size={18} />
          </span>
          <div>
            <div style={{ fontSize: 14.5, fontWeight: 700 }}>{title}</div>
            <div style={{ fontSize: 11.5, color: 'var(--il-text-hint)' }}>{en}</div>
          </div>
        </div>
        {children}
      </div>
    );
  }

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
                {['cpu', 'cuda'].map(d => (
                  <button key={d} onClick={() => setDevice(d)} style={{
                    padding: '7px 20px', borderRadius: 7, border: 'none', cursor: 'pointer', fontSize: 13, fontFamily: 'var(--il-font-mono)',
                    background: device === d ? 'var(--il-surface-el)' : 'transparent', color: device === d ? 'var(--il-text-primary)' : 'var(--il-text-sec)', fontWeight: device === d ? 600 : 400
                  }}>{d}</button>
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
        <button className="il-btn-ghost" onClick={handleReset}>기본값 복원</button>
        <button className="il-btn-follow" onClick={handleSave} style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}>
          <Ico name="save" size={18} /> 설정 저장
        </button>
      </div>
    </div>
  );
}

export default SettingsSection;
