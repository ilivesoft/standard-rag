import React, { createContext, useContext, useState, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Ico } from './Ico.jsx';

const ToastContext = createContext(null);

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);
  const idRef = useRef(0);

  const toast = useCallback((message, type = 'info') => {
    const id = ++idRef.current;
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 3000);
  }, []);

  const ICON = { success: 'check_circle', error: 'error', info: 'info' };
  const COLOR = { success: 'var(--il-success)', error: 'var(--il-live)', info: 'var(--il-blue)' };

  return (
    <ToastContext.Provider value={toast}>
      {children}
      {createPortal(
        <div style={{
          position: 'fixed', bottom: 24, right: 24,
          display: 'flex', flexDirection: 'column-reverse', gap: 10,
          zIndex: 9999, pointerEvents: 'none',
        }}>
          {toasts.map(t => (
            <div key={t.id} style={{
              background: 'var(--il-surface-el)',
              border: '1px solid var(--il-overlay)',
              borderLeft: `3px solid ${COLOR[t.type]}`,
              borderRadius: 10, padding: '12px 16px',
              display: 'flex', alignItems: 'center', gap: 10,
              fontSize: 13.5, color: 'var(--il-text-primary)',
              minWidth: 260, maxWidth: 380,
              boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
              pointerEvents: 'auto',
            }}>
              <Ico name={ICON[t.type]} size={18} style={{ color: COLOR[t.type], flexShrink: 0 }} />
              <span style={{ flex: 1 }}>{t.message}</span>
            </div>
          ))}
        </div>,
        document.body
      )}
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  return ctx ?? (() => {});
}
