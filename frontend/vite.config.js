import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/query': 'http://localhost:8000',
      '/ingest': 'http://localhost:8000',
      '/index': 'http://localhost:8000',
      '/collections': 'http://localhost:8000',
      '/conversations': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/evaluate': 'http://localhost:8000',
    }
  }
});
