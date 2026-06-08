export async function fetchDocuments() {
  const r = await fetch('/index/documents');
  if (!r.ok) return [];
  const data = await r.json();
  return data.documents || data || [];
}

export async function fetchCollections() {
  const r = await fetch('/collections');
  if (!r.ok) return [];
  const data = await r.json();
  return data.collections || data || [];
}

export async function fetchHealth() {
  const r = await fetch('/health');
  if (!r.ok) return null;
  return r.json();
}

export async function fetchConversations() {
  const r = await fetch('/conversations');
  if (!r.ok) return [];
  const data = await r.json();
  return data.conversations || data || [];
}

export async function fetchConversation(id) {
  const r = await fetch(`/conversations/${id}`);
  if (!r.ok) return null;
  return r.json();
}

export async function createConversation(data) {
  const r = await fetch('/conversations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  if (!r.ok) return null;
  return r.json();
}

export async function renameConversation(id, title) {
  const r = await fetch(`/conversations/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title })
  });
  return r.ok;
}

export async function deleteConversation(id) {
  const r = await fetch(`/conversations/${id}`, { method: 'DELETE' });
  return r.ok;
}

export async function uploadFile(file, collection = 'default') {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('collection_name', collection);
  const r = await fetch('/ingest/file', { method: 'POST', body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function queryStream(question, options, onToken, onDone) {
  const r = await fetch('/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, ...options })
  });
  if (!r.ok) throw new Error(await r.text());
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const payload = JSON.parse(line.slice(6));
          if (payload.token) onToken(payload.token);
          if (payload.done) onDone(payload);
        } catch {}
      }
    }
  }
}

export async function queryFull(question, options) {
  const r = await fetch('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, ...options })
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function runEvaluate(questions) {
  const r = await fetch('/evaluate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ questions })
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
