const API_BASE = '/api';

export interface QueryResponse {
  query: string;
  answer: string;
  from_cache: boolean;
  context_docs?: { id?: string; text: string; distance?: number; images?: string[]; source?: string; section_header?: string | null }[];
  model?: string;
  cached_at?: string;
}

export interface StatsResponse {
  vector_store: {
    name: string;
    count: number;
    chunks_count?: number;
    documents_count?: number;
    persist_directory?: string;
    loaded_files_dir?: string;
  };
  cache: {
    total_entries: number;
    oldest_entry: string | null;
    newest_entry: string | null;
    db_size_mb: number;
  };
  model: string;
  mode?: string;
}

export async function sendQuery(query: string, useCache = true): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, use_cache: useCache }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Ошибка запроса');
  }
  return res.json();
}

export async function getStats(): Promise<StatsResponse> {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw new Error('Ошибка получения статистики');
  return res.json();
}

export async function clearCache(): Promise<void> {
  const res = await fetch(`${API_BASE}/cache/clear`, { method: 'POST' });
  if (!res.ok) throw new Error('Ошибка очистки кеша');
}

export interface ReindexResponse {
  status: string;
  message: string;
  loaded_count: number;
  files_loaded: string[];
}

export async function reindex(): Promise<ReindexResponse> {
  const res = await fetch(`${API_BASE}/reindex`, { method: 'POST' });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || 'Ошибка переиндексации');
  }
  return data;
}

export async function checkHealth(): Promise<{ status: string; pipeline_ready: boolean }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('API недоступен');
  return res.json();
}

export async function uploadFile(file: File): Promise<{ status: string; message: string }> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Ошибка загрузки');
  }
  return res.json();
}
