import { useState, useCallback, useEffect } from 'react'
import { sendQuery, getStats, clearCache, reindex, checkHealth, uploadFile, type QueryResponse, type StatsResponse } from './api'
import './App.css'

interface Message {
  id: string
  query: string
  response: QueryResponse
}

function App() {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [showStats, setShowStats] = useState(false)
  const [useCache, setUseCache] = useState(true)
  const [apiStatus, setApiStatus] = useState<'checking' | 'ok' | 'error'>('checking')
  const [uploading, setUploading] = useState(false)
  const [reindexing, setReindexing] = useState(false)

  const extractCitedSections = (answer: string): string[] => {
    const lineMatch = answer.match(/Информация актуальна для разделов:\s*([^\n\r]+)/i)
    if (!lineMatch) return []
    const sectionCodes = lineMatch[1].match(/\d+(?:\.\d+)*/g) || []
    return [...new Set(sectionCodes)]
  }

  const extractDominantSection = (answer: string): string | null => {
    const lineMatch = answer.match(/Информация актуальна для разделов:\s*([^\n\r]+)/i)
    if (!lineMatch) return null
    const sectionCodes = lineMatch[1].match(/\d+(?:\.\d+)*/g) || []
    if (sectionCodes.length === 0) return null
    const counts = new Map<string, number>()
    const firstPos = new Map<string, number>()
    sectionCodes.forEach((code, idx) => {
      counts.set(code, (counts.get(code) || 0) + 1)
      if (!firstPos.has(code)) firstPos.set(code, idx)
    })
    let best: string | null = null
    let bestCount = -1
    let bestPos = Number.MAX_SAFE_INTEGER
    for (const [code, cnt] of counts.entries()) {
      const pos = firstPos.get(code) ?? Number.MAX_SAFE_INTEGER
      if (cnt > bestCount || (cnt === bestCount && pos < bestPos)) {
        best = code
        bestCount = cnt
        bestPos = pos
      }
    }
    return best
  }

  const normalizeSectionsLine = (answer: string): string => {
    return answer.replace(/Информация актуальна для разделов:\s*([^\n\r]+)/i, (_full, rawTail: string) => {
      const sectionCodes = rawTail.match(/\d+(?:\.\d+)*/g) || []
      const unique = [...new Set(sectionCodes)]
      if (unique.length === 0) {
        return "Информация актуальна для разделов: отсутствуют."
      }
      return `Информация актуальна для разделов: ${unique.join(', ')}.`
    })
  }

  const sectionMatches = (header: string | null | undefined, cited: string) => {
    if (!header) return false
    const c = cited.trim()
    const m = header.trim().match(/^(\d+(?:\.\d+)*)\b/)
    if (!m) return false
    const sectionCode = m[1]
    return sectionCode === c
  }

  const getImages = (d: { images?: string[] | string }) => {
    const imgs = d?.images
    if (Array.isArray(imgs)) return imgs.filter(Boolean)
    if (typeof imgs === 'string') {
      try {
        const parsed = JSON.parse(imgs) as unknown
        return Array.isArray(parsed) ? parsed.filter(Boolean) : []
      } catch {
        return []
      }
    }
    return []
  }

  const fetchStats = useCallback(async () => {
    try {
      const data = await getStats()
      setStats(data)
      setShowStats(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Ошибка загрузки статистики')
    }
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const q = query.trim()
    if (!q || loading) return

    setLoading(true)
    setError(null)
    try {
      const response = await sendQuery(q, useCache)
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), query: q, response },
      ])
      setQuery('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Ошибка запроса')
    } finally {
      setLoading(false)
    }
  }

  const handleClearCache = async () => {
    if (!confirm('Очистить кеш? Повторные запросы будут обрабатываться заново.')) return
    try {
      await clearCache()
      setStats(null)
      setShowStats(false)
      await fetchStats()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Ошибка очистки кеша')
    }
  }

  const handleReindex = async () => {
    if (!confirm('Переиндексировать все документы? Это займёт 1–2 минуты.')) return
    setReindexing(true)
    setError(null)
    try {
      const result = await reindex()
      setStats(null)
      setShowStats(false)
      await fetchStats()
      if (result.loaded_count === 0) {
        setError(result.message || 'В папке data/ нет PDF или TXT файлов. Загрузите документы через кнопку выше.')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Ошибка переиндексации')
    } finally {
      setReindexing(false)
    }
  }

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || uploading || apiStatus === 'error') return
    if (!file.name.toLowerCase().match(/\.(pdf|txt)$/)) {
      setError('Поддерживаются только PDF и TXT')
      return
    }
    setUploading(true)
    setError(null)
    try {
      await uploadFile(file)
      await fetchStats()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Ошибка загрузки файла')
    } finally {
      setUploading(false)
      e.target.value = ''
    }
  }

  useEffect(() => {
    checkHealth()
      .then((h) => setApiStatus(h.pipeline_ready ? 'ok' : 'error'))
      .catch(() => setApiStatus('error'))
  }, [])

  return (
    <div className="app">
      <header className="header">
        <h1>RAG Assistant</h1>
        <p className="subtitle">Retrieval-Augmented Generation — ответы на основе вашей базы документов</p>
        <div className="header-actions">
          <label className="toggle">
            <input
              type="checkbox"
              checked={useCache}
              onChange={(e) => setUseCache(e.target.checked)}
            />
            <span>Использовать кеш</span>
          </label>
          <button onClick={fetchStats} className="btn btn-secondary">
            Статистика
          </button>
          <button onClick={handleClearCache} className="btn btn-outline">
            Очистить кеш
          </button>
          <button onClick={handleReindex} className="btn btn-outline" disabled={reindexing || apiStatus === 'error'}>
            {reindexing ? 'Переиндексация...' : 'Переиндексировать'}
          </button>
          <label className="btn btn-secondary" style={{ marginLeft: 0 }}>
            <input
              type="file"
              accept=".pdf,.txt"
              onChange={handleUpload}
              disabled={uploading || apiStatus === 'error'}
              style={{ display: 'none' }}
            />
            {uploading ? 'Загрузка...' : 'Загрузить PDF/TXT'}
          </label>
        </div>
      </header>

      {apiStatus === 'error' && (
        <div className="banner banner-error">
          API недоступен. Запустите backend: <code>python -m uvicorn backend.main:app --reload --port 8000</code>
        </div>
      )}

      {error && (
        <div className="banner banner-error" onClick={() => setError(null)}>
          {error}
        </div>
      )}

      <main className="main">
        <form onSubmit={handleSubmit} className="query-form">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Задайте вопрос по базе документов..."
            disabled={loading || apiStatus === 'error'}
            className="query-input"
          />
          <button type="submit" disabled={loading || !query.trim()} className="btn btn-primary">
            {loading ? 'Ожидание...' : 'Отправить'}
          </button>
        </form>

        <section className="messages">
          {messages.length === 0 && !loading && (
            <div className="empty-state">
              <p>Начните с вопроса, например:</p>
              <ul>
                {[
                  'назначение устройства ZONT CONNECT',
                  'Функциональные возможности системы ZONT',
                  'Установка и активация SIM-карты',
                  'Настройка каналов связи с сервером',
                ].map((q) => (
                  <li key={q} onClick={() => setQuery(q)}>
                    {q}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {messages.map((m) => (
            <article key={m.id} className="message-card">
              <div className="message-query">
                <span className="label">Вопрос</span>
                <p>{m.query}</p>
              </div>
              <div className="message-answer">
                <span className="label">
                  Ответ {m.response.from_cache && <span className="badge">из кеша</span>}
                </span>
                <p>{normalizeSectionsLine(m.response.answer)}</p>
                {(() => {
                  const docs = m.response.context_docs || []
                  const sources = [...new Set(docs.map((d) => d.source).filter(Boolean))] as string[]
                  return sources.length > 0 ? (
                    <p className="doc-links">
                      Документ{sources.length > 1 ? 'ы' : ''}:{' '}
                      {sources.map((src, i) => (
                        <span key={i}>
                          {i > 0 && ', '}
                          <a href={`/api/documents/${encodeURIComponent(src)}`} target="_blank" rel="noopener noreferrer">
                            {src}
                          </a>
                        </span>
                      ))}
                    </p>
                  ) : null
                })()}
              </div>
              {(() => {
                const docs = m.response.context_docs || []
                const dominantSection = extractDominantSection(m.response.answer)
                const citedSections = extractCitedSections(m.response.answer)
                const relevantDocs =
                  dominantSection
                    ? docs.filter((d) => sectionMatches(d.section_header, dominantSection))
                    : citedSections.length > 0
                      ? docs.filter((d) => citedSections.some((c) => sectionMatches(d.section_header, c)))
                      : []
                const images = [...new Set(relevantDocs.flatMap((d) => getImages(d)))]
                return (
                  <>
                    {images.length > 0 && (
                      <div className="context-images">
                        <span className="label">Изображения из раздела</span>
                        <div className="images-grid">
                          {images.map((imgPath, i) => (
                            <a
                              key={i}
                              href={`/api/images/${imgPath}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="context-image-link"
                            >
                              <img src={`/api/images/${imgPath}`} alt={`Иллюстрация ${i + 1}`} />
                            </a>
                          ))}
                        </div>
                      </div>
                    )}
                    {docs.length > 0 && (
                      <details className="context-docs">
                        <summary>Использованный контекст ({docs.length})</summary>
                        <ul>
                          {docs.map((doc, i) => (
                            <li key={i}>
                              <span className="doc-preview">{doc.text.slice(0, 200)}...</span>
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                  </>
                )
              })()}
            </article>
          ))}
        </section>
      </main>

      {showStats && stats && (
        <aside className="stats-panel">
          <div className="stats-header">
            <h2>Статистика</h2>
            <button onClick={() => setShowStats(false)} className="btn-close">×</button>
          </div>
          <dl>
            <dt>Векторное хранилище</dt>
            <dd>Коллекция: {stats.vector_store.name}</dd>
            <dd>Документов: {stats.vector_store.documents_count ?? stats.vector_store.count}</dd>
            <dd>Чанков: {stats.vector_store.chunks_count ?? stats.vector_store.count}</dd>
            <dt>Кеш</dt>
            <dd>Записей: {stats.cache.total_entries}</dd>
            <dd>Размер: {stats.cache.db_size_mb.toFixed(2)} MB</dd>
            <dt>Модель</dt>
            <dd>{stats.model}</dd>
          </dl>
        </aside>
      )}
    </div>
  )
}

export default App
