# RAG Assistant

RAG-приложение для ответов по документам (`PDF`/`TXT`) с веб-интерфейсом.

- **Backend:** `FastAPI`
- **Frontend:** `React + Vite + TypeScript`
- **Vector DB:** `Pinecone`
- **LLM/Embeddings:** `OpenAI` (приоритет) или `GigaChat` (fallback)
- **Cache:** локальный `SQLite`

---

## Что умеет система

- Загружать и индексировать `PDF`/`TXT` из `data/` и через UI.
- Извлекать текст и изображения из PDF и привязывать их к чанкам.
- Отвечать на вопросы с использованием векторного поиска + LLM.
- Показывать в UI:
  - ответ,
  - использованный контекст,
  - изображения, относящиеся к найденным разделам.
- Хранить кэш ответов и управлять им из UI.
- Выполнять полную переиндексацию документов.

---

## Архитектура (коротко)

1. Пользователь отправляет запрос (`/api/query`).
2. `RAGPipeline`:
   - смотрит кэш (`cache.py`),
   - делает поиск в Pinecone (`vector_store.py`),
   - формирует промпт,
   - получает ответ от LLM,
   - кладет результат в кэш.
3. Frontend показывает ответ, контекст и релевантные изображения.

---

## Структура проекта

```text
RAGassistant/
├── backend/
│   └── main.py                    # FastAPI API
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # UI
│   │   └── api.ts                 # HTTP-клиент API
│   └── package.json
├── data/
│   ├── loaded_files.json          # Трекер загруженных файлов
│   └── images/                    # Извлеченные из PDF изображения
├── rag_pipeline.py                # Оркестрация RAG (cache -> search -> LLM)
├── vector_store.py                # Работа с Pinecone + embeddings
├── pdf_processor.py               # Извлечение текста/изображений, чанкинг, метаданные
├── cache.py                       # SQLite-кэш ответов
├── loaded_files.py                # Инкрементальная загрузка документов
├── gigachat_client.py             # Клиент GigaChat
├── evaluate_ragas.py              # Скрипт оценки качества (RAGAS)
├── requirements.txt
├── .env
└── README.md
```

---

## Backend: функционал и API

Основной файл: `backend/main.py`

### Эндпоинты

- `GET /api/health`
  - Проверка состояния API.
- `POST /api/query`
  - Вход: `query`, `use_cache`.
  - Выход: `answer`, `from_cache`, `context_docs`, `model`.
- `GET /api/stats`
  - Возвращает:
    - статистику векторного хранилища (`documents_count`, `chunks_count`),
    - статистику кэша,
    - модель/режим.
- `POST /api/reindex`
  - Полная переиндексация:
    - очистка кэша,
    - удаление старых чанков по `source`,
    - повторная загрузка всех `PDF/TXT`.
- `POST /api/cache/clear`
  - Полная очистка кэша ответов.
- `POST /api/upload`
  - Загрузка одного `PDF/TXT` и инкрементальная индексация.
- `GET /api/images/{path}`
  - Отдача извлеченных изображений из `data/images`.
- `GET /api/documents/{filename}`
  - Отдача исходного документа (`.pdf`/`.txt`).

### RAG pipeline (`rag_pipeline.py`)

- Провайдер выбирается автоматически:
  - `OPENAI_API_KEY` -> OpenAI
  - иначе `GIGACHAT_AUTH_KEY + GIGACHAT_RQUID` -> GigaChat
- Формат ответа включает контекстные документы с метаданными:
  - `section_header`, `chunk_number`, `source`, `images`.

### Vector store (`vector_store.py`)

- Хранит чанки в Pinecone вместе с метаданными.
- Выполняет поиск + локальный rerank кандидатов.
- Возвращает статистику:
  - `documents_count`: количество загруженных файлов,
  - `chunks_count`: количество векторов (чанков).

#### Метаданные чанка в Pinecone

При индексации каждый вектор сохраняется с полем `metadata`, которое включает:

- `text` - текст чанка (контент, который использует LLM в контексте).
- `source` - имя исходного файла, например `Texnicheskaya-dokumentaciya-ZONT-Connect-Plus.pdf`.
- `chunk_number` - порядковый номер чанка внутри документа (начиная с `1`).
- `section_header` - заголовок секции/подсекции, например `6.2 Установка и активация SIM-карты`.
- `page_start` - отображаемый номер начальной страницы чанка.
- `page_end` - отображаемый номер конечной страницы чанка.
- `images` - список путей к связанным изображениям (в хранилище сохраняется как JSON-строка).

Пример `metadata`:

```json
{
  "text": "6.2 Установка и активация SIM-карты ...",
  "source": "Texnicheskaya-dokumentaciya-ZONT-Connect-Plus.pdf",
  "chunk_number": 10,
  "section_header": "6.2 Установка и активация SIM-карты",
  "page_start": "13",
  "page_end": "13",
  "images": "[\"Texnicheskaya-dokumentaciya-ZONT-Connect-Plus/p13_img1.png\"]"
}
```

При выдаче через API поле `images` нормализуется в массив строк для фронтенда.

### PDF processing (`pdf_processor.py`)

- Очистка служебных элементов (колонтитулы/логотипы).
- Извлечение изображений в `data/images/<pdf_stem>/`.
- Разбиение текста по секциям/подсекциям.
- Привязка `page_start/page_end` и `images` к каждому чанку.

---

## Frontend: функционал

Основной файл: `frontend/src/App.tsx`

- Поле вопроса + отправка запроса.
- Переключатель использования кэша.
- Кнопки:
  - `Статистика`
  - `Очистить кеш`
  - `Переиндексировать`
  - `Загрузить PDF/TXT`
- Карточка ответа:
  - текст ответа,
  - ссылка на документ(ы),
  - изображения релевантного раздела,
  - раскрываемый блок использованного контекста.

API-клиент во `frontend/src/api.ts`.

---

## Конфигурация (`.env`)

Быстрый старт:

```bash
# Windows (PowerShell)
Copy-Item .env.example .env
```

После копирования обязательно заполните в `.env`:

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME` (если используете нестандартное имя индекса)
- один провайдер LLM/embeddings:
  - `OPENAI_API_KEY`, либо
  - `GIGACHAT_AUTH_KEY` + `GIGACHAT_RQUID`

Минимально необходимо:

```env
# Pinecone
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=rag-demo-index
PINECONE_ENVIRONMENT=us-east-1-aws

# Один из провайдеров LLM/embeddings:
OPENAI_API_KEY=...
# или
GIGACHAT_AUTH_KEY=...
GIGACHAT_RQUID=...

# Пути
DATA_DIR=data
CACHE_DB_PATH=rag_cache.db
```

Рекомендуемые параметры:

```env
LLM_MODEL=gpt-4o-mini
GIGACHAT_MODEL=GigaChat
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500

EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSION=3072

CHUNK_SIZE=500
CHUNK_OVERLAP=100
SEARCH_TOP_K=3
SEARCH_FETCH_K=20
```

Важно: размерность Pinecone-индекса должна совпадать с `EMBEDDING_DIMENSION`.

---

## Запуск

### 1) Установка Python-зависимостей

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Запуск backend

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 3) Запуск frontend

```bash
cd frontend
npm install
npm run dev
```

Открыть: `http://localhost:5173`

---

## Работа с документами

- Положите `PDF/TXT` в `data/` или загрузите через UI.
- Инкрементальная загрузка отслеживается в `data/loaded_files.json`.
- Для полной пересборки индекса используйте `Переиндексировать`.

---

## Отладка и типичные проблемы

- **В ответах старые данные** -> очистите кэш (`/api/cache/clear`).
- **Дубли в поиске** -> проверьте, нет ли дубликатов файлов в `data/` (например `file.pdf` и `file111.pdf`).
- **Record count в Pinecone не обновился сразу** -> подождите 1-2 минуты и обновите консоль.
- **Нет изображений в ответе** -> проверьте, что у соответствующих чанков в Pinecone метаданные `images` не пустые.

---

## Дополнительно

- Оценка качества retrieval/answering:

```bash
python evaluate_ragas.py
```

Требуется рабочий `OPENAI_API_KEY`.

### RAGAS: Python 3.11 окружение

Для `ragas` рекомендуется отдельное окружение на Python 3.11 (особенно на Windows).

```bash
# 1) создать и активировать venv на Python 3.11
py -3.11 -m venv venv_py311
venv_py311\Scripts\activate

# 2) обновить pip и установить зависимости проекта
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) запустить оценку
python evaluate_ragas.py
```

Если у вас несколько версий Python, проверяйте активную:

```bash
python --version
```
