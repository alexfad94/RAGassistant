"""
FastAPI backend для RAG ассистента.
Предоставляет REST API для работы с векторной БД и RAG pipeline.
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь для импорта модулей
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Импорт из корня проекта
from rag_pipeline import RAGPipeline


# Глобальный экземпляр pipeline (инициализируется при старте)
pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация и очистка при старте/остановке."""
    global pipeline
    try:
        data_dir_name = os.getenv("DATA_DIR", "data")
        cache_path = os.getenv("CACHE_DB_PATH", "rag_cache.db")
        data_dir = project_root / data_dir_name
        cache_db_path = str(project_root / cache_path) if not os.path.isabs(cache_path) else cache_path
        pipeline = RAGPipeline(
            cache_db_path=cache_db_path,
            data_file=str(data_dir),
            model=os.getenv("LLM_MODEL"),
            loaded_files_dir=str(data_dir),
        )
        print("RAG Pipeline инициализирован")
    except Exception as e:
        print(f"Ошибка инициализации: {e}")
        pipeline = None
    yield
    pipeline = None


app = FastAPI(
    title="RAG Assistant API",
    description="API для работы с векторной БД и RAG ассистентом",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic модели ---


class QueryRequest(BaseModel):
    query: str
    use_cache: bool = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    from_cache: bool
    context_docs: list | None = None
    model: str | None = None
    cached_at: str | None = None


class StatsResponse(BaseModel):
    vector_store: dict
    cache: dict
    model: str
    mode: str | None = None


# --- Endpoints ---


@app.get("/api/health")
async def health():
    """Проверка доступности API."""
    return {"status": "ok", "pipeline_ready": pipeline is not None}


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Отправить вопрос в RAG систему."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline не инициализирован")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

    try:
        result = pipeline.query(request.query.strip(), use_cache=request.use_cache)
        # context_docs из кеша — список строк, из RAG — список dict
        raw_docs = result.get("context_docs") or []
        context_docs = []
        for d in raw_docs:
            if isinstance(d, dict):
                context_docs.append({
                    "id": d.get("id"),
                    "text": d.get("text", d.get("content", str(d))),
                    "distance": d.get("distance"),
                    "images": d.get("images", []),
                    "chunk_number": d.get("chunk_number"),
                    "section_header": d.get("section_header"),
                    "source": d.get("source"),
                })
            else:
                context_docs.append({"id": None, "text": str(d), "distance": None, "images": [], "chunk_number": None, "section_header": None, "source": None})
        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            from_cache=result["from_cache"],
            context_docs=context_docs,
            model=result.get("model"),
            cached_at=result.get("cached_at"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Получить статистику системы (векторное хранилище, кеш)."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline не инициализирован")

    try:
        stats = pipeline.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/images/{path:path}")
async def serve_image(path: str):
    """Отдача изображений из папки data/images."""
    data_dir_name = os.getenv("DATA_DIR", "data")
    images_dir = project_root / data_dir_name / "images"
    full_path = (images_dir / path).resolve()
    if not str(full_path).startswith(str(images_dir.resolve())):
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    return FileResponse(full_path)


@app.get("/api/documents/{filename:path}")
async def serve_document(filename: str):
    """Отдача документов (PDF/TXT) из папки data."""
    data_dir_name = os.getenv("DATA_DIR", "data")
    data_dir = project_root / data_dir_name
    full_path = (data_dir / filename).resolve()
    if not str(full_path).startswith(str(data_dir.resolve())):
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Документ не найден")
    if full_path.suffix.lower() not in (".pdf", ".txt"):
        raise HTTPException(status_code=400, detail="Поддерживаются только PDF и TXT")
    media_type = "application/pdf" if full_path.suffix.lower() == ".pdf" else "text/plain; charset=utf-8"
    return FileResponse(full_path, media_type=media_type, filename=full_path.name)


@app.post("/api/reindex")
async def reindex():
    """Принудительная переиндексация всех документов (очистить кеш и загрузить заново)."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline не инициализирован")
    try:
        pipeline.cache.clear()
        data_dir_name = os.getenv("DATA_DIR", "data")
        data_dir = project_root / data_dir_name
        result = pipeline.vector_store.reindex_all(data_dir)
        return {
            "status": "ok",
            "message": result.get("message", "Документы переиндексированы"),
            "loaded_count": result.get("loaded_count", 0),
            "files_loaded": result.get("files_loaded", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cache/clear")
async def clear_cache():
    """Очистить кеш."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline не инициализирован")

    try:
        pipeline.cache.clear()
        return {"status": "ok", "message": "Кеш очищен"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Загрузить PDF или TXT в базу документов."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline не инициализирован")

    name = file.filename or "document"
    if not name.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Поддерживаются только PDF и TXT")

    data_dir_name = os.getenv("DATA_DIR", "data")
    data_dir = project_root / data_dir_name
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / name

    try:
        content = await file.read()
        dest.write_bytes(content)
        pipeline.vector_store.load_new_documents(data_dir)
        return {"status": "ok", "message": f"Файл {name} загружен и проиндексирован"}
    except Exception as e:
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
