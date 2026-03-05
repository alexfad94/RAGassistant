"""
RAG pipeline with OpenAI or GigaChat fallback.
Flow: cache -> vector search -> LLM -> cache.
"""

from typing import Dict, Any, List
import os
from pathlib import Path

from dotenv import load_dotenv

# Загрузка .env до импорта vector_store (SEARCH_TOP_K и др.)
load_dotenv(Path(__file__).resolve().parent / ".env")

from vector_store import VectorStore
from cache import RAGCache
from loaded_files import get_loaded_files


class RAGPipeline:
    """RAG pipeline: OpenAI if OPENAI_API_KEY, else GigaChat if GIGACHAT keys set."""

    def __init__(
        self,
        cache_db_path: str | None = None,
        data_file: str | None = None,
        model: str | None = None,
        loaded_files_dir: str | None = None,
    ):
        self._detect_provider()
        self.model = model or (self._openai_model if self._provider == "openai" else self._gigachat_model)
        self.loaded_files_dir = Path(loaded_files_dir or os.getenv("DATA_DIR", "data"))

        print("Initializing vector store...")
        self.vector_store = VectorStore(loaded_files_dir=str(self.loaded_files_dir))

        data_path = Path(data_file or os.getenv("DATA_DIR", "data")).resolve()
        if data_path.is_dir():
            print("Checking for new files in data/...")
            stats = self.vector_store.get_collection_stats()
            index_count = stats.get("count", 0)
            loaded = get_loaded_files(self.loaded_files_dir)
            if index_count == 0 and loaded:
                print("Index is empty but loaded_files has entries — forcing reindex")
                self.vector_store.reindex_all(data_path)
            else:
                self.vector_store.load_new_documents(data_path)
        else:
            stats = self.vector_store.get_collection_stats()
            count = stats.get("count", 0)
            if count == 0:
                print(f"Loading documents from {data_file}...")
                self.vector_store.load_documents(data_file)

        print("Initializing cache...")
        self.cache = RAGCache(db_path=cache_db_path or os.getenv("CACHE_DB_PATH", "rag_cache.db"))
        print(f"RAG Pipeline initialized ({self._provider} mode)")

    def _detect_provider(self):
        if os.getenv("OPENAI_API_KEY"):
            self._provider = "openai"
            self._openai_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
            self._openai_client = __import__("openai").OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif os.getenv("GIGACHAT_AUTH_KEY") and os.getenv("GIGACHAT_RQUID"):
            self._provider = "gigachat"
            self._gigachat_model = os.getenv("GIGACHAT_MODEL", "GigaChat")
            from gigachat_client import GigaChatClient
            self._gigachat_client = GigaChatClient()
        else:
            raise ValueError("Set OPENAI_API_KEY or GIGACHAT_AUTH_KEY+GIGACHAT_RQUID")

    def _create_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            header = doc.get("section_header") or ""
            label = f"Документ {i}" + (f" [заголовок: {header}]" if header else "")
            context_parts.append(f"{label}:\n{doc['text']}\n")
        context = "\n".join(context_parts)
        return f"""Ты - полезный AI ассистент. Ответь на вопрос пользователя на основе предоставленного контекста.

Контекст:
{context}

Вопрос: {query}

Инструкции:
- В первую очередь используй информацию из документов, заголовок раздела которых совпадает с темой вопроса (смотри поле [заголовок: ...]).
- Если таких документов несколько и они из разных разделов (например, 1.3 и 6.3) — дай консолидированный ответ, объединив информацию, и в конце явно укажи: «Информация актуальна для разделов: X.X, Y.Y» (перечисли номера разделов из заголовков без дублей).
- Отвечай только на основе предоставленного контекста
- Если в контексте нет информации для ответа, скажи об этом
- Будь точным и кратким
- Отвечай на русском языке

Ответ:"""

    def _generate_answer(self, prompt: str) -> str:
        if self._provider == "openai":
            r = self._openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты - полезный AI ассистент, который отвечает на вопросы на основе предоставленного контекста."},
                    {"role": "user", "content": prompt}
                ],
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "500")),
            )
            return r.choices[0].message.content.strip()
        else:
            messages = [
                {"role": "system", "content": "Ты - полезный AI ассистент, который отвечает на вопросы на основе предоставленного контекста."},
                {"role": "user", "content": prompt}
            ]
            return self._gigachat_client.chat_completion(
                messages=messages,
                model=self.model,
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "500")),
            ).strip()

    def query(self, user_query: str, use_cache: bool = True) -> Dict[str, Any]:
        print(f"\n{'='*60}\nЗапрос: {user_query}\n{'='*60}")

        if use_cache:
            cached_result = self.cache.get(user_query)
            if cached_result:
                print("[+] Ответ найден в кеше")
                ctx = cached_result.get("context") or []
                context_docs = []
                for c in ctx:
                    if isinstance(c, dict):
                        context_docs.append({
                            "text": c.get("text", ""),
                            "images": c.get("images", []),
                            "chunk_number": c.get("chunk_number"),
                            "section_header": c.get("section_header"),
                            "source": c.get("source"),
                        })
                    else:
                        context_docs.append({"text": str(c), "images": [], "chunk_number": None, "section_header": None, "source": None})
                return {
                    "query": user_query,
                    "answer": cached_result["answer"],
                    "from_cache": True,
                    "context_docs": context_docs,
                    "cached_at": cached_result.get("created_at"),
                }
            print("[-] Ответ не найден в кеше")

        print("[*] Поиск релевантных документов...")
        top_k = int(os.getenv("SEARCH_TOP_K", "3").strip())
        context_docs = self.vector_store.search(user_query, top_k=top_k)
        print(f"[+] Найдено {len(context_docs)} релевантных документов")

        prompt = self._create_prompt(user_query, context_docs)
        print(f"[*] Генерация ответа через {self.model}...")
        answer = self._generate_answer(prompt)
        print("[+] Ответ получен")

        if use_cache:
            context_for_cache = [
                {"text": d["text"], "images": d.get("images", []), "chunk_number": d.get("chunk_number"), "section_header": d.get("section_header"), "source": d.get("source")}
                for d in context_docs
            ]
            self.cache.set(user_query, answer, context_for_cache)

        return {
            "query": user_query,
            "answer": answer,
            "from_cache": False,
            "context_docs": context_docs,
            "model": self.model,
            "mode": self._provider.upper(),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "model": self.model,
            "mode": self._provider.upper(),
        }
