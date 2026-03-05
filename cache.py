"""
Модуль кеширования для RAG ассистента.
Использует SQLite для хранения пар вопрос-ответ с временными метками.
"""

import sqlite3
import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, Any
import os


class RAGCache:
    """Кеш для хранения результатов RAG запросов."""

    def __init__(self, db_path: str = "rag_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                query_hash TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def _get_query_hash(self, query: str) -> str:
        normalized_query = " ".join(query.lower().strip().split())
        return hashlib.sha256(normalized_query.encode()).hexdigest()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        query_hash = self._get_query_hash(query)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT query, answer, context, created_at
            FROM cache WHERE query_hash = ?
        """, (query_hash,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return {
                "query": result[0],
                "answer": result[1],
                "context": json.loads(result[2]) if result[2] else None,
                "created_at": result[3],
                "from_cache": True
            }
        return None

    def set(self, query: str, answer: str, context: list = None):
        query_hash = self._get_query_hash(query)
        context_json = json.dumps(context) if context else None
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO cache (query_hash, query, answer, context, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (query_hash, query, answer, context_json, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def clear(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache")
        conn.commit()
        conn.close()

    def get_stats(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cache")
        count = cursor.fetchone()[0]
        cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM cache")
        dates = cursor.fetchone()
        conn.close()
        return {
            "total_entries": count,
            "oldest_entry": dates[0] if dates[0] else None,
            "newest_entry": dates[1] if dates[1] else None,
            "db_size_mb": os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
        }
