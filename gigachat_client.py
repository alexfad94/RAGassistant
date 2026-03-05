"""
Клиент для работы с GigaChat API от Сбера.
Управляет авторизацией и запросами к API.
"""

import requests
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")


class GigaChatClient:
    """Клиент для работы с GigaChat API."""

    def __init__(self, auth_key: str = None, rq_uid: str = None):
        self.auth_key = auth_key or os.getenv("GIGACHAT_AUTH_KEY")
        self.rq_uid = rq_uid or os.getenv("GIGACHAT_RQUID")

        if not self.auth_key:
            raise ValueError("GIGACHAT_AUTH_KEY не установлен")
        if not self.rq_uid:
            raise ValueError("GIGACHAT_RQUID не установлен")

        self.oauth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self.api_url = "https://gigachat.devices.sberbank.ru/api/v1"
        self.access_token = None
        self.token_expires_at = None
        self._refresh_token()

    def _refresh_token(self):
        payload = {'scope': 'GIGACHAT_API_PERS'}
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': self.rq_uid,
            'Authorization': f'Basic {self.auth_key}'
        }
        try:
            response = requests.post(
                self.oauth_url,
                headers=headers,
                data=payload,
                verify=False
            )
            response.raise_for_status()
            data = response.json()
            self.access_token = data['access_token']
            self.token_expires_at = datetime.now() + timedelta(minutes=29)
            print("✓ GigaChat access token получен")
        except Exception as e:
            raise Exception(f"Ошибка получения access token: {e}")

    def _ensure_token_valid(self):
        if not self.access_token or datetime.now() >= self.token_expires_at:
            print("⟳ Обновление access token...")
            self._refresh_token()

    def _get_headers(self) -> Dict[str, str]:
        self._ensure_token_valid()
        return {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }

    def chat_completion(self, messages: List[Dict[str, str]], model: str = "GigaChat",
                       temperature: float = 0.3, max_tokens: int = 500) -> str:
        url = f"{self.api_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                verify=False
            )
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except Exception as e:
            raise Exception(f"Ошибка запроса к GigaChat: {e}")

    def get_embeddings(self, texts: List[str], model: str = "Embeddings") -> List[List[float]]:
        url = f"{self.api_url}/embeddings"
        payload = {"model": model, "input": texts}
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                verify=False
            )
            response.raise_for_status()
            data = response.json()
            return [item['embedding'] for item in data['data']]
        except Exception as e:
            print(f"⚠️  Embeddings API недоступен, fallback: {e}")
            import hashlib
            embeddings = []
            for text in texts:
                hash_obj = hashlib.sha256(text.encode())
                hash_bytes = hash_obj.digest()
                vector = [(hash_bytes[i % len(hash_bytes)] / 255.0) - 0.5 for i in range(1024)]
                embeddings.append(vector)
            return embeddings

    def get_models(self) -> List[Dict[str, Any]]:
        url = f"{self.api_url}/models"
        try:
            response = requests.get(url, headers=self._get_headers(), verify=False)
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
        except Exception as e:
            print(f"Ошибка получения списка моделей: {e}")
            return []
