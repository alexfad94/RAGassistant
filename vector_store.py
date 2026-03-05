"""
Pinecone vector store with OpenAI or GigaChat embeddings.
Supports PDF and TXT loading with chunking and metadata (source, images).
"""

import json
import hashlib
import os
import re
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def _make_vector_id(source: str, doc_id: str | int, chunk_id: int, text: str) -> str:
    """Generate unique vector ID for Pinecone."""
    content = f"{source}|{doc_id}|{chunk_id}|{text[:200]}"
    return hashlib.sha256(content.encode()).hexdigest()[:64]


class VectorStore:
    """Pinecone vector store with OpenAI or GigaChat embeddings."""

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        environment: str | None = None,
        loaded_files_dir: str | Path = "data",
    ):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "rag-index")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
        self.loaded_files_dir = Path(loaded_files_dir)

        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is required")

        self._init_embedder()
        self._init_pinecone()
        self.index = self.pc.Index(self.index_name)
        print(f"VectorStore initialized, index: {self.index_name}")

    def _init_embedder(self):
        """Initialize embedding client: OpenAI if key set, else GigaChat."""
        if os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            self._embedder_type = "openai"
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self._embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
            dim = os.getenv("EMBEDDING_DIMENSION")
            self._embedding_dim = int(dim) if dim else (3072 if "large" in self._embedding_model else 1536)
        elif os.getenv("GIGACHAT_AUTH_KEY") and os.getenv("GIGACHAT_RQUID"):
            from gigachat_client import GigaChatClient
            self._embedder_type = "gigachat"
            self._gigachat_client = GigaChatClient()
            dim = os.getenv("EMBEDDING_DIMENSION")
            self._embedding_dim = int(dim) if dim else 1024
        else:
            raise ValueError("Set OPENAI_API_KEY or GIGACHAT_AUTH_KEY+GIGACHAT_RQUID")

    def _init_pinecone(self):
        from pinecone import Pinecone, ServerlessSpec, CloudProvider
        self.pc = Pinecone(api_key=self.api_key)
        if not self.pc.has_index(name=self.index_name):
            region = self.environment.replace("-aws", "").replace("-gcp", "") or "us-east-1"
            print(f"Creating Pinecone index {self.index_name} (dim={self._embedding_dim})...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self._embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=CloudProvider.AWS, region=region),
            )
            print(f"Index {self.index_name} created")

    def _create_embedding(self, text: str) -> List[float]:
        if self._embedder_type == "openai":
            r = self._openai_client.embeddings.create(input=text, model=self._embedding_model)
            return r.data[0].embedding
        else:
            embs = self._gigachat_client.get_embeddings([text])
            return embs[0]

    def _create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if self._embedder_type == "openai":
            r = self._openai_client.embeddings.create(input=texts, model=self._embedding_model)
            return [d.embedding for d in r.data]
        else:
            return self._gigachat_client.get_embeddings(texts)

    def _chunk_text(self, text: str, chunk_size: int | None = None, overlap: int | None = None) -> List[str]:
        chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "500"))
        overlap = overlap or int(os.getenv("CHUNK_OVERLAP", "100"))
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                current_chunk = (current_chunk + "\n\n" + paragraph) if current_chunk else paragraph
            elif current_chunk:
                chunks.append(current_chunk)
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = (overlap_text + "\n\n" + paragraph) if overlap_text else paragraph
            else:
                if len(paragraph) > chunk_size:
                    sentence_chunks = self._split_long_paragraph(paragraph, chunk_size, overlap)
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk)
        return [c for c in chunks if len(c) >= 50]

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        if len(text) <= overlap_size:
            return text
        overlap_candidate = text[-overlap_size:]
        for delim in [". ", "! ", "? ", "\n"]:
            pos = overlap_candidate.find(delim)
            if pos != -1 and pos > 0:
                return overlap_candidate[pos + len(delim) :].strip()
        return overlap_candidate.strip()

    def _split_long_paragraph(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        sentences = re.split(r"([.!?]+\s+)", paragraph)
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])
        chunks = []
        current_chunk = ""
        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk = (current_chunk + " " + sentence) if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = (overlap_text + " " + sentence) if overlap_text else sentence
                else:
                    current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def load_documents(self, file_path: str, images_dir: str | None = None):
        """Load documents from TXT or PDF file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            self._load_pdf(str(file_path), images_dir)
        else:
            self._load_txt(str(file_path))

    def reindex_all(self, data_dir: str | Path) -> dict:
        """Принудительная переиндексация: удалить все векторы, очистить images и загрузить заново.
        Возвращает: {loaded_count, files_loaded, message}."""
        import shutil
        from loaded_files import clear_loaded_files, get_loaded_files, get_files_to_load, mark_file_loaded

        data_dir = Path(data_dir).resolve()
        if not data_dir.is_dir():
            return {"loaded_count": 0, "files_loaded": [], "message": "Папка data/ не найдена"}
        loaded = get_loaded_files(self.loaded_files_dir)
        for filename in loaded:
            self._delete_by_source(filename)
        clear_loaded_files(self.loaded_files_dir)
        images_dir = data_dir / "images"
        if images_dir.is_dir():
            shutil.rmtree(images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)
        print("Reindex: cleared data/images/")
        print("Reindex: cleared loaded_files, reloading all...")
        result = self._load_new_documents_with_result(data_dir)
        return result

    def _load_new_documents_with_result(self, data_dir: Path) -> dict:
        """Load documents and return count + list of loaded files."""
        from loaded_files import get_loaded_files, get_files_to_load, mark_file_loaded

        data_dir = data_dir.resolve()
        loaded = get_loaded_files(self.loaded_files_dir)
        to_load = get_files_to_load(data_dir, loaded)
        pdfs = list(data_dir.glob("*.pdf"))
        txts = list(data_dir.glob("*.txt"))
        print(f"Load: data_dir={data_dir}, pdfs={[p.name for p in pdfs]}, txts={[p.name for p in txts]}, loaded={list(loaded.keys())}, to_load={len(to_load)}")
        if to_load:
            print(f"Load: files to load: {[fp.name for fp, _ in to_load]}")
        if not to_load:
            msg = "Нет новых или изменённых файлов" if loaded else "В папке data/ нет PDF или TXT файлов"
            print(f"Load: {msg}")
            return {"loaded_count": 0, "files_loaded": [], "message": msg}
        files_loaded = []
        for file_path, is_new in to_load:
            filename = file_path.name
            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                continue
            if not is_new:
                self._delete_by_source(filename)
            print(f"Loading {'new' if is_new else 'updated'} file: {filename}")
            self._load_file_incremental(file_path, mtime, mark_file_loaded)
            files_loaded.append(filename)
        return {"loaded_count": len(files_loaded), "files_loaded": files_loaded, "message": f"Загружено {len(files_loaded)} файл(ов)"}

    def load_new_documents(self, data_dir: str | Path):
        """Load new or modified files from directory."""
        self._load_new_documents_with_result(Path(data_dir))

    def _delete_by_source(self, source_filename: str):
        """Delete vectors by source (query with filter to get IDs, then delete)."""
        try:
            # Query with high top_k to get all matching IDs (Pinecone limit ~10000)
            filter_dict = {"source": {"$eq": source_filename}}
            query_emb = self._create_embedding(source_filename[:100])  # Dummy query for filter
            results = self.index.query(
                vector=query_emb,
                top_k=10000,
                include_metadata=True,
                filter=filter_dict,
            )
            ids_to_delete = [m.id for m in results.matches]
            if ids_to_delete:
                self.index.delete(ids=ids_to_delete)
                print(f"  Deleted {len(ids_to_delete)} chunks from {source_filename}")
        except Exception as e:
            print(f"  Warning on delete: {e}")

    def _load_file_incremental(self, file_path: Path, mtime: float, mark_fn):
        suffix = file_path.suffix.lower()
        stem = file_path.stem
        images_dir = file_path.parent / "images" if suffix == ".pdf" else None
        if suffix == ".pdf":
            self._load_pdf(str(file_path), str(images_dir) if images_dir else None, id_prefix=f"doc_{stem}")
        else:
            self._load_txt(str(file_path), id_prefix=f"doc_{stem}")
        mark_fn(self.loaded_files_dir, file_path.name, mtime)

    def _load_txt(self, file_path: str, id_prefix: str = "doc"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = self._chunk_text(text)
        print(f"  Text split into {len(chunks)} chunks")
        self._upsert_chunks(
            chunks,
            [{"source": Path(file_path).name, "images": "[]", "chunk_number": i + 1} for i in range(len(chunks))],
            id_prefix,
        )
        print(f"  Loaded {len(chunks)} chunks from {Path(file_path).name}")

    def _load_pdf(self, file_path: str, images_dir: str | None = None, id_prefix: str = "doc"):
        from pdf_processor import process_pdf, ProcessedChunk

        file_path = Path(file_path)
        stem = file_path.stem
        data_dir = file_path.parent
        images_dir = Path(images_dir) if images_dir else data_dir / "images"
        chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
        chunks_data: List[ProcessedChunk] = process_pdf(
            str(file_path), images_dir=images_dir, chunk_size=chunk_size, overlap=overlap
        )
        prefix = id_prefix if id_prefix != "doc" else f"doc_{stem}"
        texts = [pc.text for pc in chunks_data]
        metadatas = [
            {
                "source": pc.source_file,
                "page_start": str(pc.page_start),
                "page_end": str(pc.page_end),
                "images": json.dumps(pc.images),
                "chunk_number": i + 1,
                "section_header": pc.section_header or "",
            }
            for i, pc in enumerate(chunks_data)
        ]
        self._upsert_chunks(texts, metadatas, prefix)
        print(f"  Loaded {len(chunks_data)} chunks from {file_path.name}")

    def _upsert_chunks(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        id_prefix: str,
    ):
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]
            embeddings = self._create_embeddings_batch(batch_texts)
            vectors = []
            for j, (text, emb, meta) in enumerate(zip(batch_texts, embeddings, batch_metas)):
                chunk_id = i + j
                vid = _make_vector_id(meta.get("source", ""), id_prefix, chunk_id, text)
                meta_copy = {"text": text, **meta}
                vectors.append({"id": vid, "values": emb, "metadata": meta_copy})
            self.index.upsert(vectors=vectors)
            if (i + batch_size) % 50 == 0 or i + batch_size >= len(texts):
                print(f"  Upserted {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    def search(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        top_k = top_k or int(os.getenv("SEARCH_TOP_K", "3").strip())
        fetch_k_default = max(top_k * 8, 20)
        fetch_k = int(os.getenv("SEARCH_FETCH_K", str(fetch_k_default)).strip())
        fetch_k = max(top_k, fetch_k)
        query_emb = self._create_embedding(query)
        results = self.index.query(
            vector=query_emb,
            top_k=fetch_k,
            include_metadata=True,
        )
        query_lc = query.lower().strip()
        query_tokens = {t for t in re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", query_lc) if len(t) >= 3}

        def overlap_bonus(text: str, weight: float, cap: float) -> float:
            if not text or not query_tokens:
                return 0.0
            tokens = {t for t in re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text.lower()) if len(t) >= 3}
            if not tokens:
                return 0.0
            overlap = len(query_tokens.intersection(tokens))
            return min(cap, overlap * weight)

        documents = []
        for match in results.matches:
            meta = match.metadata or {}
            images_raw = meta.get("images", [])
            images = json.loads(images_raw) if isinstance(images_raw, str) else (images_raw or [])
            chunk_num = meta.get("chunk_number")
            section_header = meta.get("section_header") or None
            if section_header == "":
                section_header = None
            text = meta.get("text", "")
            base_score = float(match.score) if match.score is not None else 0.0
            rerank_bonus = 0.0
            if section_header:
                header_lc = section_header.lower()
                if query_lc and query_lc in header_lc:
                    rerank_bonus += 0.35
                rerank_bonus += overlap_bonus(header_lc, weight=0.09, cap=0.30)
            text_preview = text[:800]
            if query_lc and query_lc in text_preview.lower():
                rerank_bonus += 0.12
            rerank_bonus += overlap_bonus(text_preview, weight=0.03, cap=0.20)
            documents.append({
                "id": match.id,
                "text": text,
                "distance": base_score,
                "source": meta.get("source"),
                "images": images,
                "chunk_number": int(chunk_num) if chunk_num is not None else None,
                "section_header": section_header,
                "_rerank_score": base_score + rerank_bonus,
            })
        documents.sort(key=lambda d: d.get("_rerank_score", d["distance"]), reverse=True)
        for d in documents:
            d.pop("_rerank_score", None)
        return documents[:top_k]

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            from loaded_files import get_loaded_files

            stats = self.index.describe_index_stats()
            total = getattr(stats, "total_vector_count", None)
            if total is None and hasattr(stats, "namespaces"):
                total = sum(
                    getattr(ns, "vector_count", getattr(ns, "record_count", 0))
                    for ns in stats.namespaces.values()
                )
            loaded_files = get_loaded_files(self.loaded_files_dir)
            documents_count = len(loaded_files)
            return {
                "name": self.index_name,
                "count": total or 0,
                "chunks_count": total or 0,
                "documents_count": documents_count,
                "loaded_files_dir": str(self.loaded_files_dir),
            }
        except Exception as e:
            return {"error": str(e), "name": self.index_name, "count": 0}
