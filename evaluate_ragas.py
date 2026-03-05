"""
Оценка качества RAG системы через RAGAS.
RAGAS requires OpenAI for evaluation. Use as CLI entry: python evaluate_ragas.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from datasets import Dataset
from ragas import evaluate

try:
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._context_precision import ContextPrecision
    faithfulness = Faithfulness
    context_precision = ContextPrecision
except ImportError:
    try:
        from ragas.metrics.collections import faithfulness, context_precision
    except ImportError:
        from ragas.metrics import faithfulness, context_precision

from rag_pipeline import RAGPipeline


EVALUATION_QUESTIONS = [
    "Что такое машинное обучение?",
    "Какие основные типы машинного обучения существуют?",
    "Что такое нейронная сеть?",
    "Как работают трансформеры в NLP?",
    "Что такое RAG и как он работает?"
]


def prepare_dataset(pipeline: RAGPipeline, questions: list) -> Dataset:
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []

    print("[*] Получение ответов от RAG системы...\n")
    for i, question in enumerate(questions, 1):
        print(f"  {i}/{len(questions)}: {question}")
        result = pipeline.query(question, use_cache=False)
        questions_list.append(question)
        answers_list.append(result["answer"])
        context_texts = [doc["text"] for doc in result["context_docs"]]
        contexts_list.append(context_texts)
        ground_truths_list.append(result["answer"][:100])
        print(f"     [+] Ответ получен")

    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list
    }
    return Dataset.from_dict(dataset_dict)


def evaluate_rag_system():
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()

    if not os.getenv("OPENAI_API_KEY"):
        print("[ОШИБКА] OPENAI_API_KEY не установлен (RAGAS требует OpenAI для оценки)")
        sys.exit(1)

    try:
        print("[*] Инициализация RAG системы...\n")
        project_root = Path(__file__).parent
        data_dir = os.getenv("DATA_DIR", "data")
        cache_path = os.getenv("CACHE_DB_PATH", "rag_cache.db")
        cache_db_path = str(project_root / cache_path) if not os.path.isabs(cache_path) else cache_path
        data_dir_abs = str(project_root / data_dir) if not os.path.isabs(data_dir) else data_dir
        pipeline = RAGPipeline(
            cache_db_path=cache_db_path,
            data_file=data_dir_abs,
            loaded_files_dir=data_dir_abs,
            model=os.getenv("LLM_MODEL"),
        )
        print("\n[OK] RAG система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        sys.exit(1)

    dataset = prepare_dataset(pipeline, EVALUATION_QUESTIONS)
    print("\n[*] Запуск оценки метрик RAGAS...")
    metrics_to_use = [faithfulness(), context_precision()]

    try:
        result = evaluate(dataset=dataset, metrics=metrics_to_use)
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        sys.exit(1)

    import math
    faithfulness_values = [v for v in result['faithfulness'] if not (isinstance(v, float) and math.isnan(v))]
    context_precision_values = [v for v in result['context_precision'] if not (isinstance(v, float) and math.isnan(v))]
    avg_faithfulness = sum(faithfulness_values) / len(faithfulness_values) if faithfulness_values else 0
    avg_context_precision = sum(context_precision_values) / len(context_precision_values) if context_precision_values else 0

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)
    print(f"\n[МЕТРИКИ] Средние значения:")
    print(f"   Faithfulness:          {avg_faithfulness:.4f}")
    print(f"   Context Precision:     {avg_context_precision:.4f}")
    avg_score = (avg_faithfulness + avg_context_precision) / 2
    print(f"\n{'─'*70}")
    print(f"[ИТОГО] Средний балл: {avg_score:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_rag_system()
