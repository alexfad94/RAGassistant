"""
Отслеживание загруженных файлов для инкрементальной загрузки в векторную БД.
Stores .loaded_files.json in project root or data/ directory.
"""

import json
from pathlib import Path
from typing import Set, Dict


def _get_tracker_path(base_dir: str | Path) -> Path:
    """Path to loaded_files.json. Uses data/loaded_files.json when base_dir is data/."""
    base = Path(base_dir).resolve()
    return base / "loaded_files.json"


def get_loaded_files(base_dir: str | Path) -> Dict[str, float]:
    """
    Получить словарь загруженных файлов: {имя_файла: mtime}.
    base_dir: project root or data/ directory.
    """
    path = _get_tracker_path(base_dir)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("files", {})
    except (json.JSONDecodeError, IOError):
        return {}


def mark_file_loaded(base_dir: str | Path, filename: str, mtime: float):
    """Отметить файл как загруженный."""
    path = _get_tracker_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"files": get_loaded_files(base_dir)}
    data["files"][filename] = mtime
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clear_loaded_files(base_dir: str | Path):
    """Очистить список загруженных файлов (для принудительной переиндексации)."""
    path = _get_tracker_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"files": {}}, f, indent=2, ensure_ascii=False)


def get_files_to_load(data_dir: Path, loaded: Dict[str, float]) -> list[tuple[Path, bool]]:
    """
    Список файлов для загрузки: (path, is_new).
    is_new=True — новый файл, is_new=False — изменённый (нужна перезагрузка).
    """
    data_dir = Path(data_dir).resolve()
    pdfs = list(data_dir.glob("*.pdf"))
    txts = list(data_dir.glob("*.txt"))
    all_files = sorted(pdfs + txts, key=lambda p: p.name)
    result = []
    for fp in all_files:
        name = fp.name
        try:
            current_mtime = fp.stat().st_mtime
        except OSError:
            continue
        if name not in loaded:
            result.append((fp, True))
        elif loaded[name] != current_mtime:
            result.append((fp, False))
    return result
