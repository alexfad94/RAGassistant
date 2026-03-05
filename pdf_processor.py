"""
Обработка PDF: извлечение текста и изображений с очисткой.
- Удаление титульного листа
- Содержание (оглавление) включается в full_text для корректного char→page маппинга,
  но в чанки попадает только body (через _get_body_text_and_offset)
- Удаление колонтитулов
- Удаление логотипов
- Чанкинг с раздела 1
- Сохранение изображений в отдельную папку
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


# Паттерны для удаления колонтитулов
HEADER_FOOTER_PATTERNS = [
    r"Страница\s+\d+\s+из\s+\d+",
    r"Стр\.\s*\d+",
    r"^\d+\s*$",  # Только номер страницы
    r"Вернуться в содержание",
    r"ZONT CONNECT\+",
    r"Техническая документация",
    r"ML\.TD\.\w+\.\d+",  # Код документа типа ML.TD.BCPL.01
]

# Паттерны для определения начала раздела 1
SECTION_1_PATTERNS = [
    r"^\s*1\.\s+",           # 1. Название
    r"^\s*1\s+",             # 1 Название
    r"^\s*Раздел\s+1\b",
    r"^\s*1\s+[А-ЯA-Z]",
]

# Минимальная площадь изображения (в пикселях²) — меньше считаем логотипом
LOGO_MAX_AREA = 15000  # ~122x122
# Площадь для средних логотипов в колонтитулах (до ~200x200)
LOGO_HEADER_MAX_AREA = 50000
# Максимальное соотношение сторон для логотипа (квадрат ± 30%)
LOGO_ASPECT_RATIO = (0.7, 1.4)
# Зона колонтитулов: верхние и нижние 15% страницы
HEADER_FOOTER_ZONE = 0.15

# Паттерны для извлечения отображаемого номера страницы из текста (до препроцессинга)
# Поддержка обычных и неразрывных пробелов
_PAGE_FROM_TEXT_RE = re.compile(
    r"(?:Страница[\s\u00a0]+(\d+)[\s\u00a0]*(?:из[\s\u00a0]*\d+)?|Стр\.[\s\u00a0]*(\d+))",
    re.IGNORECASE,
)


def _extract_page_number_from_text(text: str) -> int | None:
    """Извлекает отображаемый номер страницы из текста (Страница X из Y, Стр. X).
    Берём последнее вхождение — обычно в колонтитуле внизу страницы."""
    matches = list(_PAGE_FROM_TEXT_RE.finditer(text))
    if matches:
        m = matches[-1]
        val = m.group(1) or m.group(2)
        return int(val) if val else None
    return None


@dataclass
class ProcessedChunk:
    """Чанк с метаданными."""
    text: str
    source_file: str
    page_start: int
    page_end: int
    images: List[str]  # Относительные пути к изображениям
    section_header: str | None = None  # Заголовок раздела (1. Назначение, 2. Функции и т.д.)


def _is_logo_image(rect_width: float, rect_height: float, max_area: int = LOGO_HEADER_MAX_AREA) -> bool:
    """Проверка, является ли изображение логотипом (маленькое, квадратное)."""
    area = rect_width * rect_height
    if area < LOGO_MAX_AREA:
        return True
    if rect_height == 0:
        return False
    ratio = rect_width / rect_height
    return LOGO_ASPECT_RATIO[0] <= ratio <= LOGO_ASPECT_RATIO[1] and area < max_area


def _is_header_footer_image(page, xref: int) -> bool:
    """Проверка, находится ли изображение в зоне колонтитулов (верх/низ страницы)."""
    try:
        rects = page.get_image_rects(xref)
        if not rects:
            return False
        page_height = page.rect.height
        header_limit = page_height * HEADER_FOOTER_ZONE
        footer_limit = page_height * (1 - HEADER_FOOTER_ZONE)
        for r in rects:
            if r.y1 <= header_limit:
                return True
            if r.y0 >= footer_limit:
                return True
            if r.height < page_height * 0.12 and (r.y0 + r.y1) / 2 < page_height * 0.2:
                return True
            if r.height < page_height * 0.12 and (r.y0 + r.y1) / 2 > page_height * 0.8:
                return True
    except Exception:
        pass
    return False


def _is_in_content_zone(page, xref: int) -> bool:
    """Проверка, находится ли изображение в основной зоне контента (не колонтитулы)."""
    try:
        rects = page.get_image_rects(xref)
        if not rects:
            return False
        page_height = page.rect.height
        content_top = page_height * HEADER_FOOTER_ZONE
        content_bottom = page_height * (1 - HEADER_FOOTER_ZONE)
        for r in rects:
            center_y = (r.y0 + r.y1) / 2
            if content_top <= center_y <= content_bottom:
                return True
    except Exception:
        pass
    return False


def _clean_header_footer(text: str) -> str:
    """Удаление строк, похожих на колонтитулы."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            cleaned.append(line)
            continue
        is_header_footer = False
        for pattern in HEADER_FOOTER_PATTERNS:
            if re.search(pattern, line_stripped, re.IGNORECASE):
                is_header_footer = True
                break
        if not is_header_footer:
            cleaned.append(line)
    return "\n".join(cleaned)


def _is_title_page(text: str, page_num: int) -> bool:
    """Титульный лист — обычно первая страница с малым текстом и заголовком."""
    if page_num > 1:
        return False
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return len(lines) <= 5


# Паттерн строки оглавления: "1. Название", "1.1 Подраздел", "Приложение 1. Название"
_TOC_ENTRY_RE = re.compile(
    r"^\s*(\d+(\.\d+)*\.?\s*|Приложение\s+\d+\.?\s*)[А-Яа-яA-Za-z0-9\s\-–—,.]{1,90}$",
    re.MULTILINE,
)


def _is_toc_page(text: str) -> bool:
    """Страница с заголовком содержания."""
    toc_indicators = [
        "содержание",
        "оглавление",
        "table of contents",
        "вернуться в содержание",
    ]
    text_lower = text.lower()
    return any(ind in text_lower for ind in toc_indicators)


def _is_toc_content(text: str) -> bool:
    """
    Текст — это оглавление (список разделов), если большинство непустых строк
    выглядят как пункты оглавления: короткие, с нумерацией.
    Если есть строка с основным текстом (>120 символов) — это не оглавление.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return True
    for line in lines:
        if len(line) > 120:
            return False
    toc_count = 0
    for line in lines:
        if len(line) < 100 and _TOC_ENTRY_RE.match(line):
            toc_count += 1
        elif len(line) < 70 and re.match(r"^\s*\d+(\.\d+)*\.?\s*", line):
            toc_count += 1
        elif re.match(r"^\s*Приложение\s+\d+", line, re.IGNORECASE):
            toc_count += 1
        elif len(line) < 60 and (line.isupper() or re.match(r"^[А-ЯA-Z]", line)):
            toc_count += 1
    return toc_count >= max(1, len(lines) * 0.5)


def _get_body_text_and_offset(text: str) -> Tuple[str, int]:
    """
    Возвращает (текст основного контента без оглавления, смещение в полном тексте).
    body_text — срез full_text, чтобы позиции совпадали с char_to_page_and_images.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ("", 0)
    first_body_idx = None
    for i, para in enumerate(paragraphs):
        if len(para) > 120:
            first_body_idx = i
            break
        if not _is_toc_content(para):
            first_body_idx = i
            break
    if first_body_idx is None:
        return ("", 0)
    start = first_body_idx
    if first_body_idx > 0 and re.match(r"^\s*\d+(\.\d+)*\.?\s+[А-Яа-яA-Z]", paragraphs[first_body_idx - 1]):
        start = first_body_idx - 1
    if not paragraphs[start:]:
        return ("", 0)
    needle = paragraphs[start][:100] if len(paragraphs[start]) > 100 else paragraphs[start]
    # Ищем вхождение в body: первое может быть в TOC, берём то, что после ~15% текста
    first_occ = text.find(needle)
    if first_occ < 0:
        pos = 0
        for i in range(start):
            idx = text.find(paragraphs[i], pos)
            if idx < 0:
                break
            pos = idx + len(paragraphs[i])
            while pos < len(text) and text[pos] in " \t\n":
                pos += 1
        offset = pos
    else:
        body_start_approx = int(len(text) * 0.15)
        if first_occ < body_start_approx:
            second_occ = text.find(needle, first_occ + len(needle))
            offset = second_occ if second_occ >= 0 else first_occ
        else:
            offset = first_occ
    body_text = text[offset:]
    return (body_text, offset)


def _starts_section_1(text: str) -> bool:
    """Проверка, начинается ли текст с раздела 1."""
    first_lines = "\n".join(text.split("\n")[:3])
    for pattern in SECTION_1_PATTERNS:
        if re.search(pattern, first_lines, re.MULTILINE):
            return True
    return False


def process_pdf(
    pdf_path: str,
    images_dir: str | Path,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[ProcessedChunk]:
    """
    Обработка PDF: очистка, извлечение текста и изображений.
    """
    if fitz is None:
        raise ImportError("Install PyMuPDF: pip install pymupdf")

    pdf_path = Path(pdf_path)
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    doc_images_dir = images_dir / pdf_path.stem
    doc_images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    source_name = pdf_path.name
    page_labels: Dict[int, str] = {}  # page_index (1-based) -> отображаемый номер
    all_text_parts: List[Tuple[int, str, List[str]]] = []
    all_text_is_toc: List[bool] = []
    found_section_1 = False
    pages_skipped = 0
    max_skip_before_fallback = 10
    in_toc_zone = False

    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text()
        # Извлекаем отображаемый номер из текста (Страница 52 из 133) ДО любой очистки
        displayed = _extract_page_number_from_text(raw_text)
        if displayed is not None:
            page_labels[page_num + 1] = str(displayed)
        else:
            try:
                label = page.get_label()
                lbl = str(label).strip() if label else None
                if lbl and lbl.isdigit():
                    page_labels[page_num + 1] = lbl
                else:
                    page_labels[page_num + 1] = str(page_num + 1)
            except (AttributeError, TypeError):
                page_labels[page_num + 1] = str(page_num + 1)
        text = _clean_header_footer(raw_text)

        if page_num == 0 and _is_title_page(text, page_num):
            pages_skipped += 1
            continue
        if _is_toc_page(text):
            in_toc_zone = True
        if in_toc_zone and _is_toc_content(text):
            # Оглавление не удаляем — включаем в full_text для корректного char→page маппинга.
            # _get_body_text_and_offset вернёт body_text и toc_offset; чанкинг только по body.
            pass
        elif in_toc_zone and not _is_toc_page(text):
            in_toc_zone = False

        is_toc_page = _is_toc_page(text) or (in_toc_zone and _is_toc_content(text))
        if not is_toc_page and not found_section_1:
            if pages_skipped >= max_skip_before_fallback:
                found_section_1 = True
            elif _starts_section_1(text):
                found_section_1 = True
            else:
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                s = span.get("text", "").strip()
                                for pat in SECTION_1_PATTERNS:
                                    if re.match(pat, s):
                                        found_section_1 = True
                                        break
                if not found_section_1:
                    pages_skipped += 1
                    continue

        image_paths = []
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            xref = img[0]
            if _is_header_footer_image(page, xref):
                continue
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            width = base_image["width"]
            height = base_image["height"]
            # В зоне контента отбрасываем только совсем мелкие картинки.
            # Квадратные скриншоты (например, из раздела 6.2) должны сохраняться.
            area = width * height
            if _is_in_content_zone(page, xref):
                if area < LOGO_MAX_AREA:
                    continue
            elif _is_logo_image(width, height):
                continue
            ext = base_image["ext"]
            img_filename = f"p{page_num + 1}_img{img_index}.{ext}"
            img_path = doc_images_dir / img_filename
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            rel_path = f"{pdf_path.stem}/{img_filename}"
            image_paths.append(rel_path)

        if text.strip():
            all_text_parts.append((page_num + 1, text, image_paths))
            all_text_is_toc.append(is_toc_page)

    total_pages = len(doc)
    doc.close()

    if not all_text_parts:
        print("Warning: no content after cleanup")
        return []

    parts_joined = []
    char_to_page_and_images: List[Tuple[int, List[str]]] = []
    current_pos = 0
    exact_body_offset: int | None = None
    for i, (page_num, text, imgs) in enumerate(all_text_parts):
        if i > 0:
            parts_joined.append("\n\n")
            current_pos += 2
            prev_page, prev_imgs = all_text_parts[i - 1][0], all_text_parts[i - 1][2]
            char_to_page_and_images.extend([(prev_page, prev_imgs), (prev_page, prev_imgs)])
        if exact_body_offset is None and not all_text_is_toc[i]:
            exact_body_offset = current_pos
        parts_joined.append(text)
        current_pos += len(text)
        for _ in text:
            char_to_page_and_images.append((page_num, imgs))

    full_text = "".join(parts_joined)
    if exact_body_offset is not None:
        text_to_chunk = full_text[exact_body_offset:]
        toc_offset = exact_body_offset
    else:
        text_to_chunk, toc_offset = _get_body_text_and_offset(full_text)
    if not text_to_chunk.strip():
        print("Warning: no content after TOC")
        return []
    chunks = _chunk_by_sections(
        text_to_chunk, chunk_size, overlap,
        all_text_parts, source_name, char_to_page_and_images, page_labels,
        char_offset=toc_offset,
    )
    print(f"PDF processed: {source_name}, pages: {total_pages}, skipped: {pages_skipped}, chunks: {len(chunks)}")
    return chunks


# Паттерн нумерованного заголовка: 1., 1.1, 2.3.1, Раздел 2., Приложение 1.
_SECTION_HEADER_RE = re.compile(
    r"^\s*((?:Раздел\s+)?\d+(?:\.\d+)*\.?|Приложение\s+\d+\.?)\s+([А-ЯA-Z][^\n]{1,150})\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _split_by_sections(text: str) -> List[Tuple[int, str, str, int, int]]:
    """
    Разбивает текст на секции по нумерованным заголовкам.
    Возвращает список (level, header_text, content, start_offset, end_offset).
    """
    sections: List[Tuple[int, str, str, int, int]] = []
    matches = list(_SECTION_HEADER_RE.finditer(text))
    if not matches:
        return sections

    for i, m in enumerate(matches):
        sec_start = m.start()
        sec_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        line_end = text.find("\n", m.start())
        if line_end < 0 or line_end > sec_end:
            line_end = sec_end
        header = text[m.start():line_end].strip()
        content_start = line_end + 1 if line_end < sec_end else sec_end
        content = text[content_start:sec_end].strip()
        if not content:
            continue
        num_part = m.group(1).strip()
        level = num_part.count(".") if "." in num_part else 1
        if "Приложение" in num_part:
            level = 1
        sections.append((level, header, content, sec_start, sec_end))
    return sections


def _chunk_by_sections(
    text: str,
    chunk_size: int,
    overlap: int,
    text_parts: List[Tuple[int, str, List[str]]],
    source_name: str,
    char_to_page_and_images: List[Tuple[int, List[str]]],
    page_labels: Dict[int, str] | None = None,
    char_offset: int = 0,
) -> List[ProcessedChunk]:
    """
    Разбиение по нумерованным заголовкам. Если один раздел не влезает в чанк,
    разбиваем по абзацам до следующего раздела.
    Overlap применяется только при разбиении раздела на чанки; для целого раздела
    (заканчивается нумерованным заголовком) overlap не используется при расчёте page_start/page_end.
    """
    sections = _split_by_sections(text)
    if not sections:
        return _chunk_with_metadata_fallback(
            text, chunk_size, overlap, source_name, char_to_page_and_images, page_labels,
            char_offset=char_offset,
        )

    chunks: List[ProcessedChunk] = []
    # Физические (1-based) диапазоны страниц для каждого чанка.
    # Нужны для привязки изображений: page_labels могут отличаться от физической нумерации.
    chunk_page_ranges: List[Tuple[int, int]] = []
    # section_header -> (chunk_indices, char_start, char_end) для привязки изображений
    section_info: Dict[str, Tuple[List[int], int, int]] = {}
    section_page_ranges_by_chunk: Dict[int, Tuple[int, int]] = {}
    search_pos = 0
    section_search_pos = 0

    page_labels = page_labels or {}

    def get_page_at(p: int) -> int:
        if not char_to_page_and_images:
            return 1
        idx = max(0, min(char_offset + p, len(char_to_page_and_images) - 1))
        return char_to_page_and_images[idx][0]

    def get_page_display(page_idx: int) -> int:
        label = page_labels.get(page_idx, str(page_idx))
        try:
            return int(label)
        except ValueError:
            return page_idx

    def add_chunk(
        chunk_text: str,
        section_header: str | None = None,
        use_overlap_for_page_range: bool = False,
        start_override: int | None = None,
        end_override: int | None = None,
    ) -> None:
        nonlocal search_pos
        if start_override is not None and end_override is not None:
            start, end = start_override, end_override
        else:
            idx = text.find(chunk_text, search_pos)
            if idx < 0:
                needle = chunk_text[:80] if len(chunk_text) > 80 else chunk_text
                idx = text.find(needle, search_pos)
            if idx < 0:
                idx = search_pos
            start, end = idx, idx + len(chunk_text)
        # Overlap для page_start/page_end только при разбиении раздела на чанки
        if use_overlap_for_page_range:
            content_start = min(start + overlap, end - 1) if end > start else start
            content_end = max(content_start, end - 1 - overlap) if end > start else start
        else:
            content_start = start
            content_end = end - 1 if end > start else start
        step = max(1, (content_end - content_start + 1) // 20)
        positions = list(range(content_start, content_end + 1, step))
        if content_end >= content_start:
            positions.append(content_end)
        pages_in_chunk = [get_page_at(i) for i in positions]
        p_start = min(pages_in_chunk) if pages_in_chunk else get_page_at(start)
        p_end = max(pages_in_chunk) if pages_in_chunk else get_page_at(end - 1) if end > start else p_start
        chunk_idx = len(chunks)
        chunks.append(ProcessedChunk(
            text=chunk_text,
            source_file=source_name,
            page_start=get_page_display(p_start),
            page_end=get_page_display(p_end),
            images=[],
            section_header=section_header,
        ))
        chunk_page_ranges.append((p_start, p_end))
        if section_header:
            if section_header not in section_info:
                section_info[section_header] = ([chunk_idx], start, end)
            else:
                indices, sec_start, sec_end = section_info[section_header]
                section_info[section_header] = (indices + [chunk_idx], min(sec_start, start), max(sec_end, end))
        search_pos = end

    for level, header, content, sec_start, sec_end in sections:
        block = (header + "\n\n" + content) if header else content
        section_chunk_start = len(chunks)
        # Диапазон секции известен по точным offset, без text.find на повторяющемся тексте.
        sec_idx = sec_start
        sec_page_range: Tuple[int, int] | None = None
        if header:
            sec_p_start = get_page_at(sec_start)
            sec_p_end = get_page_at(sec_end - 1) if sec_end > sec_start else sec_p_start
            sec_page_range = (sec_p_start, sec_p_end)
            section_search_pos = sec_end

        if len(block) <= chunk_size:
            if len(block) >= 50:
                add_chunk(
                    block,
                    section_header=header or None,
                    use_overlap_for_page_range=False,
                    start_override=sec_start,
                    end_override=sec_end,
                )
        else:
            paras = [p.strip() for p in content.split("\n\n") if p.strip()]
            if not paras:
                paras = [content]
            header_prefix = (header + "\n\n") if header else ""
            # Когда известна позиция секции в исходном тексте, считаем page-range по реальным
            # границам контента параграфов (без искусственно добавленных header/overlap).
            para_spans: List[Tuple[int, int, str]] = []
            cursor = 0
            for para in paras:
                p_idx = content.find(para, cursor)
                if p_idx < 0:
                    p_idx = cursor
                para_spans.append((p_idx, p_idx + len(para), para))
                cursor = p_idx + len(para)

            current = header_prefix
            current_start_local = 0
            current_end_local = 0
            for p_start, p_end, para in para_spans:
                add_sep = 2 if current else 0
                if len(current) + add_sep + len(para) <= chunk_size:
                    current = (current + "\n\n" + para) if current else para
                    current_end_local = max(current_end_local, p_end)
                else:
                    if current and len(current) >= 50:
                        add_chunk(
                            current,
                            section_header=header or None,
                            use_overlap_for_page_range=True,
                            start_override=sec_idx + current_start_local,
                            end_override=sec_idx + max(current_end_local, current_start_local + 1),
                        )
                    overlap_text = current[-overlap:].strip() if len(current) > overlap else ""
                    # Заголовок раздела сохраняется в каждом чанке при разбиении
                    current = (header_prefix + overlap_text + "\n\n" + para).strip() if overlap_text else (header_prefix + para)
                    current_start_local = p_start
                    current_end_local = p_end
            if current and len(current) >= 50:
                add_chunk(
                    current,
                    section_header=header or None,
                    use_overlap_for_page_range=True,
                    start_override=sec_idx + current_start_local,
                    end_override=sec_idx + max(current_end_local, current_start_local + 1),
                )
        section_chunk_end = len(chunks)
        if sec_page_range is not None:
            for ci in range(section_chunk_start, section_chunk_end):
                section_page_ranges_by_chunk[ci] = sec_page_range

    # Привязываем изображения по page_start/page_end каждого чанка (не по всей секции)
    page_to_imgs: Dict[int, List[str]] = {}
    seen_pages: set = set()
    for page_idx, imgs in char_to_page_and_images:
        if page_idx not in seen_pages and imgs:
            seen_pages.add(page_idx)
            page_to_imgs[page_idx] = list(imgs)
    for i, c in enumerate(chunks):
        p_start, p_end = chunk_page_ranges[i]
        for page_idx in range(p_start, p_end + 1):
            c.images.extend(page_to_imgs.get(page_idx, []))
        # Fallback: если изображений нет, берём изображения из диапазона всей секции.
        # Это страхует случаи, когда у длинной секции (с overlap) page range чанка определился неточно.
        if not c.images and i in section_page_ranges_by_chunk:
            s_start, s_end = section_page_ranges_by_chunk[i]
            for page_idx in range(s_start, s_end + 1):
                c.images.extend(page_to_imgs.get(page_idx, []))
        if c.images:
            c.images = list(dict.fromkeys(c.images))

    return chunks


def _chunk_with_metadata_fallback(
    text: str,
    chunk_size: int,
    overlap: int,
    source_name: str,
    char_to_page_and_images: List[Tuple[int, List[str]]],
    page_labels: Dict[int, str] | None = None,
    char_offset: int = 0,
) -> List[ProcessedChunk]:
    """Fallback: разбиение по абзацам, если нет нумерованных заголовков."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    chunk_ranges: List[Tuple[int, int]] = []
    current_chunk = ""
    chunk_start_pos = 0
    pos = 0

    page_labels = page_labels or {}

    def get_page_at(p: int) -> int:
        if not char_to_page_and_images:
            return 1
        idx = max(0, min(char_offset + p, len(char_to_page_and_images) - 1))
        return char_to_page_and_images[idx][0]

    def get_page_display(page_idx: int) -> int:
        label = page_labels.get(page_idx, str(page_idx))
        try:
            return int(label)
        except ValueError:
            return page_idx

    def count_chars_on_page(start: int, end: int, page_idx: int) -> int:
        count = 0
        for i in range(max(0, char_offset + start), min(char_offset + end, len(char_to_page_and_images))):
            if char_to_page_and_images[i][0] == page_idx:
                count += 1
        return count

    for para in paragraphs:
        add_sep = 2 if current_chunk else 0
        if len(current_chunk) + add_sep + len(para) <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + para) if current_chunk else para
            pos += add_sep + len(para)
        else:
            if current_chunk and len(current_chunk) >= 50:
                end_pos = pos
                chunk_ranges.append((chunk_start_pos, end_pos))
                end_for_page = max(chunk_start_pos, end_pos - 1 - overlap)
                chunks.append(ProcessedChunk(
                    text=current_chunk,
                    source_file=source_name,
                    page_start=get_page_display(get_page_at(chunk_start_pos)),
                    page_end=get_page_display(get_page_at(end_for_page)),
                    images=[],
                ))
            overlap_text = current_chunk[-overlap:].strip() if len(current_chunk) > overlap else ""
            overlap_len = len(overlap_text)
            current_chunk = (overlap_text + "\n\n" + para).strip() if overlap_text else para
            chunk_start_pos = pos - overlap_len if overlap_text else pos
            pos = chunk_start_pos + len(current_chunk)
            pos += 2

    if current_chunk and len(current_chunk) >= 50:
        chunk_ranges.append((chunk_start_pos, pos))
        chunks.append(ProcessedChunk(
            text=current_chunk,
            source_file=source_name,
            page_start=get_page_display(get_page_at(chunk_start_pos)),
            page_end=get_page_display(get_page_at(pos - 1)),
            images=[],
        ))

    # Назначаем изображения чанку с макс. перекрытием по странице
    page_to_imgs: Dict[int, List[str]] = {}
    seen_pages: set = set()
    for page_idx, imgs in char_to_page_and_images:
        if page_idx not in seen_pages and imgs:
            seen_pages.add(page_idx)
            page_to_imgs[page_idx] = list(imgs)
    seen_imgs: set = set()
    for page_idx, imgs in page_to_imgs.items():
        for img in imgs:
            if img in seen_imgs:
                continue
            best_chunk_idx = -1
            best_count = -1
            for ci, (start, end) in enumerate(chunk_ranges):
                cnt = count_chars_on_page(start, end, page_idx)
                if cnt > best_count:
                    best_count = cnt
                    best_chunk_idx = ci
            if best_chunk_idx >= 0 and best_count > 0:
                chunks[best_chunk_idx].images.append(img)
                seen_imgs.add(img)

    return chunks
