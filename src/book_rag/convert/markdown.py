"""书籍归一：把 EPUB / TXT 转换成带章节标题的 Markdown（索引的唯一输入格式）。

为什么需要这一层（见 ``docs/plans/book-rag-library-expansion-brief.md`` S1）：
- EPUB 的 ``spine`` 文档粒度不等于"章"，直接解析会让一章横跨多个 section；
- 转成 Markdown 后章节结构固化在标题层级里，索引层只需一条 ``MarkdownParser`` 路径；
- 产物是人类可读、可人工修正的纯文本。

标题层级约定：
    ``#`` 书名 / 部（如「第一部分」/ 套装里的单本书名）
    ``##`` 章（section 粒度就到这一级，**不切小节**）
    更深的目录节点降级为正文行，不产生 heading。

对"书名与章同级"的平铺目录（如套装合集），按标题模式 + 书名表把章挂到书名之下，
而不是简单按树深度映射——否则章会被误判成小节。
"""

from __future__ import annotations

import json
import re
import shutil
import urllib.parse
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from book_rag.io.matter_titles import (
    BACK_TITLES,
    FRONT_TITLES,
    match_any_title,
    normalize_title,
)
from book_rag.io.parsers import (
    _EPUB_SKIP_NAMES,
    _is_toc_page,
    _strip_contents_prefix,
)

#: 源格式 → 是否支持
SUPPORTED_SOURCE_SUFFIXES = {".epub", ".txt", ".md", ".markdown"}

#: 转换映射 / 报告文件名（记录源文件 → 产物，保证重跑幂等）
CONVERSION_MAP_FILENAME = "_conversion.json"

#: 章级标题模式：「第N章/节/回…」或「Chapter N」或「01 xxx」这类编号章
_CHAPTER_TITLE_RE = re.compile(
    r"^(?:第[0-9零〇一二三四五六七八九十百千万两]+[章节回卷部篇集]|chapter\s+\d+|\d{1,2}[\s.、])",
    re.IGNORECASE,
)

#: 文件名非法字符（Windows 保留字符 + 路径分隔符）
_ILLEGAL_FILENAME_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

#: 有效文本下限：低于此长度视为空白页 / 纯图片页（真实章节远高于此值）
_MIN_BODY_CHARS = 5

#: 书名括号里超过这个长度视为宣传语，丢弃（如「（"这本书很好"——某名人）」）
_BLURB_MIN_CHARS = 15

#: 目录页首行特征（无 heading 的目录文档靠它识别）
_TOC_WORD_RE = re.compile(r"^(?:目录|目錄|目次|contents|table\s+of\s+contents)$", re.IGNORECASE)

#: 版权页特征词（配合正文长度上限，避免误伤正文段落）
_COPYRIGHT_RE = re.compile(
    r"版权所有|侵权必究|图书在版编目|出版发行|ISBN|isbn|CIP\s*数据", re.IGNORECASE
)


@dataclass
class ConvertResult:
    """单本书的转换结果（可直接序列化为 JSON 报告）。"""

    source: str
    output: str = ""
    book_title: str = ""
    chapters: int = 0  # 产出的 ``##`` 标题数
    source_chars: int = 0  # 转换前可见文本字符数（含被丢弃的文档）
    body_chars: int = 0  # Markdown 正文（不含标题行）字符数
    heading_chars: int = 0  # 标题文字总字符数
    dropped_docs: list[str] = field(default_factory=list)
    dropped_chars: int = 0
    notes: list[str] = field(default_factory=list)
    error: str = ""


# ---------------------------------------------------------------------------
# 通用处理
# ---------------------------------------------------------------------------


def sanitize_filename(name: str) -> str:
    """把书名清洗为可用的文件名主干（去非法字符、折叠空白、限长）。"""
    cleaned = _ILLEGAL_FILENAME_RE.sub(" ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    # 书名号/引号在文件系统里合法，但留着重名率低，这里只做长度兜底
    if len(cleaned) > 80:
        cleaned = cleaned[:80].rstrip(" .、，,·-—")
    return cleaned or "untitled"


def clean_book_title(raw: str) -> str:
    """清洗 EPUB 元数据书名：去掉过长的括号宣传语、折叠空白。

    ``投资最重要的事（"这是一本难得一见的有用的书。"——沃伦巴菲特）`` →
    ``投资最重要的事``；短括号（如「（新版）」）保留。
    """
    title = re.sub(r"\s+", " ", raw or "").strip()
    for open_ch, close_ch in (("（", "）"), ("(", ")")):
        idx = title.find(open_ch)
        if idx > 0 and open_ch in title:
            close_idx = title.rfind(close_ch)
            if close_idx > idx and close_idx - idx - 1 >= _BLURB_MIN_CHARS:
                title = title[:idx].strip()
    return title or raw.strip()


def _normalize_text(text: str) -> str:
    """规范化文本：逐行去首尾空白、连续空行折叠为一个、去首尾空行。"""
    out: list[str] = []
    pending_blank = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            pending_blank = True
            continue
        if out and pending_blank:
            out.append("")
        out.append(line)
        pending_blank = False
    return "\n".join(out).strip()


def _is_matter_title(title: str) -> bool:
    """标题是否属于可丢弃的 front/back matter（KEEP 词表优先保留）。"""
    hit = match_any_title(normalize_title(title))
    if hit is None:
        return False
    kind, keyword = hit
    if kind == "keep":
        return False
    return keyword in FRONT_TITLES or keyword in BACK_TITLES


def extract_book_names(title: str) -> list[str]:
    """从合集的括号清单里提取单本书名（如「（A、B、C）」→ [A, B, C]）。

    只在括号内出现至少两个用「、」或「，」分隔的条目时生效，避免把普通
    副标题误当成书名清单。括号按**成对匹配**，title 里出现多组括号时只取
    第一组（如「…（A、B、C）（套装三册）」取 A、B、C）。
    """
    for open_ch, close_ch in (("（", "）"), ("(", ")")):
        start = title.find(open_ch)
        if start < 0:
            continue
        end = title.find(close_ch, start + 1)
        if end <= start:
            continue
        inner = title[start + 1 : end]
        parts = [p.strip() for p in re.split(r"[、，,]", inner) if p.strip()]
        if len(parts) >= 2:
            return parts
    return []


def _is_book_title(title: str, book_names: list[str]) -> bool:
    """判断目录节点是否为「单本书名」（合集场景）。"""
    stripped = title.strip()
    return any(stripped.startswith(name) for name in book_names) if book_names else False


# ---------------------------------------------------------------------------
# 目录层级推断
# ---------------------------------------------------------------------------


@dataclass
class TocEntry:
    """目录中的一个条目（已推断出 Markdown 层级）。"""

    title: str
    href: str | None
    level: int  # 1 = ``#``，2 = ``##``，>=3 表示降级为正文（不打 heading）


@dataclass
class _RawNode:
    """从 ebooklib 目录树展开出的原始节点（深度从 1 起）。"""

    title: str
    href: str | None
    depth: int


def _flatten_toc(nodes, depth: int = 1) -> list[_RawNode]:
    """把 ebooklib 的混合目录结构（Link / (Section, children)）展平为有序列表。"""
    flat: list[_RawNode] = []
    for node in nodes:
        if isinstance(node, tuple):
            section, children = node[0], list(node[1])
            flat.append(
                _RawNode(
                    title=(getattr(section, "title", "") or "").strip(),
                    href=_href_path(getattr(section, "href", "") or ""),
                    depth=depth,
                )
            )
            flat.extend(_flatten_toc(children, depth + 1))
        else:
            flat.append(
                _RawNode(
                    title=(getattr(node, "title", "") or "").strip(),
                    href=_href_path(getattr(node, "href", "") or ""),
                    depth=depth,
                )
            )
    return flat


def _href_path(href: str) -> str | None:
    """href → 去 anchor / query 并 URL 解码的文档路径。"""
    cleaned = href.split("#", 1)[0].split("?", 1)[0].strip()
    return urllib.parse.unquote(cleaned) or None


def infer_levels(
    flat: list[_RawNode], book_names: list[str]
) -> tuple[list[TocEntry], bool]:
    """为展平的目录条目推断 Markdown 层级。

    规则：
    - 目录全平铺（无任何子节点）且条目 > 3：书名自成一档，条目全部按章处理；
    - 存在「单本书名」节点（合集）：书名当 ``#``，其后的同级章挂到书名之下；
    - 其余情况按树深度映射：1 → ``#``，2 → ``##``，更深降级为正文。

    Returns:
        ``(entries, flat_mode)``，``flat_mode`` 表示命中了全平铺特例。
    """
    has_children = any(node.depth > 1 for node in flat)
    has_book_scope = any(_is_book_title(node.title, book_names) for node in flat)
    flat_mode = not has_children and len(flat) > 3 and not has_book_scope

    entries: list[TocEntry] = []
    scope_book = False  # 当前是否处在「某本合集子书」的作用域内
    for node in flat:
        title = node.title.strip()
        if not title:
            continue
        if flat_mode:
            level = 2
        elif node.depth == 1:
            if _is_book_title(title, book_names):
                scope_book = True
                level = 1
            elif scope_book:
                # 合集里书名与章同级：章降一级挂到书名之下
                level = 2
            else:
                level = 1
        else:
            level = node.depth + 1 if scope_book else node.depth
        entries.append(TocEntry(title=title, href=node.href, level=level))
    return entries, flat_mode


# ---------------------------------------------------------------------------
# EPUB → Markdown
# ---------------------------------------------------------------------------


def _soup_of(item):
    """解析 EPUB 文档并剥掉非正文标签（含 ``head``/``title``，否则标题会混进正文）。"""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(item.content, "html.parser")
    for tag in soup(["script", "style", "head", "title"]):
        tag.decompose()
    return soup


def _epub_text(item) -> str:
    """从 EPUB 文档提取纯文本（剥 script/style/head，保留段落换行）。"""
    return _soup_of(item).get_text("\n")


def _first_heading_text(item) -> str:
    """取文档内首个 heading 的文本（无目录映射时用来识别章节名/版权页等）。"""
    heading = _soup_of(item).find(["h1", "h2", "h3", "h4", "h5", "h6"])
    return heading.get_text(strip=True) if heading is not None else ""


def _norm_for_compare(text: str) -> str:
    """比较用的标题归一化：去掉所有空白（含全角空格）与首尾标点装饰。"""
    return re.sub(r"\s+", "", text or "").strip("#＊*　 ")


def _strip_leading_heading(item, heading_text: str) -> None:
    """就地删除与章节标题重复的首个 heading 元素（避免标题双写）。

    真实 EPUB 的 heading 常被拆成多行 / 带全角空格（如「第一章　工作的意义」
    被拆成两行），因此比较前两侧都做空白归一化。
    """
    soup = _soup_of(item)
    heading = soup.find(["h1", "h2", "h3", "h4", "h5", "h6"])
    if heading is None:
        return
    if _norm_for_compare(heading.get_text(strip=True)) == _norm_for_compare(heading_text):
        heading.decompose()
        item.content = str(soup).encode("utf-8")


def epub_to_markdown(path: Path) -> tuple[str, ConvertResult]:
    """把 EPUB 转换为 Markdown 文本。

    Returns:
        ``(markdown, result)``；``result`` 携带统计与诊断信息（不含 output 路径）。
    """
    from ebooklib import epub

    result = ConvertResult(source=str(path))
    reader = epub.read_epub(str(path))

    try:
        raw_title = reader.title or path.stem
    except (AttributeError, KeyError, ValueError):
        raw_title = path.stem
    book_title = clean_book_title(raw_title)
    result.book_title = book_title

    flat = _flatten_toc(list(reader.toc or []))
    book_names = extract_book_names(raw_title)
    entries, flat_mode = infer_levels(flat, book_names)
    if flat_mode:
        result.notes.append("目录全平铺：书名自成一档，节点按章处理")
    if book_names:
        result.notes.append(f"合集书名清单：{'、'.join(book_names)}")

    # 文档路径 → 条目（同一文档命中多级节点时取层级最浅的，即「章」那一级）
    by_path: dict[str, TocEntry] = {}
    by_basename: dict[str, TocEntry] = {}
    for entry in entries:
        if not entry.href:
            continue
        current = by_path.get(entry.href)
        if current is None or entry.level < current.level:
            by_path[entry.href] = entry
        by_basename.setdefault(entry.href.rsplit("/", 1)[-1], entry)

    spine_ids = [entry for entry, _ in (reader.spine or [])]
    items_by_id = {item.id: item for item in reader.get_items()}

    lines: list[str] = [f"# {book_title}", ""] if flat_mode else []
    last_heading: str | None = None
    h1_from_nav = 0  # 由目录条目产生的 `#` 标题数（单层结构时它们就是章）

    for item_id in spine_ids:
        item = items_by_id.get(item_id)
        if item is None or item.get_type() != 9:  # 9 = ITEM_DOCUMENT
            continue
        file_name = (item.file_name or "").replace("\\", "/").lstrip("/")
        stem_name = file_name.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        if stem_name in _EPUB_SKIP_NAMES or stem_name.startswith("cover"):
            result.dropped_docs.append(f"{file_name}(跳过：封面/导航)")
            continue

        entry = by_path.get(file_name) or by_basename.get(file_name.rsplit("/", 1)[-1])
        raw_text = _epub_text(item)
        if _is_toc_page_from(raw_text):
            result.dropped_chars += len(_normalize_text(raw_text))
            result.dropped_docs.append(f"{file_name}(跳过：目录页)")
            continue

        text = _normalize_text(_strip_contents_prefix(raw_text))
        # 章节名优先级：目录条目 → 文档内首个 heading（文件名只在兜底时用）
        title = entry.title if entry else _first_heading_text(item)
        if title and _is_matter_title(title):
            result.dropped_chars += len(text)
            result.dropped_docs.append(f"{file_name}(跳过：{title})")
            continue

        if len(text) < _MIN_BODY_CHARS:
            if entry is not None and entry.level == 1 and title:
                # 空节（如「第一部分」只有标题、正文在下一文档）：保留 heading
                # 以维持层级，否则后续章会失去父级、路径退化为裸章名
                lines.extend([f"# {title}", ""])
                last_heading = f"# {title}"
                result.heading_chars += len(title)
                h1_from_nav += 1
                result.notes.append(f"{title} 无正文，保留标题以维持层级")
            else:
                result.dropped_chars += len(text)
                result.dropped_docs.append(f"{file_name}(跳过：有效文本不足 {len(text)} 字)")
            continue

        if not title:
            # 无章节名可依：一律不入正文（否则 5000 字的目录页、版权页会被
            # 并进上一章，直接污染检索）。原因 + 预览记入报告，便于人工复核。
            result.dropped_chars += len(text)
            preview = text[:40].replace("\n", " ")
            result.dropped_docs.append(
                f"{file_name}(跳过：{_untitled_reason(text)}｜{preview})"
            )
            continue

        heading_level = entry.level if entry else 2
        if heading_level >= 3:
            # 目录比章更细：为了不丢定位信息，提升到章级并记录
            heading_level = 2
            result.notes.append(f"{title} 原为小节级，提升为章级")

        _strip_leading_heading(item, title)
        body = _normalize_text(_strip_contents_prefix(_epub_text(item)))
        result.source_chars += len(body) + len(title)

        marker = "#" * heading_level
        key = f"{marker} {title}"
        if key != last_heading:
            lines.extend([f"{marker} {title}", ""])
            last_heading = key
            result.heading_chars += len(title)
            if heading_level == 2:
                result.chapters += 1
            else:
                h1_from_nav += 1
        if body:
            lines.extend([body, ""])
            result.body_chars += len(body)

    # 单层结构的书（没有「部」这一级）章就落在 `#` 上，此时章数按 `#` 计
    if result.chapters == 0 and h1_from_nav:
        result.chapters = h1_from_nav
        result.notes.append("单层结构：章级节点位于 `#`")

    markdown = "\n".join(lines).strip() + "\n"
    if result.chapters == 0:
        raise ValueError("无法从目录 / 标题推断章节结构（无任何章级节点），需人工处理")
    return markdown, result


def _untitled_reason(text: str) -> str:
    """给「无章节名」的文档归因，便于人工判断是否误丢正文。"""
    first_line = text.split("\n", 1)[0].strip()
    if _TOC_WORD_RE.match(first_line):
        return "目录页"
    if len(text) < 3000 and _COPYRIGHT_RE.search(text):
        return "版权页"
    return "无可依章节名（无 heading 且不在目录中）"


def _is_toc_page_from(text: str) -> bool:
    """按文本形态判断目录页（复用解析层启发式，但只吃纯文本）。"""
    from bs4 import BeautifulSoup

    if not text.strip():
        return False
    soup = BeautifulSoup(f"<div>{text}</div>", "html.parser")
    return _is_toc_page(soup)


# ---------------------------------------------------------------------------
# TXT → Markdown
# ---------------------------------------------------------------------------


def txt_to_markdown(path: Path) -> tuple[str, ConvertResult]:
    """把 TXT 转换为 Markdown：``第N章`` 等标题行升为 ``##``，其余为正文。"""
    from book_rag.io.parsers import TextParser

    result = ConvertResult(source=str(path))
    parsed = TextParser().parse(path)
    book_title = clean_book_title(parsed.title)
    result.book_title = book_title

    lines: list[str] = [f"# {book_title}", ""]
    last_heading: str | None = None
    for section in parsed.sections:
        text = _normalize_text(section.text)
        if not text:
            continue
        title = section.path.strip()
        result.source_chars += len(text)
        if title and title != book_title and title != path.stem:
            key = f"## {title}"
            if key != last_heading:
                lines.extend([f"## {title}", ""])
                last_heading = key
                result.heading_chars += len(title)
                result.chapters += 1
        lines.extend([text, ""])
        result.body_chars += len(text)

    markdown = "\n".join(lines).strip() + "\n"
    return markdown, result


# ---------------------------------------------------------------------------
# 批量转换
# ---------------------------------------------------------------------------


def markdown_to_markdown(path: Path) -> tuple[str, ConvertResult]:
    """已是 Markdown 的源文件：只做标题清洗与规范化，不重排结构。"""
    result = ConvertResult(source=str(path))
    text = _normalize_text(path.read_text(encoding="utf-8"))
    result.book_title = clean_book_title(path.stem)
    result.body_chars = len(text)
    result.source_chars = len(text)
    return text + "\n", result


def convert_book(
    path: Path, out_dir: Path, *, filename: str | None = None
) -> ConvertResult:
    """转换单本书并写入 ``out_dir``；失败时把错误写进 ``result.error``。

    Args:
        path: 源文件。
        out_dir: Markdown 输出目录。
        filename: 输出文件名主干；缺省用清洗后的书名。批量转换时由
            :func:`convert_library` 依据 ``_conversion.json`` 传入，保证幂等。
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".epub":
            markdown, result = epub_to_markdown(path)
        elif suffix == ".txt":
            markdown, result = txt_to_markdown(path)
        elif suffix in (".md", ".markdown"):
            markdown, result = markdown_to_markdown(path)
        else:
            raise ValueError(f"不支持转换的格式：{suffix}")
    except Exception as exc:  # noqa: BLE001 - 单本失败不阻断整批
        return ConvertResult(source=str(path), error=f"{type(exc).__name__}: {exc}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_filename(filename or result.book_title or path.stem)
    target = out_dir / f"{stem}.md"
    target.write_text(markdown, encoding="utf-8")
    result.output = str(target)
    return result


def _load_conversion_map(dest_dir: Path) -> dict[str, str]:
    """读取历史转换映射 ``{源文件名: 输出文件名}``；缺失或损坏时返回空表。"""
    path = dest_dir / CONVERSION_MAP_FILENAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    entries = data.get("books") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return {}
    mapping: dict[str, str] = {}
    for entry in entries:
        if isinstance(entry, dict) and entry.get("source_name") and entry.get("output_name"):
            mapping[str(entry["source_name"])] = str(entry["output_name"])
    return mapping


def convert_library(
    source_dir: Path,
    dest_dir: Path,
    originals_dir: Path | None = None,
    *,
    move_originals: bool = True,
) -> list[ConvertResult]:
    """批量转换 ``source_dir`` 下的书籍，可选把原始文件归档到 ``originals_dir``。

    幂等性依赖 ``dest_dir/_conversion.json``：同一源文件重跑会覆盖同名产物，
    不会产生 ``(2)`` 副本；只有不同源文件撞名时才加序号。

    Args:
        source_dir: 待转换目录（递归扫描）。
        dest_dir: Markdown 输出目录。
        originals_dir: 原始文件归档目录；``None`` 时不归档。
        move_originals: True 移动、False 复制。移动失败不丢文件（先写后移）。
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    previous = _load_conversion_map(dest_dir)
    taken = set(previous.values())

    files = sorted(
        p
        for suffix in SUPPORTED_SOURCE_SUFFIXES
        for p in source_dir.rglob(f"*{suffix}")
        if p.is_file()
    )

    results: list[ConvertResult] = []
    for path in files:
        known = previous.get(path.name)
        filename = known[:-3] if known and known.endswith(".md") else None
        if known is None:
            result = convert_book(path, dest_dir)
            if not result.error:
                stem = Path(result.output).stem
                taken.add(Path(result.output).name)
                filename = stem
        else:
            result = convert_book(path, dest_dir, filename=filename)
        if result.error:
            results.append(result)
            continue
        output_name = Path(result.output).name
        result.notes.append(f"输出：{output_name}")

        if originals_dir is not None:
            originals_dir.mkdir(parents=True, exist_ok=True)
            target = originals_dir / path.name
            counter = 2
            while target.exists() and path.name != target.name:
                target = originals_dir / f"{path.stem} ({counter}){path.suffix}"
                counter += 1
            try:
                if not target.exists():
                    if move_originals:
                        shutil.move(str(path), str(target))
                    else:
                        shutil.copy2(path, target)
            except OSError as exc:
                result.notes.append(f"原始文件归档失败：{exc}")
        results.append(result)

    _write_conversion_map(dest_dir, results, previous)
    return results


def _write_conversion_map(
    dest_dir: Path, results: list[ConvertResult], previous: dict[str, str]
) -> None:
    """把本次转换结果写成 ``_conversion.json``（同时充当 S1 转换报告）。"""
    entries = []
    for result in results:
        if result.error or not result.output:
            continue
        entries.append(
            {
                "source_name": Path(result.source).name,
                "output_name": Path(result.output).name,
                "book_title": result.book_title,
                "chapters": result.chapters,
                "source_chars": result.source_chars,
                "body_chars": result.body_chars,
                "heading_chars": result.heading_chars,
                "dropped_docs": result.dropped_docs,
                "dropped_chars": result.dropped_chars,
                "notes": result.notes,
            }
        )
    # 保留本次未扫到但历史上转换过的条目，避免增量转换丢映射
    seen = {entry["source_name"] for entry in entries}
    for source_name, output_name in previous.items():
        if source_name not in seen and (dest_dir / output_name).is_file():
            entries.append({"source_name": source_name, "output_name": output_name})

    payload = {
        "converted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "count": len(entries),
        "books": entries,
    }
    (dest_dir / CONVERSION_MAP_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# 渲染工具（供 CLI 输出）
# ---------------------------------------------------------------------------
