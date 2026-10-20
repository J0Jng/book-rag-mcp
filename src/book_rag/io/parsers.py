"""文档解析器：将 PDF / EPUB / TXT / Markdown 统一解析为结构化的 BookSection 列表。

所有解析器的输出契约一致：
- ``ParsedBook.title`` 一律取文件名主干（EPUB 优先取元数据书名）；
- ``sections`` 按阅读顺序排列，``position`` 从 0 起严格递增；
- 每个 section 的 ``text`` 均为非空文本。
"""

from __future__ import annotations

import re
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from book_rag.io.matter_titles import (
    HEADING_MAX_CHARS,
    match_any_title,
    normalize_title,
)


@dataclass
class BookSection:
    """书中的一个最小检索单元（章节 / 段落 / 页面片段）。

    Attributes:
        path: 章节路径，如 ``"第一章 起点"`` 或 ``"测试之书 > 第一章 起点"``。
        level: 结构层级深度（Markdown 标题层级、TXT 章节、PDF 页面等）。
        text: 该 section 的正文文本（非空）。
        position: 全书内的全局顺序号，从 0 起递增。
    """

    path: str
    level: int
    text: str
    position: int


@dataclass
class ParsedBook:
    """解析后的书籍。

    Attributes:
        title: 书名（默认取文件名主干；EPUB 优先取元数据）。
        language: 主导语言（ISO 639-1），由上层采样检测后填充，解析阶段为 None。
        sections: 按阅读顺序排列的 section 列表。
    """

    title: str
    language: str | None = None
    sections: list[BookSection] = field(default_factory=list)


class BookParser(ABC):
    """解析器抽象基类。"""

    @abstractmethod
    def parse(self, path: Path) -> ParsedBook:
        """解析给定路径的书籍文件，返回结构化结果。"""


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


class MarkdownParser(BookParser):
    """按 ``#`` 标题层级切分 Markdown 文档。

    标题行自身不进入正文；首个标题之前的内容归入以书名（首个一级标题或
    文件主干）命名的导语 section；连续标题之间不产生空 section。
    """

    def parse(self, path: Path) -> ParsedBook:
        title = path.stem
        lines = path.read_text(encoding="utf-8").splitlines()

        sections: list[BookSection] = []
        # 标题栈：缓存各级标题文本，用于拼接 section 路径
        stack: list[tuple[int, str]] = []
        current_text: list[str] = []

        def flush() -> None:
            """将当前累积的正文落为一个 section（正文为空则丢弃）。"""
            text = "\n".join(current_text).strip()
            current_text.clear()
            if not text:
                return
            section_path = " > ".join(name for _, name in stack) if stack else title
            level = stack[-1][0] if stack else 0
            sections.append(
                BookSection(path=section_path, level=level, text=text, position=len(sections))
            )

        for line in lines:
            match = _MD_HEADING_RE.match(line)
            if match:
                flush()
                level = len(match.group(1))
                name = match.group(2).strip()
                # 弹出同级及更深层级的标题，维持层级栈
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, name))
            else:
                current_text.append(line)
        flush()

        return ParsedBook(title=title, sections=sections)


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------

# 章节标题行：如「第一章 启程」「第12节」「Chapter 3 The Journey」
_CHAPTER_RE = re.compile(
    r"^(?:第[0-9零〇一二三四五六七八九十百千万两]+[章节卷回集部]|chapter\s+\d+)(?:[\s:：].*)?$",
    re.IGNORECASE,
)


def _is_matter_heading(para: str) -> bool:
    """判断整段是否为 front/back matter 标题行（也作为章节边界）。

    仅当整段是**单行**且长度不超过 :data:`HEADING_MAX_CHARS`、归一化后命中
    matter 词表时成立。这样「推荐序 / 致谢 / 前言」等在没有章节标记的纯文本
    里也能切出独立 section，交由 ``clean_matter`` 判定；正文段落（多行或长句）
    不会命中，避免把正文句子误切成章节。
    """
    line = para.strip()
    if not line or "\n" in line or len(line) > HEADING_MAX_CHARS:
        return False
    return match_any_title(normalize_title(line)) is not None


class TextParser(BookParser):
    """按空行分段解析纯文本，并把「第 X 章 / Chapter X」行、以及
    「推荐序 / 前言 / 致谢」等 front/back matter 标题行识别为章节边界。

    章节标题行作为 section 路径，不入正文；无章节标题的段落归入以书名
    命名的默认 section。matter 标题行同样切成独立 section，交由
    ``clean_matter`` 判定是否剥离（保留词如「前言」仍会保留）。
    """

    def parse(self, path: Path) -> ParsedBook:
        title = path.stem
        text = path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        sections: list[BookSection] = []
        current_path = title
        current_level = 0
        current_text: list[str] = []

        def flush() -> None:
            if not current_text:
                return
            sections.append(
                BookSection(
                    path=current_path,
                    level=current_level,
                    text="\n".join(current_text).strip(),
                    position=len(sections),
                )
            )
            current_text.clear()

        for para in paragraphs:
            if _CHAPTER_RE.match(para) or _is_matter_heading(para):
                flush()
                current_path = para
                current_level = 1
            elif current_level == 0:
                # 尚未进入任何章节：每个段落独立成一个默认 section
                flush()
                current_text.append(para)
                flush()
            else:
                current_text.append(para)
        flush()

        return ParsedBook(title=title, sections=sections)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


class PdfParser(BookParser):
    """用 pypdf 逐页提取文本，再按空行拆分为段落级 section。

    section 路径形如 ``"第 3 页"``，``level`` 恒为 1；提取不到文本的页面
    （扫描件等）直接跳过。
    """

    def parse(self, path: Path) -> ParsedBook:
        from pypdf import PdfReader

        title = path.stem
        reader = PdfReader(str(path))

        sections: list[BookSection] = []
        for page_no, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for para in re.split(r"\n\s*\n", page_text):
                para = para.strip()
                if not para:
                    continue
                sections.append(
                    BookSection(
                        path=f"第 {page_no} 页",
                        level=1,
                        text=para,
                        position=len(sections),
                    )
                )

        return ParsedBook(title=title, sections=sections)


# ---------------------------------------------------------------------------
# EPUB
# ---------------------------------------------------------------------------

# 常见的非正文文档（目录 / 封面 / 版权页），解析时跳过
_EPUB_SKIP_NAMES = {"nav", "ncx", "toc", "titlepage", "cover", "copyright"}

# 目录页标题行（正文几乎只剩这个词的文档视为目录页）
_TOC_TITLE_RE = re.compile(r"^(?:contents|目录|目錄|table of contents)$", re.IGNORECASE)

# 正文文档开头的排版页眉残留行：整行只有 Contents / 目录 / 目錄 / Table of Contents
_CONTENTS_PREFIX_RE = re.compile(
    r"^(?:contents|目录|目錄|table\s+of\s+contents)$", re.IGNORECASE
)

# 献词页特征（目录树之外的前置文档）：谨以此书献给 / 献给 / Dedicated to 等
_DEDICATION_RE = re.compile(
    r"(谨以此书献给|谨献给|仅以此书献给|此书献给|献给|dedicated\s+to|dedication|to\s+my\b)",
    re.IGNORECASE,
)


def _is_dedication(text: str) -> bool:
    """判定文本是否为献词页：篇幅短 + 开头命中献词特征。"""
    return len(text) < 2000 and bool(_DEDICATION_RE.search(text[:500]))


def _strip_contents_prefix(text: str) -> str:
    """仅当文档可见文本开头是一行孤立的目录页眉时剥离该行。

    有些 EPUB 的每个正文文档开头都带一行 ``Contents``（排版页眉残留），
    紧接章节标题。这里只在文本开头（strip 后）命中 Contents/目录/目錄/
    Table of Contents（忽略大小写，允许前后空白）时剥离该行及其后的
    空白；正文中（非开头位置）合法出现的 ``Contents`` 不受影响。
    """
    first_line, sep, rest = text.strip().partition("\n")
    if _CONTENTS_PREFIX_RE.match(first_line.strip()):
        return rest.lstrip() if sep else ""
    return text


@dataclass
class _TocNode:
    """EPUB 目录树中的一个节点（来自 nav / ncx）。

    Attributes:
        title: 目录条目标题。
        href: 归一化后的内容文档路径（去 anchor / query / URL 编码），可为 None。
        level: 层级深度（顶层章 = 1，节 = 2，依此类推）。
        children: 子节点列表。
    """

    title: str
    href: str | None
    level: int
    children: list[_TocNode] = field(default_factory=list)


def _href_to_file_name(href: str) -> str | None:
    """把目录条目的 href 归一化为可与 ``item.file_name`` 匹配的文档路径。

    去掉 ``#anchor`` 与 query 部分，并做 URL 解码；空路径返回 None。
    """
    if not href:
        return None
    cleaned = href.split("#", 1)[0].split("?", 1)[0].strip()
    cleaned = urllib.parse.unquote(cleaned)
    return cleaned or None


def _build_toc_tree(reader) -> list[_TocNode]:
    """从 EPUB 自带的目录（EPUB3 nav / EPUB2 ncx）提取目录树。

    ebooklib 解析后的 ``reader.toc`` 是混合结构：叶子为 ``Link``（含
    href/title），带子级的条目为 ``(Section, children)`` 元组。这里统一
    归一化为 :class:`_TocNode` 嵌套树；缺失或异常时返回空表，不抛异常。
    """
    try:
        raw_toc = list(reader.toc or [])
    except Exception:
        return []

    def to_node(entry, level: int) -> _TocNode | None:
        if isinstance(entry, tuple):
            section, children = entry[0], list(entry[1])
            node = _TocNode(
                title=(getattr(section, "title", "") or "").strip(),
                href=_href_to_file_name(getattr(section, "href", "") or ""),
                level=level,
            )
            for child in children:
                child_node = to_node(child, level + 1)
                if child_node is not None:
                    node.children.append(child_node)
            return node
        # Link（或直接的 Section 叶子）
        title = (getattr(entry, "title", "") or "").strip()
        href = _href_to_file_name(getattr(entry, "href", "") or "")
        return _TocNode(title=title, href=href, level=level)

    tree: list[_TocNode] = []
    for entry in raw_toc:
        node = to_node(entry, level=1)
        if node is not None:
            tree.append(node)
    return tree


def _set_toc_levels(nodes: list[_TocNode], level: int) -> None:
    """递归重设节点层级（书名节点提升后子级需要整体上移一级）。"""
    for node in nodes:
        node.level = level
        _set_toc_levels(node.children, level + 1)


# 章级标题模式：「第N章/篇/部/卷 …」或「Chapter N …」（匹配标题开头即命中）
_EPUB_CHAPTER_TITLE_RE = re.compile(
    r"^(?:第[0-9零〇一二三四五六七八九十百千万两]+[章篇部卷]|chapter\s+\d+)",
    re.IGNORECASE,
)

# 前置/后置部分标题模式（整题匹配，允许标题后跟空白/冒号引出的副题）
_EPUB_FRONT_BACK_TITLE_RE = re.compile(
    r"^(?:推荐序|序言|前言|引言|导读|序|封面|扉页|版权|致谢|后记|结语|附录)(?:[\s:：].*)?$"
)


def _infer_levels_from_titles(nodes: list[_TocNode]) -> list[_TocNode]:
    """目录树全平铺时，按标题模式重建「章 > 节」层级（仅一层子节）。

    规则：章级标题（「第N章 …」/「Chapter N …」）与前置/后置标题
    （引言、致谢等）为 level 1 的顶层节点，后者不更新「当前章」；
    其余标题挂到最近的章级节点下（level 2），无当前章则保持 level 1。
    """
    tree: list[_TocNode] = []
    current_chapter: _TocNode | None = None
    for node in nodes:
        title = node.title.strip()
        node.children = []
        if _EPUB_CHAPTER_TITLE_RE.match(title) or _EPUB_FRONT_BACK_TITLE_RE.match(title):
            node.level = 1
            tree.append(node)
            if _EPUB_CHAPTER_TITLE_RE.match(title):
                current_chapter = node
        elif current_chapter is not None:
            node.level = current_chapter.level + 1
            current_chapter.children.append(node)
        else:
            node.level = 1
            tree.append(node)
    return tree


def _build_toc_path_map(tree: list[_TocNode]) -> dict[str, tuple[str, int]]:
    """把目录树展开为 ``file_name -> (多级路径, level)`` 映射。

    同一文档命中多个层级节点时保留路径最长（level 最高）的那个。
    """
    mapping: dict[str, tuple[str, int]] = {}

    def walk(node: _TocNode, ancestors: list[str]) -> None:
        chain = [*ancestors, node.title] if node.title else ancestors
        if node.href:
            path = " > ".join(chain)
            prev = mapping.get(node.href)
            if prev is None or node.level > prev[1] or (
                node.level == prev[1] and len(path) > len(prev[0])
            ):
                mapping[node.href] = (path, node.level)
        for child in node.children:
            walk(child, chain)

    for node in tree:
        walk(node, [])
    return mapping


def _lookup_toc_path(mapping: dict[str, tuple[str, int]], file_name: str | None):
    """按 item 的 file_name 查目录树映射，目录层级不齐时退化为按文件名匹配。"""
    if not file_name or not mapping:
        return None
    fname = file_name.replace("\\", "/").lstrip("/")
    hit = mapping.get(fname)
    if hit is not None:
        return hit
    base = fname.rsplit("/", 1)[-1]
    for key, value in mapping.items():
        if key.rsplit("/", 1)[-1] == base:
            return value
    return None


def _is_toc_page(soup) -> bool:
    """启发式判定一个文档是否为目录页。

    规则（满足其一即判定）：
    1. 文档内含 >= 3 个指向其他文档的链接（纯 ``#anchor`` 的文内跳转
       不算），且链接文本占可见文本的大头（>= 60%）——呈「逐条链接到
       正文章节」的目录列表形态；正文里仅少量交叉引用链接不会被误判。
    2. 可见文本几乎只剩「目录 / Contents」标题本身。
    """
    try:
        doc_links = [
            a for a in soup.find_all("a", href=True)
            if (a.get("href") or "").strip() and not a.get("href").strip().startswith("#")
        ]
        if len(doc_links) >= 3:
            total_text = soup.get_text(" ", strip=True)
            link_text_len = sum(len(a.get_text(strip=True)) for a in doc_links)
            if total_text and link_text_len / max(len(total_text), 1) >= 0.6:
                return True

        lines = [ln.strip() for ln in soup.get_text("\n", strip=True).splitlines() if ln.strip()]
        if lines and all(_TOC_TITLE_RE.match(ln) for ln in lines):
            return True
    except Exception:
        return False
    return False


class EpubParser(BookParser):
    """用 ebooklib + BeautifulSoup 提取各章节文本。

    按 spine 顺序遍历正文文档。优先从 EPUB 自带目录（nav / ncx）构建
    目录树，把每个 spine 文档映射为多级章节路径（``" > "`` 连接）与真实
    层级；识别为目录页的文档整体剥离。缺失 nav/ncx 或目录结构异常时，
    退化为按文档内标题（h1-h6）或条目标题命名、level 恒为 1 的旧行为。
    """

    def parse(self, path: Path) -> ParsedBook:
        from bs4 import BeautifulSoup
        from ebooklib import epub

        reader = epub.read_epub(str(path))
        try:
            title = reader.title or path.stem
        except (AttributeError, KeyError, ValueError):
            title = path.stem

        # spine 中按 id 记录阅读顺序；id 与 item.id_name 对应
        spine_ids: list[str] = []
        for entry, _linear in reader.spine:
            spine_ids.append(entry if isinstance(entry, str) else str(entry))

        items_by_id = {item.id: item for item in reader.get_items()}

        # 从 nav/ncx 构建目录树并展开为 file_name -> (path, level) 映射；
        # 顶层若只是书名节点则跳过（书名已由 ParsedBook.title 承载）
        toc_tree = _build_toc_tree(reader)
        if len(toc_tree) == 1 and toc_tree[0].children and toc_tree[0].title == title:
            toc_tree = toc_tree[0].children
            _set_toc_levels(toc_tree, 1)
        # EPUB2 等来源的 toc 可能全平铺（无任何嵌套）；此时按标题模式
        # 推断「章 > 节」层级。带层级 nav 的书不受影响（不会触发）。
        if len(toc_tree) > 1 and all(not node.children for node in toc_tree):
            toc_tree = _infer_levels_from_titles(toc_tree)
        path_map = _build_toc_path_map(toc_tree)

        sections: list[BookSection] = []
        for item_id in spine_ids:
            item = items_by_id.get(item_id)
            if item is None or item.get_type() != 9:  # 9 = ebooklib.ITEM_DOCUMENT
                continue
            name = (item.file_name or "").rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
            if name in _EPUB_SKIP_NAMES:
                continue

            soup = BeautifulSoup(item.content, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()

            # 目录页（内容主要为指向正文章节的导航链接）整体剥离
            if _is_toc_page(soup):
                continue

            heading = soup.find(["h1", "h2", "h3", "h4", "h5", "h6"])
            mapped = _lookup_toc_path(path_map, item.file_name)
            if mapped is not None:
                section_path, level = mapped
                # 与路径末段重复的标题行不入正文，避免与 path 双重计入
                if heading is not None and heading.get_text(strip=True) == section_path.rsplit(
                    " > ", 1
                )[-1]:
                    heading.decompose()
            else:
                # 无目录映射：回退到旧行为（首个 heading / spine 条目标题 / 文件名）
                section_name = heading.get_text(strip=True) if heading else (item.title or name)
                if heading is not None:
                    heading.decompose()
                section_path, level = section_name, 1

            text = soup.get_text("\n").strip()
            # 剥离文档开头残留的 Contents 页眉行（真实书每个正文文档都带）
            text = _strip_contents_prefix(text)
            if not text:
                continue
            # 目录树之外的匿名献词页（front-matter）整文档剥离
            if mapped is None and _is_dedication(text):
                continue
            sections.append(
                BookSection(path=section_path, level=level, text=text, position=len(sections))
            )

        return ParsedBook(title=title, sections=sections)


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

_PARSER_REGISTRY: dict[str, type[BookParser]] = {
    ".md": MarkdownParser,
    ".markdown": MarkdownParser,
    ".txt": TextParser,
    ".pdf": PdfParser,
    ".epub": EpubParser,
}


def get_parser(suffix: str) -> BookParser:
    """按文件后缀返回对应的解析器实例。

    Args:
        suffix: 文件后缀，带不带前导点均可，大小写不敏感。

    Raises:
        ValueError: 后缀不受支持。
    """
    suffix = suffix.lower()
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    parser_cls = _PARSER_REGISTRY.get(suffix)
    if parser_cls is None:
        supported = ", ".join(sorted(_PARSER_REGISTRY))
        raise ValueError(f"不支持的文件后缀 {suffix!r}，支持：{supported}")
    return parser_cls()
