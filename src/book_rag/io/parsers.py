"""文档解析器：将 PDF / EPUB / TXT / Markdown 统一解析为结构化的 BookSection 列表。

所有解析器的输出契约一致：
- ``ParsedBook.title`` 一律取文件名主干（EPUB 优先取元数据书名）；
- ``sections`` 按阅读顺序排列，``position`` 从 0 起严格递增；
- 每个 section 的 ``text`` 均为非空文本。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


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


class TextParser(BookParser):
    """按空行分段解析纯文本，并把「第 X 章 / Chapter X」行识别为章节边界。

    章节标题行作为 section 路径，不入正文；无章节标题的段落归入以书名
    命名的默认 section。
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
            if _CHAPTER_RE.match(para):
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


class EpubParser(BookParser):
    """用 ebooklib + BeautifulSoup 提取各章节文本。

    按 spine 顺序遍历正文文档；section 路径取文档内标题（h1-h6）或条目标题。
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

            # 章节路径：优先文档内标题，其次 spine 条目标题
            heading = soup.find(["h1", "h2", "h3", "h4", "h5", "h6"])
            section_name = heading.get_text(strip=True) if heading else (item.title or name)
            if heading is not None:
                heading.decompose()

            text = soup.get_text("\n").strip()
            if not text:
                continue
            sections.append(
                BookSection(path=section_name, level=1, text=text, position=len(sections))
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
