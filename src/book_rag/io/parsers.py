"""文档解析器：按格式把文件解析成纯文本。

每个解析器返回 ``(title, text)``。解析失败抛出 :class:`ParseError`。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pypdf import PdfReader


class ParseError(Exception):
    """文档解析失败。"""


class DocParser(ABC):
    """解析器基类。"""

    exts: frozenset[str]

    @abstractmethod
    def parse(self, path: Path) -> tuple[str, str]:
        """解析文件，返回 (标题, 全文文本)。"""
        raise NotImplementedError


class PdfParser(DocParser):
    """PDF 解析：用 pypdf 抽取每页文本。"""

    exts = frozenset({".pdf"})

    def parse(self, path: Path) -> tuple[str, str]:
        try:
            reader = PdfReader(str(path))
        except Exception as exc:  # noqa: BLE001 - 统一转成 ParseError
            raise ParseError(f"无法打开 PDF: {path.name}: {exc}") from exc
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        title = path.name
        return title, "\n\n".join(pages)


class EpubParser(DocParser):
    """EPUB 解析：用 ebooklib 抽取各章节 HTML 文本。

    注：解析 HTML 依赖 ``beautifulsoup4``，在 parse 时惰性导入，避免启动开销。
    """

    exts = frozenset({".epub"})

    def parse(self, path: Path) -> tuple[str, str]:
        try:
            import ebooklib
            from bs4 import BeautifulSoup
            from ebooklib import epub
        except ImportError as exc:
            raise ParseError("EPUB 解析需要 pip install ebooklib beautifulsoup4") from exc

        try:
            book = epub.read_epub(str(path))
        except Exception as exc:  # noqa: BLE001
            raise ParseError(f"无法打开 EPUB: {path.name}: {exc}") from exc

        chunks = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            text = soup.get_text("\n", strip=True)
            if text.strip():
                chunks.append(text)
        title = book.get_metadata("DC", "title")
        title = title[0][0] if title else path.name
        return title, "\n\n".join(chunks)


class TextParser(DocParser):
    """纯文本解析：直接读文件，猜测编码。"""

    exts = frozenset({".txt"})

    _ENCODINGS = ("utf-8", "utf-8-sig", "gbk", "latin-1")

    def parse(self, path: Path) -> tuple[str, str]:
        raw = path.read_bytes()
        text = None
        for enc in self._ENCODINGS:
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = raw.decode("utf-8", errors="replace")
        return path.name, text


class MarkdownParser(DocParser):
    """Markdown 解析：直接读文件（保留结构便于切分）。"""

    exts = frozenset({".md", ".markdown"})

    def parse(self, path: Path) -> tuple[str, str]:
        for enc in TextParser._ENCODINGS:
            try:
                text = path.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
        return path.name, text