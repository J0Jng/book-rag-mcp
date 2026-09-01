"""书籍扫描与加载：把书籍库中的文件解析成结构化文本块。"""

from __future__ import annotations

from pathlib import Path

from book_rag.config import SUPPORTED_EXTS
from book_rag.io.parsers import DocParser, EpubParser, MarkdownParser, PdfParser, TextParser


class SupportedFormatError(Exception):
    """文件格式不受支持。"""


# 解析器注册表：扩展名 -> 解析器实例
_PARSERS: dict[str, DocParser] = {
    ext: parser
    for parser in (
        PdfParser(),
        EpubParser(),
        TextParser(),
        MarkdownParser(),
    )
    for ext in parser.exts
}


def get_parser(path: Path) -> DocParser | None:
    """按扩展名返回对应解析器，不支持时返回 None。"""
    return _PARSERS.get(path.suffix.lower())


def list_books(library: Path) -> list[Path]:
    """递归扫描书籍库中所有受支持的书籍文件。"""
    if not library.exists():
        return []
    return sorted(
        (p for p in library.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS),
        key=lambda p: str(p).lower(),
    )


def load_document(path: Path) -> tuple[str, str, str]:
    """加载单个文档，返回 (相对路径, 标题, 全文文本)。

    ``path`` 可以是书籍库内的相对路径。
    """
    path = Path(path)
    if not path.is_absolute():
        # 视为相对书籍库根目录
        from book_rag.config import load_config

        cfg = load_config()
        resolved = cfg.library / path
        if not resolved.exists():
            raise FileNotFoundError(f"文档不存在: {path}")
        path = resolved

    parser = get_parser(path)
    if parser is None:
        raise SupportedFormatError(f"不支持的格式: {path.suffix} (支持: {sorted(SUPPORTED_EXTS)})")
    title, text = parser.parse(path)
    rel = str(path)
    return rel, title, text