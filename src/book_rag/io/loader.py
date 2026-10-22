"""书籍文件发现与加载入口。

对外提供三个能力：
- ``discover_books``：递归扫描书籍库目录，找出所有受支持的书籍文件；
- ``load_book``：按后缀路由到对应解析器，得到 :class:`ParsedBook`；
- ``compute_file_hash``：计算文件 SHA256，用于增量索引比对。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from book_rag.io.parsers import ParsedBook, get_parser

#: 受支持的书籍文件后缀（统一小写，带前导点）
#: 只收能直接读取文字的格式；PDF 需要 OCR / 版面还原，明确不支持。
SUPPORTED_SUFFIXES = {".epub", ".txt", ".md", ".markdown"}


def discover_books(library_path: Path) -> list[Path]:
    """递归扫描目录，返回所有受支持书籍文件的路径列表（按路径排序保证稳定）。

    Args:
        library_path: 书籍库根目录。

    Returns:
        排序后的书籍文件路径列表；目录不存在时返回空列表。
    """
    if not library_path.is_dir():
        return []

    books: set[Path] = set()
    for suffix in SUPPORTED_SUFFIXES:
        books.update(p for p in library_path.rglob(f"*{suffix}") if p.is_file())
    return sorted(books)


def load_book(path: Path) -> ParsedBook:
    """按文件后缀选择解析器，解析为 :class:`ParsedBook`。

    Raises:
        ValueError: 文件后缀不受支持。
        FileNotFoundError: 文件不存在。
    """
    if not path.is_file():
        raise FileNotFoundError(f"书籍文件不存在：{path}")
    parser = get_parser(path.suffix)
    return parser.parse(path)


def compute_file_hash(path: Path) -> str:
    """流式计算文件内容的 SHA256 十六进制摘要。

    Args:
        path: 任意文件路径。

    Returns:
        64 位小写十六进制哈希字符串。
    """
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()
