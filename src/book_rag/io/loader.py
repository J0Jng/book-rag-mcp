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

    **跳过**以下划线或点开头的目录（相对书籍库根目录）：``_originals/``（原始
    EPUB 归档）、``.book_rag_index/``（索引本体）等属于库内基础设施，不是书。

    Args:
        library_path: 书籍库根目录。

    Returns:
        排序后的书籍文件路径列表；目录不存在时返回空列表。
    """
    if not library_path.is_dir():
        return []

    books: set[Path] = set()
    for suffix in SUPPORTED_SUFFIXES:
        for path in library_path.rglob(f"*{suffix}"):
            if not path.is_file():
                continue
            if _has_internal_dir(path, library_path):
                continue
            # 库根的分词词典是检索基础设施，不能作为 TXT 书籍入库。
            if path.parent == library_path and path.name.lower() == "jieba_dict.txt":
                continue
            books.add(path)
    return sorted(books)


def _has_internal_dir(path: Path, root: Path) -> bool:
    """路径（相对 root）中是否含有下划线 / 点开头的内部目录。"""
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    return any(part.startswith(("_", ".")) for part in relative.parts[:-1])


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
