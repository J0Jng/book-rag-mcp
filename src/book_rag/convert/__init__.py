"""书籍归一（convert）：原始格式 → Markdown，作为索引的唯一输入。

公开 API：
    :func:`convert_book` 单本转换
    :func:`convert_library` 批量转换 + 原始文件归档
    :data:`SUPPORTED_SOURCE_SUFFIXES` 支持的源格式
"""

from book_rag.convert.markdown import (
    CONVERSION_MAP_FILENAME,
    SUPPORTED_SOURCE_SUFFIXES,
    ConvertResult,
    clean_book_title,
    convert_book,
    convert_library,
    extract_book_names,
    infer_levels,
    sanitize_filename,
)

__all__ = [
    "CONVERSION_MAP_FILENAME",
    "SUPPORTED_SOURCE_SUFFIXES",
    "ConvertResult",
    "clean_book_title",
    "convert_book",
    "convert_library",
    "extract_book_names",
    "infer_levels",
    "sanitize_filename",
]
