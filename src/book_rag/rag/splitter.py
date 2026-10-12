"""结构感知切分：把解析出的 BookSection 切成用于检索的 Document 块。

优先在自然边界（空行、换行、句号）处断开，不硬截断句子；
每个 chunk 保留书籍级与章节级元数据，供检索结果定位原文位置。
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from book_rag.io.parsers import BookSection

# 分隔符优先级：段落 > 换行 > 中文/全角句号 > 半角句号 > 空格 > 兜底逐字符
_SEPARATORS = ["\n\n", "\n", "。", "．", ". ", " ", ""]


def split_book(
    sections: list[BookSection],
    chunk_size: int,
    chunk_overlap: int,
    *,
    title: str = "",
    source: str = "",
) -> list[Document]:
    """将书籍的 section 列表切分为检索用 Document 块。

    Args:
        sections: 解析器输出的章节 / 段落列表，按阅读顺序排列。
        chunk_size: 单个 chunk 的最大字符数。
        chunk_overlap: 相邻 chunk 的重叠字符数。
        title: 书名，写入每个 chunk 的 metadata；缺省为空字符串。
        source: 书籍文件的相对路径（相对书籍库根目录），写入 metadata。

    Returns:
        Document 列表，阅读顺序排列。每个 chunk 的 metadata 包含：
        ``source``（相对路径）、``title``（书名）、``section``（章节路径）、
        ``position``（全书 chunk 的全局顺序号，从 0 起严格递增）。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        keep_separator=True,
    )

    documents: list[Document] = []
    for section in sections:
        for chunk_text in splitter.split_text(section.text):
            if not chunk_text.strip():
                continue
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": source,
                        "title": title,
                        "section": section.path,
                        "position": len(documents),
                    },
                )
            )
    return documents
