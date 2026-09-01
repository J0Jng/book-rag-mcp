"""文本切分：把长文档切成可嵌入/检索的块。

用 LangChain 的 RecursiveCharacterTextSplitter，按语义边界（章节、段落、句子）
递归切分，避免切断语义。
"""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 语义边界分隔符（中英文都考虑）
_SEPARATORS = [
    "\n\n",        # 段落
    "\n",          # 行
    "。",          # 中文句号
    "！", "？",    # 中文标点
    ". ", "! ", "? ",  # 英文句点
    "；", "；",
    "; ",
    " ",          # 空格（最终兜底）
]

_PARAGRAPH_RE = re.compile(r"\n{2,}")


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    """把全文切成块，返回文本块列表。

    参数:
        text: 全文。
        chunk_size: 每个块的大致字符数。
        chunk_overlap: 相邻块重叠字符数（保留上下文，避免关键句被切断）。
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for chunk in splitter.split_text(paragraph):
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)
    return chunks