"""FastMCP 服务定义：把 RAG 内核暴露为 MCP 工具。

工具设计面向 Agent（Claude Code / Codex / Cursor 等），工具名/描述语义清晰：
- ``list_books``：列出书籍库中已索引的书籍
- ``index_books``：建立/刷新整个书籍库的向量索引
- ``add_book``：添加并索引单个书籍文件
- ``search_books``：语义检索书籍内容（核心工具）
"""

from __future__ import annotations

from typing import Any

from book_rag.config import load_config
from book_rag.rag.retriever import Retriever


def _fmt_result(result: Any) -> dict:
    """把检索结果序列化成 JSON 可返回的字典。"""
    return {
        "query": result.query,
        "total_chunks": result.total_chunks,
        "hits": [
            {
                "text": hit.text,
                "score": round(hit.score, 4),
                "source": hit.source,
                "chunk_index": hit.chunk_index,
            }
            for hit in result.hits
        ],
    }


def create_server(ctrl: Retriever | None = None) -> Any:
    """构建并返回 FastMCP 服务实例。

    参数:
        ctrl: 可选注入的 Retriever（便于测试）；为 None 时创建默认实例。
    """
    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("需要安装 fastmcp：uv sync 或 pip install fastmcp") from exc

    cfg = load_config()
    server = FastMCP(
        "Book-RAG",
        instructions=(
            "书籍 RAG 检索服务。先调用 list_books 看有哪些书；需要精确内容时用 "
            "search_books 做语义检索；新增书籍用 add_book，批量导入用 index_books。"
        ),
    )

    @server.tool()
    def list_books() -> list[str]:
        """列出书籍库中已索引的所有书籍文件（相对路径）。"""
        r = ctrl if ctrl is not None else retriever()
        return r.list_documents()

    @server.tool()
    def index_books() -> dict[str, Any]:
        """扫描书籍库目录，建立或刷新所有书籍的向量索引。返回统计。"""
        r = ctrl if ctrl is not None else retriever()
        return r.index_library()

    @server.tool()
    def add_book(file_path: str) -> dict[str, Any]:
        """添加单个书籍文件并索引。file_path 为书籍库内相对路径或绝对路径。"""
        r = ctrl if ctrl is not None else retriever()
        return r.add_book(file_path)

    @server.tool()
    def search_books(query: str, k: int = 5) -> dict[str, Any]:
        """语义检索书籍内容。返回与查询最相关的若干文本块（含来源与相似度）。"""
        r = ctrl if ctrl is not None else retriever()
        return _fmt_result(r.search(query, k=k))

    return server


def retriever() -> Retriever:
    """全局默认 Retriever（懒加载单例，避免重复加载嵌入模型）。"""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER


_RETRIEVER: Retriever | None = None