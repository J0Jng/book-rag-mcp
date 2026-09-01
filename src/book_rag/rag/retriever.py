"""检索接口：组合切分 + 向量库，面向 Agent 的高级检索。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from book_rag.rag.embeddings import EmbeddingModel
from book_rag.rag.vector_store import FaissVectorStore, ScoredChunk


@dataclass
class SearchResult:
    """一次检索的结构化结果。"""

    query: str
    total_chunks: int
    hits: list[ScoredChunk] = field(default_factory=list)


class Retriever:
    """顶层检索器：负责建库、增删书籍、查询。

    这是 Agent 与 RAG 内核之间的主入口（MCP 工具直接调用它）。
    """

    def __init__(self, model: EmbeddingModel | None = None, store: FaissVectorStore | None = None) -> None:
        import logging

        self._log = logging.getLogger("book_rag.retriever")
        self.model = model or EmbeddingModel()
        self.store = store or FaissVectorStore(self.model)
        from book_rag.io.loader import list_books  # noqa: F401  (延迟,避免循环导入)

    # ---- 索引管理 ----

    def index_library(self) -> dict:
        """扫描书籍库并建立/更新索引，返回统计。"""
        from book_rag.config import load_config
        from book_rag.io.loader import list_books, load_document
        from book_rag.rag.splitter import split_text

        cfg = load_config()
        books = list_books(cfg.library)
        stats = {"scanned": len(books), "indexed_files": 0, "chunks": 0, "skipped": []}

        for book in books:
            rel, title, text = load_document(book)
            chunks = split_text(text, cfg.chunk_size, cfg.chunk_overlap)
            if not chunks:
                stats["skipped"].append(rel)
                continue
            self.store.add_texts(
                chunks,
                source=rel,
                metadata={"title": title, "chunk_index": list(range(len(chunks)))},
            )
            stats["indexed_files"] += 1
            stats["chunks"] += len(chunks)
        return stats

    def add_book(self, rel_or_abs: str) -> dict:
        """向向量库添加单个书籍文件。"""
        from book_rag.config import load_config
        from book_rag.io.loader import load_document
        from book_rag.rag.splitter import split_text

        cfg = load_config()
        rel, title, text = load_document(rel_or_abs)
        chunks = split_text(text, cfg.chunk_size, cfg.chunk_overlap)
        self.store.add_texts(
            chunks,
            source=rel,
            metadata={"title": title, "chunk_index": list(range(len(chunks)))},
        )
        return {"file": rel, "title": title, "chunks": len(chunks)}

    # ---- 检索 ----

    def search(self, query: str, k: int = 5) -> SearchResult:
        """言义检索，返回最相近的 k 个块。"""
        hits = self.store.search(query, k=k)
        return SearchResult(query=query, total_chunks=self.store.size, hits=hits)

    def list_documents(self) -> list[str]:
        """列出向量库中已索引的所有来源文件。"""
        return sorted({meta.get("source", "") for meta in self.store._meta.values()})  # noqa: SLF001

    @property
    def total_chunks(self) -> int:
        return self.store.size