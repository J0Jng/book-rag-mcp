"""跨书籍检索：按 query 语言路由书籍索引，合并各书结果并按相关度排序。

设计要点（见 docs/plans/book-rag-mvp-impl-brief.md 关键决策 3 / 文件清单 9）：
- 扫描索引根目录读取每本书的 ``meta.json``，得到候选书籍及其语言 / 路径；
- 默认只搜索与 query 同语言的书籍，``search_all_languages=True`` 时搜索全部；
- 对每本匹配的书用其构建时的模型（``meta.model_name``）编码 query，
  在该书独立的 FAISS 索引上执行 ``similarity_search_with_score``；
- FAISS 返回的 L2 距离越小越相关，统一映射为 ``1 / (1 + distance)`` 的
  相似度分（越大越相关）后跨书合并排序，保证 score 语义一致、直观。

注意：不同语言的书籍使用不同 embedding 模型，跨模型的分数只能用于
各自书内的相对排序，跨书绝对可比性有限——这也是默认按语言过滤的原因之一。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from book_rag.rag.embeddings import get_embeddings
from book_rag.rag.index import BookIndexMeta, load_book_index
from book_rag.rag.language import detect_language, normalize_language


@dataclass
class SearchResult:
    """单条检索结果。

    Attributes:
        content: 命中的原文片段。
        book: 书名（chunk 元数据中的 title，缺失时退回书籍相对路径）。
        section: 章节路径（解析器生成的 section.path）。
        source: 书籍文件的相对路径（相对书籍库根目录）。
        score: 相似度得分，由 FAISS L2 距离映射为 ``1 / (1 + d)``，越大越相关。
        position: 该 chunk 在全书中的全局顺序号。
    """

    content: str
    book: str
    section: str
    source: str
    score: float
    position: int


def _list_indexed_metas(index_dir: Path) -> list[BookIndexMeta]:
    """扫描索引根目录，读取每本书的 meta.json 元数据。"""
    metas: list[BookIndexMeta] = []
    if not index_dir.is_dir():
        return metas
    for idx_dir in sorted(index_dir.iterdir()):
        meta_path = idx_dir / "meta.json"
        if idx_dir.is_dir() and meta_path.is_file():
            metas.append(
                BookIndexMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
            )
    return metas


def _matches_book(meta: BookIndexMeta, book: str) -> bool:
    """判断书籍是否匹配 book 过滤参数：对书籍路径做不区分大小写的部分匹配。

    书籍以文件形式存储时文件名即书名，因此对 ``meta.book_path`` 的
    部分匹配同时覆盖了「按书名」与「按路径」两种过滤意图。
    """
    return book.lower() in meta.book_path.lower()


def _to_similarity(distance: float) -> float:
    """把 FAISS 的 L2 距离映射为越大越相关的相似度分。"""
    return 1.0 / (1.0 + distance)


def search_books(
    query: str,
    *,
    config,
    top_k: int = 5,
    book: str | None = None,
    search_all_languages: bool = False,
) -> list[SearchResult]:
    """在所有书籍索引中检索 query，返回按相关度排序的前 top_k 条结果。

    Args:
        query: 查询文本。
        config: 全局配置，提供索引目录等。
        top_k: 返回的最大结果数；对每本书先各取前 top_k 候选，再全局排序截断。
        book: 书籍过滤参数，支持对书名 / 路径的部分匹配；None 表示不过滤。
        search_all_languages: 为 True 时搜索全部语言的书籍；默认只搜索
            与 query 同语言的书籍（query 语言检测失败时视为不过滤）。

    Returns:
        SearchResult 列表，按 score 从高到低排列。
    """
    query_lang = normalize_language(detect_language(query))

    metas = _list_indexed_metas(config.index_dir)
    if book:
        metas = [m for m in metas if _matches_book(m, book)]
    if not search_all_languages and query_lang != "unknown":
        metas = [m for m in metas if m.language == query_lang]

    results: list[SearchResult] = []
    for meta in metas:
        loaded = load_book_index(config.index_dir, Path(meta.book_path))
        if loaded is None:
            continue
        vector_store, _ = loaded
        for doc, distance in vector_store.similarity_search_with_score(query, k=top_k):
            md = doc.metadata
            results.append(
                SearchResult(
                    content=doc.page_content,
                    book=md.get("title") or meta.book_path,
                    section=md.get("section", ""),
                    source=md.get("source", meta.book_path),
                    score=_to_similarity(distance),
                    position=md.get("position", 0),
                )
            )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
