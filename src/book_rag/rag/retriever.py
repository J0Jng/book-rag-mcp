"""跨书籍 Hybrid 检索：向量 + BM25 → RRF 融合 → 单次 cross-encoder 重排。

设计要点（见 docs/plans/book-rag-library-expansion-brief.md S3）：
- 过滤出候选书籍（book / workspace / genre / 语言）后，对每本书分别取
  **向量 top N** 与 **BM25 top N**；
- 两路结果用 RRF（``k=60``）在**全局**候选集上融合，按
  ``1/(60+rank)`` 累加分数，同一 ``(source, position)`` 只保留一条；
- 融合后的 top M 候选只做**一次** cross-encoder 重排（``BAAI/bge-reranker-v2-m3``），
  再截断到 top_k；严禁每本书各自重排后合并；
- 重排可用 ``BOOK_RAG_RERANK=0`` 关闭；模型加载 / 推理失败时自动降级为
  融合排序，并记录真实原因；
- 索引加载走进程内缓存（FAISS + meta + BM25），按文件 mtime 失效，
  同一 query 的第二次检索不会重复 ``FAISS.load_local``。

语言检测保留：用于默认只搜同语言书籍（``search_all_languages=True`` 可关闭），
但不再决定 embedding 模型——所有书共享 bge-m3 向量空间。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from langchain_core.documents import Document

from book_rag.rag.bm25 import USER_DICT_FILENAME, ensure_user_dict
from book_rag.rag.index import (
    BookIndexMeta,
    _documents_by_position,
    load_book_index_cached,
)
from book_rag.rag.language import detect_language, normalize_language
from book_rag.rag.metadata import load_catalog
from book_rag.rag.reranker import rerank_scores
from book_rag.rag.structure import matches_section

#: RRF 融合常数（与常见实现一致，k=60 对长尾更平滑）
RRF_K = 60


@dataclass
class SearchResult:
    """单条检索结果。

    Attributes:
        content: 命中的原文片段。
        book: 书名（chunk 元数据中的 title，缺失时退回书籍相对路径）。
        section: 章节路径（解析器生成的 section.path）。
        source: 书籍文件的相对路径（相对书籍库根目录）。
        score: 最终相关度。开启重排时为 cross-encoder 的 sigmoid 分数（0–1）；
            关闭 / 降级时为 RRF 融合分（越大越相关，量级较小）。
        score_kind: ``rrf`` 或 ``rerank``；两类排序分不能比较，也不是相关概率。
        position: 该 chunk 在全书中的全局顺序号。
        workspaces: 该书所属 workspace 列表（多 workspace 已合并）。
        genre: 该书体裁。
        tags: 该书标签。
    """

    content: str
    book: str
    section: str
    source: str
    score: float
    position: int
    workspaces: list[str] = field(default_factory=list)
    genre: str = "unknown"
    tags: list[str] = field(default_factory=list)
    score_kind: str = "rrf"


def _list_indexed_metas(
    index_dir: Path, library_path: Path | None = None
) -> list[BookIndexMeta]:
    """扫描索引根目录，读取每本书的 meta.json 元数据。

    传入 ``library_path`` 时，按其中的 ``workspaces.yaml`` **实时**覆盖元数据，
    保证 yaml 修改后无需重建索引、甚至无需刷新即可生效。旧 meta.json 缺少
    workspace 字段时按 ``_unclassified`` / ``unknown`` / ``[]`` 加载。
    """
    metas: list[BookIndexMeta] = []
    if not index_dir.is_dir():
        return metas
    catalog = load_catalog(library_path) if library_path is not None else None
    for idx_dir in sorted(index_dir.iterdir()):
        if not idx_dir.name.startswith("book-"):
            continue
        meta_path = idx_dir / "meta.json"
        if idx_dir.is_dir() and meta_path.is_file():
            meta = BookIndexMeta.from_dict(
                json.loads(meta_path.read_text(encoding="utf-8"))
            )
            if catalog is not None:
                resolved = catalog.resolve(meta.book_path)
                meta = replace(
                    meta,
                    workspaces=list(resolved.workspaces),
                    genre=resolved.genre,
                    tags=list(resolved.tags),
                )
            metas.append(meta)
    return metas


def _matches_book(meta: BookIndexMeta, book: str) -> bool:
    """判断书籍是否匹配 book 过滤参数：对书籍路径做不区分大小写的部分匹配。

    书籍以文件形式存储时文件名即书名，因此对 ``meta.book_path`` 的
    部分匹配同时覆盖了「按书名」与「按路径」两种过滤意图。
    """
    return book.lower() in meta.book_path.lower()


def _matches_workspace(meta: BookIndexMeta, workspace: str | list[str] | None) -> bool:
    """判断书籍是否属于指定 workspace；None / 空值表示不过滤。"""
    if not workspace:
        return True
    wanted = [workspace] if isinstance(workspace, str) else list(workspace)
    wanted = [item for item in wanted if item]
    return not wanted or any(item in meta.workspaces for item in wanted)


def _matches_genre(meta: BookIndexMeta, genre: str | None) -> bool:
    """判断书籍体裁是否匹配；None / 空串表示不过滤。"""
    return not genre or meta.genre == genre


def _to_similarity(distance: float) -> float:
    """把 FAISS 的 L2 距离映射为越大越相关的相似度分。"""
    return 1.0 / (1.0 + distance)


def _dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    """按 ``(source, position)`` 去重，并合并同一 chunk 的 workspaces / tags。

    一本书可属多个 workspace；当同一 chunk 从多条路径被召回时，只保留一条，
    ``workspaces`` / ``tags`` 取并集（保序），分数取最大值，genre 冲突时归为
    ``mixed``。
    """
    merged: dict[tuple[str, int], SearchResult] = {}
    order: list[tuple[str, int]] = []
    for result in results:
        key = (result.source, result.position)
        existing = merged.get(key)
        if existing is None:
            merged[key] = replace(
                result,
                workspaces=list(result.workspaces),
                tags=list(result.tags),
            )
            order.append(key)
            continue

        workspaces = list(existing.workspaces)
        for workspace in result.workspaces:
            if workspace not in workspaces:
                workspaces.append(workspace)
        tags = list(existing.tags)
        for tag in result.tags:
            if tag not in tags:
                tags.append(tag)
        genres = {existing.genre, result.genre} - {"unknown"}
        if not genres:
            genre = "unknown"
        elif len(genres) == 1:
            genre = next(iter(genres))
        else:
            genre = "mixed"
        merged[key] = replace(
            existing,
            workspaces=workspaces,
            tags=tags,
            genre=genre,
            score=max(existing.score, result.score),
        )
    return [merged[key] for key in order]


@dataclass
class _Candidate:
    """融合阶段的单条候选（向量 / BM25 共用，按 position 去重）。

    ``bm25_score`` 是书内归一化后的 BM25 分（0–1），仅用于 RRF 同分时打破平局：
    bge-m3 让跨书向量距离可比，但 BM25 原始分不可比，故先按书内最大值归一。
    """

    document: Document
    meta: BookIndexMeta
    rrf_score: float = 0.0
    bm25_score: float = 0.0


def _vector_hits(vector_store, query: str, k: int, section: str | None, query_vector=None):
    """取一本书的向量 top-k；section 过滤时先取全书再按章节前缀筛选。"""
    total = int(vector_store.index.ntotal)
    if total <= 0 or k <= 0:
        return []
    if query_vector is None:
        query_vector = vector_store.embedding_function.embed_query(query)
    if section:
        hits = vector_store.similarity_search_with_score_by_vector(query_vector, k=total)
        hits = [
            (doc, distance)
            for doc, distance in hits
            if matches_section(doc.metadata.get("section", ""), section)
        ]
        return hits[:k]
    return vector_store.similarity_search_with_score_by_vector(query_vector, k=min(k, total))


def _bm25_hits(bm25, vector_store, query: str, k: int, section: str | None):
    """取一本书的 BM25 top-k；section 过滤由检索谓词在打分后完成。"""
    if bm25 is None or k <= 0:
        return []
    matcher = None
    if section:
        matcher = lambda doc: matches_section(
            doc.metadata.get("section", ""), section
        )
    return bm25.search(
        query, _documents_by_position(vector_store), k=k, section_matcher=matcher
    )


def _candidate_key(document: Document, meta: BookIndexMeta) -> tuple[str, int]:
    """候选的去重键：``(source, position)``。"""
    return (
        str(document.metadata.get("source", meta.book_path)),
        int(document.metadata.get("position", 0)),
    )


def _add_rrf(
    candidates: dict[tuple[str, int], _Candidate],
    document: Document,
    meta: BookIndexMeta,
    rank: int,
    *,
    bm25_score: float = 0.0,
) -> None:
    """把一条命中按其在本路结果中的名次累加 RRF 分数。"""
    key = _candidate_key(document, meta)
    candidate = candidates.get(key)
    if candidate is None:
        candidate = _Candidate(document=document, meta=meta)
        candidates[key] = candidate
    candidate.rrf_score += 1.0 / (RRF_K + rank)
    candidate.bm25_score = max(candidate.bm25_score, bm25_score)


def _to_search_result(
    candidate: _Candidate, score: float, score_kind: str = "rrf"
) -> SearchResult:
    """把融合候选转换为对外结果，并合并该书的 workspace / genre / tags。"""
    md = candidate.document.metadata
    meta = candidate.meta
    return SearchResult(
        content=candidate.document.page_content,
        book=md.get("title") or meta.book_path,
        section=md.get("section", ""),
        source=md.get("source", meta.book_path),
        score=float(score),
        position=int(md.get("position", 0)),
        workspaces=list(meta.workspaces),
        genre=meta.genre,
        tags=list(meta.tags),
        score_kind=score_kind,
    )


def _limit_with_coverage(items: list, limit: int, *, source) -> list:
    """显式多书查询预留每本书最高名次，再按全局排名补齐；总数仍受 limit 限制。"""
    first = {}
    for i, item in enumerate(items):
        first.setdefault(source(item), i)
    selected = set(first.values())
    if len(selected) > limit:
        raise ValueError("top_k 必须至少等于匹配的书籍数，才能覆盖每本书")
    for i in range(len(items)):
        if len(selected) >= limit:
            break
        selected.add(i)
    return [item for i, item in enumerate(items) if i in selected]


def search_books(
    query: str,
    *,
    config,
    top_k: int = 5,
    book: str | None = None,
    books: list[str] | None = None,
    search_all_languages: bool = False,
    section: str | None = None,
    workspace: str | list[str] | None = None,
    genre: str | None = None,
) -> list[SearchResult]:
    """Hybrid 检索：每书 向量 top N ‖ BM25 top N → RRF 全局融合 → 单次重排 → top_k。

    Args:
        query: 查询文本。
        config: 全局配置，提供索引目录、候选池大小与 rerank 开关。
        top_k: 返回的最大结果数。
        book: 书籍过滤参数，支持对书名 / 路径的部分匹配；None 表示不过滤。
        search_all_languages: 为 True 时搜索全部语言的书籍；默认只搜索
            与 query 同语言的书籍（query 语言检测失败时视为不过滤）。
        section: 章节过滤参数，对 chunk 的 ``section`` 元数据做前缀匹配
            （如 ``"第一章"`` 命中 ``"第一章 起点"``）；None / 空串表示不过滤。
        workspace: workspace 过滤，可传单个 id 或 id 列表；None / 空值表示不过滤。
            未在 ``workspaces.yaml`` 声明的书属于 ``_unclassified``。
        genre: 体裁过滤（fiction / knowledge / mixed / unknown）；None 表示不过滤。

    Returns:
        SearchResult 列表，按最终相关度从高到低排列；同一 ``(source, position)``
        只会出现一次，``workspaces`` 为 yaml 解析后的值。
    """
    if book is not None and books is not None:
        raise ValueError("book 与 books 不可同时指定")
    if books is not None:
        if not isinstance(books, list) or any(not isinstance(b, str) or not b.strip() for b in books):
            raise ValueError("books 必须是非空书名字符串的列表")
        if not books:
            return []
    if not query or not query.strip() or top_k <= 0:
        return []

    query_lang = normalize_language(detect_language(query))
    metas = _list_indexed_metas(config.index_dir, config.library_path)
    if book:
        metas = [m for m in metas if _matches_book(m, book)]
    if books is not None:
        metas = [m for m in metas if any(_matches_book(m, b) for b in books)]
    if workspace:
        metas = [m for m in metas if _matches_workspace(m, workspace)]
    if genre:
        metas = [m for m in metas if _matches_genre(m, genre)]
    if not search_all_languages and query_lang != "unknown":
        metas = [m for m in metas if m.language == query_lang]
    if not metas:
        return []
    coverage = books is not None and len(metas) > 1
    if coverage and top_k < len(metas):
        raise ValueError("top_k 必须至少等于匹配的书籍数，才能覆盖每本书")

    ensure_user_dict(config.library_path / USER_DICT_FILENAME)
    pool_size = max(1, int(getattr(config, "candidate_pool", 20)))
    rerank_size = max(top_k, int(getattr(config, "rerank_candidates", 20)))
    # 两路候选先按书收集，再在全局排序后赋名次：bge-m3 的跨书向量距离可比，
    # 若按「每本书各自 rank=1」做 RRF，所有书的 top-1 会同分，融合失去意义。
    vector_pool: list[tuple[float, Document, BookIndexMeta]] = []
    bm25_pool: list[tuple[float, float, Document, BookIndexMeta]] = []
    query_vectors: dict[str, list[float]] = {}

    for meta in metas:
        loaded = load_book_index_cached(config.index_dir, Path(meta.book_path))
        if loaded is None:
            continue
        vector_store, _, bm25 = loaded
        if meta.model_name not in query_vectors:
            query_vectors[meta.model_name] = vector_store.embedding_function.embed_query(query)
        for document, distance in _vector_hits(
            vector_store, query, pool_size, section, query_vectors[meta.model_name]
        ):
            vector_pool.append((float(distance), document, meta))
        bm25_hits = _bm25_hits(bm25, vector_store, query, pool_size, section)
        if bm25_hits:
            max_score = max(score for _, score in bm25_hits) or 1.0
            for document, score in bm25_hits:
                bm25_pool.append((score / max_score, score, document, meta))

    candidates: dict[tuple[str, int], _Candidate] = {}
    vector_pool.sort(key=lambda item: item[0])
    for rank, (_distance, document, meta) in enumerate(vector_pool, start=1):
        _add_rrf(candidates, document, meta, rank)
    bm25_pool.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for rank, (normalized, _score, document, meta) in enumerate(bm25_pool, start=1):
        _add_rrf(candidates, document, meta, rank, bm25_score=normalized)

    if not candidates:
        return []

    fused = sorted(
        candidates.values(), key=lambda item: (item.rrf_score, item.bm25_score), reverse=True
    )
    fused = _limit_with_coverage(
        fused, rerank_size, source=lambda item: item.meta.book_path,
    ) if coverage else fused[:rerank_size]

    # 只对融合后的全局候选集重排一次；失败 / 关闭时保留 RRF 排序
    if getattr(config, "rerank_enabled", False):
        scores = rerank_scores(query, [item.document.page_content for item in fused])
        if scores is not None:
            ranked = sorted(zip(fused, scores), key=lambda pair: pair[1], reverse=True)
            ranked = _limit_with_coverage(
                ranked, top_k, source=lambda pair: pair[0].meta.book_path,
            ) if coverage else ranked[:top_k]
            return [_to_search_result(item, score, "rerank") for item, score in ranked]

    selected = _limit_with_coverage(
        fused, top_k, source=lambda item: item.meta.book_path,
    ) if coverage else fused[:top_k]
    return [_to_search_result(item, item.rrf_score) for item in selected]
