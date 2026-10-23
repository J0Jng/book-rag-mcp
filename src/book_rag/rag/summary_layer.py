"""可单独恢复的离线摘要层；核心 chunk 向量不会被重算。"""

from __future__ import annotations

import json
import logging
import os
from functools import partial
from pathlib import Path

from book_rag.io.atomic import write_json_atomic
from book_rag.io.parsers import BookSection
from book_rag.rag.concepts import build_concept_entries, build_concept_index
from book_rag.rag.summarizer import SUMMARY_VERSION, _group_sections, summarize_sections

logger = logging.getLogger(__name__)


def write_summary_layer(
    sections: list[BookSection],
    documents: list,
    llm,
    *,
    idx_dir: Path,
    source: str,
    title: str,
    genre: str,
    book_hash: str,
    embeddings_factory,
    workers: int = 1,
) -> dict:
    """逐章保存 pending；仅完整结果发布为 summaries.json，失败只记状态与日志。"""
    expected = len(_group_sections(sections, title=title))
    status = {
        "source": source,
        "book_hash": book_hash,
        "genre": genre,
        "summary_version": SUMMARY_VERSION,
        "expected": expected,
        "completed": 0,
        "complete": False,
        "concepts_complete": False,
        "llm_model": getattr(llm, "model", None),
        "llm_base_url": getattr(llm, "base_url", None),
    }
    status_path = idx_dir / "summary_status.json"
    if llm is None and status_path.is_file():
        try:
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            if (previous.get("complete") and previous.get("book_hash") == book_hash
                    and previous.get("genre") == genre
                    and previous.get("summary_version") == SUMMARY_VERSION
                    and (idx_dir / "summaries.json").is_file()):
                logger.warning("LLM 未配置，保留现有完整摘要：%s", source)
                return previous
        except (OSError, ValueError):
            pass
    if genre not in ("fiction", "knowledge") or llm is None:
        status["reason"] = "genre_unclassified" if llm else "llm_not_configured"
        logger.warning("摘要层跳过：%s，%s；核心索引可用", source, status["reason"])
        write_json_atomic(status_path, status)
        return status
    checkpoint = idx_dir / "summaries.pending.json"
    existing = []
    for path in (idx_dir / "summaries.json", checkpoint):
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    existing.extend(item for item in data if isinstance(item, dict))
            except (OSError, ValueError):
                logger.warning("忽略无效摘要断点：%s", source)
    saved = {entry.get("input_hash"): entry for entry in existing if entry.get("input_hash")}

    def progress(entries):
        saved.update({entry["input_hash"]: entry for entry in entries})
        write_json_atomic(checkpoint, list(saved.values()))
        status["completed"] = len(entries)
        write_json_atomic(status_path, status)

    write_json_atomic(status_path, status)
    try:
        summaries = summarize_sections(
            sections,
            llm,
            title=title,
            genre=genre,
            source=source,
            documents=documents,
            existing=existing,
            on_progress=progress,
            workers=workers,
        )
        status["completed"] = len(summaries)
        if len(summaries) != expected or not expected:
            status["reason"] = "incomplete_summaries"
            return status
        write_json_atomic(idx_dir / "summaries.json", summaries)
        status["complete"] = True
        if genre == "knowledge":
            concepts = build_concept_entries(summaries, source=source)
            status["concept_count"] = len(concepts)
            if concepts:
                build_concept_index(concepts, embeddings_factory(), idx_dir)
            status["concepts_complete"] = True
        return status
    except Exception as exc:  # noqa: BLE001 - optional layer must fail open
        logger.warning("摘要层降级：%s，%s；核心索引可用", source, type(exc).__name__)
        status["reason"] = "summary_layer_error"
        return status
    finally:
        write_json_atomic(status_path, status)


def enrich_indexes(config, *, book: str | None = None) -> list[dict]:
    """显式离线命令：只读取现有向量/快照，不触发核心索引构建或书籍移动。"""
    from book_rag.rag.embeddings import get_embeddings
    from book_rag.rag.index import (
        get_book_index_dir,
        load_book_index_cached,
        load_section_snapshot,
    )
    from book_rag.rag.llm import load_llm
    from book_rag.rag.retriever import _list_indexed_metas, _matches_book

    llm = load_llm()
    workers = int(os.environ.get("BOOK_RAG_SUMMARY_WORKERS", "4"))
    if not 1 <= workers <= 8:
        raise ValueError("BOOK_RAG_SUMMARY_WORKERS 必须在 1–8 之间")
    statuses = []
    for meta in _list_indexed_metas(config.index_dir, config.library_path):
        if book and not _matches_book(meta, book):
            continue
        idx_dir = get_book_index_dir(config.index_dir, Path(meta.book_path))
        snapshot = load_section_snapshot(idx_dir)
        if snapshot is None:
            logger.warning("摘要跳过：章节快照缺失，%s", meta.book_path)
            statuses.append(
                {"source": meta.book_path, "complete": False, "reason": "missing_sections"}
            )
            continue
        if llm is None:
            documents = []
        else:
            loaded = load_book_index_cached(config.index_dir, Path(meta.book_path))
            store = loaded[0]
            documents = [
                store.docstore.search(store.index_to_docstore_id[i])
                for i in range(store.index.ntotal)
            ]
        statuses.append(
            write_summary_layer(
                [BookSection(**s) for s in snapshot],
                documents,
                llm,
                idx_dir=idx_dir,
                source=meta.book_path,
                title=Path(meta.book_path).stem,
                genre=meta.genre,
                book_hash=meta.book_hash,
                embeddings_factory=partial(get_embeddings, meta.model_name),
                workers=workers,
            )
        )
    return statuses
