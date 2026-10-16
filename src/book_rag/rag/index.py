"""书籍级 FAISS 索引的构建、持久化与增量同步。

设计要点（见 docs/plans/book-rag-mvp-impl-brief.md 关键决策 1 / 5）：
- 每本书一个独立索引子目录（``index.faiss`` + ``index.pkl`` + ``meta.json``），
  新增 / 变更 / 删除一本书只影响该书自己的目录；
- 全局 ``manifest.json`` 记录每本书的 sha256 与 mtime，作为增量比对的依据；
- 书籍身份统一使用相对书籍库根目录的 POSIX 路径字符串（不在库内的退回
  原路径），索引目录名由该路径的 sha256 派生，跨平台稳定且不含特殊字符。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from langchain_community.vectorstores import FAISS

from book_rag.io.loader import compute_file_hash, discover_books, load_book
from book_rag.rag.concepts import build_concept_entries, build_concept_index
from book_rag.rag.embeddings import get_embeddings, get_embeddings_for_language
from book_rag.rag.language import choose_embedding_model, detect_language, normalize_language
from book_rag.rag.llm import load_llm
from book_rag.rag.summarizer import summarize_sections
from book_rag.rag.splitter import split_book
from book_rag.rag.structure import build_toc, save_structure

# 采样检测语言时累计的正文字符数上限
_LANGUAGE_SAMPLE_CHARS = 1000


@dataclass
class BookIndexMeta:
    """单本书的索引元数据，可序列化为 JSON（``meta.json`` 的内容）。

    Attributes:
        book_path: 书籍相对书籍库根目录的 POSIX 路径（库外书籍为原路径）。
        book_hash: 书籍文件内容的 SHA256，用于增量比对。
        language: 书籍主导语言（归一化后的 ISO 639-1 码，如 ``"zh"``）。
        model_name: 构建索引时实际使用的 embedding 模型名。
        chunk_count: 切分出的 Document 块数。
        updated_at: 索引构建时间（UTC ISO 8601 字符串）。
    """

    book_path: str
    book_hash: str
    language: str
    model_name: str
    chunk_count: int
    updated_at: str

    def to_dict(self) -> dict:
        """序列化为可写入 JSON 的字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BookIndexMeta:
        """从 JSON 字典反序列化。"""
        return cls(**data)


@dataclass
class RefreshResult:
    """一次增量同步的统计信息，各字段为书籍路径字符串列表。"""

    built: list[str] = field(default_factory=list)  # 本次新建 / 重建的书籍
    skipped: list[str] = field(default_factory=list)  # 未变更、跳过的书籍
    deleted: list[str] = field(default_factory=list)  # 已从库中移除、清理索引的书籍


def _utc_now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串。"""
    return datetime.now(UTC).isoformat()


def _book_key(book_path: Path, library_path: Path) -> Path:
    """计算书籍的稳定身份路径：优先取相对书籍库根目录的路径，否则原样返回。"""
    try:
        return book_path.resolve().relative_to(library_path.resolve())
    except ValueError:
        return book_path


def get_book_index_dir(index_dir: Path, book_path: Path) -> Path:
    """由书籍路径派生其索引子目录：``<index_dir>/book-<sha256 前 16 位>``。

    对路径字符串（统一为 POSIX 分隔符）取 hash，保证同一本书在任何
    平台、任何会话中目录名稳定，且不含文件系统特殊字符。

    Args:
        index_dir: 索引根目录。
        book_path: 书籍路径（建议与 ``meta.book_path`` 同形式，即相对路径）。
    """
    key = str(book_path).replace("\\", "/")
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return index_dir / f"book-{digest}"


def _sample_language(sections) -> str:
    """采样前若干 section 的正文检测语言，超出上限字符数即停止。"""
    texts: list[str] = []
    total = 0
    for section in sections:
        texts.append(section.text)
        total += len(section.text)
        if total >= _LANGUAGE_SAMPLE_CHARS:
            break
    return normalize_language(detect_language("\n".join(texts)))


def build_book_index(book_path: Path, config) -> BookIndexMeta:
    """解析、切分、向量化一本书并落盘索引与元数据。

    流程：load_book → 检测语言（parsed.language 为空时采样检测）→
    split_book → FAISS.from_documents → save_local → 落盘 toc.json → 写 meta.json。

    Args:
        book_path: 书籍文件路径。
        config: 全局配置，提供书籍库根目录、索引目录与切分参数。

    Returns:
        构建完成的索引元数据。

    Raises:
        ValueError: 书籍解析后没有可索引的文本块。
    """
    parsed = load_book(book_path)
    language = parsed.language or _sample_language(parsed.sections)
    parsed.language = language

    key = _book_key(book_path, config.library_path)
    docs = split_book(
        parsed.sections,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        title=parsed.title,
        source=key.as_posix(),
    )
    if not docs:
        raise ValueError(f"书籍没有可索引的文本块：{book_path}")

    embeddings = get_embeddings_for_language(language)
    idx_dir = get_book_index_dir(config.index_dir, key)
    idx_dir.mkdir(parents=True, exist_ok=True)
    FAISS.from_documents(docs, embeddings).save_local(str(idx_dir))

    # 结构层：目录树由解析结果聚合而成，随索引落盘（纯结构化，无 LLM）
    save_structure(idx_dir, build_toc(parsed.sections))

    # 摘要层：LLM 可用时离线抽取章节摘要与概念索引；任何失败只降级跳过，
    # 绝不阻塞索引构建（无 LLM / 抽取失败时 TOC 层仍完整可用）
    llm = load_llm()
    if llm is not None:
        try:
            summaries = summarize_sections(parsed.sections, llm, title=parsed.title)
            if summaries:
                (idx_dir / "summaries.json").write_text(
                    json.dumps(summaries, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                concepts = build_concept_entries(summaries, source=key.as_posix())
                build_concept_index(concepts, embeddings, idx_dir)
        except Exception:  # noqa: BLE001 - 摘要/概念层降级：失败只跳过
            pass

    meta = BookIndexMeta(
        book_path=key.as_posix(),
        book_hash=compute_file_hash(book_path),
        language=language,
        model_name=choose_embedding_model(language),
        chunk_count=len(docs),
        updated_at=_utc_now_iso(),
    )
    (idx_dir / "meta.json").write_text(
        json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def load_book_index(index_dir: Path, book_path: Path):
    """加载一本书的索引与元数据。

    Args:
        index_dir: 索引根目录。
        book_path: 书籍路径，需与构建时 ``meta.book_path`` 的形式一致
            （即相对书籍库根目录的路径）。

    Returns:
        ``(FAISS 索引, BookIndexMeta)`` 元组；索引不存在时返回 ``None``。
    """
    idx_dir = get_book_index_dir(index_dir, book_path)
    meta_path = idx_dir / "meta.json"
    if not meta_path.is_file():
        return None

    meta = BookIndexMeta.from_dict(
        json.loads(meta_path.read_text(encoding="utf-8"))
    )
    embeddings = get_embeddings(meta.model_name)
    vector_store = FAISS.load_local(
        str(idx_dir), embeddings, allow_dangerous_deserialization=True
    )
    return vector_store, meta


def refresh_all_indexes(config) -> RefreshResult:
    """增量同步书籍库与索引：新书建索引、变更重建、已删除的清理。

    以 ``manifest.json`` 中记录的文件 hash 为比对依据（mtime 仅作记录）。
    """
    config.index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.index_dir / "manifest.json"
    manifest: dict[str, dict] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    result = RefreshResult()
    discovered_keys: set[str] = set()

    for book in discover_books(config.library_path):
        key = _book_key(book, config.library_path).as_posix()
        discovered_keys.add(key)
        book_hash = compute_file_hash(book)
        entry = manifest.get(key)
        if entry is not None and entry.get("hash") == book_hash:
            result.skipped.append(key)
        else:
            build_book_index(book, config)
            result.built.append(key)
        manifest[key] = {"hash": book_hash, "mtime": book.stat().st_mtime}

    for key in list(manifest):
        if key not in discovered_keys:
            idx_dir = get_book_index_dir(config.index_dir, Path(key))
            if idx_dir.is_dir():
                shutil.rmtree(idx_dir)
            del manifest[key]
            result.deleted.append(key)

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
