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
import tempfile
import threading
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path

from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from book_rag.io.atomic import replace_with_retry
from book_rag.io.atomic import write_json_atomic as _write_json_atomic
from book_rag.io.cleaning import clean_matter
from book_rag.io.loader import compute_file_hash, discover_books, load_book
from book_rag.rag.bm25 import (
    BM25_FILENAME,
    USER_DICT_FILENAME,
    BM25Index,
    build_bm25_payload,
    dictionary_fingerprint,
    ensure_user_dict,
    load_bm25_index,
    save_bm25_index,
)
from book_rag.rag.embeddings import get_embeddings, get_embeddings_for_language
from book_rag.rag.language import (
    DEFAULT_EMBEDDING_MODEL,
    choose_embedding_model,
    detect_language,
    normalize_language,
)
from book_rag.rag.llm import load_llm
from book_rag.rag.metadata import (
    UNCLASSIFIED_WORKSPACE,
    UNKNOWN_GENRE,
    BookMetadata,
    WorkspaceCatalog,
    load_catalog,
)
from book_rag.rag.splitter import split_book
from book_rag.rag.structure import build_toc, save_structure
from book_rag.rag.summary_layer import write_summary_layer

# 采样检测语言时累计的正文字符数上限
_LANGUAGE_SAMPLE_CHARS = 1000
SECTIONS_FILENAME = "sections.json"
_INDEX_IO_LOCK = threading.RLock()
_REFRESH_LOCK = threading.RLock()


def _remove_index_directory(path: Path, root: Path) -> None:
    """删除前确认是真实索引根目录的直接子目录，禁止跟随越界符号链接。"""
    if path.resolve().parent != root.resolve():
        raise ValueError(f"拒绝删除索引根目录之外的路径：{path}")
    shutil.rmtree(path)


def load_section_snapshot(idx_dir: Path) -> list[dict] | None:
    """读取索引时保存的章节正文，绝不从重叠 chunk 猜测原文。"""
    path = idx_dir / SECTIONS_FILENAME
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        return None
    return data["sections"]


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
        workspaces: 所属 workspace 列表；未在 ``workspaces.yaml`` 声明时为
            ``["_unclassified"]``。
        genre: 体裁（fiction / knowledge / mixed / unknown）。
        tags: 自由标签列表。
    """

    book_path: str
    book_hash: str
    language: str
    model_name: str
    chunk_count: int
    updated_at: str
    workspaces: list[str] = field(
        default_factory=lambda: [UNCLASSIFIED_WORKSPACE]
    )
    genre: str = UNKNOWN_GENRE
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """序列化为可写入 JSON 的字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BookIndexMeta:
        """从 JSON 字典反序列化，兼容缺少 workspace 字段的旧 ``meta.json``。

        旧索引没有 ``workspaces`` / ``genre`` / ``tags`` 字段，按未声明书处理：
        ``["_unclassified"]`` / ``"unknown"`` / ``[]``。未知的额外键会被忽略，
        避免未来新增字段导致旧代码无法加载。
        """
        known_fields = {item.name for item in fields(cls)}
        payload = {key: value for key, value in data.items() if key in known_fields}
        payload.setdefault("workspaces", [UNCLASSIFIED_WORKSPACE])
        payload.setdefault("genre", UNKNOWN_GENRE)
        payload.setdefault("tags", [])
        return cls(**payload)


@dataclass
class RefreshResult:
    """一次增量同步的统计信息，各字段为书籍路径字符串列表。"""

    built: list[str] = field(default_factory=list)  # 本次新建 / 重建的书籍
    skipped: list[str] = field(default_factory=list)  # 未变更、跳过的书籍
    deleted: list[str] = field(default_factory=list)  # 已从库中移除、清理索引的书籍
    metadata_updated: list[str] = field(default_factory=list)  # 仅重写 meta.json 的书籍


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


def _build_book_index_at(book_path: Path, config, idx_dir: Path) -> BookIndexMeta:
    """解析、切分、向量化一本书并落盘索引与元数据。

    流程：load_book → clean_matter（front/back matter 清洗，可配置关闭）→
    检测语言（parsed.language 为空时采样检测）→ split_book →
    FAISS.from_documents → save_local → 落盘 toc.json → 写 cleaning_report.json →
    写 meta.json。

    Args:
        book_path: 书籍文件路径。
        config: 全局配置，提供书籍库根目录、索引目录与切分参数。

    Returns:
        构建完成的索引元数据。

    Raises:
        ValueError: 书籍解析后没有可索引的文本块。
    """
    parsed = load_book(book_path)
    # 用户词典影响 BM25 分词，必须在构建 / 检索前加载（两份代码共用同一词表）
    ensure_user_dict(config.library_path / USER_DICT_FILENAME)
    # 向量化前清洗 front/back matter；必须放在语言采样之前，
    # 避免推荐序 / 致谢污染语言检测样本。
    if config.clean_matter:
        parsed, report = clean_matter(parsed)
    else:
        report = None
    language = parsed.language or _sample_language(parsed.sections)
    parsed.language = language

    key = _book_key(book_path, config.library_path)
    # 元数据真源是 workspaces.yaml；与向量索引完全解耦，未声明的书归入 _unclassified
    book_metadata = load_catalog(config.library_path).resolve(key.as_posix())
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
    idx_dir.mkdir(parents=True, exist_ok=True)
    FAISS.from_documents(docs, embeddings).save_local(str(idx_dir))

    # 关键词索引随书落盘：分词结果与 chunk position 一一对应，增量重建时一并覆盖
    save_bm25_index(idx_dir, docs)

    # 结构层：目录树由解析结果聚合而成，随索引落盘（纯结构化，无 LLM）
    save_structure(idx_dir, build_toc(parsed.sections))
    _write_json_atomic(idx_dir / SECTIONS_FILENAME, {
        "version": 1,
        "source": key.as_posix(),
        "sections": [asdict(section) for section in parsed.sections],
    })

    # 清洗审计：记录被剥离的 front/back matter，便于排查与复核
    if report is not None:
        (idx_dir / "cleaning_report.json").write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 摘要层：LLM 可用时离线抽取章节摘要与概念索引；任何失败只降级跳过，
    # 绝不阻塞索引构建（无 LLM / 抽取失败时 TOC 层仍完整可用）
    write_summary_layer(
        parsed.sections, docs, load_llm(), idx_dir=idx_dir, source=key.as_posix(),
        title=parsed.title, genre=book_metadata.genre, book_hash=compute_file_hash(book_path),
        embeddings_factory=lambda: embeddings,
    )

    meta = BookIndexMeta(
        book_path=key.as_posix(),
        book_hash=compute_file_hash(book_path),
        language=language,
        model_name=choose_embedding_model(language),
        chunk_count=len(docs),
        updated_at=_utc_now_iso(),
        workspaces=book_metadata.workspaces,
        genre=book_metadata.genre,
        tags=book_metadata.tags,
    )
    _write_json_atomic(idx_dir / "meta.json", meta.to_dict())
    return meta


def build_book_index(book_path: Path, config) -> BookIndexMeta:
    """在临时目录完整构建并校验，再发布；失败保留上一版索引。"""
    config.index_dir.mkdir(parents=True, exist_ok=True)
    key = _book_key(book_path, config.library_path)
    target = get_book_index_dir(config.index_dir, key)
    original_hash = compute_file_hash(book_path)
    with tempfile.TemporaryDirectory(prefix=".building-", dir=config.index_dir) as temporary:
        staged = Path(temporary) / target.name
        meta = _build_book_index_at(book_path, config, staged)
        if original_hash != meta.book_hash or original_hash != compute_file_hash(book_path):
            raise RuntimeError(f"构建期间书籍发生变化，请重试：{key}")
        # 读回索引验证正文 position、向量数量与 BM25 的一一对应。
        loaded = load_book_index(Path(temporary), key)
        bm25 = load_bm25_index(staged)
        if loaded is None or bm25 is None:
            raise ValueError(f"索引不完整：{key}")
        store, _ = loaded
        positions = set(_documents_by_position(store))
        if (store.index.ntotal != meta.chunk_count
                or positions != set(range(meta.chunk_count))
                or set(bm25.positions) != positions
                or load_section_snapshot(staged) is None):
            raise ValueError(f"索引 position / 数量不一致：{key}")
        backup = config.index_dir / f".previous-{target.name}"
        with _INDEX_IO_LOCK:
            if backup.exists():
                if not target.exists():
                    replace_with_retry(backup, target)
                else:
                    _remove_index_directory(backup, config.index_dir)
            if target.exists():
                replace_with_retry(target, backup)
            try:
                replace_with_retry(staged, target)
            except BaseException:
                if backup.exists():
                    replace_with_retry(backup, target)
                raise
            clear_index_cache()
        if backup.exists():
            _remove_index_directory(backup, config.index_dir)
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


@dataclass
class CachedBookIndex:
    """进程内缓存的单本书索引：向量库 + 元数据 + BM25。"""

    signature: tuple
    vector_store: FAISS
    meta: BookIndexMeta
    bm25: BM25Index | None


# 书索引进程内缓存：key 为 (索引根目录, 书籍路径)，value 含失效签名
_INDEX_CACHE: dict[tuple[str, str], CachedBookIndex] = {}
_INDEX_CACHE_LOCK = threading.Lock()


def clear_index_cache() -> None:
    """清空进程内索引缓存（索引重建 / 删除后调用）。"""
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE.clear()


def _cache_signature(idx_dir: Path) -> tuple:
    """缓存签名包含向量、docstore、元数据、BM25 的 mtime 及词典指纹。"""

    def mtime_ns(name: str) -> int | None:
        path = idx_dir / name
        return path.stat().st_mtime_ns if path.is_file() else None

    return tuple(mtime_ns(name) for name in (
        "index.faiss", "index.pkl", "meta.json", BM25_FILENAME,
    )) + (dictionary_fingerprint(),)


def _documents_by_position(vector_store: FAISS) -> dict[int, object]:
    """从 FAISS docstore 提取 position → Document 映射（供 BM25 命中回查）。"""
    return {
        int(doc.metadata.get("position", -1)): doc
        for doc in vector_store.docstore._dict.values()
    }


def _load_book_index_cached(
    index_dir: Path, book_path: Path
) -> tuple[FAISS, BookIndexMeta, BM25Index | None] | None:
    """带进程内缓存的索引加载；文件未变化时直接复用，不再 ``FAISS.load_local``。

    缓存以索引文件的 mtime 与词典指纹为失效依据：任何一次重建（FAISS / meta / BM25）
    都会改变签名并触发重新加载。BM25 文件缺失时（旧索引）按 docstore 现场构建，
    不写盘——持久化由下一次增量同步补齐。
    """
    idx_dir = get_book_index_dir(index_dir, book_path)
    meta_path = idx_dir / "meta.json"
    if not meta_path.is_file():
        return None

    key = (str(Path(index_dir).resolve()), str(book_path).replace("\\", "/"))
    signature = _cache_signature(idx_dir)
    with _INDEX_CACHE_LOCK:
        cached = _INDEX_CACHE.get(key)
    if cached is not None and cached.signature == signature:
        return cached.vector_store, cached.meta, cached.bm25

    loaded = load_book_index(index_dir, book_path)
    if loaded is None:
        return None
    vector_store, meta = loaded
    bm25 = load_bm25_index(idx_dir)
    if bm25 is None:
        # 旧索引没有 bm25.json：按 docstore 现场构建内存索引（下次同步会补落盘）
        payload = build_bm25_payload(list(_documents_by_position(vector_store).values()))
        if any(item["tokens"] for item in payload["docs"]):
            bm25 = BM25Index(
                bm25=BM25Okapi([item["tokens"] for item in payload["docs"]]),
                positions=[item["position"] for item in payload["docs"]],
            )

    entry = CachedBookIndex(
        signature=signature, vector_store=vector_store, meta=meta, bm25=bm25
    )
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE[key] = entry
    return vector_store, meta, bm25


def load_book_index_cached(index_dir: Path, book_path: Path):
    """序列化进程内发布与加载，防止切换窗口读到半套文件。"""
    with _INDEX_IO_LOCK:
        return _load_book_index_cached(index_dir, book_path)


def apply_book_metadata(meta: BookIndexMeta, metadata: BookMetadata) -> BookIndexMeta:
    """把 ``workspaces.yaml`` 解析出的元数据合并进索引元数据（返回新对象）。"""
    return replace(
        meta,
        workspaces=list(metadata.workspaces),
        genre=metadata.genre,
        tags=list(metadata.tags),
    )


def sync_index_metadata(
    index_dir: Path,
    book_key: str,
    catalog: WorkspaceCatalog,
) -> bool:
    """按 yaml 真源重写一本书的 ``meta.json``，不触碰向量索引。

    Returns:
        ``True`` 表示 meta.json 有变化并被重写；索引不存在或无变化时返回 ``False``。
    """
    meta_path = get_book_index_dir(index_dir, Path(book_key)) / "meta.json"
    if not meta_path.is_file():
        return False
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = BookIndexMeta.from_dict(raw)
    updated = apply_book_metadata(meta, catalog.resolve(book_key))
    missing_fields = any(
        field_name not in raw for field_name in ("workspaces", "genre", "tags")
    )
    if updated == meta and not missing_fields:
        return False
    _write_json_atomic(meta_path, updated.to_dict())
    return True


def _refresh_all_indexes(config) -> RefreshResult:
    """增量同步书籍库与索引：新书建索引、变更重建、已删除的清理。

    以 ``manifest.json`` 中记录的文件 hash 为比对依据（mtime 仅作记录）。
    """
    config.index_dir.mkdir(parents=True, exist_ok=True)
    # 索引文件可能被重建 / 删除，进入同步前先清空进程内缓存
    clear_index_cache()
    manifest_path = config.index_dir / "manifest.json"
    manifest: dict[str, dict] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    result = RefreshResult()
    discovered_keys: set[str] = set()
    catalog = load_catalog(config.library_path)
    ensure_user_dict(config.library_path / USER_DICT_FILENAME)

    for book in discover_books(config.library_path):
        key = _book_key(book, config.library_path).as_posix()
        discovered_keys.add(key)
        book_hash = compute_file_hash(book)
        entry = manifest.get(key)
        idx_dir = get_book_index_dir(config.index_dir, Path(key))
        backup = config.index_dir / f".previous-{idx_dir.name}"
        # 上次进程若在 Windows 目录切换窗口被强制终止，先恢复完整旧版本。
        with _INDEX_IO_LOCK:
            if backup.exists() and not idx_dir.exists():
                replace_with_retry(backup, idx_dir)
        needs_build = entry is None or entry.get("hash") != book_hash
        if not needs_build:
            # 旧索引 / 旧向量模型：文件缺失或模型不是统一的 bge-m3 时重建
            meta_path = idx_dir / "meta.json"
            if any(not (idx_dir / name).is_file() for name in (
                "meta.json", "index.faiss", "index.pkl", "toc.json", SECTIONS_FILENAME,
            )):
                needs_build = True
            else:
                try:
                    existing = BookIndexMeta.from_dict(
                        json.loads(meta_path.read_text(encoding="utf-8"))
                    )
                    needs_build = (
                        existing.model_name != DEFAULT_EMBEDDING_MODEL
                        or existing.book_hash != book_hash
                        or existing.book_path != key
                    )
                except (OSError, ValueError, TypeError):
                    needs_build = True
        if needs_build:
            build_book_index(book, config)
            result.built.append(key)
        else:
            result.skipped.append(key)
            # 旧索引没有 BM25 落盘文件时补建（仅重建 BM25，不动向量索引）
            if load_bm25_index(idx_dir) is None:
                loaded = load_book_index(config.index_dir, Path(key))
                if loaded is not None:
                    save_bm25_index(
                        idx_dir, list(loaded[0].docstore._dict.values())
                    )
        # yaml 变更不需要重建 embedding：仅重写 meta.json 即可生效
        if sync_index_metadata(config.index_dir, key, catalog):
            result.metadata_updated.append(key)
        manifest[key] = {"hash": book_hash, "mtime": book.stat().st_mtime}
        _write_json_atomic(manifest_path, manifest)

    for key in list(manifest):
        if key not in discovered_keys:
            idx_dir = get_book_index_dir(config.index_dir, Path(key))
            if idx_dir.is_dir():
                _remove_index_directory(idx_dir, config.index_dir)
            del manifest[key]
            result.deleted.append(key)

    _write_json_atomic(manifest_path, manifest)
    return result


def refresh_all_indexes(config) -> RefreshResult:
    """同一进程串行同步；每本书发布后立即持久化 manifest 进度。"""
    with _REFRESH_LOCK:
        return _refresh_all_indexes(config)
