"""index 模块测试。

覆盖三层行为：
- ``get_book_index_dir``：目录名稳定且互不冲突；
- ``build_book_index`` / ``load_book_index``：索引与 meta.json 的读写闭环；
- ``refresh_all_indexes``：增量行为（新书建索引、未变更跳过、变更重建、删除清理）。

为避免下载真实 HuggingFace 模型，所有 embedding 均替换为基于 sha256 的
确定性假向量；语言检测替换为固定结果以保证测试可复现。
"""

import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from book_rag.config import BookRagConfig
from book_rag.io.loader import compute_file_hash
from book_rag.rag import index as index_mod
from book_rag.rag.index import (
    BookIndexMeta,
    build_book_index,
    get_book_index_dir,
    load_book_index,
    refresh_all_indexes,
)

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_DIM = 16


class FakeHashEmbeddings(Embeddings):
    """基于 sha256 的确定性假 embedding，向量维度固定，构建/加载可复现。

    继承 ``langchain_core.embeddings.Embeddings``，以满足 FAISS 的
    isinstance 检查（FAISS 对非 Embeddings 对象按可调用函数处理）。
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [b / 255.0 for b in (digest * 2)[:FAKE_DIM]]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BookRagConfig:
    """构造临时书籍库 + 索引目录，并 mock 掉真实模型与语言检测。"""
    library = tmp_path / "library"
    library.mkdir()
    shutil.copy(FIXTURES / "sample_book.md", library / "sample_book.md")
    shutil.copy(FIXTURES / "sample_book.txt", library / "sample_book.txt")

    monkeypatch.setattr(
        index_mod, "get_embeddings_for_language", lambda lang: FakeHashEmbeddings()
    )
    monkeypatch.setattr(index_mod, "get_embeddings", lambda name: FakeHashEmbeddings())
    monkeypatch.setattr(index_mod, "detect_language", lambda text: "zh-cn")
    return BookRagConfig(
        library_path=library, index_dir=tmp_path / "index", rerank_enabled=False
    )


# ---------------------------------------------------------------------------
# get_book_index_dir
# ---------------------------------------------------------------------------


def test_get_book_index_dir_stable_and_distinct(config: BookRagConfig):
    """同一本书目录名稳定，不同书籍目录名互不相同，且位于 index_dir 之下。"""
    md = config.library_path / "sample_book.md"
    txt = config.library_path / "sample_book.txt"

    d_md_1 = get_book_index_dir(config.index_dir, md)
    d_md_2 = get_book_index_dir(config.index_dir, md)
    d_txt = get_book_index_dir(config.index_dir, txt)

    assert d_md_1 == d_md_2  # 稳定
    assert d_md_1 != d_txt  # 不同书不冲突
    assert d_md_1.parent == config.index_dir
    # 目录名应是安全的 hash 形式，而非直接拼接原始路径
    assert "sample_book" not in d_md_1.name


# ---------------------------------------------------------------------------
# build_book_index / load_book_index
# ---------------------------------------------------------------------------


def test_build_book_index_creates_faiss_and_meta(config: BookRagConfig):
    """build 后应生成 index.faiss 与 meta.json，meta 字段与实际一致。"""
    book = config.library_path / "sample_book.md"
    meta = build_book_index(book, config)

    idx_dir = get_book_index_dir(config.index_dir, Path(meta.book_path))
    assert (idx_dir / "index.faiss").is_file()
    assert (idx_dir / "meta.json").is_file()
    assert (idx_dir / "bm25.json").is_file()  # S3：BM25 随书落盘

    assert meta.book_path == "sample_book.md"  # 相对书籍库的稳定路径
    assert meta.book_hash == compute_file_hash(book)
    assert meta.language == "zh"  # mock 返回 zh-cn，归一化后为 zh
    assert meta.model_name == "BAAI/bge-m3"  # S3 起统一单模型
    assert meta.chunk_count > 0
    assert meta.updated_at  # ISO 时间戳非空


def test_build_book_index_meta_json_roundtrip(config: BookRagConfig):
    """meta.json 落盘内容应能反序列化为等价的 BookIndexMeta。"""
    book = config.library_path / "sample_book.md"
    meta = build_book_index(book, config)

    idx_dir = get_book_index_dir(config.index_dir, Path(meta.book_path))
    data = json.loads((idx_dir / "meta.json").read_text(encoding="utf-8"))
    assert BookIndexMeta.from_dict(data) == meta


def test_build_book_index_writes_cleaning_report(config: BookRagConfig):
    """build 时剥离 front/back matter，落盘 cleaning_report.json 且索引不含被剥内容。"""
    book = config.library_path / "matter_book.md"
    book.write_text(
        "# 推荐序\n\n这是一段推荐序内容，不应进入索引。\n\n"
        "# 第一章 起点\n\n清晨的雾气笼罩着山谷，旅人背起行囊。\n\n"
        "# 致谢\n\n感谢所有帮助过这本书的人。\n",
        encoding="utf-8",
    )

    meta = build_book_index(book, config)
    idx_dir = get_book_index_dir(config.index_dir, Path(meta.book_path))

    report_path = idx_dir / "cleaning_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["aborted"] is False
    assert {item["path"] for item in report["removed"]} == {"推荐序", "致谢"}
    assert {item["kind"] for item in report["removed"]} == {"front", "back"}
    assert report["total_sections"] == 3
    assert report["kept_sections"] == 1

    # 索引里的 chunk 只来自清洗后的正文
    vector_store, _ = load_book_index(config.index_dir, Path(meta.book_path))
    texts = [doc.page_content for doc in vector_store.docstore._dict.values()]
    assert len(texts) == meta.chunk_count
    assert any("清晨的雾气" in text for text in texts)
    assert all("推荐序内容" not in text and "感谢所有" not in text for text in texts)

    # 目录树顶层不应出现被剥的 front/back matter
    toc = json.loads((idx_dir / "toc.json").read_text(encoding="utf-8"))
    top_names = {child["name"] for child in toc["children"]}
    assert "推荐序" not in top_names
    assert "致谢" not in top_names
    assert "第一章 起点" in top_names


def test_load_book_index_returns_none_when_missing(config: BookRagConfig):
    """索引不存在时应返回 None，而不是抛异常。"""
    assert load_book_index(config.index_dir, Path("never_built.md")) is None


def test_load_book_index_roundtrip(config: BookRagConfig):
    """build 后 load 应还原出 meta 与可检索的 FAISS 索引。"""
    book = config.library_path / "sample_book.txt"
    meta = build_book_index(book, config)

    loaded = load_book_index(config.index_dir, Path(meta.book_path))
    assert loaded is not None
    vector_store, loaded_meta = loaded

    assert loaded_meta == meta
    hits = vector_store.similarity_search("清晨的雾气", k=1)
    assert len(hits) == 1
    assert hits[0].metadata["source"] == "sample_book.txt"
    assert hits[0].metadata["title"] == "sample_book"


# ---------------------------------------------------------------------------
# refresh_all_indexes
# ---------------------------------------------------------------------------


def test_refresh_all_indexes_builds_all_new_books(config: BookRagConfig):
    """首次刷新：两本书全部新建索引，manifest 记录 hash 与 mtime。"""
    result = refresh_all_indexes(config)

    assert sorted(result.built) == ["sample_book.md", "sample_book.txt"]
    assert result.skipped == []
    assert result.deleted == []

    manifest_path = config.index_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == {"sample_book.md", "sample_book.txt"}
    for key, entry in manifest.items():
        assert entry["hash"] == compute_file_hash(config.library_path / key)
        assert isinstance(entry["mtime"], float)


def test_refresh_all_indexes_skips_unchanged(config: BookRagConfig):
    """未变更的书二次刷新应跳过，不重建索引。"""
    refresh_all_indexes(config)
    result = refresh_all_indexes(config)

    assert result.built == []
    assert sorted(result.skipped) == ["sample_book.md", "sample_book.txt"]
    assert result.deleted == []


def test_refresh_all_indexes_rebuilds_modified_book(config: BookRagConfig):
    """修改一本书后，仅该书被重建，其余书跳过。"""
    refresh_all_indexes(config)

    txt = config.library_path / "sample_book.txt"
    txt.write_text(txt.read_text(encoding="utf-8") + "\n\n旅人终于走出了森林。\n", encoding="utf-8")

    result = refresh_all_indexes(config)

    assert result.built == ["sample_book.txt"]
    assert result.skipped == ["sample_book.md"]
    assert result.deleted == []


def test_refresh_all_indexes_deletes_removed_books(config: BookRagConfig):
    """删除一本书后，其索引目录被清理且从 manifest 移除。"""
    refresh_all_indexes(config)

    txt = config.library_path / "sample_book.txt"
    txt_index_dir = get_book_index_dir(config.index_dir, Path("sample_book.txt"))
    assert txt_index_dir.is_dir()
    txt.unlink()

    result = refresh_all_indexes(config)

    assert result.deleted == ["sample_book.txt"]
    assert result.built == []
    assert "sample_book.txt" not in result.skipped
    assert not txt_index_dir.exists()

    manifest = json.loads(
        (config.index_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest) == {"sample_book.md"}


# ---------------------------------------------------------------------------
# 元数据（workspaces.yaml）与向后兼容
# ---------------------------------------------------------------------------


def _write_workspaces(config: BookRagConfig, text: str) -> None:
    """在临时书籍库根目录写入 workspaces.yaml。"""
    (config.library_path / "workspaces.yaml").write_text(text, encoding="utf-8")


def test_book_index_meta_from_dict_backward_compatible():
    """旧 meta.json 缺少新字段时按 _unclassified / unknown / [] 加载。"""
    old = {
        "book_path": "old.md",
        "book_hash": "abc",
        "language": "zh",
        "model_name": "BAAI/bge-large-zh-v1.5",
        "chunk_count": 10,
        "updated_at": "2026-01-01T00:00:00+00:00",
    }

    meta = BookIndexMeta.from_dict(old)

    assert meta.workspaces == ["_unclassified"]
    assert meta.genre == "unknown"
    assert meta.tags == []
    assert meta.to_dict()["book_path"] == "old.md"


def test_book_index_meta_ignores_unknown_future_fields():
    """未知的额外字段被忽略，不阻塞旧代码加载新 meta.json。"""
    data = {
        "book_path": "old.md",
        "book_hash": "abc",
        "language": "zh",
        "model_name": "m",
        "chunk_count": 1,
        "updated_at": "t",
        "workspaces": ["fiction"],
        "genre": "fiction",
        "tags": ["经典"],
        "future_field": 123,
    }

    meta = BookIndexMeta.from_dict(data)

    assert meta.workspaces == ["fiction"]
    assert meta.genre == "fiction"
    assert meta.tags == ["经典"]


def test_build_book_index_applies_workspace_metadata(config: BookRagConfig):
    """建索引时按 workspaces.yaml 写入 workspaces / genre / tags。"""
    _write_workspaces(
        config,
        """
workspaces:
  fiction: {label: 小说, genre: fiction}
books:
  sample_book.md:
    workspaces: [fiction]
    tags: [经典]
""",
    )

    meta = build_book_index(config.library_path / "sample_book.md", config)

    assert meta.workspaces == ["fiction"]
    assert meta.genre == "fiction"
    assert meta.tags == ["经典"]

    idx_dir = get_book_index_dir(config.index_dir, Path(meta.book_path))
    on_disk = json.loads((idx_dir / "meta.json").read_text(encoding="utf-8"))
    assert on_disk["workspaces"] == ["fiction"]


def test_refresh_updates_metadata_without_rebuilding_embedding(config: BookRagConfig):
    """只改 yaml：refresh 跳过内容未变的书，但仍重写 meta.json，不碰向量索引。"""
    refresh_all_indexes(config)

    md_key = "sample_book.md"
    idx_dir = get_book_index_dir(config.index_dir, Path(md_key))
    index_mtime = (idx_dir / "index.faiss").stat().st_mtime_ns
    assert json.loads((idx_dir / "meta.json").read_text(encoding="utf-8"))["workspaces"] == [
        "_unclassified"
    ]

    _write_workspaces(
        config,
        """
workspaces:
  fiction: {label: 小说, genre: fiction}
books:
  sample_book.md:
    workspaces: [fiction]
    tags: [经典]
""",
    )

    result = refresh_all_indexes(config)

    assert result.built == []  # 未重建 embedding
    assert md_key in result.skipped
    assert md_key in result.metadata_updated
    assert (idx_dir / "index.faiss").stat().st_mtime_ns == index_mtime

    on_disk = json.loads((idx_dir / "meta.json").read_text(encoding="utf-8"))
    assert on_disk["workspaces"] == ["fiction"]
    assert on_disk["genre"] == "fiction"
    assert on_disk["tags"] == ["经典"]


def test_refresh_rewrites_old_meta_missing_workspace_fields(config: BookRagConfig):
    """旧 meta.json 缺字段时，refresh 会补齐默认值（未声明书仍显式可见）。"""
    refresh_all_indexes(config)
    txt_key = "sample_book.txt"
    idx_dir = get_book_index_dir(config.index_dir, Path(txt_key))
    meta_path = idx_dir / "meta.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    for field_name in ("workspaces", "genre", "tags"):
        data.pop(field_name, None)
    meta_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    result = refresh_all_indexes(config)

    assert txt_key in result.metadata_updated
    updated = json.loads(meta_path.read_text(encoding="utf-8"))
    assert updated["workspaces"] == ["_unclassified"]
    assert updated["genre"] == "unknown"
    assert updated["tags"] == []


def test_refresh_resumes_after_second_book_failure(config: BookRagConfig, monkeypatch):
    """第二本失败时第一本进度已持久化，重试不重新向量化第一本。"""
    real_build = index_mod.build_book_index
    calls = []

    def fail_second(book, cfg):
        calls.append(book.name)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return real_build(book, cfg)

    monkeypatch.setattr(index_mod, "build_book_index", fail_second)
    with pytest.raises(RuntimeError, match="interrupted"):
        refresh_all_indexes(config)
    manifest = json.loads((config.index_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 1
    first = next(iter(manifest))
    monkeypatch.setattr(index_mod, "build_book_index", real_build)
    resumed = refresh_all_indexes(config)
    assert first in resumed.skipped
    assert len(resumed.built) == 1


def test_failed_rebuild_preserves_published_index(config: BookRagConfig, monkeypatch):
    """构建中 BM25 写入失败不得覆盖上一版 FAISS 或 meta。"""
    book = config.library_path / "sample_book.md"
    meta = build_book_index(book, config)
    target = get_book_index_dir(config.index_dir, Path(meta.book_path))
    previous = {path.name: path.read_bytes() for path in target.iterdir()}

    def interrupted(*args):
        raise RuntimeError("write failure")

    monkeypatch.setattr(index_mod, "save_bm25_index", interrupted)
    with pytest.raises(RuntimeError, match="write failure"):
        build_book_index(book, config)
    assert {path.name: path.read_bytes() for path in target.iterdir()} == previous
    assert not list(config.index_dir.glob(".building-*"))


@pytest.mark.parametrize("filename", ["index.pkl", "toc.json", "sections.json"])
def test_refresh_repairs_missing_required_file(config: BookRagConfig, filename: str):
    """存在 manifest 不代表完整；必要文件丢失时必须重建修复。"""
    refresh_all_indexes(config)
    target = get_book_index_dir(config.index_dir, Path("sample_book.md"))
    (target / filename).unlink()
    result = refresh_all_indexes(config)
    assert "sample_book.md" in result.built
    assert (target / filename).is_file()


def test_refresh_recovers_interrupted_directory_switch(config: BookRagConfig):
    """模拟进程在旧目录改名后被终止，下次刷新先恢复上一版完整目录。"""
    refresh_all_indexes(config)
    target = get_book_index_dir(config.index_dir, Path("sample_book.md"))
    backup = config.index_dir / f".previous-{target.name}"
    index_mod.replace_with_retry(target, backup)
    result = refresh_all_indexes(config)
    assert result.built == []
    assert target.is_dir() and not backup.exists()


def test_refresh_repairs_corrupt_metadata(config: BookRagConfig):
    """旧进程写坏的 meta 不应永久阻塞恢复。"""
    refresh_all_indexes(config)
    target = get_book_index_dir(config.index_dir, Path("sample_book.md"))
    (target / "meta.json").write_text("{", encoding="utf-8")
    assert "sample_book.md" in refresh_all_indexes(config).built
