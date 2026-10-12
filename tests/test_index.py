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
    return BookRagConfig(library_path=library, index_dir=tmp_path / "index")


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

    assert meta.book_path == "sample_book.md"  # 相对书籍库的稳定路径
    assert meta.book_hash == compute_file_hash(book)
    assert meta.language == "zh"  # mock 返回 zh-cn，归一化后为 zh
    assert meta.model_name == "BAAI/bge-large-zh-v1.5"
    assert meta.chunk_count > 0
    assert meta.updated_at  # ISO 时间戳非空


def test_build_book_index_meta_json_roundtrip(config: BookRagConfig):
    """meta.json 落盘内容应能反序列化为等价的 BookIndexMeta。"""
    book = config.library_path / "sample_book.md"
    meta = build_book_index(book, config)

    idx_dir = get_book_index_dir(config.index_dir, Path(meta.book_path))
    data = json.loads((idx_dir / "meta.json").read_text(encoding="utf-8"))
    assert BookIndexMeta.from_dict(data) == meta


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
