"""BookRagConfig 单元测试。"""

from pathlib import Path

from book_rag.config import BookRagConfig


def test_default_field_values():
    """未显式传参时使用默认分块与 top_k 配置。"""
    config = BookRagConfig(library_path=Path("books"), index_dir=Path("books/.book_rag_index"))
    assert config.chunk_size == 512
    assert config.chunk_overlap == 64
    assert config.default_top_k == 5


def test_from_env_reads_env_vars(monkeypatch):
    """设置环境变量时，library_path 与 index_dir 均取自环境。"""
    monkeypatch.setenv("BOOK_RAG_LIBRARY", "/tmp/my-library")
    monkeypatch.setenv("BOOK_RAG_INDEX_DIR", "/tmp/my-index")
    config = BookRagConfig.from_env()
    assert config.library_path == Path("/tmp/my-library")
    assert config.index_dir == Path("/tmp/my-index")


def test_from_env_default_index_dir(monkeypatch):
    """未设置 BOOK_RAG_INDEX_DIR 时，默认为书籍库目录下的 .book_rag_index。"""
    monkeypatch.setenv("BOOK_RAG_LIBRARY", "F:/tmp/lib")
    monkeypatch.delenv("BOOK_RAG_INDEX_DIR", raising=False)
    config = BookRagConfig.from_env()
    assert config.library_path == Path("F:/tmp/lib")
    assert config.index_dir == Path("F:/tmp/lib") / ".book_rag_index"


def test_rerank_enabled_by_default(monkeypatch):
    """未设置 BOOK_RAG_RERANK 时默认开启重排。"""
    monkeypatch.delenv("BOOK_RAG_RERANK", raising=False)

    assert BookRagConfig.from_env().rerank_enabled is True


def test_rerank_can_be_disabled_by_env(monkeypatch):
    """BOOK_RAG_RERANK=0 / false / off / no 时关闭重排。"""
    for value in ("0", "false", "OFF", "no"):
        monkeypatch.setenv("BOOK_RAG_RERANK", value)
        assert BookRagConfig.from_env().rerank_enabled is False

    monkeypatch.setenv("BOOK_RAG_RERANK", "1")
    assert BookRagConfig.from_env().rerank_enabled is True
