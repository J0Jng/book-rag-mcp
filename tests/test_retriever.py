"""retriever 模块测试。

覆盖四类行为：
- 基本检索：query 命中书中概念时返回非空结果，字段完整且内容来自对应书籍；
- ``book`` 参数：支持部分匹配书名 / 路径，无匹配时返回空；
- 语言路由：默认只搜同语言书籍，``search_all_languages=True`` 搜索全部；
- ``top_k``：限制返回结果数量。

为避免下载真实 HuggingFace 模型，索引构建与 query 编码均使用与
test_index.py 相同的基于 sha256 的确定性假向量；语言检测替换为固定结果。
"""

import hashlib
import math
import shutil
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from book_rag.config import BookRagConfig
from book_rag.rag import index as index_mod
from book_rag.rag import retriever as retriever_mod
from book_rag.rag.index import refresh_all_indexes
from book_rag.rag.retriever import SearchResult, _dedupe_results, search_books

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_DIM = 16


class FakeHashEmbeddings(Embeddings):
    """基于 sha256 的确定性假 embedding，与 test_index.py 保持一致，

    保证「索引时文档向量」与「检索时 query 向量」来自同一函数，可复现。
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
    """构造临时书籍库并建好两本书的索引，同时 mock 掉模型与语言检测。"""
    library = tmp_path / "library"
    library.mkdir()
    shutil.copy(FIXTURES / "sample_book.md", library / "sample_book.md")
    shutil.copy(FIXTURES / "sample_book.txt", library / "sample_book.txt")

    # 索引构建侧（index 模块内部引用）
    monkeypatch.setattr(
        index_mod, "get_embeddings_for_language", lambda lang: FakeHashEmbeddings()
    )
    monkeypatch.setattr(index_mod, "get_embeddings", lambda name: FakeHashEmbeddings())
    monkeypatch.setattr(index_mod, "detect_language", lambda text: "zh-cn")
    # 检索侧（retriever 模块内部引用）
    monkeypatch.setattr(retriever_mod, "detect_language", lambda text: "zh-cn")

    cfg = BookRagConfig(
        library_path=library, index_dir=tmp_path / "index", rerank_enabled=False
    )
    refresh_all_indexes(cfg)
    return cfg


# ---------------------------------------------------------------------------
# 基本检索
# ---------------------------------------------------------------------------


def test_search_returns_nonempty_results_with_fields(config: BookRagConfig):
    """query 命中书中概念：结果非空，字段完整，内容确实来自库内书籍。"""
    results = search_books("森林深处没有路标", config=config, top_k=3)

    assert results  # 非空
    for r in results:
        assert isinstance(r, SearchResult)
        assert r.content.strip()  # 有原文片段
        assert r.book  # 书名非空
        assert r.source in {"sample_book.md", "sample_book.txt"}
        assert r.section  # 章节路径非空
        assert r.score >= 0.0
        assert isinstance(r.position, int)


def test_search_result_content_comes_from_books(config: BookRagConfig):
    """返回的 content 应来自对应书籍原文（忽略切分时的空白差异）。"""
    results = search_books("清晨的雾气", config=config, top_k=3)

    assert results
    library = config.library_path
    for r in results:
        original = (library / r.source).read_text(encoding="utf-8")
        # splitter 可能把空行合并为单个换行，按行拆分后逐行比对
        assert all(line in original for line in r.content.splitlines())


# ---------------------------------------------------------------------------
# book 参数过滤
# ---------------------------------------------------------------------------


def test_book_filter_partial_match(config: BookRagConfig):
    """book 参数支持部分匹配：只返回匹配书籍的结果。"""
    results = search_books("旅人", config=config, top_k=5, book="md")

    assert results
    for r in results:
        assert r.source == "sample_book.md"


def test_book_filter_no_match_returns_empty(config: BookRagConfig):
    """book 参数无任何匹配时返回空列表，而不是抛异常。"""
    results = search_books("旅人", config=config, top_k=5, book="不存在的书")

    assert results == []


# ---------------------------------------------------------------------------
# 语言路由
# ---------------------------------------------------------------------------


def test_language_filter_excludes_other_language_books(
    config: BookRagConfig, monkeypatch: pytest.MonkeyPatch
):
    """query 检测为英文时，默认不应返回中文书籍的结果。"""
    monkeypatch.setattr(retriever_mod, "detect_language", lambda text: "en-US")

    results = search_books("mist over the valley", config=config, top_k=5)

    assert results == []


def test_search_all_languages_searches_everything(
    config: BookRagConfig, monkeypatch: pytest.MonkeyPatch
):
    """search_all_languages=True 时，跨语言也返回结果。"""
    monkeypatch.setattr(retriever_mod, "detect_language", lambda text: "en-US")

    results = search_books(
        "mist over the valley", config=config, top_k=5, search_all_languages=True
    )

    assert results


# ---------------------------------------------------------------------------
# top_k 限制
# ---------------------------------------------------------------------------


def test_top_k_limits_result_count(config: BookRagConfig):
    """返回结果数不超过 top_k。"""
    assert len(search_books("旅人", config=config, top_k=1)) == 1
    assert len(search_books("旅人", config=config, top_k=3)) <= 3


def test_empty_library_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """索引目录不存在 / 为空时返回空列表，而不是抛异常。"""
    monkeypatch.setattr(retriever_mod, "detect_language", lambda text: "zh-cn")
    cfg = BookRagConfig(
        library_path=tmp_path / "empty",
        index_dir=tmp_path / "no_index",
        rerank_enabled=False,
    )
    assert search_books("任何问题", config=cfg, top_k=5) == []


# ---------------------------------------------------------------------------
# workspace / genre 过滤与多 workspace 去重
# ---------------------------------------------------------------------------


def _write_workspaces(config: BookRagConfig, workspaces: str, books: str) -> None:
    """写入临时 workspaces.yaml。"""
    (config.library_path / "workspaces.yaml").write_text(
        f"workspaces:\n{workspaces}\nbooks:\n{books}\n", encoding="utf-8"
    )


def test_workspace_filter_only_returns_declared_book(config: BookRagConfig):
    """workspace=fiction 只返回声明为 fiction 的书。"""
    _write_workspaces(
        config,
        "  fiction: {label: 小说, genre: fiction}\n"
        "  science: {label: 科学, genre: knowledge}",
        "  sample_book.md: {workspaces: [fiction]}\n"
        "  sample_book.txt: {workspaces: [science]}",
    )

    results = search_books("旅人", config=config, top_k=10, workspace="fiction")

    assert results
    assert {r.source for r in results} == {"sample_book.md"}
    assert all(r.workspaces == ["fiction"] for r in results)
    assert all(r.genre == "fiction" for r in results)


def test_undeclared_book_filterable_as_unclassified(config: BookRagConfig):
    """未声明书可通过 workspace=_unclassified 显式检索到。"""
    _write_workspaces(
        config,
        "  fiction: {label: 小说, genre: fiction}",
        "  sample_book.md: {workspaces: [fiction]}",
    )

    results = search_books(
        "旅人", config=config, top_k=10, workspace="_unclassified"
    )

    assert results
    assert {r.source for r in results} == {"sample_book.txt"}
    assert all(r.workspaces == ["_unclassified"] for r in results)
    assert all(r.genre == "unknown" for r in results)


def test_genre_filter(config: BookRagConfig):
    """genre 过滤按推导后的体裁生效。"""
    _write_workspaces(
        config,
        "  fiction: {label: 小说, genre: fiction}\n"
        "  science: {label: 科学, genre: knowledge}",
        "  sample_book.md: {workspaces: [fiction]}\n"
        "  sample_book.txt: {workspaces: [science]}",
    )

    knowledge = search_books("旅人", config=config, top_k=10, genre="knowledge")
    fiction = search_books("旅人", config=config, top_k=10, genre="fiction")

    assert {r.source for r in knowledge} == {"sample_book.txt"}
    assert {r.source for r in fiction} == {"sample_book.md"}
    assert search_books("旅人", config=config, top_k=10, genre="mixed") == []


def test_yaml_change_takes_effect_without_reindex(config: BookRagConfig):
    """建索引后只改 yaml，不调用 refresh 也应立即生效（元数据与向量解耦）。"""
    assert {
        r.source
        for r in search_books("旅人", config=config, top_k=10, workspace="fiction")
    } == set()  # 无 yaml 时全部 _unclassified

    _write_workspaces(
        config,
        "  fiction: {label: 小说, genre: fiction}",
        "  sample_book.md: {workspaces: [fiction]}",
    )

    results = search_books("旅人", config=config, top_k=10, workspace="fiction")

    assert results
    assert {r.source for r in results} == {"sample_book.md"}


def test_multi_workspace_book_exposes_merged_workspaces(config: BookRagConfig):
    """多 workspace 的书：结果带全部 workspace，genre 跨体裁时为 mixed。"""
    _write_workspaces(
        config,
        "  fiction: {label: 小说, genre: fiction}\n"
        "  social: {label: 社会学, genre: knowledge}",
        "  sample_book.md: {workspaces: [fiction, social], tags: [经典]}",
    )

    for workspace in ("fiction", "social"):
        results = search_books("旅人", config=config, top_k=10, workspace=workspace)
        assert results
        assert all(r.workspaces == ["fiction", "social"] for r in results)
        assert all(r.genre == "mixed" for r in results)
        assert all(r.tags == ["经典"] for r in results)


def test_dedupe_results_merges_same_source_position():
    """同一 (source, position) 只保留一条，workspaces / tags 合并、分数取最大。"""
    first = SearchResult(
        content="片段",
        book="A",
        section="第一章",
        source="a.md",
        score=0.5,
        position=7,
        workspaces=["fiction"],
        genre="fiction",
        tags=["经典"],
    )
    second = SearchResult(
        content="片段",
        book="A",
        section="第一章",
        source="a.md",
        score=0.8,
        position=7,
        workspaces=["social"],
        genre="knowledge",
        tags=["社会学"],
    )
    other = SearchResult(
        content="另一段",
        book="A",
        section="第一章",
        source="a.md",
        score=0.3,
        position=8,
        workspaces=["fiction"],
        genre="fiction",
    )

    merged = _dedupe_results([first, second, other])

    assert len(merged) == 2
    deduped = merged[0]
    assert (deduped.source, deduped.position) == ("a.md", 7)
    assert deduped.workspaces == ["fiction", "social"]
    assert deduped.tags == ["经典", "社会学"]
    assert deduped.genre == "mixed"
    assert deduped.score == 0.8
    assert merged[1].position == 8
