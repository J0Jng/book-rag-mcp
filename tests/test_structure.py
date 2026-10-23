"""结构层（structure）测试：TOC 构建 / 落盘加载 / section 过滤检索 / 章节原文拉取。

覆盖五类行为：
- ``build_toc``：由 ``BookSection.path``（``" > "`` 分隔）+ ``level`` 构造嵌套目录树，
  叶子聚合 position，重复路径（如 EPUB 同文档多段）合并为一个叶子；
- ``save_structure`` / ``load_structure``：``toc.json`` 落盘与加载闭环，缺失返回 None；
- ``matches_section``：chunk 章节路径的 前缀匹配语义；
- ``build_book_index`` 落盘 ``toc.json``：建索引后目录树随索引持久化；
- ``search_books(section=...)`` 与 mcp_server 三个新工具：
  ``get_book_structure`` / ``search_within_section`` / ``get_section_content``，
  返回内容全部锚定书籍原文。

为避免下载真实 HuggingFace 模型，embedding 使用与 test_index.py 相同的
基于 sha256 的确定性假向量；语言检测替换为固定结果。
"""

import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from book_rag.config import BookRagConfig
from book_rag.io.parsers import BookSection
from book_rag.rag import index as index_mod
from book_rag.rag import retriever as retriever_mod
from book_rag.rag.index import build_book_index, get_book_index_dir, refresh_all_indexes
from book_rag.rag.retriever import search_books
from book_rag.rag.structure import build_toc, load_structure, matches_section, save_structure
from book_rag.servers import mcp_server

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_DIM = 16


class FakeHashEmbeddings(Embeddings):
    """基于 sha256 的确定性假 embedding，与 test_index.py 保持一致。"""

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
    """构造临时书籍库 + 索引目录，并 mock 掉真实模型与语言检测（不建索引）。"""
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

    return BookRagConfig(
        library_path=library, index_dir=tmp_path / "index", rerank_enabled=False
    )


# ---------------------------------------------------------------------------
# build_toc
# ---------------------------------------------------------------------------


def test_build_toc_nests_by_path():
    """嵌套路径按 ``" > "`` 拆分构造树；叶子带 position；中间节点不携带 position。"""
    sections = [
        BookSection(path="第一章 起点", level=1, text="正文甲", position=0),
        BookSection(
            path="第一章 起点 > 第一节 出发", level=2, text="正文乙", position=1
        ),
        BookSection(
            path="第一章 起点 > 第二节 露营", level=2, text="正文丙", position=2
        ),
        BookSection(path="第二章 归途", level=1, text="正文丁", position=3),
    ]

    toc = build_toc(sections)

    assert toc["children"][0]["name"] == "第一章 起点"
    assert toc["children"][0]["level"] == 1
    # 「第一章 起点」自身也有正文 section：叶子同时携带自己的 position
    assert toc["children"][0]["positions"] == [0]
    child_names = [c["name"] for c in toc["children"][0]["children"]]
    assert child_names == ["第一节 出发", "第二节 露营"]
    first = toc["children"][0]["children"][0]
    assert first["level"] == 2
    assert first["positions"] == [1]
    assert toc["children"][1] == {
        "name": "第二章 归途",
        "level": 1,
        "positions": [3],
        "children": [],
    }


def test_build_toc_aggregates_repeated_paths():
    """同一 section 路径出现多次（如 EPUB 同一文档多段）时聚合为一个叶子，positions 升序。"""
    sections = [
        BookSection(path="第 1 页", level=1, text="段落一", position=0),
        BookSection(path="第 1 页", level=1, text="段落二", position=1),
        BookSection(path="第 2 页", level=1, text="段落三", position=2),
    ]

    toc = build_toc(sections)

    assert [c["name"] for c in toc["children"]] == ["第 1 页", "第 2 页"]
    assert toc["children"][0]["positions"] == [0, 1]
    assert toc["children"][1]["positions"] == [2]


def test_build_toc_empty_sections():
    """空 section 列表返回只含根节点的空树，不抛异常。"""
    toc = build_toc([])

    assert toc == {"name": "", "level": 0, "positions": [], "children": []}


# ---------------------------------------------------------------------------
# save_structure / load_structure
# ---------------------------------------------------------------------------


def test_save_and_load_structure_roundtrip(tmp_path: Path):
    """save 落盘 toc.json，load 还原出等价的目录树。"""
    toc = build_toc([BookSection(path="第一章 起点", level=1, text="正文", position=0)])

    path = save_structure(tmp_path, toc)

    assert path == tmp_path / "toc.json"
    assert path.is_file()
    # 落盘的是合法 JSON（UTF-8、非 ASCII 不转义）
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["children"][0]["name"] == "第一章 起点"
    assert load_structure(tmp_path) == toc


def test_load_structure_returns_none_when_missing(tmp_path: Path):
    """toc.json 不存在时返回 None，而不是抛异常。"""
    assert load_structure(tmp_path) is None


# ---------------------------------------------------------------------------
# matches_section
# ---------------------------------------------------------------------------


def test_matches_section_prefix_semantics():
    """chunk 章节路径的匹配语义：路径前缀 / 完整路径 / 单级章节名（或其前缀）。"""
    # 完整路径链（MarkdownParser 产出形如「测试之书 > 第一章 起点」）
    assert matches_section("测试之书 > 第一章 起点", "测试之书 > 第一章 起点")  # 完整路径
    assert matches_section("测试之书 > 第一章 起点", "测试之书")  # 路径前缀
    assert matches_section("测试之书 > 第一章 起点", "第一章 起点")  # 单级章节名
    assert matches_section("测试之书 > 第一章 起点", "第一章")  # 章节名前缀
    assert matches_section("第二章 迷途", "第二章 迷途")  # 无链路径的精确匹配
    assert matches_section("第二章 迷途", "第二章")  # 前缀匹配
    assert not matches_section("第一章 起点", "第二章")  # 不同章节
    assert not matches_section("", "第二章")  # chunk 无章节信息
    assert matches_section("第二章 迷途", " 第二章 ")  # 两侧空白不影响匹配


# ---------------------------------------------------------------------------
# build_book_index 落盘 toc.json
# ---------------------------------------------------------------------------


def test_build_book_index_writes_toc_json(config: BookRagConfig):
    """build_book_index 后索引目录应包含 toc.json，内容覆盖该书的章节。"""
    book = config.library_path / "sample_book.md"
    meta = build_book_index(book, config)

    idx_dir = get_book_index_dir(config.index_dir, Path(meta.book_path))
    assert (idx_dir / "toc.json").is_file()

    toc = load_structure(idx_dir)
    assert toc is not None
    # sample_book.md 解析出的路径链为「测试之书 > 第 X 章 …」，目录树相应嵌套
    assert [c["name"] for c in toc["children"]] == ["测试之书"]
    assert [c["name"] for c in toc["children"][0]["children"]] == [
        "第一章 起点",
        "第二章 迷途",
    ]


# ---------------------------------------------------------------------------
# search_books 的 section 过滤
# ---------------------------------------------------------------------------


def test_search_books_section_filter_limits_results(config: BookRagConfig):
    """section 过滤后只返回章节路径前缀匹配的 chunk。"""
    refresh_all_indexes(config)

    results = search_books("旅人", config=config, top_k=5, section="第二章 迷途")

    assert results  # 第二章内有「旅人在溪边休息」等正文，应命中
    for r in results:
        assert "第二章 迷途" in r.section


def test_search_books_section_filter_with_book(config: BookRagConfig):
    """section 与 book 过滤可叠加：只返回指定书指定章节的结果。"""
    refresh_all_indexes(config)

    results = search_books(
        "清晨的雾气", config=config, top_k=5, book="md", section="第一章 起点"
    )

    assert results
    for r in results:
        assert r.source == "sample_book.md"
        assert "第一章 起点" in r.section


def test_search_books_section_no_match_returns_empty(config: BookRagConfig):
    """section 无任何匹配 chunk 时返回空列表，而不是抛异常。"""
    refresh_all_indexes(config)

    results = search_books("旅人", config=config, top_k=5, section="不存在的章节")

    assert results == []


# ---------------------------------------------------------------------------
# mcp_server：get_book_structure / search_within_section / get_section_content
# ---------------------------------------------------------------------------


def test_get_book_structure_returns_markdown_tree(config: BookRagConfig):
    """返回 Markdown 缩进列表，包含各级章节名。"""
    refresh_all_indexes(config)

    out = mcp_server.get_book_structure("sample_book.md", config=config)

    assert "sample_book.md" in out
    assert "- 测试之书" in out
    # 各级章节名以缩进列表呈现：两章嵌套在书名导语之下
    assert "  - 第一章 起点" in out
    assert "  - 第二章 迷途" in out


def test_get_book_structure_unknown_book(config: BookRagConfig):
    """无匹配书籍时返回提示，而不是抛异常。"""
    out = mcp_server.get_book_structure("不存在的书", config=config)

    assert "未找到匹配的书籍" in out


def test_search_within_section_returns_markdown(config: BookRagConfig):
    """search_within_section 返回 Markdown，原文片段来自指定章节。"""
    refresh_all_indexes(config)

    out = mcp_server.search_within_section(
        "旅人在溪边休息", book="txt", section="第二章 迷途", config=config
    )

    assert out.startswith("找到")
    assert "第二章 迷途" in out
    quoted = [line[2:] for line in out.splitlines() if line.startswith("> ")]
    assert quoted
    original = (config.library_path / "sample_book.txt").read_text(encoding="utf-8")
    for line in quoted:
        assert line in original


def test_get_section_content_aggregates_chunks(config: BookRagConfig):
    """get_section_content 聚合该 section 全部 chunk 原文（按 position 排序）。"""
    refresh_all_indexes(config)

    out = mcp_server.get_section_content("sample_book.txt", "第二章 迷途", config=config)

    assert "第二章 迷途" in out
    # 章节内的两段正文都必须出现，且不混入其他章节内容
    assert "森林深处没有路标" in out
    assert "旅人在溪边休息" in out
    assert "第一章 启程" not in out
    original = (config.library_path / "sample_book.txt").read_text(encoding="utf-8")
    for line in out.splitlines():
        if line and not line.startswith("#"):
            assert line in original  # 除标题行外全部来自书籍原文


def test_get_section_content_unknown_book_or_section(config: BookRagConfig):
    """书籍或章节无匹配时返回提示，而不是抛异常。"""
    refresh_all_indexes(config)

    assert "未找到匹配的书籍" in mcp_server.get_section_content(
        "不存在的书", "第二章", config=config
    )
    assert "未找到" in mcp_server.get_section_content(
        "sample_book.md", "不存在的章节", config=config
    )


# ---------------------------------------------------------------------------
# create_mcp 工具注册
# ---------------------------------------------------------------------------


async def test_create_mcp_registers_structure_tools(config: BookRagConfig):
    """create_mcp 注册三个结构层 MCP 工具，且背后的函数可调用。"""
    mcp = mcp_server.create_mcp(config)

    structure_tool = await mcp.get_tool("get_book_structure")
    within_tool = await mcp.get_tool("search_within_section")
    content_tool = await mcp.get_tool("get_section_content")

    out = structure_tool.fn("sample_book.md")
    assert "第一章 起点" in out
    searched = within_tool.fn("旅人在溪边休息", book="txt", section="第二章 迷途")
    assert searched.startswith("找到")
    content = content_tool.fn("sample_book.txt", "第二章 迷途")
    assert "旅人在溪边休息" in content
