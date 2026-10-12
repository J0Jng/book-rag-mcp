"""mcp_server 模块测试。

覆盖四类行为：
- ``search_book_knowledge``：返回非空 Markdown，包含书名 / 章节路径 / 相关度 /
  来自书籍原文的片段；无命中时返回「未找到」提示；受 top_k 限制；
- ``list_indexed_books``：返回 ``"书名 (语言: zh, 块数: N)"`` 格式列表；
- ``refresh_book_index``：返回含新建 / 跳过 / 删除统计的文本；
- ``create_mcp``：创建时完成一次增量同步，并注册三个工具。

为避免下载真实 HuggingFace 模型，embedding 使用与 test_index.py 相同的
基于 sha256 的确定性假向量；语言检测替换为固定结果。
"""

import hashlib
import math
import re
import shutil
from pathlib import Path

import pytest

from langchain_core.embeddings import Embeddings

from book_rag.config import BookRagConfig
from book_rag.rag import index as index_mod
from book_rag.rag import retriever as retriever_mod
from book_rag.rag.index import refresh_all_indexes
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
    monkeypatch.setattr(retriever_mod, "get_embeddings", lambda name: FakeHashEmbeddings())
    monkeypatch.setattr(retriever_mod, "detect_language", lambda text: "zh-cn")

    return BookRagConfig(library_path=library, index_dir=tmp_path / "index")


@pytest.fixture
def indexed_config(config: BookRagConfig) -> BookRagConfig:
    """在 config 之上先建好两本书的索引。"""
    refresh_all_indexes(config)
    return config


# ---------------------------------------------------------------------------
# search_book_knowledge
# ---------------------------------------------------------------------------


def test_search_book_knowledge_returns_markdown(indexed_config: BookRagConfig):
    """命中查询：返回非空 Markdown，包含书名、章节、相关度与书中原文片段。"""
    out = mcp_server.search_book_knowledge("清晨的雾气", config=indexed_config)

    assert isinstance(out, str)
    assert out.strip()  # 非空
    assert out.startswith("找到")
    assert "sample_book" in out  # 书名 / 来源
    assert "相关度" in out

    # 引用块（原文片段）逐行比对：必须来自库内书籍原文
    originals = [
        (indexed_config.library_path / "sample_book.md").read_text(encoding="utf-8"),
        (indexed_config.library_path / "sample_book.txt").read_text(encoding="utf-8"),
    ]
    quoted = [line[2:] for line in out.splitlines() if line.startswith("> ")]
    assert quoted  # 至少有一条原文片段
    for line in quoted:
        assert any(line in original for original in originals)


def test_search_book_knowledge_respects_top_k(indexed_config: BookRagConfig):
    """top_k=1 时只返回一条结果（一个结果标题）。"""
    out = mcp_server.search_book_knowledge("旅人", config=indexed_config, top_k=1)

    assert out.count("\n### ") + out.startswith("### ") == 1


def test_search_book_knowledge_no_results(indexed_config: BookRagConfig):
    """无命中（book 过滤无匹配）时返回「未找到」提示，而不是抛异常。"""
    out = mcp_server.search_book_knowledge("旅人", config=indexed_config, book="不存在的书")

    assert "未找到" in out


# ---------------------------------------------------------------------------
# list_indexed_books
# ---------------------------------------------------------------------------


def test_list_indexed_books_format(indexed_config: BookRagConfig):
    """返回 ``"书名 (语言: zh, 块数: N)"`` 格式的列表，覆盖两本书。"""
    items = mcp_server.list_indexed_books(config=indexed_config)

    assert isinstance(items, list)
    assert len(items) == 2
    pattern = re.compile(r"^.+ \(语言: zh, 块数: \d+\)$")
    names = set()
    for item in items:
        assert pattern.match(item), f"格式不符合预期: {item}"
        names.add(item.split(" (")[0])
    assert names == {"sample_book.md", "sample_book.txt"}


def test_list_indexed_books_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """索引目录不存在时返回空列表，而不是抛异常。"""
    monkeypatch.setattr(retriever_mod, "detect_language", lambda text: "zh-cn")
    cfg = BookRagConfig(library_path=tmp_path / "empty", index_dir=tmp_path / "no_index")

    assert mcp_server.list_indexed_books(config=cfg) == []


# ---------------------------------------------------------------------------
# refresh_book_index
# ---------------------------------------------------------------------------


def test_refresh_book_index_reports_stats(indexed_config: BookRagConfig):
    """新增一本书后刷新：统计文本包含新建数量与书名。"""
    (indexed_config.library_path / "new_book.txt").write_text(
        "新的故事。\n\n第二段内容。\n", encoding="utf-8"
    )

    out = mcp_server.refresh_book_index(config=indexed_config)

    assert "索引同步完成" in out
    assert "新建/重建 1 本" in out
    assert "new_book.txt" in out
    assert "跳过 2 本" in out


# ---------------------------------------------------------------------------
# create_mcp
# ---------------------------------------------------------------------------


async def test_create_mcp_refreshes_on_startup_and_registers_tools(
    config: BookRagConfig,
):
    """create_mcp 创建时即做一次增量同步，并注册三个 MCP 工具。"""
    from fastmcp import FastMCP

    mcp = mcp_server.create_mcp(config)

    assert isinstance(mcp, FastMCP)
    # 启动同步生效：两本书的索引已建立
    assert (config.index_dir / "manifest.json").is_file()
    metas_dirs = [d for d in config.index_dir.iterdir() if d.is_dir()]
    assert len(metas_dirs) == 2

    search_tool = await mcp.get_tool("search_book_knowledge")
    list_tool = await mcp.get_tool("list_indexed_books")
    refresh_tool = await mcp.get_tool("refresh_book_index")

    # 工具背后的函数可直接调用
    items = list_tool.fn()
    assert isinstance(items, list) and len(items) == 2
    out = search_tool.fn("清晨的雾气")
    assert isinstance(out, str) and out.startswith("找到")
    stats = refresh_tool.fn()
    assert "索引同步完成" in stats
