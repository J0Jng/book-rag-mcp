"""长章保护测试：get_section_content 的 15,000 字上限与 offset 续取。

对应《工作、消费主义和新穷人》第四章 23,426 字的真实用例：章级切分保留超长章，
工具层必须截断并允许分页续取，避免一次返回整章撑爆 Agent 上下文。
"""

import hashlib
import math
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from book_rag.config import BookRagConfig
from book_rag.rag import index as index_mod
from book_rag.rag.index import refresh_all_indexes
from book_rag.servers import mcp_server
from book_rag.servers.mcp_server import SECTION_CONTENT_LIMIT

FAKE_DIM = 16
SENTENCE = "这是一段用于验证长章保护与偏移续取的测试正文，包含足够的汉字长度。"
LONG_BODY = "".join(f"第{i}句：{SENTENCE}" for i in range(600))


class FakeHashEmbeddings(Embeddings):
    """确定性假 embedding，避免测试下载真实模型。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [value / 255.0 for value in (digest * 2)[:FAKE_DIM]]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / norm for value in raw]


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BookRagConfig:
    """建一本含 2 万字以上超长章的书。"""
    library = tmp_path / "library"
    library.mkdir()
    (library / "长章之书.md").write_text(
        f"# 长章之书\n\n## 第四章 超长章节\n\n{LONG_BODY}\n", encoding="utf-8"
    )

    monkeypatch.setattr(
        index_mod, "get_embeddings_for_language", lambda language: FakeHashEmbeddings()
    )
    monkeypatch.setattr(index_mod, "get_embeddings", lambda name: FakeHashEmbeddings())
    monkeypatch.setattr(index_mod, "detect_language", lambda text: "zh-cn")

    cfg = BookRagConfig(
        library_path=library, index_dir=tmp_path / "index", rerank_enabled=False
    )
    refresh_all_indexes(cfg)
    return cfg


def _full_section_text(config: BookRagConfig) -> str:
    """独立从源文件解析原文，不能复用待测函数的拼接算法。"""
    from book_rag.io.parsers import MarkdownParser

    book = MarkdownParser().parse(config.library_path / "长章之书.md")
    return next(section.text for section in book.sections if "第四章" in section.path)


def test_long_section_is_truncated_with_offset_hint(config: BookRagConfig):
    """超过 15,000 字时只返回前 15,000 字，并提示 offset 续取。"""
    full = _full_section_text(config)
    assert len(full) > SECTION_CONTENT_LIMIT  # 前置条件：确实是超长章

    out = mcp_server.get_section_content("长章之书", "第四章", config=config)

    assert "内容已截断" in out
    assert f"offset={SECTION_CONTENT_LIMIT}" in out
    assert "单次上限 15000 字" in out
    assert full[:SECTION_CONTENT_LIMIT] in out
    assert out.split("\n\n", 1)[1].split("\n\n（", 1)[0] == full[:SECTION_CONTENT_LIMIT]
    # 输出体量受控（正文上限 + 少量标题 / 提示文字）
    assert len(out) < SECTION_CONTENT_LIMIT + 500


def test_offset_continues_from_exact_position(config: BookRagConfig):
    """offset=15000 续取到后文，且与首次返回不重叠。"""
    full = _full_section_text(config)

    out = mcp_server.get_section_content(
        "长章之书", "第四章", config=config, offset=SECTION_CONTENT_LIMIT
    )

    assert "已返回本章末尾" in out
    assert full[SECTION_CONTENT_LIMIT : SECTION_CONTENT_LIMIT + 40] in out
    assert "内容已截断" not in out


def test_offset_beyond_end_is_reported(config: BookRagConfig):
    """offset 超出本章长度时给出明确提示，而不是返回空正文。"""
    out = mcp_server.get_section_content(
        "长章之书", "第四章", config=config, offset=10_000_000
    )

    assert "已到达或超出本章末尾" in out


async def test_mcp_tool_exposes_offset(config: BookRagConfig):
    """MCP 注册的 get_section_content 工具暴露 offset 参数并透传给实现。"""
    mcp = mcp_server.create_mcp(config)
    tool = await mcp.get_tool("get_section_content")

    out = tool.fn("长章之书", "第四章", offset=SECTION_CONTENT_LIMIT)

    assert "已返回本章末尾" in out


def test_pages_reconstruct_original_exactly(config: BookRagConfig):
    """分页正文拼接必须逐字等于源章节，包含真实重复句子也不误去重。"""
    original = _full_section_text(config)
    pages = []
    for offset in range(0, len(original), SECTION_CONTENT_LIMIT):
        out = mcp_server.get_section_content(
            "长章之书", "第四章", config=config, offset=offset, limit=1_000_000
        )
        expected = original[offset:offset + SECTION_CONTENT_LIMIT]
        body = out.split("\n\n", 1)[1].split("\n\n（", 1)[0]
        assert body == expected
        pages.append(body)
    assert "".join(pages) == original


def test_missing_snapshot_requests_refresh(config: BookRagConfig):
    """旧索引不能通过带重叠片段伪造原文。"""
    idx_dir = index_mod.get_book_index_dir(config.index_dir, Path("长章之书.md"))
    (idx_dir / index_mod.SECTIONS_FILENAME).unlink()
    assert "请刷新索引" in mcp_server.get_section_content("长章之书", "第四章", config=config)


@pytest.mark.parametrize("length", [14999, 15000, 15001])
def test_exact_page_boundaries(config: BookRagConfig, length: int):
    """验证上限前、上限处与上限后一字；重复文本不丢失。"""
    source = config.library_path / "长章之书.md"
    body = "原" * length
    source.write_text(f"# 长章之书\n\n## 第四章\n{body}\n", encoding="utf-8")
    refresh_all_indexes(config)
    out = mcp_server.get_section_content("长章之书", "第四章", config=config)
    assert ("内容已截断" in out) == (length > SECTION_CONTENT_LIMIT)
    assert out.split("\n\n", 1)[1].split("\n\n（", 1)[0] == body[:15000]
