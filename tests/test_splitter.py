"""splitter 模块测试。使用 tests/fixtures/sample_book.md 的解析结果进行切分。"""

from pathlib import Path

from langchain_core.documents import Document

from book_rag.io.parsers import MarkdownParser
from book_rag.rag.splitter import split_book

FIXTURE_MD = Path(__file__).parent / "fixtures" / "sample_book.md"


def _split(chunk_size: int = 512, chunk_overlap: int = 64):
    """解析 fixture 并切分，返回 (ParsedBook.title, docs)。"""
    parsed = MarkdownParser().parse(FIXTURE_MD)
    docs = split_book(
        parsed.sections,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        title=parsed.title,
        source="fixtures/sample_book.md",
    )
    return parsed.title, docs


def test_split_book_returns_nonempty_documents():
    """切分结果应为非空的 Document 列表。"""
    _, docs = _split()
    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)


def test_split_book_metadata_complete():
    """每个 chunk 的 metadata 必须包含 source / title / section / position。"""
    title, docs = _split()

    for doc in docs:
        assert doc.metadata["source"] == "fixtures/sample_book.md"
        assert doc.metadata["title"] == title
        assert doc.metadata["section"]  # 章节路径非空
        assert isinstance(doc.metadata["position"], int)


def test_split_book_page_content_nonempty():
    """每个 chunk 的正文不得为空 / 纯空白。"""
    _, docs = _split()
    for doc in docs:
        assert doc.page_content.strip()


def test_split_book_positions_globally_ordered():
    """position 应为全书范围内的全局顺序号，从 0 起严格递增。"""
    _, docs = _split()
    positions = [doc.metadata["position"] for doc in docs]
    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions)
    assert positions[0] == 0


def test_split_book_sections_split_independently():
    """不同 section 的 chunk 不应混入彼此正文（按章节路径可区分）。"""
    _, docs = _split()
    sections = {doc.metadata["section"] for doc in docs}
    # fixture 有导语 + 两个章节，切分后至少应出现 2 个不同的章节路径
    assert len(sections) >= 2
