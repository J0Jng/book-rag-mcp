"""文档解析器测试。

TXT / Markdown 使用 tests/fixtures 下的最小 fixture；
PDF / EPUB 在 tmp_path 中程序化生成最小文件，避免向仓库提交二进制文件。
"""

from pathlib import Path

import pytest

from book_rag.io.parsers import (
    EpubParser,
    MarkdownParser,
    PdfParser,
    TextParser,
    get_parser,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# 辅助：手工构造最小 PDF（每页一段 ASCII 文本），不引入额外依赖
# ---------------------------------------------------------------------------


def _make_minimal_pdf(path: Path, page_texts: list[str]) -> None:
    """生成一个只含纯文本页面的最小合法 PDF 文件。"""

    def esc(s: str) -> str:
        return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    n = len(page_texts)
    font_num = 3 + 2 * n  # Font 对象编号：紧跟所有 Page / Contents 对象之后

    objs: list[bytes] = []
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode("ascii"))
    for i, text in enumerate(page_texts):
        stream = f"BT /F1 12 Tf 72 720 Td ({esc(text)}) Tj ET".encode("ascii")
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {4 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 {font_num} 0 R >> >> >>".encode("ascii")
        )
        objs.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for num, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode("ascii")
    path.write_bytes(bytes(out))


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def test_markdown_parser_on_fixture():
    parsed = MarkdownParser().parse(FIXTURES / "sample_book.md")
    assert parsed.title == "sample_book"
    assert parsed.sections, "至少应解析出一个 section"
    assert all(s.text.strip() for s in parsed.sections)
    # position 从 0 起严格递增
    assert [s.position for s in parsed.sections] == list(range(len(parsed.sections)))
    # 标题路径应包含章节名
    all_paths = " | ".join(s.path for s in parsed.sections)
    assert "第一章 起点" in all_paths
    assert "第二章 迷途" in all_paths
    all_text = "\n".join(s.text for s in parsed.sections)
    assert "清晨的雾气" in all_text
    assert "森林深处" in all_text


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------


def test_text_parser_on_fixture():
    parsed = TextParser().parse(FIXTURES / "sample_book.txt")
    assert parsed.title == "sample_book"
    assert len(parsed.sections) >= 2, "应识别出两个章节"
    chapter1 = next(s for s in parsed.sections if "第一章" in s.path)
    chapter2 = next(s for s in parsed.sections if "第二章" in s.path)
    assert "清晨的雾气" in chapter1.text
    assert "森林深处" in chapter2.text


def test_text_parser_paragraphs_without_chapter(tmp_path):
    """无章节标题的纯文本应按空行分段，归入默认路径。"""
    book = tmp_path / "plain.txt"
    book.write_text("第一段内容。\n\n第二段内容。\n\n第三段内容。\n", encoding="utf-8")
    parsed = TextParser().parse(book)
    assert len(parsed.sections) == 3
    assert all(s.text for s in parsed.sections)
    assert parsed.sections[0].position == 0


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def test_pdf_parser(tmp_path):
    pdf_path = tmp_path / "sample_book.pdf"
    _make_minimal_pdf(pdf_path, ["The morning fog covered the valley.", "Second page text."])
    parsed = PdfParser().parse(pdf_path)
    assert parsed.title == "sample_book"
    assert parsed.sections, "至少应解析出一个 section"
    assert all(s.text.strip() for s in parsed.sections)
    assert [s.position for s in parsed.sections] == list(range(len(parsed.sections)))
    all_text = "\n".join(s.text for s in parsed.sections)
    assert "morning fog" in all_text
    assert "Second page" in all_text


# ---------------------------------------------------------------------------
# EPUB
# ---------------------------------------------------------------------------


def test_epub_parser(tmp_path):
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_title("测试 EPUB")
    book.set_language("zh")
    book.add_author("测试作者")
    chapter = epub.EpubHtml(title="第一章 启程", file_name="chap_01.xhtml", lang="zh")
    chapter.content = (
        "<html><body><h1>第一章 启程</h1>"
        "<p>清晨的雾气笼罩着山谷。</p></body></html>"
    )
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    out = tmp_path / "sample_book.epub"
    epub.write_epub(str(out), book)

    parsed = EpubParser().parse(out)
    assert parsed.title == "测试 EPUB"
    assert parsed.sections, "至少应解析出一个 section"
    assert any("第一章" in s.path for s in parsed.sections)
    assert any("清晨的雾气" in s.text for s in parsed.sections)


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def test_get_parser_factory():
    assert isinstance(get_parser(".md"), MarkdownParser)
    assert isinstance(get_parser(".markdown"), MarkdownParser)
    assert isinstance(get_parser(".txt"), TextParser)
    assert isinstance(get_parser(".TXT"), TextParser)
    assert isinstance(get_parser(".pdf"), PdfParser)
    assert isinstance(get_parser(".epub"), EpubParser)
    with pytest.raises(ValueError):
        get_parser(".docx")
