"""书籍归一（EPUB/TXT → Markdown）的单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from book_rag.convert import (
    clean_book_title,
    convert_book,
    convert_library,
    extract_book_names,
    sanitize_filename,
)
from book_rag.io.loader import discover_books
from book_rag.io.parsers import MarkdownParser

# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------


def test_extract_book_names_from_omnibus_title():
    title = "诺贝尔经济学奖经典收藏版（自由选择、生活中的经济学、增长的极限）（套装三册）"
    assert extract_book_names(title) == ["自由选择", "生活中的经济学", "增长的极限"]


def test_extract_book_names_ignores_plain_subtitle():
    assert extract_book_names("投资最重要的事（珍藏版）") == []
    assert extract_book_names("杀死一只知更鸟") == []


def test_clean_book_title_strips_long_blurb():
    assert (
        clean_book_title('投资最重要的事（"这是一本难得一见的有用的书。"——沃伦巴菲特）')
        == "投资最重要的事"
    )


def test_clean_book_title_keeps_short_parenthetical():
    assert clean_book_title("人类简史（新版）") == "人类简史（新版）"


def test_sanitize_filename_replaces_illegal_chars():
    assert sanitize_filename('a/b:c*d?"e"') == "a b c d e"
    assert sanitize_filename("   ") == "untitled"
    assert len(sanitize_filename("长" * 200)) <= 80


# ---------------------------------------------------------------------------
# 测试用 EPUB 构造
# ---------------------------------------------------------------------------


def _make_epub(path: Path, title: str, docs: list[tuple[str, str]], toc) -> None:
    """按 (file_name, html) 列表构造最小 EPUB，spine 顺序即 docs 顺序。"""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_title(title)
    book.set_language("zh")
    book.add_author("测试作者")

    items = []
    for file_name, html in docs:
        item = epub.EpubHtml(title=file_name, file_name=file_name, lang="zh")
        item.content = html
        book.add_item(item)
        items.append(item)

    book.toc = toc
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]
    epub.write_epub(str(path), book)


@pytest.fixture
def part_chapter_epub(tmp_path) -> Path:
    """「部 → 章」两级目录。"""
    from ebooklib import epub

    out = tmp_path / "part.epub"
    docs = [
        (
            "part_01.xhtml",
            (
                "<html><body><h1>第一部</h1>"
                "<p>这是第一部的开篇说明文字，用于占位测试内容。</p></body></html>"
            ),
        ),
        (
            "chap_02.xhtml",
            "<html><body><h1>第二章 归途</h1><p>归途的列车驶向远方。</p></body></html>",
        ),
        (
            "chap_03.xhtml",
            "<html><body><h1>第三章 重逢</h1><p>站台上有人等候。</p></body></html>",
        ),
    ]
    toc = [
        (
            epub.Section("第一部", href="part_01.xhtml"),
            [
                epub.Link("chap_02.xhtml", "第二章 归途", "c2"),
                epub.Link("chap_03.xhtml", "第三章 重逢", "c3"),
            ],
        )
    ]
    _make_epub(out, "两级目录测试", docs, toc)
    return out


@pytest.fixture
def flat_toc_epub(tmp_path) -> Path:
    """全平铺目录（无嵌套），应退化为「书名 + 章」。"""
    from ebooklib import epub

    out = tmp_path / "flat.epub"
    docs = [
        (f"c{i}.xhtml", f"<html><body><h1>章节{i}</h1><p>第{i}章正文内容说明。</p></body></html>")
        for i in range(1, 6)
    ]
    toc = [epub.Link(f"c{i}.xhtml", f"章节{i}", f"c{i}") for i in range(1, 6)]
    _make_epub(out, "平铺目录测试", docs, toc)
    return out


# ---------------------------------------------------------------------------
# EPUB → Markdown
# ---------------------------------------------------------------------------


def test_epub_part_and_chapter_headings(part_chapter_epub):
    from book_rag.convert.markdown import epub_to_markdown

    markdown, result = epub_to_markdown(part_chapter_epub)

    assert not result.error
    # 「部」这一级占 `#`，章占 `##`；书名由文件名承载，不额外写一行 `#`
    assert markdown.startswith("# 第一部")
    assert "\n## 第二章 归途\n" in markdown
    assert "\n## 第三章 重逢\n" in markdown
    # 标题不双写：章标题在正文中只出现一次
    assert markdown.count("第二章 归途") == 1
    assert result.chapters == 2


def test_epub_flat_toc_falls_back_to_book_plus_chapters(flat_toc_epub):
    from book_rag.convert.markdown import epub_to_markdown

    markdown, result = epub_to_markdown(flat_toc_epub)

    assert markdown.startswith("# 平铺目录测试")
    assert "\n## 章节1\n" in markdown
    assert "\n## 章节5\n" in markdown
    assert "# 章节1" not in markdown.replace("## 章节1", "")
    assert result.chapters == 5


def test_epub_body_text_preserved(part_chapter_epub):
    from book_rag.convert.markdown import epub_to_markdown

    markdown, _ = epub_to_markdown(part_chapter_epub)
    assert "归途的列车驶向远方。" in markdown
    assert "站台上有人等候。" in markdown


def test_epub_skips_front_matter_documents(tmp_path):
    from ebooklib import epub

    from book_rag.convert.markdown import epub_to_markdown

    out = tmp_path / "front.epub"
    docs = [
        (
            "front_01.xhtml",
            "<html><body><h1>版权</h1><p>本书版权归某某出版社所有，禁止翻印。</p></body></html>",
        ),
        (
            "chap_01.xhtml",
            "<html><body><h1>第一章 启程</h1><p>清晨的山谷里起了雾。</p></body></html>",
        ),
    ]
    toc = [epub.Link("chap_01.xhtml", "第一章 启程", "c1")]
    _make_epub(out, "版权页测试", docs, toc)

    markdown, result = epub_to_markdown(out)
    assert "禁止翻印" not in markdown
    assert any("版权" in item for item in result.dropped_docs)


def test_epub_omnibus_keeps_book_scope(tmp_path):
    """合集：书名与章同为顶层节点时，章要挂到书名之下，而不是并列。"""
    from ebooklib import epub

    from book_rag.convert.markdown import epub_to_markdown

    out = tmp_path / "omnibus.epub"
    docs = [
        (f"d{i}.xhtml", f"<html><body><h1>t{i}</h1><p>正文{i}的说明内容。</p></body></html>")
        for i in range(1, 7)
    ]
    toc = [
        epub.Link("d1.xhtml", "甲书（珍藏版）", "d1"),
        epub.Link("d2.xhtml", "第1章 甲的开端", "d2"),
        epub.Link("d3.xhtml", "第2章 甲的继续", "d3"),
        epub.Link("d4.xhtml", "乙书（新版）", "d4"),
        epub.Link("d5.xhtml", "第1章 乙的开端", "d5"),
        epub.Link("d6.xhtml", "第2章 乙的继续", "d6"),
    ]
    _make_epub(out, "合集测试（甲书、乙书）（套装两册）", docs, toc)

    markdown, _result = epub_to_markdown(out)
    lines = markdown.splitlines()
    assert "# 甲书（珍藏版）" in lines
    assert "# 乙书（新版）" in lines
    assert "## 第1章 甲的开端" in lines
    assert "## 第1章 乙的开端" in lines

    # 解析回来：章的路径应带书名层级
    parsed = MarkdownParser().parse(_write_tmp(markdown, tmp_path / "omnibus.md"))
    paths = [section.path for section in parsed.sections]
    assert any(p.startswith("甲书（珍藏版） > 第1章") for p in paths)
    assert any(p.startswith("乙书（新版） > 第1章") for p in paths)


def test_epub_drops_untitled_toc_and_copyright_docs(tmp_path):
    """无 heading 的目录页 / 版权页不应被并进正文（会污染检索）。"""
    from ebooklib import epub

    from book_rag.convert.markdown import epub_to_markdown

    out = tmp_path / "untitled.epub"
    docs = [
        (
            "toc_text.xhtml",
            (
                "<html><body><p>目录</p><p>第一章 启程</p><p>第二章 归途</p>"
                "<p>第三章 重逢</p><p>第四章 别离</p></body></html>"
            ),
        ),
        (
            "pub_info.xhtml",
            (
                "<html><body><p>本书纸版由某某出版社出版，版权所有，侵权必究。"
                "ISBN 978-7-111-00000-0</p></body></html>"
            ),
        ),
        (
            "chap_01.xhtml",
            "<html><body><h1>第一章 启程</h1><p>清晨的山谷里起了雾，旅人收拾行装。</p></body></html>",
        ),
    ]
    toc = [epub.Link("chap_01.xhtml", "第一章 启程", "c1")]
    _make_epub(out, "无标题文档测试", docs, toc)

    markdown, result = epub_to_markdown(out)
    assert "第一章 启程" in markdown
    assert "侵权必究" not in markdown
    assert "第三章 重逢" not in markdown  # 目录正文不应混入
    dropped = " ".join(result.dropped_docs)
    assert "toc_text.xhtml" in dropped
    assert "pub_info.xhtml" in dropped


def test_epub_without_any_chapter_raises(tmp_path):
    """完全推不出章级结构时应当显式失败，而不是产出一份空书。"""

    from book_rag.convert.markdown import epub_to_markdown

    out = tmp_path / "nochapter.epub"
    docs = [
        (f"d{i}.xhtml", f"<html><body><p>无标题段落{i}，仅正文没有 heading。</p></body></html>")
        for i in range(3)
    ]
    _make_epub(out, "无章节测试", docs, [])
    with pytest.raises(ValueError):
        epub_to_markdown(out)


def _write_tmp(markdown: str, path: Path) -> Path:
    path.write_text(markdown, encoding="utf-8")
    return path


def test_generated_markdown_parses_into_expected_sections(part_chapter_epub, tmp_path):
    from book_rag.convert.markdown import epub_to_markdown

    markdown, _ = epub_to_markdown(part_chapter_epub)
    md_path = _write_tmp(markdown, tmp_path / "out.md")
    parsed = MarkdownParser().parse(md_path)

    assert [s.path for s in parsed.sections] == [
        "第一部",
        "第一部 > 第二章 归途",
        "第一部 > 第三章 重逢",
    ]


# ---------------------------------------------------------------------------
# 落盘 / 幂等 / 归档
# ---------------------------------------------------------------------------


def test_convert_book_is_idempotent(part_chapter_epub, tmp_path):
    out_dir = tmp_path / "library"
    first = convert_book(part_chapter_epub, out_dir)
    second = convert_book(part_chapter_epub, out_dir)

    assert first.output == second.output
    assert Path(first.output).read_text(encoding="utf-8") == Path(second.output).read_text(
        encoding="utf-8"
    )
    assert len(list(out_dir.glob("*.md"))) == 1


def test_convert_library_moves_originals_and_writes_map(part_chapter_epub, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    moved = src / part_chapter_epub.name
    moved.write_bytes(part_chapter_epub.read_bytes())
    dest = tmp_path / "library"
    originals = tmp_path / "_originals"

    results = convert_library(src, dest, originals)

    assert len(results) == 1 and not results[0].error
    assert (originals / part_chapter_epub.name).is_file()
    assert not (src / part_chapter_epub.name).exists()
    assert (dest / "_conversion.json").is_file()
    assert "output_name" in (dest / "_conversion.json").read_text(encoding="utf-8")


def test_convert_library_rerun_keeps_single_output(part_chapter_epub, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    source = src / part_chapter_epub.name
    source.write_bytes(part_chapter_epub.read_bytes())
    dest = tmp_path / "library"
    originals = tmp_path / "_originals"

    convert_library(src, dest, originals)
    # 二次转换：把归档文件放回源目录模拟重跑
    (originals / part_chapter_epub.name).replace(source)
    convert_library(src, dest, originals)

    assert len(list(dest.glob("*.md"))) == 1


def test_discover_books_skips_internal_dirs(tmp_path):
    library = tmp_path / "books"
    (library / "library").mkdir(parents=True)
    (library / "_originals").mkdir()
    (library / ".book_rag_index").mkdir()

    (library / "library" / "好书.md").write_text("# 好书\n\n正文。\n", encoding="utf-8")
    (library / "_originals" / "原书.epub").write_bytes(b"fake")
    (library / ".book_rag_index" / "索引.epub").write_bytes(b"fake")

    discovered = [p.name for p in discover_books(library)]
    assert discovered == ["好书.md"]


# ---------------------------------------------------------------------------
# TXT → Markdown
# ---------------------------------------------------------------------------


def test_txt_chapters_become_headings(tmp_path):
    from book_rag.convert.markdown import txt_to_markdown

    path = tmp_path / "小说.txt"
    path.write_text(
        "第一章 启程\n\n清晨出发了。\n\n第二章 归途\n\n列车驶向远方。\n", encoding="utf-8"
    )

    markdown, result = txt_to_markdown(path)
    assert markdown.startswith("# 小说")
    assert "\n## 第一章 启程\n" in markdown
    assert "\n## 第二章 归途\n" in markdown
    assert result.chapters == 2
