"""文档解析器测试。

TXT / Markdown 使用 tests/fixtures 下的最小 fixture；
EPUB 在 tmp_path 中程序化生成最小文件，避免向仓库提交二进制文件。

PDF 明确不支持（需要 OCR / 版面还原），工厂函数应拒绝 ``.pdf``。
"""

from pathlib import Path

import pytest

from book_rag.io.parsers import (
    EpubParser,
    MarkdownParser,
    TextParser,
    get_parser,
)

FIXTURES = Path(__file__).parent / "fixtures"


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


def _make_epub_with_toc_tree(path: Path) -> None:
    """构造一个含三级目录（nav/ncx）+ 目录页文档的最小 EPUB。

    目录树：「第一章 启程」→「1.1 出发」「1.2 露营」；「第二章 归途」。
    另有一个名为 catalog.xhtml 的目录页文档（不在文件名黑名单内），
    正文文档对应上述目录节点；第二章正文附带两个指向其他章节的链接，
    用于验证 _is_toc_page 不误删含少量链接的正文。
    """
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_title("多级目录测试")
    book.set_language("zh")
    book.add_author("测试作者")

    c1 = epub.EpubHtml(title="第一章 启程", file_name="chap_01.xhtml", lang="zh")
    c1.content = "<html><body><h1>第一章 启程</h1><p>清晨整装待发。</p></body></html>"
    c11 = epub.EpubHtml(title="1.1 出发", file_name="chap_01_01.xhtml", lang="zh")
    c11.content = "<html><body><h2>1.1 出发</h2><p>山路蜿蜒向上。</p></body></html>"
    c12 = epub.EpubHtml(title="1.2 露营", file_name="chap_01_02.xhtml", lang="zh")
    c12.content = "<html><body><h2>1.2 露营</h2><p>夜幕下搭起帐篷。</p></body></html>"
    c2 = epub.EpubHtml(title="第二章 归途", file_name="chap_02.xhtml", lang="zh")
    c2.content = (
        "<html><body><h1>第二章 归途</h1>"
        "<p>归途的列车驶向远方。参见<a href='chap_01.xhtml'>第一章</a>"
        "与<a href='chap_01_01.xhtml'>1.1 出发</a>。</p></body></html>"
    )
    catalog = epub.EpubHtml(title="目录", file_name="catalog.xhtml", lang="zh")
    catalog.content = (
        "<html><body><h1>目录</h1><ul>"
        "<li><a href='chap_01.xhtml'>第一章 启程</a></li>"
        "<li><a href='chap_01_01.xhtml'>1.1 出发</a></li>"
        "<li><a href='chap_01_02.xhtml'>1.2 露营</a></li>"
        "<li><a href='chap_02.xhtml'>第二章 归途</a></li>"
        "</ul></body></html>"
    )
    for item in (c1, c11, c12, c2, catalog):
        book.add_item(item)

    book.toc = [
        (
            epub.Section("第一章 启程", href="chap_01.xhtml"),
            [
                epub.Link("chap_01_01.xhtml", "1.1 出发", "c11"),
                epub.Link("chap_01_02.xhtml", "1.2 露营", "c12"),
            ],
        ),
        epub.Link("chap_02.xhtml", "第二章 归途", "c2"),
    ]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", catalog, c1, c11, c12, c2]
    epub.write_epub(str(path), book)


def test_epub_parser_toc_tree_and_toc_page_strip(tmp_path):
    epub_path = tmp_path / "toc_book.epub"
    _make_epub_with_toc_tree(epub_path)

    parsed = EpubParser().parse(epub_path)
    assert parsed.title == "多级目录测试"

    sections = parsed.sections
    # nav 与目录页文档（catalog.xhtml）均被剥离，只剩 4 个正文文档
    assert len(sections) == 4
    # 目录页残留消失：没有任何 section 以目录页的「目录」标题开头
    assert all(not s.text.strip().startswith("目录") for s in sections)

    # 目录树映射出多级路径与真实层级
    by_path = {s.path: s for s in sections}
    assert "第一章 启程 > 1.1 出发" in by_path
    assert by_path["第一章 启程 > 1.1 出发"].level == 2
    assert "第一章 启程 > 1.2 露营" in by_path
    assert by_path["第一章 启程 > 1.2 露营"].level == 2
    assert "第一章 启程" in by_path
    assert by_path["第一章 启程"].level == 1
    assert "第二章 归途" in by_path
    assert by_path["第二章 归途"].level == 1

    # position 从 0 起严格递增；text 均非空
    assert [s.position for s in sections] == list(range(len(sections)))
    assert all(s.text.strip() for s in sections)


def _make_epub_with_contents_header(path: Path) -> None:
    """构造正文文档开头带 Contents 排版页眉残留的最小 EPUB。

    第一章文档正文以一行孤立的 ``Contents`` 开头（页眉残留，位于章节
    标题之前）；第二章文档正文中间出现 ``Contents`` 字样（合法内容），
    用于验证剥离只作用于文档开头、不误伤正文。
    """
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_title("页眉残留测试")
    book.set_language("zh")
    book.add_author("测试作者")

    # title 留空：ebooklib 会把 EpubHtml 的 title 写进 <head><title>，
    # 其内容会混入 get_text 的可见文本，干扰「文档开头即 Contents」的形态
    c1 = epub.EpubHtml(title="", file_name="chap_01.xhtml", lang="zh")
    c1.content = (
        "<html><body>"
        "<p>Contents</p>"
        "<h1>第1章 分析你的生活</h1>"
        "<p>在斯坦福大学的设计工作室有这样一个标牌。</p>"
        "</body></html>"
    )
    c2 = epub.EpubHtml(title="", file_name="chap_02.xhtml", lang="zh")
    c2.content = (
        "<html><body><h1>第2章 归途</h1>"
        "<p>他翻开书的目录（Contents）页查看后续章节。</p>"
        "</body></html>"
    )
    for item in (c1, c2):
        book.add_item(item)
    book.toc = [
        epub.Link("chap_01.xhtml", "第1章 分析你的生活", "c1"),
        epub.Link("chap_02.xhtml", "第2章 归途", "c2"),
    ]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = [c1, c2]
    epub.write_epub(str(path), book)


def test_epub_parser_strips_leading_contents_header(tmp_path):
    """正文文档开头的孤立 Contents 页眉行应被剥离，正文完整保留。"""
    epub_path = tmp_path / "header_book.epub"
    _make_epub_with_contents_header(epub_path)

    parsed = EpubParser().parse(epub_path)
    assert parsed.title == "页眉残留测试"

    by_path = {s.path: s for s in parsed.sections}
    assert "第1章 分析你的生活" in by_path
    c1_text = by_path["第1章 分析你的生活"].text
    # 页眉残留被剥离：section 文本不再以 Contents 开头
    assert not c1_text.strip().startswith("Contents")
    # 正文完整保留，且从章节标题之后的正文开始
    assert c1_text.startswith("在斯坦福大学")
    assert "在斯坦福大学的设计工作室有这样一个标牌。" in c1_text

    # 正文中间的 Contents 字样不被误剥离
    assert "第2章 归途" in by_path
    assert "Contents" in by_path["第2章 归途"].text


def test_strip_contents_prefix_variants():
    """_strip_contents_prefix 只剥离开头一行孤立的目录页眉，各变体均覆盖。"""
    from book_rag.io.parsers import _strip_contents_prefix

    assert _strip_contents_prefix("Contents\n第1章 正文") == "第1章 正文"
    assert _strip_contents_prefix("  Contents \n\n第1章 正文") == "第1章 正文"
    assert _strip_contents_prefix("目录\n第1章 正文") == "第1章 正文"
    assert _strip_contents_prefix("目錄\n第1章 正文") == "第1章 正文"
    assert _strip_contents_prefix("Table of Contents\nChapter 1") == "Chapter 1"
    assert _strip_contents_prefix("table of contents\nChapter 1") == "Chapter 1"
    # 整篇只剩页眉本身：剥离后为空（由 parse 的空文本分支跳过）
    assert _strip_contents_prefix("Contents") == ""
    # 正文中（非开头）出现的 Contents 原样保留
    assert (
        _strip_contents_prefix("第1章 正文\nContents 出现在段中")
        == "第1章 正文\nContents 出现在段中"
    )
    # 首行不是孤立的页眉（同一行还跟了其他内容）时不剥离
    assert (
        _strip_contents_prefix("Contents 一词出现在句首")
        == "Contents 一词出现在句首"
    )
    assert _strip_contents_prefix("第1章 正文") == "第1章 正文"


def _make_epub_with_flat_ncx(path: Path) -> None:
    """构造一个 EPUB2 形态（仅 ncx、无 nav）且目录全平铺的最小 EPUB。

    目录是 5 个顶层 leaf、零嵌套（模拟大量中文 EPUB2 书：ebooklib 解析后
    没有任何 (Section, children) 元组）：「引言」「第1章 启程」「出发」「露营」
    「第2章 归途」。层级只能靠章标题模式推断。
    """
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_title("平铺目录测试")
    book.set_language("zh")
    book.add_author("测试作者")

    intro = epub.EpubHtml(title="引言", file_name="intro.xhtml", lang="zh")
    intro.content = "<html><body><h1>引言</h1><p>这是一段引言。</p></body></html>"
    c1 = epub.EpubHtml(title="第1章 启程", file_name="chap_01.xhtml", lang="zh")
    c1.content = "<html><body><h1>第1章 启程</h1><p>清晨整装待发。</p></body></html>"
    c11 = epub.EpubHtml(title="出发", file_name="chap_01_01.xhtml", lang="zh")
    c11.content = "<html><body><h2>出发</h2><p>山路蜿蜒向上。</p></body></html>"
    c12 = epub.EpubHtml(title="露营", file_name="chap_01_02.xhtml", lang="zh")
    c12.content = "<html><body><h2>露营</h2><p>夜幕下搭起帐篷。</p></body></html>"
    c2 = epub.EpubHtml(title="第2章 归途", file_name="chap_02.xhtml", lang="zh")
    c2.content = "<html><body><h1>第2章 归途</h1><p>归途的列车驶向远方。</p></body></html>"
    for item in (intro, c1, c11, c12, c2):
        book.add_item(item)

    book.toc = [
        epub.Link("intro.xhtml", "引言", "intro"),
        epub.Link("chap_01.xhtml", "第1章 启程", "c1"),
        epub.Link("chap_01_01.xhtml", "出发", "c11"),
        epub.Link("chap_01_02.xhtml", "露营", "c12"),
        epub.Link("chap_02.xhtml", "第2章 归途", "c2"),
    ]
    book.add_item(epub.EpubNcx())
    book.spine = [intro, c1, c11, c12, c2]
    epub.write_epub(str(path), book)


def test_epub_parser_flat_ncx_infers_levels_from_titles(tmp_path):
    """全平铺 toc + 章标题模式：应按标题推断出「章 > 节」真树形层级。"""
    epub_path = tmp_path / "flat_ncx_book.epub"
    _make_epub_with_flat_ncx(epub_path)

    parsed = EpubParser().parse(epub_path)
    assert parsed.title == "平铺目录测试"

    sections = parsed.sections
    # ncx 不在 spine，5 个正文文档全部保留
    assert len(sections) == 5

    by_path = {s.path: s for s in sections}
    # 小节挂到当前章下：level 2、多级路径
    assert "第1章 启程 > 出发" in by_path
    assert by_path["第1章 启程 > 出发"].level == 2
    assert "第1章 启程 > 露营" in by_path
    assert by_path["第1章 启程 > 露营"].level == 2
    # 章级节点保持 level 1
    assert by_path["第1章 启程"].level == 1
    assert by_path["第2章 归途"].level == 1
    # 前置部分（引言）是独立 level 1 节点，不挂到任何章下
    assert by_path["引言"].level == 1

    # position 从 0 起严格递增；text 均非空
    assert [s.position for s in sections] == list(range(len(sections)))
    assert all(s.text.strip() for s in sections)


def _make_epub_with_dedication_page(path: Path) -> None:
    """构造含「目录树之外的匿名献词页」的最小 EPUB。

    chapter1.xhtml 是献词页：不在 ``book.toc``、无 heading、无 item.title
    （title 留空，避免 ebooklib 写入 <head><title> 混入可见文本），正文开头
    为献词特征句，模仿真实书的 front-matter 形态；chap_01.xhtml 是正常章节
    （在 ``book.toc``、有 h1 标题与正文）。spine 顺序为献词页在前。
    """
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_title("献词页测试")
    book.set_language("zh")
    book.add_author("测试作者")

    dedicate = epub.EpubHtml(title="", file_name="chapter1.xhtml", lang="zh")
    dedicate.content = (
        "<html><body>"
        "<p>谨以此书献给所有与我们分享人生故事的优秀学员。</p>"
        "</body></html>"
    )
    chapter = epub.EpubHtml(title="第1章 启程", file_name="chap_01.xhtml", lang="zh")
    chapter.content = (
        "<html><body><h1>第1章 启程</h1>"
        "<p>清晨整装待发。</p></body></html>"
    )
    for item in (dedicate, chapter):
        book.add_item(item)

    book.toc = [epub.Link("chap_01.xhtml", "第1章 启程", "c1")]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = [dedicate, chapter]
    epub.write_epub(str(path), book)


def test_epub_parser_strips_dedication_page(tmp_path):
    """目录树之外的匿名献词页应被整文档剥离，剩余 section position 从 0 重排。"""
    epub_path = tmp_path / "dedication_book.epub"
    _make_epub_with_dedication_page(epub_path)

    parsed = EpubParser().parse(epub_path)
    assert parsed.title == "献词页测试"

    sections = parsed.sections
    # 献词页被剥离，只剩正常章节
    assert len(sections) == 1
    assert sections[0].path == "第1章 启程"
    # 剥离后 position 从 0 重排
    assert sections[0].position == 0
    assert "清晨整装待发" in sections[0].text


def test_epub_parser_keeps_unmapped_body(tmp_path):
    """防误伤：无目录映射但内容为正常非献词正文的文档必须保留。"""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_title("防误伤测试")
    book.set_language("zh")
    book.add_author("测试作者")

    extra = epub.EpubHtml(title="", file_name="extra.xhtml", lang="zh")
    extra.content = (
        "<html><body>"
        "<p>这是一段没有目录映射的普通正文，记录书稿排版时遗漏的补充材料，"
        "内容平实，供读者参考。</p>"
        "</body></html>"
    )
    book.add_item(extra)
    book.toc = []
    book.add_item(epub.EpubNcx())
    book.spine = [extra]
    out = tmp_path / "unmapped_body.epub"
    epub.write_epub(str(out), book)

    parsed = EpubParser().parse(out)
    # 不被 mapped is None 误剥：唯一的文档保留，path 来自回退命名（文件名主干）
    assert len(parsed.sections) == 1
    assert parsed.sections[0].path == "extra"
    assert "补充材料" in parsed.sections[0].text


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def test_get_parser_factory():
    assert isinstance(get_parser(".md"), MarkdownParser)
    assert isinstance(get_parser(".markdown"), MarkdownParser)
    assert isinstance(get_parser(".txt"), TextParser)
    assert isinstance(get_parser(".TXT"), TextParser)
    assert isinstance(get_parser(".epub"), EpubParser)
    with pytest.raises(ValueError):
        get_parser(".docx")
    # PDF 明确不支持：需要 OCR / 版面还原，见 docs/ROADMAP.md
    with pytest.raises(ValueError):
        get_parser(".pdf")
