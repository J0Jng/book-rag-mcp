"""front/back matter 清洗「题目集」——快速层（不需要模型）。

每道题是一个独立可执行的断言，题面写在 docstring 里。题目分两组：

- 第 1 组「格式无关性」：EPUB / Markdown / TXT / PDF 各自的清洗信号强弱；
- 第 2 组「判定边界」：保留优先、位置带、词根误伤、副题、fail-open、幂等、全角。

已知局限（题 3、题 4）用 ``xfail(strict=True)`` 标注：当前实现确实做不到，
一旦将来补上会变成 XPASS 从而提醒更新这份题目集。
"""

from pathlib import Path

import pytest

from book_rag.io.cleaning import clean_matter
from book_rag.io.parsers import (
    BookSection,
    EpubParser,
    MarkdownParser,
    ParsedBook,
    TextParser,
)

# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _book(paths: list[str], *, text: str | None = None) -> ParsedBook:
    """把标题列表构造成一本最小书（每个 section 正文可辨识）。"""
    return ParsedBook(
        title="题目集测试书",
        sections=[
            BookSection(
                path=path,
                level=1,
                text=text if text is not None else f"{path} 的正文内容，用于断言。",
                position=i,
            )
            for i, path in enumerate(paths)
        ],
    )


def _leaves(parsed: ParsedBook) -> list[str]:
    """取所有 section 的最末级标题（保持阅读顺序）。"""
    return [s.path.split(">")[-1].strip() for s in parsed.sections]


def _make_epub(path: Path) -> None:
    """构造含「推荐序 / 第1章 / 致谢」三份文档的最小 EPUB（三节都在目录树里）。"""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_title("题集 EPUB")
    book.set_language("zh")

    specs = [
        ("pre.xhtml", "推荐序", "他人撰写的推荐序内容。"),
        ("c1.xhtml", "第1章 起点", "清晨的雾气笼罩着山谷。"),
        ("ack.xhtml", "致谢", "感谢所有帮助过我们的人。"),
    ]
    items = []
    for file_name, title, body in specs:
        item = epub.EpubHtml(title=title, file_name=file_name, lang="zh")
        item.content = f"<html><body><h1>{title}</h1><p>{body}</p></body></html>"
        book.add_item(item)
        items.append(item)

    book.toc = [
        epub.Link(file_name, title, f"q{i}")
        for i, (file_name, title, _body) in enumerate(specs)
    ]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = items
    epub.write_epub(str(path), book)


# ---------------------------------------------------------------------------
# 第 1 组：格式无关性
# ---------------------------------------------------------------------------


def test_q1_markdown_strips_front_and_back_matter(tmp_path):
    """【题 1】Markdown：`# 推荐序` + 正文 + `# 致谢` → 只留正文。"""
    book = tmp_path / "q1.md"
    book.write_text(
        "# 推荐序\n\n他人撰写的推荐文字。\n\n"
        "# 第一章 起点\n\n清晨的雾气笼罩着山谷。\n\n"
        "# 致谢\n\n感谢所有帮助过我们的人。\n",
        encoding="utf-8",
    )

    cleaned, report = clean_matter(MarkdownParser().parse(book))

    assert _leaves(cleaned) == ["第一章 起点"]
    assert {r.path for r in report.removed} == {"推荐序", "致谢"}
    assert {r.kind for r in report.removed} == {"front", "back"}
    assert report.aborted is False


def test_q2_epub_strips_front_and_back_matter(tmp_path):
    """【题 2】EPUB：目录树里的「推荐序 / 致谢」→ 剥离，正文保留。"""
    epub_path = tmp_path / "q2.epub"
    _make_epub(epub_path)

    parsed = EpubParser().parse(epub_path)
    assert len(parsed.sections) == 3, "前置条件：解析出三个 section"

    cleaned, report = clean_matter(parsed)

    assert _leaves(cleaned) == ["第1章 起点"]
    assert {r.path for r in report.removed} == {"推荐序", "致谢"}


@pytest.mark.xfail(
    strict=True,
    reason="已知局限：TXT 无章节标题时不产生标题 section，清洗层拿不到判定信号",
)
def test_q3_txt_without_headings_cannot_be_cleaned(tmp_path):
    """【题 3】TXT（无章节标题）：期望剥离「推荐序」，当前实现做不到。

    TextParser 在进入章节前把每段独立成 section、path 一律用书名，
    因此清洗层看不到「推荐序」这个标题——这是题目集要暴露的能力缺口。
    """
    book = tmp_path / "q3.txt"
    book.write_text(
        "推荐序\n\n他人撰写的推荐文字。\n\n"
        "第一章 起点\n\n清晨的雾气笼罩着山谷。\n\n"
        "致谢\n\n感谢所有帮助过我们的人。\n",
        encoding="utf-8",
    )

    cleaned, _report = clean_matter(TextParser().parse(book))

    # 注意：TXT 的 section.path 一律是书名，"推荐序"只存在于正文里，
    # 因此必须按正文断言，否则会假通过。
    assert "推荐序" not in "\n".join(s.text for s in cleaned.sections)


@pytest.mark.xfail(
    strict=True,
    reason="已知局限：PDF 只有页级 path（'第 N 页'），清洗层拿不到标题信号",
)
def test_q4_pdf_page_level_sections_cannot_be_cleaned():
    """【题 4】PDF（页级 path）：期望剥离版权页，当前实现做不到。

    模拟 PdfParser 的输出形态：每个 section 的 path 是「第 N 页」，
    标题信息在正文里而不在 path 上，标题词表匹配不到任何东西。
    """
    parsed = ParsedBook(
        title="题目集 PDF",
        sections=[
            BookSection(path="第 1 页", level=1, text="版权所有，侵权必究。", position=0),
            BookSection(path="第 2 页", level=1, text="清晨的雾气笼罩着山谷。", position=1),
            BookSection(path="第 3 页", level=1, text="感谢所有帮助过我们的人。", position=2),
        ],
    )

    cleaned, _report = clean_matter(parsed)

    assert len(cleaned.sections) == 1


# ---------------------------------------------------------------------------
# 第 2 组：判定边界
# ---------------------------------------------------------------------------


def test_q5_keep_list_wins_over_strip_list():
    """【题 5】「推荐序 / 前言 / 正文 / 结语 / 致谢」→ 只剥推荐序与致谢。

    前言、结语属于 KEEP 词表，即使位于首尾位置也不得剥离。
    """
    cleaned, report = clean_matter(
        _book(["推荐序", "前言", "第一章 起点", "结语", "致谢"])
    )

    assert _leaves(cleaned) == ["前言", "第一章 起点", "结语"]
    assert {r.path for r in report.removed} == {"推荐序", "致谢"}


def test_q6_position_band_protects_mid_book_thanks():
    """【题 6】20 节的书中，「致谢」出现在第 10 节（正文中段）→ 不剥离。

    位置带只覆盖前/后 3 节（max(3, ceil(20*0.15))），中段同名词条按正文保留。
    """
    paths = [f"第{i}章 正文" for i in range(20)]
    paths[10] = "致谢"

    cleaned, report = clean_matter(_book(paths))

    assert report.removed == []
    assert len(cleaned.sections) == 20
    assert _leaves(cleaned)[10] == "致谢"


def test_q7_word_prefix_is_not_a_match():
    """【题 7】词根相连的正文标题不得误伤：致谢的力量 / 索引式学习 / 版权经济学。

    「推荐系统实践」也不能被 `推荐` 误命中（`推荐` 只做整题匹配）。
    """
    titles = ["致谢的力量", "索引式学习", "版权经济学", "推荐系统实践", "文献综述"]

    cleaned, report = clean_matter(_book(titles))

    assert report.removed == []
    assert _leaves(cleaned) == titles


def test_q8_subtitle_after_keyword_is_stripped():
    """【题 8】「推荐序 你的人生是工业品，还是艺术品？」→ 按副题规则剥离。"""
    cleaned, report = clean_matter(
        _book(["推荐序 你的人生是工业品，还是艺术品？", "第一章 起点"])
    )

    assert _leaves(cleaned) == ["第一章 起点"]
    assert report.removed[0].reason == "title:推荐序"
    assert report.removed[0].chars > 0


def test_q9_fail_open_never_empties_the_book():
    """【题 9】全书只有 2 节且都被命中 → 放弃清洗，原样返回（aborted=True）。"""
    parsed = _book(["封面", "致谢"])

    cleaned, report = clean_matter(parsed)

    assert report.aborted is True
    assert report.removed, "候选记录应保留，供审计"
    assert len(cleaned.sections) == 2
    assert _leaves(cleaned) == ["封面", "致谢"]


def test_q10_cleaning_is_idempotent():
    """【题 10】对清洗结果再跑一次 → 不再剥离任何内容（幂等）。"""
    once, first = clean_matter(
        _book(["推荐序", "第一章 起点", "第二章 归途", "致谢"])
    )
    twice, second = clean_matter(once)

    assert {r.path for r in first.removed} == {"推荐序", "致谢"}
    assert second.removed == []
    assert len(twice.sections) == len(once.sections)


def test_q11_fullwidth_and_uppercase_titles_are_normalized():
    """【题 11】全角 + 大写的 `ＡＣＫＮＯＷＬＥＤＧＥＭＥＮＴＳ` / `ＣＯＮＴＥＮＴＳ` → 正常剥离。"""
    cleaned, report = clean_matter(
        _book(["ＣＯＮＴＥＮＴＳ", "第一章 起点", "ＡＣＫＮＯＷＬＥＤＧＥＭＥＮＴＳ"])
    )

    assert _leaves(cleaned) == ["第一章 起点"]
    assert len(report.removed) == 2


def test_q12_short_book_position_band_floor():
    """【题 12】3 节的短书：位置带下限 3 生效，首尾 matter 都能剥。"""
    cleaned, report = clean_matter(_book(["推荐序", "第一章 起点", "致谢"]))

    assert _leaves(cleaned) == ["第一章 起点"]
    assert {r.path for r in report.removed} == {"推荐序", "致谢"}
