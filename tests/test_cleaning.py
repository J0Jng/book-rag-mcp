"""front/back matter 清洗层单元测试。

全部直接构造 ``ParsedBook``，不依赖真实书、模型或网络。
"""

from book_rag.io.cleaning import (
    BACK_TITLES,
    FRONT_TITLES,
    KEEP_TITLES,
    MatterPolicy,
    _normalize_title,
    clean_matter,
)
from book_rag.io.parsers import BookSection, ParsedBook


def _make_book(titles: list[str], texts: list[str] | None = None) -> ParsedBook:
    """按标题列表构造一本顺序排列的 ParsedBook。"""
    assert texts is None or len(texts) == len(titles)
    sections = [
        BookSection(
            path=title,
            level=1,
            text=texts[i] if texts is not None else f"{title}的正文",
            position=i,
        )
        for i, title in enumerate(titles)
    ]
    return ParsedBook(title="测试之书", sections=sections)


def _chapters(count: int, start: int = 1) -> list[str]:
    """生成指定数量的章级正文标题。"""
    return [f"第{i}章 正文" for i in range(start, start + count)]


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------


def test_normalize_title_variants():
    """全角 / 大小写 / 多余空白 / 标点 / Markdown 装饰符均应正确归一化。"""
    cases = {
        "## 推荐序": "推荐序",
        "  CONTENTS  ": "contents",
        "Table   of\nContents": "table of contents",
        "**版权页**": "版权页",
        "《致谢》": "致谢",
        "版权页：": "版权页",
        "－ 致谢 －": "致谢",
        "ＲＥＦＥＲＥＮＣＥＳ": "references",
        "＃＃ 目录": "目录",
    }
    for raw, expected in cases.items():
        assert _normalize_title(raw) == expected, raw


# ---------------------------------------------------------------------------
# 剥离
# ---------------------------------------------------------------------------


def test_strips_front_matter_titles():
    """推荐序 / 封面 / 版权页 / 带副题的推荐序均被剥离。"""
    titles = ["推荐序", "封面", "版权页", "推荐序 你的人生是工业品，还是艺术品？"]
    titles += _chapters(19) + ["结语"]
    parsed = _make_book(titles)

    cleaned, report = clean_matter(parsed)

    removed = {item.path: item for item in report.removed}
    assert set(removed) == set(titles[:4])
    assert all(item.kind == "front" for item in removed.values())
    assert removed["推荐序 你的人生是工业品，还是艺术品？"].reason == "title:推荐序"
    assert "结语" in {section.path for section in cleaned.sections}
    assert len(cleaned.sections) == 20


def test_strips_back_matter_titles():
    """致谢 / 参考文献 / 索引 / 作者简介均被剥离。"""
    titles = ["封面"] + _chapters(19) + ["致谢", "参考文献", "索引", "作者简介"]
    parsed = _make_book(titles)

    cleaned, report = clean_matter(parsed)

    back_removed = {item.path for item in report.removed if item.kind == "back"}
    assert back_removed == {"致谢", "参考文献", "索引", "作者简介"}
    assert {item.path for item in report.removed if item.kind == "front"} == {"封面"}
    assert all(
        section.path not in back_removed for section in cleaned.sections
    )
    assert len(cleaned.sections) == 19


# ---------------------------------------------------------------------------
# 保留词优先
# ---------------------------------------------------------------------------


def test_keeps_meaningful_matter():
    """前言 / 序言 / 序 / 引言 / 导读 / 结语 / 后记 / 附录 / 术语表不被剥离。

    特意把它们放进 front / back 位置带内，验证 KEEP 优先级高于位置带。
    """
    titles = ["前言", "序言", "序", "引言"]
    titles += _chapters(15)
    titles += ["结语", "后记", "附录", "术语表", "导读"]
    parsed = _make_book(titles)

    cleaned, report = clean_matter(parsed)

    assert report.removed == []
    assert len(cleaned.sections) == len(titles)
    assert {section.path for section in cleaned.sections} == set(titles)


# ---------------------------------------------------------------------------
# 防误伤
# ---------------------------------------------------------------------------


def test_no_false_positive_on_word_prefix():
    """词根相连但无边界符的标题（致谢的力量等）不剥离。"""
    titles = ["版权经济学"] + _chapters(19) + ["致谢的力量", "文献综述", "索引式学习"]
    parsed = _make_book(titles)

    cleaned, report = clean_matter(parsed)

    assert report.removed == []
    assert len(cleaned.sections) == len(titles)


# ---------------------------------------------------------------------------
# 位置带
# ---------------------------------------------------------------------------


def test_position_band_rejects_mid_body():
    """20 个 section 的书里，致谢出现在 pos 10（band=3）应保留。"""
    titles = _chapters(10) + ["致谢"] + _chapters(9, start=11)
    parsed = _make_book(titles)
    assert len(titles) == 20

    cleaned, report = clean_matter(parsed)

    assert report.removed == []
    assert len(cleaned.sections) == 20


def test_short_book_band_floor():
    """3 个 section 的短书：min_band_sections 兜底，首尾都被剥离。"""
    parsed = _make_book(["推荐序", "第一章 起点", "致谢"])

    cleaned, report = clean_matter(parsed)

    assert {item.path for item in report.removed} == {"推荐序", "致谢"}
    assert report.aborted is False
    assert len(cleaned.sections) == 1
    assert cleaned.sections[0].path == "第一章 起点"


# ---------------------------------------------------------------------------
# position 重建 / 幂等 / 不变性
# ---------------------------------------------------------------------------


def test_positions_renumbered():
    """剥离后 position 严格 0..k-1 连续递增。"""
    parsed = _make_book(["封面", "推荐序"] + _chapters(20) + ["致谢"])

    cleaned, report = clean_matter(parsed)

    assert len(report.removed) == 3
    assert [section.position for section in cleaned.sections] == list(range(20))
    assert all(section.path.startswith("第") for section in cleaned.sections)


def test_fail_open_when_all_removed():
    """全书仅 2 个 section 且都命中 → aborted=True，sections 原样返回且非空。"""
    parsed = _make_book(["封面", "致谢"])

    cleaned, report = clean_matter(parsed)

    assert report.aborted is True
    assert len(report.removed) == 2  # 候选记录保留
    assert len(cleaned.sections) == 2
    assert cleaned.sections == parsed.sections
    assert [section.position for section in cleaned.sections] == [0, 1]


def test_policy_switches_off():
    """strip_front=False 时 front 类不剥，back 类照常剥离。"""
    parsed = _make_book(["推荐序", "封面"] + _chapters(20) + ["致谢", "参考文献"])

    cleaned, report = clean_matter(parsed, MatterPolicy(strip_front=False))

    assert {item.path for item in report.removed} == {"致谢", "参考文献"}
    assert all(item.kind == "back" for item in report.removed)
    kept_paths = {section.path for section in cleaned.sections}
    assert {"推荐序", "封面"} <= kept_paths


def test_idempotent():
    """对 clean_matter 的输出再跑一次：无新增剥离、section 数不变。"""
    parsed = _make_book(["封面"] + _chapters(20) + ["致谢"])

    first, first_report = clean_matter(parsed)
    second, second_report = clean_matter(first)

    assert len(first_report.removed) == 2
    assert second_report.removed == []
    assert len(second.sections) == len(first.sections)
    assert [s.position for s in second.sections] == [s.position for s in first.sections]
    assert [s.path for s in second.sections] == [s.path for s in first.sections]


def test_input_not_mutated():
    """传入 parsed.sections 的 position 不被原地修改。"""
    parsed = _make_book(["推荐序"] + _chapters(20) + ["致谢"])
    before = [(section.path, section.position) for section in parsed.sections]

    cleaned, _ = clean_matter(parsed)

    assert [(section.path, section.position) for section in parsed.sections] == before
    assert cleaned.sections is not parsed.sections
    assert cleaned is not parsed


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def test_report_fields():
    """to_dict() 字段完整，removed[i].chars 与被剥原文长度一致。"""
    body_texts = [f"第{i}章正文" for i in range(1, 21)]
    parsed = _make_book(
        ["推荐序", "封面"] + _chapters(20) + ["致谢"],
        texts=["推荐序正文" * 3, "封面文字"] + body_texts + ["感谢所有人"],
    )

    _, report = clean_matter(parsed)
    data = report.to_dict()

    assert set(data) == {"policy", "total_sections", "kept_sections", "aborted", "removed"}
    assert data["policy"] == {
        "strip_front": True,
        "strip_back": True,
        "position_ratio": 0.15,
        "min_band_sections": 3,
    }
    assert data["total_sections"] == 23
    assert data["kept_sections"] == 20
    assert data["aborted"] is False

    originals = {section.position: section for section in parsed.sections}
    assert {item["kind"] for item in data["removed"]} == {"front", "back"}
    for item in data["removed"]:
        assert set(item) == {"path", "kind", "reason", "position", "chars"}
        assert item["chars"] == len(originals[item["position"]].text)
        assert item["reason"].startswith("title:")


def test_title_tables_are_disjoint_from_keep_except_intended():
    """KEEP 与剥离表的词条不应出现「同词两义」的意外重叠。"""
    # 明确允许的重叠只有位置决定归属的封底 / 出版信息；其余不得交叉。
    assert KEEP_TITLES.isdisjoint(FRONT_TITLES | BACK_TITLES)
    assert (FRONT_TITLES & BACK_TITLES) == {"封底", "出版信息"}
