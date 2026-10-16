"""章节摘要测试：分组聚合 / schema 与指针 / 单章容错 / 空输入。

llm 全部 mock，不触发真实网络 / 真实模型。
"""

import pytest

from book_rag.io.parsers import BookSection
from book_rag.rag.llm import LLMError
from book_rag.rag.summarizer import summarize_sections

FIXED = {
    "summary": "本章核心思想",
    "frameworks": [{"name": "F", "description": "d"}],
    "terms": [{"name": "T", "definition": "def"}],
    "mental_models": ["m1"],
}


class MockLLM:
    """记录调用并返回固定摘要 dict；user 含 BAD 章文本时抛 LLMError。"""

    def __init__(self, payload=None):
        self.payload = payload if payload is not None else FIXED
        self.calls = []  # [(system, user), ...]

    def complete_json(self, system, user, *, max_tokens=2000):
        self.calls.append((system, user))
        if "BAD_MARKER" in user:
            raise LLMError("mock failure")
        return dict(self.payload)


def _sec(path: str, text: str, position: int, level: int = 1) -> BookSection:
    return BookSection(path=path, level=level, text=text, position=position)


# ---------------------------------------------------------------------------
# 分组与聚合
# ---------------------------------------------------------------------------


def test_groups_by_top_level_section_and_aggregates_text():
    llm = MockLLM()
    sections = [
        _sec("第一章 起点 > 1.1 引子", "甲段。", 0, level=2),
        _sec("第二章 展开 > 2.1 续", "乙段。", 1, level=2),
        _sec("第一章 起点 > 1.2 展开", "丙段。", 2, level=3),
    ]

    results = summarize_sections(sections, llm, title="书名")

    # 两个顶层章节（path 第一段）→ 两条摘要
    assert len(results) == 2
    assert [r["section"] for r in results] == ["第一章 起点", "第二章 展开"]
    assert llm.calls[0][1].index("甲段。") < llm.calls[0][1].index("丙段。")  # position 升序
    # 同组 text 用换行拼接（丙段前有换行）
    first_user = llm.calls[0][1]
    assert "甲段。\n丙段。" in first_user


def test_empty_path_sections_go_to_title_or_preamble():
    llm = MockLLM()
    sections = [_sec("", "序言文本。", 0)]

    results = summarize_sections(sections, llm, title="书名")

    assert len(results) == 1
    assert results[0]["section"] == "书名"


def test_empty_path_without_title_uses_preamble():
    llm = MockLLM()

    results = summarize_sections([_sec("", "序言文本。", 0)], llm)

    assert len(results) == 1
    assert results[0]["section"] == "(序言)"


def test_markdown_book_level_grouping():
    """Markdown 书名 H1 + 章节 H2：书名层剥离，各章独立成组。"""
    llm = MockLLM()
    sections = [
        _sec("测试之书", "导语文本。", 0, level=1),
        _sec("测试之书 > 第一章 起点", "甲段。", 1, level=2),
        _sec("测试之书 > 第二章 迷途", "乙段。", 2, level=2),
    ]

    results = summarize_sections(sections, llm, title="测试之书")

    assert len(results) == 3
    assert [r["section"] for r in results] == ["测试之书", "第一章 起点", "第二章 迷途"]
    first_user = llm.calls[0][1]
    assert "导语文本。" in first_user and "甲段。" not in first_user
    second_user = llm.calls[1][1]
    assert "甲段。" in second_user and "乙段。" not in second_user
    third_user = llm.calls[2][1]
    assert "乙段。" in third_user and "甲段。" not in third_user


def test_no_book_level_when_multiple_top_sections():
    """最浅层级有多个 section（无书名层）→ 不剥离，各成一组。"""
    llm = MockLLM()
    sections = [
        _sec("第一章", "甲段。", 0, level=1),
        _sec("第二章", "乙段。", 1, level=1),
    ]

    results = summarize_sections(sections, llm)

    assert len(results) == 2
    assert [r["section"] for r in results] == ["第一章", "第二章"]


# ---------------------------------------------------------------------------
# 输出 schema 与指针（pointer 由代码附加，不靠 LLM）
# ---------------------------------------------------------------------------


def test_output_schema_and_pointers_attached_by_code():
    llm = MockLLM()
    sections = [
        _sec("第一章 > 1.1", "文本一。", 3),
        _sec("第一章 > 1.2", "文本二。", 7),
    ]

    results = summarize_sections(sections, llm, title="书名")

    entry = results[0]
    assert entry["summary"] == "本章核心思想"
    assert entry["frameworks"] == [{"name": "F", "description": "d"}]
    assert entry["terms"] == [{"name": "T", "definition": "def"}]
    assert entry["mental_models"] == ["m1"]
    # LLM 只返回四个内容字段；section 与 pointers 由代码附加
    assert entry["section"] == "第一章"
    assert entry["pointers"] == [
        {"section": "第一章 > 1.1", "position": 3},
        {"section": "第一章 > 1.2", "position": 7},
    ]


def test_system_prompt_contains_anti_fabrication_constraints():
    llm = MockLLM()

    summarize_sections([_sec("第一章", "文本。", 0)], llm)

    system = llm.calls[0][0]
    assert "绝不编造" in system
    assert "[未检测到]" in system
    assert "JSON" in system


# ---------------------------------------------------------------------------
# 单章容错：重试 1 次，再失败跳过，不影响其他章
# ---------------------------------------------------------------------------


def test_failing_section_is_skipped_after_retry():
    llm = MockLLM()
    sections = [
        _sec("好章", "好章文本。", 0),
        _sec("坏章", "BAD_MARKER 文本。", 1),
        _sec("好章二", "好章二文本。", 2),
    ]

    results = summarize_sections(sections, llm)

    assert [r["section"] for r in results] == ["好章", "好章二"]
    # 坏章恰好调用 2 次（重试 1 次）
    bad_calls = [u for _, u in llm.calls if "BAD_MARKER" in u]
    assert len(bad_calls) == 2


def test_retry_recovers_from_transient_failure():
    calls = {"n": 0}

    class FlakyLLM:
        def complete_json(self, system, user, *, max_tokens=2000):
            calls["n"] += 1
            if calls["n"] == 1:
                raise LLMError("transient")
            return dict(FIXED)

    results = summarize_sections([_sec("第一章", "文本。", 0)], FlakyLLM())

    assert len(results) == 1
    assert results[0]["section"] == "第一章"
    assert calls["n"] == 2


def test_empty_sections_returns_empty_list():
    assert summarize_sections([], MockLLM()) == []
