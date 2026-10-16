"""章节摘要：按顶层章节聚合原文，调 LLM 抽取核心思想 / 框架 / 术语 / 心智模型。

分组与指针（``section`` / ``pointers``）由代码生成，LLM 只负责内容抽取；
单章失败重试 1 次后跳过，绝不影响其他章、绝不向上抛。
"""

from __future__ import annotations

from book_rag.io.parsers import BookSection
from book_rag.rag.llm import LLMError, LLMTruncatedError

SYSTEM_PROMPT = (
    "你是书籍章节摘要抽取器。你只依据下方提供的原文提取结构，绝不编造："
    "原文没有的框架/术语/概念一律不要写，拿不准的一律填「[未检测到]」。"
    "输出必须是合法 JSON，且不包含任何多余文本。JSON schema：\n"
    "{"
    '"summary": "1-2 句：本章最核心的一个知识点",'
    '"frameworks": [{"name": "框架名", "description": "作者原表述 + 何时用 + 如何用"}],'
    '"terms": [{"name": "术语", "definition": "一句话精确定义"}],'
    '"mental_models": ["在 X 场景使用 Y 的心智模型/反模式，一句话"]'
    "}"
)


def _group_sections(
    sections: list[BookSection], *, title: str
) -> list[tuple[str, list[BookSection]]]:
    """按顶层章节名分组，保持首次出现顺序，组内按 position 升序。

    「书名层」判定：若最浅层级（min_level）只有唯一一个 section 且存在更深层级，
    则该唯一顶层 section 是书名（如 Markdown 的 H1），深层 path 的分组键取
    第二段（跳过书名层）；否则取第一段。
    path 为空的 section（导语 / 序言）归入 ``title or "(序言)"``。
    """
    if not sections:
        return []
    levels = [s.level for s in sections]
    min_level, max_level = min(levels), max(levels)
    min_count = sum(1 for s in sections if s.level == min_level)
    has_book_level = max_level > min_level and min_count == 1

    groups: dict[str, list[BookSection]] = {}
    for sec in sections:
        if sec.path:
            parts = [p for p in sec.path.split(" > ") if p]
            name = parts[1] if (has_book_level and len(parts) > 1) else parts[0]
        else:
            name = title or "(序言)"
        groups.setdefault(name, []).append(sec)
    return [
        (name, sorted(members, key=lambda s: s.position)) for name, members in groups.items()
    ]


def summarize_sections(
    sections: list[BookSection], llm, *, title: str = ""
) -> list[dict]:
    """对每个顶层章节调 LLM 生成摘要，返回 list[dict]（schema 见 brief D2）。

    每条由代码附加：
    - ``section``：顶层章节名；
    - ``pointers``：该组所有 ``{"section": s.path, "position": s.position}``（不靠 LLM）。

    单章容错：``LLMError``/``LLMTruncatedError`` 重试 1 次，再失败该章跳过，
    不写入结果、不影响其他章、不 raise。
    """
    results: list[dict] = []
    for name, members in _group_sections(sections, title=title):
        text = "\n".join(sec.text for sec in members)
        user = f"书名：{title}\n\n本章：{name}\n\n原文：\n{text}"
        try:
            data = llm.complete_json(SYSTEM_PROMPT, user)
        except (LLMError, LLMTruncatedError):
            try:
                data = llm.complete_json(SYSTEM_PROMPT, user)
            except (LLMError, LLMTruncatedError):
                continue  # 该章跳过，不中断整体
        data["section"] = name
        data["pointers"] = [
            {"section": sec.path, "position": sec.position} for sec in members
        ]
        results.append(data)
    return results
