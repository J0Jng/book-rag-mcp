"""向量化前的 front/back matter 清洗层（格式无关、纯规则、确定性可测）。

设计要点（见 docs/plans/book-rag-matter-cleaning-pipeline-brief.md）：
- **格式无关**：只读取 ``ParsedBook.sections`` 的标题（``path`` 最末级），
  对 EPUB / TXT / Markdown 任一解析器的输出一视同仁，不依赖
  spine / nav / 文档边界等容器信号；
- **纯规则**：标题关键词（含副题）+ 位置带约束 + 保留词优先级，零 LLM、
  零 embedding、零网络调用；
- **弱信号回退**：标题本身没有信息（书名 / 空 / 历史索引遗留的页码型 path）的
  section（典型是 TXT 无章节标题时的段落），改用「正文首行 + 短文本」判定，
  仍受位置带约束——这是 TXT 也能被清洗的关键；
- **挂点**：解析器输出之后、``split_book`` 之前，清洗后的 sections 是后续
  目录树 / 切分 / 摘要 / 概念索引的唯一事实来源；
- **fail-open**：如果剥离会清空全书，则放弃本次清洗、原样返回，绝不产出
  空书或抛错；
- **不原地修改**：输入 ``ParsedBook`` 保持不变，返回新的 ``ParsedBook``。
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, replace

from book_rag.io.matter_titles import (
    BACK_TITLES,
    FRONT_TITLES,
    KEEP_TITLES,
)
from book_rag.io.matter_titles import (
    match_title as _match_title,
)
from book_rag.io.matter_titles import (
    normalize_title as _normalize_title,
)
from book_rag.io.parsers import BookSection, ParsedBook

__all__ = [
    "BACK_TITLES",
    "FRONT_TITLES",
    "KEEP_TITLES",
    "MatterPolicy",
    "MatterReport",
    "RemovedSection",
    "clean_matter",
]

# 长词优先（与词表模块同一排序语义）
_KEEP_SORTED = tuple(sorted(KEEP_TITLES, key=len, reverse=True))
_FRONT_SORTED = tuple(sorted(FRONT_TITLES, key=len, reverse=True))
_BACK_SORTED = tuple(sorted(BACK_TITLES, key=len, reverse=True))

#: 页码型 section 标题（历史 PDF 索引遗留的 ``"第 3 页"`` path，保留兼容），
#: 不含标题信息
_PAGE_PATH_RE = re.compile(r"^第\s*\d+\s*页$")

#: 弱信号回退的两道上限：整篇过长说明是正文页，首行过长说明不是标题行。
#: 首行上限取 60：英文版权行（如 "Copyright 2020 All rights reserved."）比
#: 中文标题长，30 会把它们漏掉。
_CONTENT_FALLBACK_MAX_CHARS = 2000
_CONTENT_FIRST_LINE_MAX_CHARS = 60

#: 首行切分（空白 + 中英文标点），用于取出「首个短语」做弱匹配
_FIRST_PHRASE_RE = re.compile(r"[\s，,。.：:、；;！!？?（）()\[\]【】]+")


@dataclass(frozen=True)
class MatterPolicy:
    """清洗策略（不可变，便于审计与复现）。

    Attributes:
        strip_front: 是否剥离前置部分（默认 True）。
        strip_back: 是否剥离后置部分（默认 True）。
        position_ratio: front 只看前 N%、back 只看后 N%（默认 0.15）。
        min_band_sections: 位置带下限，短书也至少覆盖前/后 N 个 section
            （默认 3），防止小书的位置带过窄。
    """

    strip_front: bool = True
    strip_back: bool = True
    position_ratio: float = 0.15
    min_band_sections: int = 3


@dataclass(frozen=True)
class RemovedSection:
    """一条被剥离 section 的审计记录。

    Attributes:
        path: 原始章节路径（``"章 > 节"`` 完整链）。
        kind: ``"front"`` 或 ``"back"``。
        reason: 命中来源与关键词，如 ``"title:推荐序"`` / ``"content:版权"``。
        position: 剥离前的原始 position。
        chars: 被剥离正文的字符数。
    """

    path: str
    kind: str
    reason: str
    position: int
    chars: int


@dataclass
class MatterReport:
    """一次清洗的审计报告，可序列化为 ``cleaning_report.json``。

    Attributes:
        removed: 被剥离 section 的记录列表。
        kept_sections: 清洗后保留的 section 数。
        total_sections: 清洗前的 section 总数。
        aborted: True 表示命中会清空全书，已放弃清洗（fail-open）。
        policy: 本次使用的策略。
    """

    removed: list[RemovedSection]
    kept_sections: int
    total_sections: int
    aborted: bool
    policy: MatterPolicy

    def to_dict(self) -> dict:
        """序列化为可写入 JSON 的字典。"""
        return {
            "policy": asdict(self.policy),
            "total_sections": self.total_sections,
            "kept_sections": self.kept_sections,
            "aborted": self.aborted,
            "removed": [asdict(item) for item in self.removed],
        }


def _leaf_title(section: BookSection) -> str:
    """取 section 路径的最末级标题（避免父路径关键词误伤）。"""
    return section.path.split(">")[-1].strip()


def _is_uninformative_leaf(leaf: str, book_title: str) -> bool:
    """判断 leaf 标题是否「不含标题信息」：空 / 页码 / 书名本身。

    这类 section 无法用标题词表判定（TXT 无章节标题时的段落、历史页码型 path），
    需要走正文首行弱信号回退。
    """
    text = leaf.strip()
    if not text:
        return True
    if _PAGE_PATH_RE.match(text):
        return True
    return bool(book_title) and text == book_title.strip()


def _content_heading(section: BookSection) -> tuple[str, str] | None:
    """弱信号：用正文首行判定无标题 section 的性质。

    判据（三重保守）：
    1. 整段不超过 :data:`_CONTENT_FALLBACK_MAX_CHARS`（matter 页通常很短）；
    2. 首行不超过 :data:`_CONTENT_FIRST_LINE_MAX_CHARS`（是标题行而非正文句）；
    3. 首行归一化后「以关键词开头」——弱信号不要求边界符，靠 1 + 2 + 位置带兜底。

    返回 ``(kind, keyword)``，其中 kind 为 ``"keep"`` / ``"front"`` / ``"back"``。
    """
    text = section.text.strip()
    if not text or len(text) > _CONTENT_FALLBACK_MAX_CHARS:
        return None

    first_line = text.split("\n", 1)[0].strip()
    if not first_line or len(first_line) > _CONTENT_FIRST_LINE_MAX_CHARS:
        return None

    # 候选：整行 + 首个短语（"Copyright 2020 All rights reserved." 取 Copyright；
    # "版权所有，侵权必究" 两者都能命中）
    candidates = [_normalize_title(first_line)]
    phrases = [p for p in _FIRST_PHRASE_RE.split(first_line) if p]
    if phrases:
        candidates.append(_normalize_title(phrases[0]))
    candidates = [c for c in candidates if c]
    if not candidates:
        return None

    for kind, titles in (
        ("keep", _KEEP_SORTED),
        ("front", _FRONT_SORTED),
        ("back", _BACK_SORTED),
    ):
        for title in titles:
            for candidate in candidates:
                if candidate.startswith(title):
                    return kind, title
    return None


def clean_matter(
    parsed: ParsedBook, policy: MatterPolicy | None = None
) -> tuple[ParsedBook, MatterReport]:
    """剥离无检索价值的 front/back matter，返回新书对象与审计报告。

    判定顺序（即优先级）：取末级标题 → 归一化 → KEEP 优先（命中即保留）→
    front/back 词表匹配 → 标题无信息时用正文首行弱信号回退 → 位置带约束。
    命中剥离后重建 ``position = 0..k-1``。

    若剥离会清空全书，则放弃清洗（``report.aborted=True``），原样返回输入
    的 ``ParsedBook`` 与候选记录。输入对象永不被原地修改。

    Args:
        parsed: 解析器输出的书籍（``sections`` 按阅读顺序排列）。
        policy: 清洗策略；缺省使用默认 :class:`MatterPolicy`。

    Returns:
        ``(新 ParsedBook, MatterReport)``。
    """
    policy = policy or MatterPolicy()
    sections = parsed.sections
    total = len(sections)
    band = max(
        policy.min_band_sections,
        math.ceil(total * policy.position_ratio),
    )

    removed: list[RemovedSection] = []
    kept: list[BookSection] = []
    for section in sections:
        leaf = _leaf_title(section)
        normalized = _normalize_title(leaf)
        back_hit = _match_title(normalized, _BACK_SORTED)

        # 1) KEEP 优先：保留词命中即永不剥离，无论位置
        if _match_title(normalized, _KEEP_SORTED) is not None:
            kept.append(section)
            continue

        # 2) 强信号：末级标题命中剥离词表
        hit: tuple[str, str] | None = None
        source = "title"
        front_hit = _match_title(normalized, _FRONT_SORTED)
        if front_hit is not None:
            hit = ("front", front_hit)
        elif back_hit is not None:
            hit = ("back", back_hit)
        elif _is_uninformative_leaf(leaf, parsed.title):
            # 3) 弱信号：标题无信息时看正文首行（TXT 无标题段落 / 历史页码型 path）
            hit = _content_heading(section)
            if hit is not None:
                source = "content"

        if hit is None or hit[0] == "keep":
            kept.append(section)
            continue

        kind, keyword = hit

        # 4) 位置带：front 只看前 band 个，back 只看后 band 个
        in_front_band = section.position < band
        in_back_band = section.position >= total - band
        should_strip = (kind == "front" and policy.strip_front and in_front_band) or (
            kind == "back" and policy.strip_back and in_back_band
        )
        if (
            not should_strip
            and kind == "front"
            and source == "title"
            and back_hit is not None
            and policy.strip_back
            and in_back_band
        ):
            # 「封底 / 出版信息」同时出现在 front/back 两表：按实际位置归 back
            kind = "back"
            should_strip = True
        if not should_strip:
            kept.append(section)
            continue

        removed.append(
            RemovedSection(
                path=section.path,
                kind=kind,
                reason=f"{source}:{keyword}",
                position=section.position,
                chars=len(section.text),
            )
        )

    # fail-open：剥离会清空全书时放弃清洗，原样返回，绝不产出空索引
    if sections and not kept:
        report = MatterReport(
            removed=removed,
            kept_sections=total,
            total_sections=total,
            aborted=True,
            policy=policy,
        )
        return parsed, report

    new_sections = [replace(section, position=i) for i, section in enumerate(kept)]
    report = MatterReport(
        removed=removed,
        kept_sections=len(new_sections),
        total_sections=total,
        aborted=False,
        policy=policy,
    )
    return replace(parsed, sections=new_sections), report
