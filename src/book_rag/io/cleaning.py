"""向量化前的 front/back matter 清洗层（格式无关、纯规则、确定性可测）。

设计要点（见 docs/plans/book-rag-matter-cleaning-pipeline-brief.md）：
- **格式无关**：只读取 ``ParsedBook.sections`` 的标题（``path`` 最末级），
  对 EPUB / PDF / TXT / Markdown 任一解析器的输出一视同仁，不依赖
  spine / nav / 文档边界等容器信号；
- **纯规则**：标题关键词（含副题）+ 位置带约束 + 保留词优先级，零 LLM、
  零 embedding、零网络调用；
- **挂点**：解析器输出之后、``split_book`` 之前，清洗后的 sections 是后续
  目录树 / 切分 / 摘要 / 概念索引的唯一事实来源；
- **fail-open**：如果剥离会清空全书，则放弃本次清洗、原样返回，绝不产出
  空书或抛错；
- **不原地修改**：输入 ``ParsedBook`` 保持不变，返回新的 ``ParsedBook``。
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass, replace

from book_rag.io.parsers import BookSection, ParsedBook

#: 保留词表：命中即永不剥离（无论位置），优先级高于 front/back 剥离表。
#: 明确保留正文性内容：前言 / 序言 / 引言 / 导读 / 结语 / 后记 / 附录 等。
KEEP_TITLES = frozenset(
    {
        "前言",
        "序言",
        "自序",
        "译者序",
        "序",
        "导读",
        "导言",
        "引言",
        "编者按",
        "楔子",
        "引子",
        "开篇",
        "结语",
        "尾声",
        "后记",
        "跋",
        "致读者",
        "结论",
        "附录",
        "术语表",
        "词汇表",
        "注释",
        "尾注",
        "脚注",
        "preface",
        "foreword",
        "introduction",
        "prologue",
        "epilogue",
        "afterword",
        "conclusion",
        "appendix",
        "appendices",
        "glossary",
        "notes",
    }
)

#: 前置部分剥离表：明确无检索价值的封面 / 版权 / 献词 / 推荐序 / 目录等。
#: 注意：``foreword / preface / introduction`` 不在此表（它们在 KEEP 中）。
FRONT_TITLES = frozenset(
    {
        "封面",
        "封底",
        "封面页",
        "封底页",
        "扉页",
        "内封",
        "书名页",
        "版权",
        "版权页",
        "版权信息",
        "出版说明",
        "出版信息",
        "献词",
        "题献",
        "献辞",
        "目录",
        "目錄",
        "目次",
        "推荐序",
        "推荐语",
        "名家推荐",
        "推荐",
        "赞誉",
        "序一",
        "序二",
        "代序",
        "cover",
        "title page",
        "half title",
        "copyright",
        "imprint",
        "dedication",
        "epigraph",
        "praise",
        "endorsements",
        "table of contents",
        "contents",
        "toc",
    }
)

#: 后置部分剥离表：明确无检索价值的致谢 / 参考文献 / 索引 / 作者简介等。
#: 注意：``结语 / 后记 / 附录`` 不在此表（它们在 KEEP 中）。
BACK_TITLES = frozenset(
    {
        "致谢",
        "鸣谢",
        "谢辞",
        "参考文献",
        "参考书目",
        "引用文献",
        "文献",
        "索引",
        "人名索引",
        "主题索引",
        "译名对照表",
        "作者简介",
        "作者介绍",
        "关于作者",
        "封底",
        "勒口",
        "出版信息",
        "acknowledgments",
        "acknowledgements",
        "references",
        "bibliography",
        "works cited",
        "index",
        "name index",
        "subject index",
        "about the author",
        "about the authors",
        "colophon",
    }
)

#: 仅按整题精确匹配的剥离词（``推荐`` 作为词根会误伤「推荐系统」等正文标题）。
_EXACT_ONLY_TITLES = frozenset({"推荐"})

#: 长词优先，避免「索引」抢先命中「人名索引」、「contents」抢先命中
#: 「table of contents」（整题相等时也会因排序先命中更长的词）。
_KEEP_SORTED = tuple(sorted(KEEP_TITLES, key=len, reverse=True))
_FRONT_SORTED = tuple(sorted(FRONT_TITLES, key=len, reverse=True))
_BACK_SORTED = tuple(sorted(BACK_TITLES, key=len, reverse=True))

# Markdown 装饰符（配合标题首尾空白一起剥离）
_MARKDOWN_DECORATION = "#*_~`"

# 标题首尾需要剥离的标点（NFKC 后全角标点多半已转为半角，这里两者都保留）
_EDGE_PUNCTUATION = "：:、,，.。·—–-|｜《》<>「」『』()（）[]【】\"'“”‘’!！?？;；/／\\"

# 关键词作为前缀命中时，其后必须紧跟的分隔符（或字符串结束）
_TITLE_BOUNDARIES = frozenset(" :：｜|-—–·、（(")

_WHITESPACE_RE = re.compile(r"\s+")


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
        reason: 命中的关键词，如 ``"title:推荐序"``。
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


def _normalize_title(raw: str) -> str:
    """归一化标题，用于关键词匹配。

    顺序：NFKC（全角→半角）→ 去 Markdown 装饰符与首尾空白 → 折叠内部
    连续空白 → ``casefold()`` → 去首尾标点与空白。
    """
    text = unicodedata.normalize("NFKC", raw)
    text = text.strip(_MARKDOWN_DECORATION + " \t\r\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = text.casefold()
    return text.strip(_EDGE_PUNCTUATION + " \t")


def _match_title(normalized: str, titles: tuple[str, ...]) -> str | None:
    """在归一化标题上匹配词表，返回命中的关键词（无命中返回 None）。

    命中规则：
    - 整题精确相等；
    - 前缀命中且关键词后紧跟边界符（空格 / 冒号 / 竖线 / 连字符 / 间隔号 /
      顿号 / 括号）或字符串结束。``"致谢的力量"``、``"版权经济学"`` 这类
      词根相连的标题因无边界符而不会命中。
    """
    if not normalized:
        return None
    for title in titles:
        if normalized == title:
            return title
        if title in _EXACT_ONLY_TITLES:
            continue
        if normalized.startswith(title) and normalized[len(title)] in _TITLE_BOUNDARIES:
            return title
    return None


def _leaf_title(section: BookSection) -> str:
    """取 section 路径的最末级标题（避免父路径关键词误伤）。"""
    return section.path.split(">")[-1].strip()


def clean_matter(
    parsed: ParsedBook, policy: MatterPolicy | None = None
) -> tuple[ParsedBook, MatterReport]:
    """剥离无检索价值的 front/back matter，返回新书对象与审计报告。

    判定顺序（即优先级）：取末级标题 → 归一化 → KEEP 优先（命中即保留）→
    front/back 词表匹配 → 位置带约束。命中剥离后重建 ``position = 0..k-1``。

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

        # KEEP 优先：保留词命中即永不剥离，无论位置
        if _match_title(normalized, _KEEP_SORTED) is not None:
            kept.append(section)
            continue

        front_hit = _match_title(normalized, _FRONT_SORTED)
        back_hit = _match_title(normalized, _BACK_SORTED)

        # 位置带：front 只看前 band 个，back 只看后 band 个。
        # 「封底 / 出版信息」同时出现在两表时，先按 front、再按 back 判定。
        if policy.strip_front and front_hit is not None and section.position < band:
            kind, hit = "front", front_hit
        elif (
            policy.strip_back
            and back_hit is not None
            and section.position >= total - band
        ):
            kind, hit = "back", back_hit
        else:
            kept.append(section)
            continue

        removed.append(
            RemovedSection(
                path=section.path,
                kind=kind,
                reason=f"title:{hit}",
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
