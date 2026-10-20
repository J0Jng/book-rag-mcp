"""front/back matter 标题词表与标题归一化匹配（零依赖，parsers / cleaning 共用）。

单独成模块的原因：``TextParser`` 需要把 matter 标题行识别为「章节边界」，
而 ``cleaning`` 已经 import ``parsers``——词表若留在 cleaning 会形成循环 import。
本模块不 import 任何项目内模块。
"""

from __future__ import annotations

import re
import unicodedata

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
        "版权所有",
        "图书在版编目",
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

#: 标题行长度上限：parsers 用它判断「整行能否作为章节边界」（严格，避免把
#: 正文句误切成章节）。cleaning 的正文首行弱信号另用更宽的上限（见 cleaning.py）。
HEADING_MAX_CHARS = 30

#: 仅按整题精确匹配的剥离词（``推荐`` 作为词根会误伤「推荐系统」等正文标题）。
EXACT_ONLY_TITLES = frozenset({"推荐"})

#: 长词优先，避免「索引」抢先命中「人名索引」、「contents」抢先命中
#: 「table of contents」（整题相等时也会因排序先命中更长的词）。
KEEP_SORTED = tuple(sorted(KEEP_TITLES, key=len, reverse=True))
FRONT_SORTED = tuple(sorted(FRONT_TITLES, key=len, reverse=True))
BACK_SORTED = tuple(sorted(BACK_TITLES, key=len, reverse=True))

# Markdown 装饰符（配合标题首尾空白一起剥离）
_MARKDOWN_DECORATION = "#*_~`"

# 标题首尾需要剥离的标点（NFKC 后全角标点多半已转为半角，这里两者都保留）
_EDGE_PUNCTUATION = (
    "：:、,，.。·—–-|｜《》<>「」『』()（）[]【】\"'“”‘’!！?？;；/／\\"
)

# 关键词作为前缀命中时，其后必须紧跟的分隔符（或字符串结束）
_TITLE_BOUNDARIES = frozenset(" :：｜|-—–·、（(")

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_title(raw: str) -> str:
    """归一化标题，用于关键词匹配。

    顺序：NFKC（全角→半角）→ 去 Markdown 装饰符与首尾空白 → 折叠内部
    连续空白 → ``casefold()`` → 去首尾标点与空白。
    """
    text = unicodedata.normalize("NFKC", raw)
    text = text.strip(_MARKDOWN_DECORATION + " \t\r\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = text.casefold()
    return text.strip(_EDGE_PUNCTUATION + " \t")


def match_title(normalized: str, titles: tuple[str, ...]) -> str | None:
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
        if title in EXACT_ONLY_TITLES:
            continue
        if normalized.startswith(title) and normalized[len(title)] in _TITLE_BOUNDARIES:
            return title
    return None


def match_any_title(normalized: str) -> tuple[str, str] | None:
    """按 KEEP → FRONT → BACK 顺序匹配，返回 ``(kind, keyword)``。

    ``kind`` 为 ``"keep"`` / ``"front"`` / ``"back"``；无命中返回 None。
    ``TextParser`` 用它判断某一行是否为「matter 标题行」（三者都算章节边界）。
    """
    for kind, titles in (
        ("keep", KEEP_SORTED),
        ("front", FRONT_SORTED),
        ("back", BACK_SORTED),
    ):
        hit = match_title(normalized, titles)
        if hit is not None:
            return kind, hit
    return None
