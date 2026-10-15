# Implementation Brief: 剥离 EPUB 文档开头的 Contents 页眉残留（P0.2）

## Background

P0 / P0.1 已使目录树成真树形，但 `get_section_content` 章节原文首段仍是 `Contents`。侦察定位：该书 EPUB 的**每一个正文文档**（`chapter*.xhtml`）正文开头都有一行 `Contents` 排版页眉残留，紧接章节标题，形如：

```
Contents
第1章 分析测评你目前的生活
在斯坦福大学的设计工作室有这样一个标牌……
```

`EpubParser` 移除 heading 后，这行 `Contents` 残留在 `text` 开头，随后被切分、embedding、污染每个 section 的第一段。消除它是 P0「干净正文 + 真树形地图」闭环的最后一块。

## Design Decisions

1. **只剥离「文档可见文本最开头」的孤立 Contents 行**，不触碰正文中合法出现的 `Contents`（[章节标题之后] 的正文里极少出现，且只按文档开头精准匹配）。
2. 匹配一行开头的变体：`Contents` / `目录` / `目錄` / `Table of Contents`（忽略大小写），允许前后空白；后面紧跟换行或章节标题时视为页眉。
3. 在 `get_text` 之后、`text.strip()` 之后做一次前缀剥离（一行级），不影响 splitter / 结构层。
4. 纯结构化、零依赖、无 LLM。

## Change List

### 1. `src/book_rag/io/parsers.py` — `EpubParser`

- 新增模块级 `_CONTENTS_PREFIX_RE` 与辅助 `_strip_contents_prefix(text: str) -> str`：仅当文本**开头（strip 后）即匹配** Contents/目录/目錄/Table of Contents（可带空白/换行）时，剥离该行；否则原样返回。
- 在 `EpubParser.parse` 的 `text = soup.get_text("\n").strip()` 之后调用它，再判断 `if not text: continue`。

### 2. `tests/test_parsers.py`

- 现有 EPUB fixture 的正文文档开头加一行 `Contents`（或新增断言），验证解析出的 section 文本**不以 `Contents` 开头**、且正文仍完整保留；补充一个「正文中间含 `Contents` 字样不被误剥离」的反例断言。

## Acceptance Criteria（可运行）

```bash
export PYTHONPATH=
.venv/Scripts/python.exe -m pytest tests/ -q     # 全绿
```

重建《人生设计课》索引后：

- `get_section_content("人生设计课", "第1章 …")` 首段不再是 `Contents`，而是正文字段。
- 章节定位 / 结构层行为不回归。

## Explicitly Do NOT Do

- ❌ 不改 `PdfParser` / `TextParser` / `MarkdownParser`。
- ❌ 不新增依赖、不改 `pyproject.toml` / `uv.lock`、不用 LLM。
- ❌ 不剥离正文中（非文档开头）合法出现的 `Contents`。
- ❌ 不改 `structure.py` / `retriever.py` / splitter。