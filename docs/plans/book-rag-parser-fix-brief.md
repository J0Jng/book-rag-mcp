# Implementation Brief: 修复 EPUB 解析器（P0 — 真树形 TOC + 剥离目录页残留）

## Background

结构层 Ticket 1 已上线，但真实冒烟暴露两个上游 parser 缺陷（根因在 `EpubParser`，不在结构层）：

1. **TOC 平铺非树形**：解析《斯坦福人生设计课》得到 60 个 section，`toc.json` 里 61 个节点 **全部 `level=1`**，章节与小节挤在同一平面（「第1章 分析测评你目前的生活」「辨识无法解决的重力问题」「重力问题」…全平铺），没有「第X章 > 小节」的树形层级。
2. **目录页残留**：`position=0` 是名为「chapter1」的 section，内容纯为「Contents」+ 空行；「推荐序」正文前 4 行也是「Contents | 空 | 空 | 空」。EPUB 目录页被当正文解析、切分、embedding，污染向量库与 `get_section_content` 开头。

**根因定位**（`src/book_rag/io/parsers.py`）：
- `L209` `_EPUB_SKIP_NAMES` 只按文件名跳过 nav/ncx/toc/titlepage/cover/copyright；该书目录页文档文件名是 `chapter1`，漏网。
- `L249` 每个 spine 文档只取**第一个** `h1-h6` 作 section 名；`L258` `level` **恒写死 `1`**；完全没利用 EPUB 自带 `nav`/`ncx` 里的目录层级。

## Design Decisions

1. **层级来自 EPUB 目录（nav/ncx），不是文档内 heading**：从 `nav`（EPUB3）或 `ncx`（EPUB2）读取目录树（章→节→子节 + href 指向内容文件），把每个 spine 内容文档映射到正确的多级 path（`">"` 连接）与真实 level。
2. **level 语义对齐 MarkdownParser**：章 = 1，节 = 2，子节 = 3，依此类推；目录树顶层若只是书名节点则跳过（书名已由 `ParsedBook.title` 承载，不进 path）。
3. **目录页剥离用启发式，不只靠文件名**：识别「内容主要为指向正文章节的导航链接 / 文本呈目录列表」的文档并跳过；保留现有 `_EPUB_SKIP_NAMES` 文件名黑名单兜底。
4. **健壮降级**：书缺失 nav/ncx 或目录结构异常时，退化为现有行为（文件名/首个 heading，level=1），**不得抛异常**。
5. **纯结构化、无 LLM、零新依赖**：只用 ebooklib 已有能力（它已提供 nav/ncx 读取接口）。

## Change List

### 1. `src/book_rag/io/parsers.py` — `EpubParser`（现 L212-261）及模块内辅助函数

- 新增辅助函数（模块级，供 `EpubParser` 调用），实现自理：
  - `_build_toc_tree(reader) -> list[Node]`：从 nav/ncx 提取目录树，每节点含 `title`、`href`（文件路径，去 `#anchor`）、`level`、`children`。EPUB3 用 `reader.get_items_of_type(ITEM_NAVIGATION)` 里的 `nav` 的 ol/li 结构；EPUB2 用 `reader.toc`（Ncx）嵌套结构；两者都无效时返回空表。
  - `_href_to_file_name(href) -> str | None`：把 href 归一化到 `file_name` 的对应形态（去 query/anchor、urldecode），用于与 `item.file_name` 匹配。
  - `_is_toc_page(soup) -> bool`：目录页启发式——文档内 `<a>` 链接文本 / 纯文本高度匹配「章节目录列表」特征（如大量「第X章…」短标题 + 逐条链接到正文文件）时判定为目录页。
- 改 `EpubParser.parse` 主循环（L236-259）：
  - 预先用 `_build_toc_tree` 建好 `file_name -> (path, level)` 映射（一个文件命中目录树中多级节点时，取路径最长的那个；否则退化为现有单级逻辑）。
  - 遍历 spine 时，若 `_is_toc_page(soup)` 为真则跳过。
  - `section.path` 用目录树映射出的多级路径（`" > "` 连接）；`level` 用目录树层级；无映射时回退现行行为（首个 heading / `item.title` / 文件名，level=1）。

### 2. `tests/test_parsers.py` — 新增多级 + 目录页 fixture 与断言

- 用 ebooklib 程序化构造一个**含 nav（三级目录）+ 目录页文档**的最小 EPUB：
  - 目录树：「第一章 启程」→「1.1 出发」「1.2 露营」；「第二章 归途」。
  - 一个目录页文档（text 为「第一章 启程 ·· 第二章 归途」的链接列表，file_name 取非黑名单名如 `catalog.xhtml`）。
  - 正文文档分别对应上述节点。
- 新增断言（沿用现有 `test_epub_parser` 风格）：
  - 目录页文档**不出现**在 `parsed.sections`（Contents 残留消失）；
  - 存在 `path == "第一章 启程 > 1.1 出发"` 的 section，且其 `level == 2`；
  - 存在 `level == 1` 的「第一章 启程」section；
  - 保留 `title`、「position 从 0 严格递增」、「text 非空」断言。

## Acceptance Criteria（可运行）

```bash
export PYTHONPATH=
.venv/Scripts/python.exe -m pytest tests/ -q     # 全绿（结构层 16 用例 + parser 用例不回归）
```

对真实《人生设计课》重建索引（删 `books/.book_rag_index` 后 `index`）后验证：

- `toc.json` 出现 `level > 1` 的节点，「第1章」下挂「辨识无法解决的重力问题」等小节（真树形）。
- `get_section_content("人生设计课", "第1章 …")` 开头不再是 `Contents` + 连续空行。
- `get_book_structure` 目录树呈现嵌套层级。
- 现有 chunk 检索与结构层行为不回归。

## Explicitly Do NOT Do

- ❌ 不引入 pymupdf 等新第三方依赖（只用 ebooklib 已有能力）。
- ❌ 不改 `PdfParser` / `TextParser` / `MarkdownParser`（PDF 章节层级提取属「后续设想」，本次不做）。
- ❌ 不用 LLM（纯结构化解析）。
- ❌ 目录页剥离不得误删正文：章节正文内含少量指向其他章节的链接，不得被 `_is_toc_page` 误判为目录页。
- ❌ 不改 `pyproject.toml` / `uv.lock` / 不新增依赖。
- ❌ 不改 `structure.py` / `retriever.py` / 结构层与检索逻辑。