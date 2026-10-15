# Implementation Brief: EPUB 章节层级推断（P0.1 — 平铺 ncx 书的真树形 TOC）

## Background

P0（`4304575`）已使 `EpubParser` 从 nav/ncx 读取目录树并映射多级 path/level，单测通过。但真实书《斯坦福人生设计课》重建索引后 `toc.json` 仍**全 `level=1` 平铺**。侦察定位到根因：

- 这本书是 **EPUB2**（只有 `toc.ncx`，无 EPUB3 `nav`），其 `reader.toc` 是 **62 个平铺 leaf、零嵌套**——ebooklib 解析后没有任何 `(Section, children)` 元组，`_build_toc_tree` 读不到层级，逐文档 fallback 回 `level=1`。
- 但章节层级其实编码在**标题模式**里：11 个「第N章 …」是章级；「辨识无法解决的重力问题」「人生设计评估」「反思你的工作观」等是上一章的小节（文件名 `chapter4.xhtml` vs `chapter4-1.xhtml` 也印证了章 vs 节的一对多关系）。

因此需要在 nav/ncx **无层级**时，用标题模式推断层级。这是大量中文 EPUB2 书的共同形态，修复有普遍价值。

## Design Decisions

1. **标题模式层级推断只作 fallback**：仅当 `_build_toc_tree` 产出的树**全平铺（无任何 children）**时触发，绝不干扰真正带层级 nav 的书（现有 fixture 路径保持不变）。
2. **章级标题模式**：`第[0-9零〇一二三四五六七八九十百千万两]+[章篇部卷]` 或 `Chapter\s+\d+`（大小写不敏感）。命中 → `level=1`，并设为「当前章」。
3. **前/后置部分模式**：`序`/`前言`/`引言`/`导读`/`封面`/`扉页`/`版权`/`推荐序`/`致谢`/`后记`/`结语`/`附录`。命中 → `level=1` 的**独立节点**，不挂到任何章下，也不更新「当前章」。
4. **其余标题（小节）**：若有「当前章」→ 挂其下（`level=2`，`path = "当前章 > 本节"`）；无当前章 → `level=1`。
5. 纯结构化、无 LLM、零新依赖；复用现有 `_TocNode` / `_build_toc_path_map` 结构。

## Change List

### 1. `src/book_rag/io/parsers.py`

- 新增章级标题正则与前置/后置标题正则（模块级，单独命名，勿与 `TextParser` 的 `_CHAPTER_RE` 混用）。
- 新增 `_infer_levels_from_titles(nodes: list[_TocNode]) -> list[_TocNode]`：把平铺节点按上文规则重建为带 `children` 的树（仅一层子节即可，不必递归更深处）；维护「当前章」指针；对章级/前置标题处理为 level 1，其余挂到当前章下 level 2。
- 改 `EpubParser.parse`（现构建目录树的段落，约 L387-395）：`_build_toc_tree(reader)` 之后，检测 `all(not n.children for n in toc_tree)`（且仅当一棵树整体平铺、节点数 > 1 时才推断，避免单节点书名误触发）→ 用 `_infer_levels_from_titles` 重建 `toc_tree` → 再 `_build_toc_path_map(toc_tree)`。

### 2. `tests/test_parsers.py`

- 新增 fixture：构造一个 **平铺 ncx**（top-level 全 leaf、无嵌套 children）+ 章标题模式的 EPUB——含「第1章 启程」「出发」「露营」「第2章 归途」这样的平铺标题，以及一个「引言」前置标题。
- 新增断言：
  - `path == "第1章 启程 > 出发"` 的 section 存在且 `level == 2`；
  - `path == "第1章 启程"` 的章节点存在且 `level == 1`；
  - 「引言」保持 `level == 1`；
  - 既有断言不回归：position 严格递增、text 非空；既有「带层级 nav」的 fixture 行为不变（那本书仍走 nav 层级，不被标题推断干扰）。

## Acceptance Criteria（可运行）

```bash
export PYTHONPATH=
.venv/Scripts/python.exe -m pytest tests/ -q     # 全绿
```

重建《人生设计课》索引（删 `books/.book_rag_index` 后 `index`，`HF_HUB_OFFLINE=1`）后：

- `toc.json` 出现 `level == 2` 节点（「第1章 …」下挂「辨识无法解决的重力问题」等小节的真树形）。
- `get_book_structure` 呈现嵌套层级（章 → 节）。
- 章节定位检索（`search_within_section` 用「小节名」过滤）仍可用。

## Explicitly Do NOT Do

- ❌ 不改 `PdfParser` / `TextParser` / `MarkdownParser`。
- ❌ 不新增第三方依赖、不改 `pyproject.toml` / `uv.lock`。
- ❌ 不用 LLM（纯正则 + 结构化推断）。
- ❌ 标题层级推断只在 toc **全平铺**时触发；不得改变带层级 nav 的书的解析结果。
- ❌ 不改 `structure.py` / `retriever.py` / 结构层与检索逻辑。