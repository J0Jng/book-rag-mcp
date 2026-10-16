# Implementation Brief: Book-RAG P0.3 — EpubParser 剥离目录树之外的献词页（front-matter）

## Background

真实书籍《斯坦福大学人生设计课》重建索引后，`get_book_structure` 返回的目录树**顶部多出一个匿名 section「chapter1」**，污染了每个接入 Agent 拿到的目录树第一项。

根因（已诊断）：
- 该 EPUB 的 spine 文档 `Text/chapter1.xhtml` 是**献词页**（正文「谨以此书献给所有与我们分享人生故事的优秀学员…」，仅 226 字）。
- 它**不在 nav/ncx 目录树里**（目录树从 封面 → 扉页 → 推荐序(=chapter2.xhtml) 直接开始，跳过了 chapter1.xhtml）。
- 它无 heading、无 `item.title`，于是 `EpubParser.parse` 的回退分支（`parsers.py` 现 478-483 行）用了文件名主干 `chapter1` 作为 section 名，`level=1`，混入正文索引。
- 这是**全书唯一一个「无目录映射」的 spine 文档**，其余 59 个 section 层级（11 章 + 节）都正确。

本质不是「Contents 页眉」类残留（P0.2 已处理），而是**「目录树之外的匿名 front-matter 文档」未识别**。这是计划中的「向量化前书籍清洗环节」的第一步（见 ROADMAP 后续设想第 4 条）。

## Design（已拍板，不可违背）

1. **判据 = 「无目录映射」+「献词特征」双条件**，前者极强（正常章节都有 nav/ncx 映射），
   后者兜底内容语义。两者同时命中才剥离，误伤概率极低。
2. **整文档剥离**（不 append 该 section），不做局部文本截断——献词页不是可检索正文。
3. 不触碰目录树构建、`_strip_contents_prefix`、`_is_toc_page`、`_EPUB_SKIP_NAMES` 等既有逻辑。

## Change List

### 1. `src/book_rag/io/parsers.py`

- 在 `_CONTENTS_PREFIX_RE`（现约 216-218 行）之后，新增献词特征正则与判定函数：

  ```python
  # 献词页特征（目录树之外的前置文档）：谨以此书献给 / 献给 / Dedicated to 等
  _DEDICATION_RE = re.compile(
      r"(谨以此书献给|谨献给|仅以此书献给|此书献给|献给|dedicated\s+to|dedication|to\s+my\b)",
      re.IGNORECASE,
  )

  def _is_dedication(text: str) -> bool:
      \"\"\"判定文本是否为献词页：篇幅短 + 开头命中献词特征。\"\"\"
      return len(text) < 2000 and bool(_DEDICATION_RE.search(text[:500]))
  ```

- 在 `EpubParser.parse` 遍历 spine 的循环内（现约 485-489 行附近），
  `text = _strip_contents_prefix(text)` 之后、`if not text: continue` 之后、`sections.append(...)` 之前，
  插入：

  ```python
  # 目录树之外的匿名献词页（front-matter）整文档剥离
  if mapped is None and _is_dedication(text):
      continue
  ```

  （`mapped` 是该循环里 `_lookup_toc_path` 的返回值，原本在 `if mapped is not None: ... else: ...` 块
  之外可见；确认插入位置能访问到 `mapped`。若当前变量作用域使其不可用，把判断移到 `else` 回退分支内
  对 `text` 计算后的同一位置——即「无目录映射」这一支里判断。）

### 2. `tests/test_parsers.py`

按现有 `_make_epub_with_*` 辅助函数模式，新增两个测试：

- `_make_epub_with_dedication_page(path)`：构造最小 EPUB，含
  - 一个献词文档 `chapter1.xhtml`（**不在 `book.toc`、无 heading**，content 开头为
    `谨以此书献给所有与我们分享人生故事的优秀学员。`，模仿真实书形态），
  - 一个正常章节 `chap_01.xhtml`（在 `book.toc`，有 `<h1>第1章 启程</h1>` + 正文），
  - `book.spine = [dedicate, chapter]`，写入 path。

- `test_epub_parser_strips_dedication_page`：
  解析后断言——
  - `len(sections) == 1`（献词页被剥离，只剩正常章节）；
  - 该 section path == `第1章 启程`；
  - `position == 0`（剥离后 position 从 0 重排）。

- `test_epub_parser_keeps_unmapped_body`（防误伤）：
  构造含一个「无目录映射 + 无 heading + 内容为正常非献词正文」文档的 EPUB，断言该文档**仍被保留**
  （不被 `mapped is None` 误剥），即 `len(sections) == 1` 且其 path 来自回退命名。

## Acceptance Criteria（可运行，全部自验）

1. `cd "F:/code/projects/Book-RAG" && export PYTHONPATH= && .venv/Scripts/python.exe -m pytest tests/ -v`
   全部通过（含新增 2 个测试 + 既有全套不回归）。
2. 真实书回归：删旧索引、用干净临时 `index_dir` 重建，`toc.json` 顶层**不含** `chapter1`，
   顶层节点为 `推荐序 / 引言 / 第1章 … 第11章 / 结语 / 致谢`，section 数从 60 减为 59。
   （真实书解析需在脚本内 `import` 前设 `HF_HUB_OFFLINE=1`；重建索引需真实 bge 模型，已缓存。）

## Explicitly Do NOT Do

- ❌ 不引入新的「清洗 pipeline / 新模块」——本次只做献词页定点剥离，系统清洗环节留待后续。
- ❌ 不改目录树构建 / `_strip_contents_prefix` / `_is_toc_page` / `_EPUB_SKIP_NAMES` 逻辑。
- ❌ 不改其他 parser（Markdown / TXT / PDF）、`index.py` / `retriever.py` / 结构层 / MCP。
- ❌ 不改 `pyproject.toml` / `uv.lock`，不新增硬依赖。
- ❌ 不针对具体书名硬编码（不得出现「人生设计课」「斯坦福」等字面量进源码）。
- ❌ 不误伤：`mapped is None` 但正常正文的文档必须保留（有防误伤测试兜底）。