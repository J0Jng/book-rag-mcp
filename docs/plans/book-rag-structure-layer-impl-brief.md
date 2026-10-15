# Implementation Brief: Book-RAG 结构层（Structure Layer）

## Background

Book-RAG 目前是**纯 flat chunk 检索**：解析 → 切分（512/64）→ 本地 embedding → FAISS → 返回原文 chunk。
chunk 仅带 `section` / `position` 元数据，**没有章节目录树、没有章节/概念摘要、没有概念索引**。
外层 Agent 只能"盲搜"，无法"先定位章节再精读"，跨章节综合题（"这本书的核心方法论是什么"）答得很差。

本 brief 加一层**结构层（Structure Layer）**，目标是：
1. 让外层 Agent 先拿"地图"（目录树 / 摘要 / 概念索引）定位，再精检索原文。
2. 证据可回溯：结构层条目都锚定原文 chunk，杜绝编造。
3. 减少语义损失：复用 `book-to-skill` 的"提取结构"哲学，但产出**可检索 + 双向链接**，而非静态笔记。

## 已确认的设计决策（不可违背，来自 Joe 拍板）

- **路径 A**：Book-RAG 保持无状态检索服务，不内置 agentic 循环。
- **A1**：循环思考（query 改写 / 拆子问题 / 反思重试 / 生成）**全部由外层 Agent 执行**，
  Book-RAG 自身不引 LLM 做循环、不引 LangGraph。Book-RAG 只提供细粒度原子工具 + playbook 纪律。
- **绝不编造**：所有返回内容必须锚定原文；检索不到就明说"未检索到"，禁止用通识补全。

## Design Decisions

1. **结构层 = 独立模块 `src/book_rag/rag/structure.py` + 独立索引子目录**，不改现有
   `retriever.py` / `index.py` 的 chunk 检索路径。结构层是"加一层"，不是重写。
2. **TOC（章节目录树）**：从解析器 `BookSection.path` + `level` 构建（构建索引时已有数据），
   序列化为 JSON 落盘，OL(1) 返回，不调 LLM。
3. **章节/概念摘要**：用 LLM **离线一次性**抽取（索引构建时），复用 `book-to-skill` 的
   章节摘要模板（核心思想 / 框架 / 术语 / 心智模型 / 反模式），但存成**可检索索引 +
   指向 `section`/`position` 的指针**。检索时不再调 LLM。
4. **LLM provider 复用 Hermes 的 custom provider**（DeepSeek），配置读 Hermes profile 的
   `config.yaml`（见 `references/reuse-hermes-custom-provider-credentials.md` 模式）。
   **可选降级**：若 provider 不可用 / 未配置，跳过摘要与概念层，TOC 层仍完整可用，
   **不报错、不阻塞索引构建**。
5. **概念索引**：摘要里抽出的框架/术语/心智模型，每条是短文本 + 指针，用现有 embedding
   建独立 FAISS 索引，供 `retrieve_concepts` 查询。
6. **绝不编造**：摘要抽取的 prompt 强制要求"原文没有的不要写，拿不准标注未检测到"；
   所有返回结果带 `source`（相对路径）/ `section` / `position` 元数据。

## Change List（按 ticket，TDD：红→绿→重构）

### Ticket 1 — TOC + 章节定位检索（纯结构化，无 LLM）【pilot，先做】

- 新增 `src/book_rag/rag/structure.py`：
  - `build_toc(sections: list[BookSection]) -> dict`：由 `BookSection.path`（`" > "` 分隔）+ `level`
    构造目录树（嵌套 dict，叶子带 `position` 范围 / `chunk` 下标）。
  - `save_structure(idx_dir, toc)` / `load_structure(idx_dir)`：TOC 落盘为 `toc.json`。
- 改 `src/book_rag/rag/index.py` `build_book_index()`（现 110-157 行）：在写 `meta.json` 之前，
  调用 `build_toc(parsed.sections)` 并落盘到 `idx_dir / "toc.json"`。
- 改 `src/book_rag/rag/retriever.py`：新增
  `search_books(..., section: str | None = None)`（现 74-123 行）：`section` 不为空时，
  用 `doc.metadata["section"]` 前缀过滤候选 chunk，再在命中范围内向量检索。
- 改 `src/book_rag/servers/mcp_server.py`：
  - 新增工具 `get_book_structure(book: str) -> str`：返回该书的目录树（Markdown 缩进列表，
    含各级章节名）。数据来自 `toc.json`，无 LLM。
  - 新增工具 `search_within_section(query, book, section) -> str`：调 `search_books(...,
    section=section)`。
  - 新增工具 `get_section_content(book, section) -> str`：按 section 拉取完整原文
    （聚合该 section 下所有 chunk 的 `page_content`，按 `position` 排序）。
- 测试：`tests/test_structure.py` 覆盖 TOC 构造、落盘/加载、section 过滤检索、章节原文拉取。

### Ticket 2 — 章节/概念摘要（LLM 离线抽取，可选降级）

- 新增 `src/book_rag/rag/summarizer.py`：
  - `summarize_sections(sections, llm) -> list[dict]`：对每章（或每 section）调 LLM 抽取
    `{section, summary, frameworks[], terms[], mental_models[], pointers[...]}`。
  - prompt 复用 book-to-skill 模板；强制"原文没有不要写"。
  - `llm` 封装放 `src/book_rag/rag/llm.py`：读 Hermes custom provider 配置，HTTP 调 DeepSeek；
    配置缺失时返回 `None`（触发降级）。
- 落盘 `summaries.json` + 概念条目嵌入独立 FAISS 索引 `concepts.faiss/.pkl`。
- 改 `mcp_server.py`：新增工具 `retrieve_concepts(query, book) -> str`：概念索引向量检索，
  返回 `{concept, pointer(section/position), 原文片段}`。
- 测试：`tests/test_summarizer.py`（LLM 用 mock；降级路径用"无 provider"直接测）。

### Ticket 3 — playbook + AGENTS.md + 集成

- `create_mcp()` 加 server 级 `instructions=`（循环纪律 + 满意标准 + 退出条件 + 绝不编造约束，
  详见下节 playbook 草稿）。
- 新增 `AGENTS.md`：给接入 agent 的完整编排说明（与 instructions 一致，人可读）。
- 集成冒烟：`get_book_structure` → `search_within_section` → `get_section_content` 链路跑通。

## playbook 草稿（Ticket 3 的 instructions 内容，可精修）

```text
你接入了一个"书籍知识检索"服务（Book-RAG）。回答某本书的问题时：
1. 先 list_indexed_books 确认书已索引，再 get_book_structure 拿目录树/摘要，判断问题落在哪。
2. 优先 search_within_section 在定位到的章节内精检索；范围不明才用全局 search_book_knowledge。
3. 结果相关度低(相似度<0.5 或明显不相关)时，改写查询词或换范围再试≤2 次——这是 corrective
   retrieval，不是放弃。
4. 跨章节"概念关系/推导"题拆成多个子查询分别检索再综合。
5. "核心框架/方法论/心智模型"类题优先 retrieve_concepts。
6. 生成答案必须基于检索到的原文，引用标注 source 与 section；检索不到就明说"书中未检索到"，
   绝不编造。
```

## Acceptance Criteria（可运行）

- `export PYTHONPATH=` + `.venv/Scripts/python.exe -m pytest tests/ -v` 全部通过。
- `uv run book-rag-mcp index` 对 `books/` 重建索引后，`books/.book_rag_index/**/toc.json` 存在且非空。
- 手动 smoke：通过 MCP 工具调用 `get_book_structure` / `search_within_section` /
  `get_section_content` / `retrieve_concepts` 均返回锚定原文的结果（无 hallucinated 内容）。
- 无 LLM provider 时：`TOC` 仍可用，`retrieve_concepts` 返回"概念索引未构建（需配置 LLM）"而非报错。
- 现有 chunk 检索 `search_book_knowledge` 行为不变（回归测试通过）。

## Explicitly Do NOT Do

- ❌ 不引 LangGraph / 不在检索路径内置 LLM 循环（A1）。
- ❌ 不改嵌入路由 / 语言检测逻辑（`language.py`、`embeddings.py`）。
- ❌ 不改 `pyproject.toml` / `uv.lock` / 不新增硬依赖（LLM 用 stdlib + 已有 httpx/requests）。
- ❌ 不硬编码 API key（读 Hermes profile 配置，缺则降级）。
- ❌ 不重写现有 `search_books` 的跨书合并/打分逻辑，只在末尾加 `section` 过滤参数。
- ❌ 返回内容不得出现原文没有的表述。