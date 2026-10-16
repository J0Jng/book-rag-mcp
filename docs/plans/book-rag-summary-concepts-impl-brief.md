# Implementation Brief: Book-RAG P1 — 薄 LLM 摘要层 + 概念索引

## Background

P0 已完成：parser 修复（Contents 剥离 / 层级推断 / 章节路径映射）与结构层 Ticket 1
（TOC 目录树 `toc.json` + 章节定位检索，纯结构化零 LLM）均已落地并提交。

P1 实现结构层的 Ticket 2：**章节/概念摘要 + 概念索引**。目标是让外层 Agent 在
`get_book_structure`（目录树）之外，还能拿到每章的「核心思想 + 框架 + 术语 + 心智模型」
并可按概念检索——回答「这本书的核心方法论是什么」这类跨章节综合题时不必盲搜原文。

摘要由 LLM **离线一次性**抽取（索引构建时），检索时不再调 LLM；概念条目锚定
`section`/`position`，杜绝编造；LLM 不可用时**优雅降级**（TOC 层照常可用）。

## 已确认的设计决策（不可违背）

来自 Joe 拍板（结构层总任务书）+ 本 brief 细化：

- **路径 A / A1**：Book-RAG 保持无状态检索服务，不内置 agentic 循环、不引 LangGraph。
- **绝不编造**：摘要抽取 prompt 强制「原文没有的不要写，拿不准标[未检测到]」；所有
  返回内容锚定原文，检索不到明说。
- **可选降级**：LLM 不可用 / 未配置 / 任一章抽取失败 → 跳过摘要与概念层，**不报错、
  不阻塞索引构建**，TOC 层仍完整可用；`retrieve_concepts` 返回「概念索引未构建」而非抛异常。

## Design

### D1. LLM 封装 `src/book_rag/rag/llm.py`（新增）

**用 stdlib `urllib.request` 做裸 HTTP，不引 langchain-openai**（langchain-openai 1.4.x
会把 `max_tokens` 改名 `max_completion_tokens`，在火山方舟端点上语义不同、会把推理
token 与输出 token 合并计预算，导致空输出——见 `hermes-claude-handoff` 的
volcengine-plan-langchain-reasoning-pitfall）。零新依赖。

**真实探针已验证**（2026-10-16，本机直连火山方舟 Coding Plan）：

| 项 | 结果 |
|---|---|
| 端点 | `POST {base_url}/chat/completions`，`base_url = https://ark.cn-beijing.volces.com/api/coding/v3` |
| 请求体 | `{model, messages, max_tokens, response_format:{"type":"json_object"}}` | 全部被接受 |
| model | `DeepSeek-V4-Pro` → 200，返回 `choices[0].message.content` |
| JSON 输出 | `response_format=json_object` 生效，`message.content` 是干净 JSON 字符串 |
| ⚠️ 推理 token | `DeepSeek-V4-Pro` 是推理模型：`usage.completion_tokens_details.reasoning_tokens` 占预算大头（实测 30 token 里 28 是推理）→ **`max_tokens` 必须设大且处理 `finish_reason=="length"` 截断** |

**`SummarizerLLM` 类**：
- `__init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0)`
- `complete_json(self, system: str, user: str, *, max_tokens: int = 2000) -> dict`：
  1. `POST f"{base_url}/chat/completions"`，头 `Authorization: Bearer {api_key}`。
  2. body：`{"model": model, "messages": [{"role":"system","content":system},
     {"role":"user","content":user}], "max_tokens": max_tokens,
     "response_format": {"type": "json_object"}}`。
  3. 读 `resp["choices"][0]["message"]["content"]` → `json.loads`。
  4. `finish_reason == "length"` → raise `LLMTruncatedError`（截断，输出不可信）。
  5. `content` 空 / JSON 解析失败 / HTTP 非 200 → raise `LLMError`。
- 不重试由调用方（summarizer）决定；llm.py 本身单次调用、异常上抛。

**`load_llm() -> SummarizerLLM | None`**（工厂，绝不向上抛、绝不阻塞）：
```python
base_url = os.environ.get("BOOK_RAG_LLM_BASE_URL") or "https://ark.cn-beijing.volces.com/api/coding/v3"
api_key  = os.environ.get("BOOK_RAG_LLM_API_KEY") or os.environ.get("HERMES_CUSTOM_CUSTOM_API_KEY")
model    = os.environ.get("BOOK_RAG_LLM_MODEL") or "DeepSeek-V4-Pro"
if not api_key:
    return None
return SummarizerLLM(base_url, api_key, model)
```
**设计说明（对原 brief「读 Hermes config.yaml」的简化）**：复用 Hermes custom provider 的
意图通过「base_url/model 取火山方舟 Coding Plan + DeepSeek-V4-Pro 默认值，key 直接复用
Hermes 的 `HERMES_CUSTOM_CUSTOM_API_KEY` 环境变量」实现，**不读 config.yaml 文件**——避免把
Hermes 安装路径耦合进项目（项目部署到无 Hermes 的机器时，环境变量 `BOOK_RAG_LLM_*` 独立生效）。
key 绝不硬编码，缺失即降级 None。若 Joe 要求严格读 config.yaml，再补。

### D2. 章节摘要 `src/book_rag/rag/summarizer.py`（新增）

- `summarize_sections(sections: list[BookSection], llm, *, title: str = "") -> list[dict]`：
  1. **按 `section.path` 的第一段**（`path.split(" > ")[0]`）分组（顶层章节名），同组按
     `position` 排序聚合 `text`（用 `"\n"` 拼接）。空 path 的 section 归入 `title` 或 `"(序言)"`。
  2. 对每章调 `llm.complete_json`，输出 schema（严格 JSON）：
     ```json
     {
       "summary": "1-2 句：本章最核心的一个知识点",
       "frameworks": [{"name": "框架名", "description": "作者原表述 + 何时用 + 如何用"}],
       "terms": [{"name": "术语", "definition": "一句话精确定义"}],
       "mental_models": ["在 X 场景使用 Y 的心智模型/反模式，一句话"]
     }
     ```
  3. 返回 list[dict]，每条**由代码附加指针**（不依赖 LLM）：
     ```python
     {"section": 章名, "summary": ..., "frameworks": ..., "terms": ...,
      "mental_models": ..., "pointers": [{"section": s.path, "position": s.position} for s in 该组成员]}
     ```
  4. **单章容错**：某章 `LLMError`/`LLMTruncatedError` → 重试 1 次；再失败 → 该章**跳过**
     （不写入结果、不影响其他章、不 raise）。整体永不因单章失败中断。

- **prompt 复用 book-to-skill 章节摘要模板**（核心思想 / 引入的框架 / 关键概念 / 心智模型 /
  反模式），但强制 JSON 输出 + 防编造约束。system 必含：
  > 你只依据下方提供的原文提取结构，绝不编造。原文没有的框架/术语/概念一律不要写；
  > 拿不准的标「[未检测到]」。输出必须是合法 JSON，且不包含任何多余文本。

### D3. 概念索引 `src/book_rag/rag/concepts.py`（新增）

- `Concept` dataclass：`text: str, kind: str, section: str, position: int, source: str`。
  `kind ∈ {"framework", "term", "mental_model"}`。
- `build_concept_entries(summaries: list[dict]) -> list[Concept]`：从每条 summary 抽出
  concepts，`text` 用「名称 + 描述」合成（框架/术语），心智模型直接用原文；`section`/`position`
  取该 summary 第一个 pointer，`source` 由调用方传入（书籍相对路径）。
- `build_concept_index(concepts: list[Concept], embeddings, idx_dir: Path, *, source: str) -> list[Concept]`：
  `concepts` 为空 → 直接返回 `[]`（不建索引）。否则
  `FAISS.from_documents([Document(page_content=c.text, metadata={...}) for c in concepts], embeddings)`
  → **`save_local(str(idx_dir), index_name="concepts")`**（生成 `concepts.faiss` + `concepts.pkl`，
  不与 chunk 的 `index.faiss` 冲突），并把 concepts 元数据落盘 `idx_dir/"concepts.json"`。
- `load_concept_index(idx_dir, embeddings) -> (vector_store, list[Concept]) | None`：
  无 `concepts.json` → `None`；否则读 `concepts.json` + `FAISS.load_local(..., index_name="concepts",
  allow_dangerous_deserialization=True)`。
- **`retrieve_concepts(query, *, config, book, top_k=5) -> str` 的实现放 `mcp_server.py`**
  （定位书需 `_find_matching_metas` + `get_book_index_dir`，若放 concepts.py 会与 index.py
  互相 import 造成循环依赖）：
  1. `_find_matching_metas` 定位书 → `get_book_index_dir` → `get_embeddings(meta.model_name)` → `load_concept_index`。
  2. `None` → 返回「概念索引未构建（需配置 LLM 后重建索引）」。
  3. 向量检索 top_k，返回 Markdown：每条概念
     `名称（类型）· 章节（相关度）+ 指针 + 概念描述`。
  concepts.py 只提供不依赖 index/retriever 的纯函数（`build_concept_entries` /
  `build_concept_index` / `load_concept_index`），保证依赖单向：index → concepts，无循环。

### D4. 索引集成 `src/book_rag/rag/index.py`

`build_book_index()`（现 L111-161）在 `save_structure(...)`（L148）之后、`meta = BookIndexMeta(...)`
之前，插入：
```python
llm = load_llm()  # 模块顶部 import；测试 monkeypatch index_mod.load_llm
if llm is not None:
    try:
        summaries = summarize_sections(parsed.sections, llm, title=parsed.title)
        if summaries:
            json.dump(summaries, ...)  -> (idx_dir / "summaries.json")
            concepts = build_concept_entries(summaries)
            build_concept_index(concepts, embeddings, idx_dir, source=key.as_posix())
    except Exception:
        pass  # 摘要/概念层失败绝不阻塞索引构建（降级）
```
`embeddings` 复用 L142 已建的实例。整个摘要块 try/except 兜底，任何异常只跳过不影响索引。
（注：摘要每章一次串行 LLM 调用，一本书多章会显著拖慢索引——这是「离线一次性」的预期成本，
pilot 先串行，批量/并发优化后续再议。）

### D5. MCP 工具 `src/book_rag/servers/mcp_server.py`

- 新增模块级 `retrieve_concepts(query, book, *, config, top_k=5) -> str`（委托
  `concepts.retrieve_concepts`）。
- 模块级引用捕获区（现 L200-206）加 `_concepts_impl = retrieve_concepts`。
- `create_mcp()` 内注册 `@mcp.tool def retrieve_concepts(query, book, top_k=5) -> str`，
  风格与现有 6 个工具一致。

## Change List（分两刀委托，pilot 先行）

### P1-A（pilot）— llm + 摘要，纯新增
1. 新增 `src/book_rag/rag/llm.py`：`SummarizerLLM` + `load_llm` + `LLMError`/`LLMTruncatedError`。
2. 新增 `src/book_rag/rag/summarizer.py`：`summarize_sections`（分组 / 聚合 / prompt / 容错）。
3. 新增 `tests/test_llm.py`：`load_llm` 配置解析（env 优先 / 回退 key_env / 缺失 None）；
   `complete_json` 用 monkeypatch `urllib.request.urlopen` 造假响应：正常 JSON、非 200、
   `finish_reason=length`、空 content、坏 JSON 五类。
4. 新增 `tests/test_summarizer.py`：mock llm（`complete_json` 返回固定 dict）验证分组聚合、
   输出 schema、pointer 自动附加、单章失败跳过不中断。**不触发真实网络/模型。**

### P1-B — 概念索引 + 集成 + 工具
1. 新增 `src/book_rag/rag/concepts.py`：`Concept` / `build_concept_entries` /
   `build_concept_index` / `retrieve_concepts`。
2. 改 `src/book_rag/rag/index.py`：`build_book_index` 接 D4 集成块（顶部 `from book_rag.rag.llm
   import load_llm`、`from book_rag.rag.summarizer import summarize_sections`、
   `from book_rag.rag.concepts import build_concept_entries, build_concept_index`）。
3. 改 `src/book_rag/servers/mcp_server.py`：D5 新增工具。
4. 新增 `tests/test_concepts.py`：`build_concept_entries` 抽取、`build_concept_index` 落盘
   concepts.faiss/pkl/json（FakeHashEmbeddings）、`retrieve_concepts` 命中 vs 降级提示。

## Acceptance Criteria（可运行）

- `export PYTHONPATH=` + `.venv/Scripts/python.exe -m pytest tests/ -q` 全绿（含新增测试，
  现基线 92 passed 不回归）。
- mock 测试覆盖「llm 不可用 → 跳过摘要/概念，索引照常构建」降级路径。
- 手动（验收阶段，需 key）：`source` Hermes `.env` 导出 `HERMES_CUSTOM_CUSTOM_API_KEY` 后，
  `uv run book-rag-mcp index` 对书重建，`idx_dir/summaries.json` 与 `concepts.json` 存在且
  非空、内容锚定原文（抽查 summary 无原文外的表述）。
- 无 key 环境：`retrieve_concepts` 返回「概念索引未构建」提示而非报错。

## Explicitly Do NOT Do

- ❌ 不引 langchain-openai / 不新增任何第三方依赖（llm.py 用 stdlib `urllib.request`；
  concept 嵌入复用已有 `sentence-transformers`/FAISS）。
- ❌ 不改 `pyproject.toml` / `uv.lock`。
- ❌ 不硬编码 API key。
- ❌ 不改 `language.py` / `embeddings.py` / `splitter.py` / 解析器。
- ❌ 不重写 `search_books` / `build_toc` 既有逻辑，只在 `build_book_index` 末尾**追加**摘要块。
- ❌ 摘要失败不得阻塞索引构建；任何 LLM/概念异常都在 `build_book_index` 内 try/except 兜底。
- ❌ 摘要/概念内容不得出现原文没有的表述。
- ❌ 不 commit / 不 push（改完等验收）。