# Implementation Brief: Book-RAG P2 — playbook + AGENTS.md + 集成

## Background

结构层 Ticket 1（TOC + 章节定位检索）与 Ticket 2（章节/概念摘要 + 概念索引）已落地并提交：
- `4cacf4b` Ticket 1：`structure.py` 目录树 + `search_within_section` / `get_section_content` / `get_book_structure`
- `32b17ad` Ticket 2：`summarizer.py` + `concepts.py` + `retrieve_concepts`（LLM 离线抽取，可选降级）

ROADMAP 演进路线 `P0 → P1 → P2` 中，P2 = **playbook 文档 / AGENTS.md**，即结构层 brief 的
**Ticket 3 —— playbook + AGENTS.md + 集成**。这是结构层最后一块，本次推进。

当前缺口：
- `create_mcp()` 创建的 FastMCP 实例没有 `instructions=`，接入 Agent 拿不到编排纪律，只会盲搜。
- 无 `AGENTS.md`，人 / 接入侧没有可读的编排说明。
- README「暴露的 MCP 工具」表仍只列 3 个工具（实际 7 个），已过时。

## Design（已拍板，不可违背）

1. **纪律外置为纯文本常量**：编排纪律（playbook）写成字面明文，作为 `mcp_server.py` 模块级常量
   `PLAYBOOK` 传入 `FastMCP(..., instructions=PLAYBOOK)`。不放别处，不引模型生成，检索路径不引 LLM
   （结构层 A1 铁律：无状态检索服务，循环思考交给外层 Agent）。

2. **playbook 措辞与现有 7 个工具精确对齐**，工具名一个都不能写错。

3. **AGENTS.md 与 instructions 内容一致**：一份纪律、两处呈现（MCP instructions 供机器注入，AGENTS.md 供人读）。

4. **README 工具表补全为 7 个工具**，并新增指向 AGENTS.md 的一行。

## PLAYBOOK 定稿（instructions 与 AGENTS.md 共用的正文，逐字照搬）

```text
你接入了一个「书籍知识检索」服务（Book-RAG），回答关于某本书的问题时遵循以下纪律：

1. 先 list_indexed_books 确认书已索引，再 get_book_structure 拿目录树，判断问题落在哪个章节。
2. 定位到章节后，优先 search_within_section 在该章节内精检索；只有范围不明时才用全局的 search_book_knowledge。
3. 需要完整上下文时，用 get_section_content 拉取整章原文（内容全部来自书籍本身）。
4. 「核心框架 / 方法论 / 心智模型 / 术语定义」类问题优先 retrieve_concepts。
5. 检索结果相关度低（相似度 < 0.5 或明显不相关）时，改写查询词或换检索范围再试，最多 2 次——这是纠正式检索，不是放弃。
6. 跨章节的「概念关系 / 推导 / 对比」题，拆成多个子查询分别检索后再综合。
7. 生成答案必须基于检索到的原文，并标注来源（source / section / position）；检索不到就明说「书中未检索到」，绝不编造、绝不用通识补全。
```

## Change List

### 1. `src/book_rag/servers/mcp_server.py`
- 在 import 区之后、`_CONFIG` 之前新增模块级常量 `PLAYBOOK`（内容 == 上面「PLAYBOOK 定稿」代码块内的正文，
  含开头导语句与 7 条纪律，末尾不要多余空行）。
- 修改 `create_mcp()`（现 271-279 行）：
  `mcp = FastMCP("book-rag")` → `mcp = FastMCP("book-rag", instructions=PLAYBOOK)`。
- 更新模块 docstring（现 1-13 行）：补一句说明 `create_mcp` 注入 `PLAYBOOK` 编排纪律（instructions）。
- **不要**改动任何工具函数、不要动 tool 注册逻辑、不要动 `refresh_all_indexes(config)` 启动同步。

### 2. 新增 `AGENTS.md`（项目根目录，与 README.md 同级）
结构（Markdown）：
```markdown
# AGENTS.md

## 项目定位
一句话：Book-RAG 是无状态书籍知识检索 MCP 服务，把本地书籍（PDF/EPUB/TXT/Markdown）解析成
可检索的知识源，通过 MCP 暴露给任意 Agent；只提供细粒度原子工具，循环思考/生成由外层 Agent 负责。

## 编排纪律（接入 Agent 必读）
[== 上面「PLAYBOOK 定稿」正文 ==]

## 工具清单
| 工具名 | 说明 |
|---|---|
| list_indexed_books | 列出已索引书籍及语言、块数 |
| get_book_structure | 返回书籍目录树（Markdown 缩进列表），先定位章节 |
| search_within_section | 在指定章节内精检索原文片段 |
| get_section_content | 拉取某章节完整原文（按阅读顺序聚合） |
| search_book_knowledge | 全局按内容检索，返回原文片段 + 相关度 + 来源 |
| retrieve_concepts | 检索框架/术语/心智模型条目（需离线概念索引，未建时返回提示） |
| refresh_book_index | 增量同步书籍库索引 |

## 铁律
- 绝不编造：所有返回内容锚定原文，检索不到明说「书中未检索到」，禁止通识补全。
- 结构层不内置 LLM 循环；改写查询/拆子问题/综合由接入 Agent 自己执行。
```

### 3. `README.md`
- 把「暴露的 MCP 工具」表（现 102-106 行）从 3 行补全为 7 行（工具名与说明与 AGENTS.md 工具清单一致，
  顺序按上面的清单排列）。
- 在「接入 Agent」小节末尾（现 98 行附近）加一句：编排纪律详见 `AGENTS.md`（接入 Agent 会自动注入 instructions）。

### 4. `tests/test_mcp_server.py`
- 在文件末尾新增测试：
  - `test_playbook_constant_contains_discipline`：`from book_rag.servers.mcp_server import PLAYBOOK`，
    断言其为非空字符串，且包含「绝不编造」「search_within_section」「retrieve_concepts」
    「get_book_structure」「list_indexed_books」五个关键词。
  - `test_create_mcp_sets_instructions`：`mcp = mcp_server.create_mcp(config)`，断言
    `mcp.instructions == PLAYBOOK`（FastMCP 实例的 `instructions` 属性已确认可读，值为构造时传入的字符串）。
- **不要**改动文件里已有的任何测试。

## Acceptance Criteria（可运行，全部自验）

```bash
cd "F:/code/projects/Book-RAG"
export PYTHONPATH=
.venv/Scripts/python.exe -m pytest tests/ -v
```
全部通过。

额外自验：
```bash
# instructions 已注入到实例
.venv/Scripts/python.exe -c "from book_rag.servers.mcp_server import create_mcp; m=create_mcp(); print('OK' if m.instructions else 'EMPTY')"
```
打印 `OK`。

## Explicitly Do NOT Do

- ❌ 不引 LangGraph，不在检索路径内置 LLM 循环（结构层 A1）。
- ❌ 不改 `language.py` / `embeddings.py` / `retriever.py` / `index.py` / `structure.py` / `summarizer.py` / `concepts.py`。
- ❌ 不改 `pyproject.toml` / `uv.lock`，不新增硬依赖。
- ❌ 不改任何已存在的工具函数实现或 tool 注册逻辑，只加 `instructions=` 与 `PLAYBOOK` 常量。
- ❌ 不改动 `config.py`、`cli.py`、`servers/web.py`。
- ❌ 不硬编码 API key / 不读 Hermes 私有配置。
- ❌ playbook / AGENTS.md 措辞必须与「PLAYBOOK 定稿」逐字一致，工具名不得写错或新增不存在的工具。