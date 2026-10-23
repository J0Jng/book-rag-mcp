# Claude 执行队列（Book-RAG 书库扩充）

> **2026-10-23 接手更新**：Joe 已授权 Codex 完成 S3 并继续 S4–S6。本文件下方保留历史提示词；“由 Claude 实现”“S2 待确认”“重建前另问”等历史流程不再代表当前授权状态。当前 S1–S6 已实现并完成阶段验证，S4–S6 未重算正式核心索引；证据及主题混合/边缘成员局限见 docs/reports/s4-s6-acceptance.md。禁止删除原始 EPUB。

> 用途：把 `docs/plans/book-rag-library-expansion-brief.md` 的阶段拆成**可直接粘贴给 Claude Code 的单步任务**。
> 规则：**一次只出一个阶段**，前一阶段验收不过不得进入下一阶段。
> 维护：Hermes（dev profile）负责设计、验收与整改清单；**实现由 Claude 执行**。

## 一、现状快照（2026-10-22，Hermes 实测）

| 项 | 状态 |
|---|---|
| S0 结构体检 | ✅ 完成（`scripts/inspect_library.py` + `docs/reports/library-structure-inspection.md`） |
| S1 Markdown 归一 | ✅ 完成（5 本书 → 484 章 / 1,289,760 字 / 3,257 chunk，零版权页污染） |
| 索引输入 | `books/library/*.md`（唯一输入）；`books/library/_conversion.json` 记录源→产物 |
| 原始归档 | `books/_originals/*.epub`（5 份，**只读**，`discover_books` 已跳过 `_` 前缀目录） |
| 索引状态 | ⚠️ **脏**：`books/.book_rag_index/` 仍指向已归档的旧 epub，**不要把它当基准**，统一在 S3 末尾重建 |
| 测试基线 | 173 passed（`-m 'not integration'`）；`ruff check src tests scripts` 全绿 |

关键实现事实（写代码前先读）：
- 章节粒度＝**章级**：`#` = 书名/部，`##` = 章；小节标题已降级为正文行，**不要**再引入小节级 section。
- 转换器 `src/book_rag/convert/markdown.py` 已处理：`<head><title>` 剥离、短章保留、目录/版权页丢弃、合集书名作用域。
- 《工作、消费主义和新穷人》第四章 23,426 字，是已知取舍，靠 S3 的工具层上限保护解决，**不要靠再切分它绕过去**。

## 二、通用规矩（每个阶段都适用）

1. **只做当前阶段**，不顺手改其他阶段的代码/文档。
2. `books/_originals/` 只读，禁止移动、删除、重命名。
3. **不要删或重建 `books/.book_rag_index/`**——索引重建只在 S3 末尾做一次，且需 Joe 确认。
4. 跑 Python/pytest 前清 `PYTHONPATH`：
   - bash：`export PYTHONPATH= && uv run pytest -q`
   - cmd：`set "PYTHONPATH=" && uv run pytest -q`
5. 依赖用 `uv add`，禁止手改 `uv.lock`；不升级已有依赖。
6. 新增/修改代码必须配测试；`ruff check src tests scripts` 必须全绿。
7. 完成后必须给：**改了哪些文件 / 跑了什么命令 / 真实输出 / 未验证项**。不允许用"应该没问题"代替实跑。
8. 不执行 `git commit` / `git push`（由 Joe 决定）。
9. 风格：中文 docstring，`line-length = 100`，与现有模块的 dataclass + 纯函数 + 薄适配层风格一致。
10. 遇到与本任务包冲突的现实（如数据形态不符），**停下来报告**，不要自行改设计。

## 三、S2 提示词（可直接粘贴）

```
项目：F:\code\projects\Book-RAG（Python 3.11+，uv 管理，Windows/MSYS 环境）

先读这三个文件，再动手：
- docs/plans/book-rag-library-expansion-brief.md（尤其「已确认的设计决策」表与 S2 一节）
- AGENTS.md
- src/book_rag/rag/index.py、src/book_rag/rag/retriever.py、src/book_rag/servers/mcp_server.py

本阶段只做 S2：元数据 + workspace。要求：

1. 新增 books/workspaces.yaml 作为元数据唯一真源（库根），并在仓库根放一份
   workspaces.example.yaml 说明 schema（因为 books/ 在 .gitignore 内）。
   schema 要点：workspaces.<id>.{label,genre}；books.<相对 md 路径>.{workspaces[],tags[]}。
   genre 取值 fiction / knowledge / mixed / unknown。
2. 扩展 BookIndexMeta：新增 workspaces: list[str] / genre: str / tags: list[str]，
   旧 meta.json 缺这些字段时必须能正常加载（默认 ["_unclassified"] / "unknown" / []）。
3. 未在 yaml 中声明的书 → workspaces=["_unclassified"]、genre="unknown"，且必须在
   list_indexed_books 输出里显式可见，禁止静默归类。
4. 改 yaml 后不重建 embedding 即可生效：元数据写入独立于向量索引（可在刷新时重写 meta.json）。
5. 检索与列举加 workspace / genre 过滤参数，MCP 工具与 Web API 同步暴露；
   Web 首页加 workspace 切换。
6. 一本书属多个 workspace 时，检索结果按 (source, position) 去重，workspaces 合并展示。
7. 配测试：yaml 解析、未声明书归 _unclassified、多 workspace 去重、meta 向后兼容、
   过滤参数生效。全量 pytest 与 ruff 必须通过。

禁止：改 S3 的检索算法、改章节切分、动 books/_originals、重建索引。
完成后报告：改动文件、命令与真实输出、未验证项。
```

## 四、S3 提示词（S2 验收通过后再用）

```
本阶段只做 S3：Hybrid 检索 + rerank + 索引缓存。

硬约束（来自已拍板决策，不得更改）：
- 向量模型统一 BAAI/bge-m3 单模型（语言路由降级为元数据，不再决定 embedding 模型）。
- 融合流水线：每本 向量 top N ‖ BM25 top N → RRF(k=60) 全局融合 → cross-encoder
  对融合后的候选集**只重排一次** → top_k。
- 严禁「每本书各自 rerank 再合并」（延迟会随书数线性增长）。
- rerank 模型 BAAI/bge-reranker-v2-m3，默认开启，支持 BOOK_RAG_RERANK=0 关闭。
- 新增依赖仅限 jieba + rank-bm25（uv add）。
- BM25 索引随书落盘，增量同步时重建；进程内缓存 (faiss, meta, bm25)，按 mtime/hash 失效。
- get_section_content 必须加长章保护：超过 15000 字时返回「前 N 字 + 偏移续取」能力，
  不得一次吐出整章（《工作、消费主义和新穷人》第四章 23,426 字就是用例）。

完成后需给出实测数据：5 本书、top_k=5 的单次延迟（开/关 rerank）、精确术语与小说人名的
检索命中样例、缓存命中证据。索引重建前先问 Joe。
```

## 五、待 Joe 确认（阻塞 S2）

1. **workspace 划分**（Hermes 建议，待拍板）：
   - `fiction`：杀死一只知更鸟
   - `finance`：投资最重要的事、诺贝尔经济学奖经典收藏版
   - `social`：工作、消费主义和新穷人
   - `life`：斯坦福大学人生设计课
2. **执行方式**：由 Joe 自己在 Claude Code 客户端发起，还是由 Hermes 通过 CLI 驱动（需逐阶段明确授权）。
3. 合集《诺贝尔经济学奖经典收藏版》genre 记为 `knowledge`（内含三本经济学著作），是否可接受。
