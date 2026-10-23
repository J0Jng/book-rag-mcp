# Book-RAG MCP

接入各种 AI Agent 的**书籍 RAG** MCP 服务。基于 **FastMCP + LangChain**，把本地书籍库（EPUB / TXT / Markdown）变成可检索的知识源，通过标准 MCP 协议暴露给任何支持 MCP 的 Agent（Claude Code / Codex / Cursor / Hermes / 自研 Agent 等）。

> **当前阶段：S1–S6 已实现并完成阶段验证。** 五本 Markdown 共 3,257 个 chunk、469 条章节摘要、2,908 条知识概念、8 个主题均已生成。服务端不生成回答；单层主题聚类保留混合/边缘成员，未实现跨书主题对齐。实测与局限见 [S4–S6 验证报告](docs/reports/s4-s6-acceptance.md)。

## 特性

- 📚 多格式书籍解析：EPUB / TXT / Markdown（只收能直接读取文字的格式，不支持 PDF）
- 📥 **Markdown 归一**：`convert` 子命令把 EPUB / TXT 转成带章节标题的 Markdown（`books/library/`），
  章节结构固化在标题层级里，原始文件归档到 `books/_originals/`；索引层只走一条 Markdown 解析路径
- 🧹 向量化前清洗：规则化剥离推荐序 / 版权页 / 致谢等 front/back matter，审计记录落 `cleaning_report.json`
- 🔍 语义检索：LangChain + sentence-transformers **本地**嵌入 + FAISS 向量库（模型首次下载后完全离线）
- 🌐 统一向量模型：所有语言使用 `BAAI/bge-m3`，语言保留为展示和过滤元数据
- 🔎 Hybrid：每书向量与 BM25 候选 → 全局 RRF(k=60) → 单次 `BAAI/bge-reranker-v2-m3` 重排；`BOOK_RAG_RERANK=0` 关闭重排
- 🗂️ workspace：库根 `workspaces.yaml` 管理多对多分组、genre 和 tags，修改元数据不重新 embedding
- 🗺️ 结构层：目录树 + 章节内精检索 + 原文快照读取；正文单次最多 15,000 字，使用 offset 续取
- 🧠 体裁摘要：小说剧情/人物/转折/悬念，知识类框架/术语/心智模型；可断点补建，未配置 LLM 自动降级
- 📖 多书检索：`books` 列表与 `group_by_book` 分书取材，原有 `book` 字符串兼容，服务端不综合
- 🧭 离线主题：复用已有向量的 KMeans + 代表片段命名；只读主题检索按书返回原文，明确标注单层聚类局限
- 🔄 增量索引：启动自动加载，自动检测目录变化并增量更新
- 🔌 标准 MCP：`stdio` + `HTTP/SSE` 双传输，一处开发处处接入
- 🧩 Agent 无关：框架无关的 RAG 内核 + 薄 MCP 适配层，方便扩展其他协议
- 🖥️ 本地 Web 界面：无需任何 Agent，浏览器里批量转换书籍 + 语义搜索查看

> 真实检索、延迟、摘要、跨书与主题证据见 `docs/reports/`，原定范围见 [扩充计划](docs/plans/book-rag-library-expansion-brief.md)。

## 架构

```
        ┌─────────────┐
        │   AI Agent   │   (Claude Code / Codex / Cursor / ...)
        └──────┬──────┘
               │ MCP 协议 (stdio / HTTP-SSE)
        ┌──────┴──────┐
        │  MCP Layer   │  FastMCP 暴露工具
        └──────┬──────┘
        ┌──────┴──────┐
        │   RAG 内核   │  LangChain: 解析→切分→嵌入→检索
        └──────┬──────┘
        ┌──────┴──────┐
        │  书籍库目录  │  解析索引 (向量库)
        └─────────────┘
```

## 快速开始

```bash
# 1. 安装依赖（含测试与 ruff）
uv sync --extra dev

# 2. 配置书籍库路径 (环境变量)
export BOOK_RAG_LIBRARY="C:/path/to/books"

# 2. 书籍归一：EPUB / TXT → books/library/*.md（原始文件归档到 books/_originals/）
#    已在 books/library/ 的 md 可跳过这一步；--copy-originals 可改为复制而非移动
uv run book-rag-mcp convert --source "C:/path/to/incoming_books"

# 2.1（可选）配置离线摘要 / 概念层用的 LLM；不配置则自动降级，TOC 与检索照常可用
export BOOK_RAG_LLM_API_KEY="ark-xxx"
export BOOK_RAG_LLM_BASE_URL="https://ark.cn-beijing.volces.com/api/coding/v3"
export BOOK_RAG_LLM_MODEL="DeepSeek-V4-Pro"

# 3. 构建/刷新索引（一次性；embedding 模型本地加载，首次需联网下载一次）
uv run book-rag-mcp index

# 4. 启动本地 Web 界面（浏览器打开 http://127.0.0.1:8000）
uv run book-rag-mcp web

# 5. 启动 MCP 服务 (stdio, 供 MCP 客户端接入)
uv run book-rag-mcp

# 或 HTTP/SSE 传输
uv run book-rag-mcp --transport http
```

> embedding 与 reranker 均在本地运行；首次使用需下载两份模型，缓存后可设置 `HF_HUB_OFFLINE=1` 强制离线。只有可选的摘要/概念层调用在线 LLM。reranker 不可用时记录原因并回退 RRF，该降级不能算作 rerank 验收通过。

CPU reranker 默认对同一模型的 Linear 层使用动态 INT8，并将 PyTorch 线程数设为 `min(12, CPU 数)`；CUDA 保持原生精度。可设置 `BOOK_RAG_RERANK_INT8=0` 保留 FP32，或用正整数 `BOOK_RAG_RERANK_CPU_THREADS` 调整线程数。这会影响进程共享的 PyTorch CPU 线程池。INT8 的排序效果与延迟须以真实书库验收为准；本机 FP32 未达到 5 秒预算。

## 本地 Web 界面（无需 Agent）

不需要任何 Agent，直接在浏览器里操作：

```bash
uv run book-rag-mcp web   # 默认 http://127.0.0.1:8000，可用 --host / --port 调整
```

- 📚 **书籍库 / 批量转换**：列出库内书籍与索引状态，一键转换建索引，并展示新建 / 跳过 / 删除统计。
- 🔍 **数据搜索**：语义检索 query，支持 top_k、书名过滤与跨语言搜索，返回带相关度与原文片段的卡片。
- 📖 **已索引书籍**：查看每本书的语言、块数与最后更新时间。

Web 界面与 MCP 服务共享同一套 RAG 内核（解析 → 切分 → embedding → FAISS 索引），只是换了一个浏览器入口；前端为单文件（无外部依赖），由 Starlette 提供。

## 接入 Agent

### Hermes (`~/.hermes/config.yaml`)

```yaml
mcp_servers:
  book-rag:
    command: "uv"
    args: ["--directory", "F:/code/projects/Book-RAG", "run", "book-rag-mcp"]
    env:
      BOOK_RAG_LIBRARY: "C:/path/to/books"
```

### Claude Code (`.mcp.json`)

```json
{
  "mcpServers": {
    "book-rag": {
      "command": "uv",
      "args": ["--directory", "F:/code/projects/Book-RAG", "run", "book-rag-mcp"],
      "env": { "BOOK_RAG_LIBRARY": "C:/path/to/books" }
    }
  }
}
```

编排纪律详见 `AGENTS.md`（接入 Agent 会自动注入 instructions）。

## 暴露的 MCP 工具

| 工具名 | 说明 |
|---|---|
| `list_indexed_books` | 列出已索引书籍及语言、块数 |
| `get_book_structure` | 返回书籍目录树（Markdown 缩进列表），先定位章节 |
| `search_within_section` | 在指定章节内精检索原文片段 |
| `get_section_content` | 读取章节原文快照，每次最多 15,000 字，按 offset 续取 |
| `search_book_knowledge` | 全局按内容检索，返回原文片段 + 相关度 + 来源 |
| `retrieve_concepts` | 仅知识类：检索框架/术语/心智模型条目（需离线概念索引，未建时返回提示） |
| `get_book_summaries` | 只读小说梗概或知识摘要及原文指针；按 section 过滤、offset 续取 |
| `search_by_topic` | 按主题名/描述或 topic-ID 返回分书原文成员；不代表跨书主题已对齐 |
| `refresh_book_index` | 增量同步书籍库索引 |

## 已知限制与不做的事

- **不支持 PDF**：只收能直接读取文字的格式（EPUB / TXT / Markdown）。PDF 尤其扫描版需要 OCR 与版面还原，属于独立问题域，见 `docs/ROADMAP.md`。
- **不生成回答**：Book-RAG 是无状态检索服务，循环思考与答案生成由接入 Agent 负责；可选离线摘要用于定位章节，引用前须核对原文。
- **引用粒度是章节 + position，没有页码**：EPUB / TXT / MD 本身无稳定页码，能定位到章节与段落即可满足需求。
- **概念 / 摘要层需要 LLM**：未配置 `BOOK_RAG_LLM_API_KEY` 时索引照常构建，`retrieve_concepts` 返回「概念索引未构建」提示（不报错、不编造）。
- **单一书籍库**：一个 `BOOK_RAG_LIBRARY` 对应一个库，库内可配置多个 workspace；未声明的书显式归入 `_unclassified`。
- **分数语义**：检索结果含 `score_kind=rrf/rerank`；分数只用于排序，不是概率，不使用统一 0.5 阈值。
- **主题粒度**：当前单层 KMeans 产生宽主题或混合主题，仍可能包含署名等边缘短片段；名称/描述仅概括代表片段，不能推导全簇统一观点。真正的跨书主题对齐需要后续二次聚类，本期不做。

## 开发与验证

```bash
uv sync --extra dev
uv run pytest              # 快速层：不下载模型
uv run pytest -m integration   # 效果层：真实 bge 模型的语义检索质量（模型未缓存自动 skip）
uv run ruff check src tests scripts

# S3 独立目录验收，输出真实证据；--build 可增量构建（禁用在线摘要）
uv run python scripts/verify_s3.py --index-dir books/.book_rag_index --output docs/reports/s3-evidence.json

# 显式补齐摘要，不重算正文向量；成功章节保存断点，失败返回非零
uv run book-rag-mcp summarize --book "杀死一只知更鸟"
uv run python scripts/verify_s4.py --build --regression

# 多书实际检索验证；显式离线主题构建和只读验证
uv run python scripts/verify_s5.py
uv run book-rag-mcp topics
uv run python scripts/verify_s6.py
```

摘要客户端读取 `BOOK_RAG_LLM_BASE_URL`、`BOOK_RAG_LLM_MODEL`、`BOOK_RAG_LLM_API_KEY`，
不自动读取其他应用的凭据文件。DeepSeek 官方接口可配置为 `https://api.deepseek.com` / `deepseek-flash`；
`BOOK_RAG_LLM_THINKING=disabled` 可关闭该提供方默认的推理模式，其他端点默认不发送此参数。
`BOOK_RAG_SUMMARY_WORKERS` 控制显式离线命令的并发（1–8，默认 4）。
小说用段落编号选择证据，再由代码映射回真实连续原文；人物名匹配只忽略排版空白，保存时恢复原文。
`summaries.pending.json` 保存断点，完整结果才发布到 `summaries.json`；
`summary_status.json` 记录完成数、书籍哈希和体裁，读取工具会拒绝过期或未完成的摘要。

多书 MCP 调用示例：`search_book_knowledge(query="风险与不确定性", books=["投资最重要的事", "诺贝尔经济学奖经典收藏版"], group_by_book=true)`。
`book` 与 `books` 互斥；空列表不会扩大为全库；`top_k` 至少等于语言/workspace/体裁过滤后的匹配书籍数。
Web `/api/search` 接受相同筛选参数，并在 `group_by_book=true` 时附加 `groups`，保留原有 `results`。

主题命令从本地 FAISS 读取全部向量并做 L2 归一化，比较候选 K 的余弦 silhouette 后选择最高者。
`topics.pending.json` 保存聚类/命名断点，完整结果发布到 `topics.json`；核心索引变动后拒绝旧主题。
`search_by_topic(query="风险", top_k=3, per_book=2)` 对主题名/描述做关键词匹配，或用 `topic-02` 等 ID 直取；
每个主题内按书返回距离中心最近的原文成员，不调用 LLM 或 embedding。

CI 见 `.github/workflows/ci.yml`（push / PR 跑 ruff + 快速层 pytest）。

库根可放 `jieba_dict.txt` 用户词典（如小说的实际译名），该文件不会作为书籍发现。
词典变更后刷新只重建 BM25，检索时也会按当前词典重建过期的内存 BM25。
章节原文保存在每书索引的 `sections.json`，与切分前正文一致；旧索引缺少快照时提示刷新。

## 项目结构

```
Book-RAG/
├── pyproject.toml          # 项目配置与依赖 [已就绪]
├── README.md               # 本文档 [已就绪]
├── IDEA.md                 # 项目灵感说明 [已就绪]
├── docs/plans/             # 实现任务书 [已就绪]
├── scripts/
│   └── inspect_library.py  # 书库结构体检（只读：章节/清洗/切分诊断）[已就绪]
├── src/book_rag/
│   ├── __init__.py         # 包声明 + 版本 [已就绪]
│   ├── cli.py              # 命令行入口 [已就绪]
│   ├── config.py           # 全局配置 [已就绪]
│   ├── convert/            # 书籍归一：EPUB/TXT → Markdown [已就绪]
│   │   ├── __init__.py
│   │   └── markdown.py     # 目录层级推断 + 正文提取 + 幂等落盘
│   ├── io/                 # 文档解析层 [已就绪]
│   │   ├── __init__.py
│   │   ├── loader.py       # 书籍发现、加载、hash [已就绪]
│   │   ├── parsers.py      # EPUB/TXT/MD 解析器 [已就绪]
│   │   ├── cleaning.py     # front/back matter 清洗层 [已就绪]
│   │   └── matter_titles.py# matter 标题词表 [已就绪]
│   ├── rag/                # RAG 内核 [已就绪]
│   │   ├── __init__.py
│   │   ├── embeddings.py   # embedding 模型缓存与路由 [已就绪]
│   │   ├── language.py     # 语言检测与模型选择 [已就绪]
│   │   ├── index.py        # FAISS 索引构建与增量同步 [已就绪]
│   │   ├── retriever.py    # 跨书检索 [已就绪]
│   │   ├── splitter.py     # 文本切分 [已就绪]
│   │   ├── structure.py    # TOC 目录树与章节匹配 [已就绪]
│   │   ├── llm.py          # 离线摘要用最小 LLM 客户端 [已就绪]
│   │   ├── summarizer.py   # 章节摘要抽取（可降级）[已就绪]
│   │   ├── summary_layer.py# 断点补建、完整发布与概念构建 [已就绪]
│   │   ├── topics.py       # 现有向量聚类、离线命名与主题原文检索 [已就绪]
│   │   └── concepts.py     # 概念索引构建与加载 [已就绪]
│   ├── servers/            # 服务适配层（MCP + Web）[已就绪]
│   │   ├── __init__.py
│   │   ├── mcp_server.py   # FastMCP 服务与工具定义 [已就绪]
│   │   └── web.py          # 本地 Web 服务（Starlette）[已就绪]
│   └── web/                # 前端静态页面 [已就绪]
│       └── index.html      # 单文件前端 [已就绪]
└── tests/                  # 单元测试（含 Web 端点测试）[已就绪]
```

## License

MIT
