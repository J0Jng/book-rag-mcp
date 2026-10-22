# Book-RAG MCP

接入各种 AI Agent 的**书籍 RAG** MCP 服务。基于 **FastMCP + LangChain**，把本地书籍库（EPUB / TXT / Markdown）变成可检索的知识源，通过标准 MCP 协议暴露给任何支持 MCP 的 Agent（Claude Code / Codex / Cursor / Hermes / 自研 Agent 等）。

> **当前状态：MVP + 结构层 / 摘要层已完成**。支持本地 EPUB / TXT / Markdown 解析、front/back matter 清洗、自动语言检测、本地 HuggingFace embedding + FAISS 索引、增量同步、目录树（TOC）与章节定位检索、可选的离线 LLM 章节摘要 + 概念索引，通过 MCP 工具返回书中相关原文片段。

## 特性

- 📚 多格式书籍解析：EPUB / TXT / Markdown（只收能直接读取文字的格式，不支持 PDF）
- 🧹 向量化前清洗：规则化剥离推荐序 / 版权页 / 致谢等 front/back matter，审计记录落 `cleaning_report.json`
- 🔍 语义检索：LangChain + sentence-transformers **本地**嵌入 + FAISS 向量库（模型首次下载后完全离线）
- 🌐 自动语言路由：中文 / 英文 / 多语言自动选择对应 embedding 模型
- 🗺️ 结构层：`toc.json` 目录树 + 章节内精检索 + 整章原文拉取（纯结构化，零 LLM）
- 🧠 摘要 / 概念层：索引时离线抽取章节摘要、框架、术语、心智模型（需配置 LLM，未配置自动降级）
- 🔄 增量索引：启动自动加载，自动检测目录变化并增量更新
- 🔌 标准 MCP：`stdio` + `HTTP/SSE` 双传输，一处开发处处接入
- 🧩 Agent 无关：框架无关的 RAG 内核 + 薄 MCP 适配层，方便扩展其他协议
- 🖥️ 本地 Web 界面：无需任何 Agent，浏览器里批量转换书籍 + 语义搜索查看

> 以上能力均已实现；未做 / 待设计的事项见 [docs/DESIGN-TODO.md](docs/DESIGN-TODO.md)。

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

> **完全本地**：embedding 用 sentence-transformers 在本地加载 HuggingFace 模型（`BAAI/bge-large-zh-v1.5` / `-en-v1.5` / `bge-m3`），检索路径不发网络请求；模型缓存后可用 `HF_HUB_OFFLINE=1` 强制离线。只有**可选的**摘要/概念层会调 LLM，且可完全不配。

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
| `get_section_content` | 拉取某章节完整原文（按阅读顺序聚合） |
| `search_book_knowledge` | 全局按内容检索，返回原文片段 + 相关度 + 来源 |
| `retrieve_concepts` | 检索框架/术语/心智模型条目（需离线概念索引，未建时返回提示） |
| `refresh_book_index` | 增量同步书籍库索引 |

## 已知限制与不做的事

- **不支持 PDF**：只收能直接读取文字的格式（EPUB / TXT / Markdown）。PDF 尤其扫描版需要 OCR 与版面还原，属于独立问题域，见 `docs/ROADMAP.md`。
- **不做生成**：Book-RAG 是无状态检索服务，只返回原文片段，循环思考与答案生成由接入 Agent 负责。
- **引用粒度是章节 + position，没有页码**：EPUB / TXT / MD 本身无稳定页码，能定位到章节与段落即可满足需求。
- **概念 / 摘要层需要 LLM**：未配置 `BOOK_RAG_LLM_API_KEY` 时索引照常构建，`retrieve_concepts` 返回「概念索引未构建」提示（不报错、不编造）。
- **单一书籍库**：一个 `BOOK_RAG_LIBRARY` 对应一个库，暂无多 workspace 分组。
- **未做的方向**：hybrid 检索、rerank、跨书综合 / 主题聚类都在 [docs/DESIGN-TODO.md](docs/DESIGN-TODO.md) 里挂着待设计。

## 开发与验证

```bash
uv sync --extra dev
uv run pytest              # 快速层：155 个用例，不下载模型
uv run pytest -m integration   # 效果层：真实 bge 模型的语义检索质量（模型未缓存自动 skip）
uv run ruff check src tests
```

CI 见 `.github/workflows/ci.yml`（push / PR 跑 ruff + 快速层 pytest）。

## 项目结构

```
Book-RAG/
├── pyproject.toml          # 项目配置与依赖 [已就绪]
├── README.md               # 本文档 [已就绪]
├── IDEA.md                 # 项目灵感说明 [已就绪]
├── docs/plans/             # 实现任务书 [已就绪]
├── src/book_rag/
│   ├── __init__.py         # 包声明 + 版本 [已就绪]
│   ├── cli.py              # 命令行入口 [已就绪]
│   ├── config.py           # 全局配置 [已就绪]
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