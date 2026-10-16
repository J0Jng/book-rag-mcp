# Book-RAG MCP

接入各种 AI Agent 的**书籍 RAG** MCP 服务。基于 **FastMCP + LangChain**，把本地书籍库（PDF / EPUB / TXT / Markdown）变成可检索的知识源，通过标准 MCP 协议暴露给任何支持 MCP 的 Agent（Claude Code / Codex / Cursor / Hermes / 自研 Agent 等）。

> **当前状态：MVP 已实现**。支持本地多格式书籍解析、自动语言检测、本地 HuggingFace embedding + FAISS 索引、增量同步与按内容检索，通过 MCP 工具返回书中相关原文片段。

## 特性

- 📚 多格式书籍解析：PDF / EPUB / TXT / Markdown
- 🔍 语义检索：LangChain + sentence-transformers 本地嵌入 + FAISS 向量库
- 🌐 自动语言路由：中文 / 英文 / 多语言自动选择对应 embedding 模型
- 🔄 增量索引：启动自动加载，自动检测目录变化并增量更新
- 🔌 标准 MCP：`stdio` + `HTTP/SSE` 双传输，一处开发处处接入
- 🧩 Agent 无关：框架无关的 RAG 内核 + 薄 MCP 适配层，方便扩展其他协议
- 🖥️ 本地 Web 界面：无需任何 Agent，浏览器里批量转换书籍 + 语义搜索查看

> 以上能力均在 MVP 中已实现。

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
# 1. 安装依赖
uv sync

# 2. 配置书籍库路径 (环境变量)
export BOOK_RAG_LIBRARY="C:/path/to/books"

# 3. 构建/刷新索引（一次性）
uv run book-rag-mcp index

# 4. 启动本地 Web 界面（浏览器打开 http://127.0.0.1:8000）
uv run book-rag-mcp web

# 5. 启动 MCP 服务 (stdio, 供 MCP 客户端接入)
uv run book-rag-mcp

# 或 HTTP/SSE 传输
uv run book-rag-mcp --transport http
```

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
│   │   └── parsers.py      # PDF/EPUB/TXT/MD 解析器 [已就绪]
│   ├── rag/                # RAG 内核 [已就绪]
│   │   ├── __init__.py
│   │   ├── embeddings.py   # embedding 模型缓存与路由 [已就绪]
│   │   ├── language.py     # 语言检测与模型选择 [已就绪]
│   │   ├── index.py        # FAISS 索引构建与增量同步 [已就绪]
│   │   ├── retriever.py    # 跨书检索 [已就绪]
│   │   └── splitter.py     # 文本切分 [已就绪]
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