# Book-RAG MCP

接入各种 AI Agent 的**书籍 RAG** MCP 服务。基于 **FastMCP + LangChain**，把本地书籍库（PDF / EPUB / TXT / Markdown）变成可检索的知识源，通过标准 MCP 协议暴露给任何支持 MCP 的 Agent（Claude Code / Codex / Cursor / Hermes / 自研 Agent 等）。

## 特性

- 📚 多格式书籍解析：PDF / EPUB / TXT / Markdown
- 🔍 语义检索：LangChain + sentence-transformers 本地嵌入 + FAISS 向量库
- 🔌 标准 MCP：`stdio` + `HTTP/SSE` 双传输，一处开发处处接入
- 🧩 Agent 无关：框架无关的 RAG 内核 + 薄 MCP 适配层，方便扩展其他协议

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

# 3. 启动 MCP 服务 (stdio, 供 MCP 客户端接入)
uv run book-rag-mcp

# 或 HTTP/SSE 传输
uv run book-rag-mcp --transport http --host 127.0.0.1 --port 8000
```

## 接入 Agent

以 Claude Code 为例 (`.mcp.json`)：

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

## 项目结构

```
Book-RAG/
├── pyproject.toml          # 项目配置与依赖
├── src/book_rag/
│   ├── __init__.py
│   ├── cli.py              # 命令行入口
│   ├── config.py           # 配置 (书籍库路径、嵌入模型等)
│   ├── io/                 # 文档解析层
│   │   ├── __init__.py
│   │   ├── loader.py       # 各格式加载器 (pdf/epub/txt/md)
│   │   └── parsers.py      # 底层解析实现
│   ├── rag/                # RAG 内核
│   │   ├── __init__.py
│   │   ├── embeddings.py   # 嵌入模型封装
│   │   ├── vector_store.py # 向量库 (FAISS)
│   │   ├── splitter.py     # 文本切分
│   │   └── retriever.py    # 检索
│   └── servers/            # MCP 适配层
│       ├── __init__.py
│       └── mcp_server.py   # FastMCP 服务与工具定义
├── tests/                  # 单元测试
└── docs/                   # 文档
```

## License

MIT