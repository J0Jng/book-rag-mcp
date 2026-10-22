"""Book-RAG MCP — 接入各种 AI Agent 的书籍 RAG 服务。

MVP 已实现：支持本地 EPUB / TXT / Markdown 书籍解析、
自动语言检测与 embedding 路由、FAISS 索引构建与增量同步、
按内容检索并通过 MCP 工具返回书中相关原文片段。
"""

__version__ = "0.1.0"