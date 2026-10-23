"""Book-RAG MCP — 接入各种 AI Agent 的书籍 RAG 服务。

MVP 已实现：支持本地 EPUB / TXT / Markdown 书籍解析、
语言元数据与统一 bge-m3 embedding、FAISS/BM25 索引构建与增量同步、
Hybrid 融合与全局单次重排，通过 MCP 工具返回带来源的原文片段。
"""

__version__ = "0.1.0"
