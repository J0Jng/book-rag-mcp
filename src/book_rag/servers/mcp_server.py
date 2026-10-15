"""FastMCP 服务：把书籍知识检索暴露为 MCP 工具。

设计要点（见 docs/plans/book-rag-mvp-impl-brief.md 文件清单 10）：
- 全局配置由 ``get_config()`` 从环境变量惰性初始化，便于测试替换；
- ``create_mcp`` 在创建服务时调用一次 ``refresh_all_indexes`` 做增量同步，
  保证服务启动后索引与书籍库一致；
- 对外暴露六个工具：search_book_knowledge / list_indexed_books / refresh_book_index，
  以及结构层的 get_book_structure / search_within_section / get_section_content。

工具的检索 / 列举 / 刷新逻辑以模块级函数实现并显式接受 ``config``，
FastMCP 注册的只是薄封装——测试无需启动服务即可覆盖全部行为。
"""

from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP

from book_rag.config import BookRagConfig
from book_rag.rag.index import (
    RefreshResult,
    get_book_index_dir,
    load_book_index,
    refresh_all_indexes,
)
from book_rag.rag.retriever import (
    SearchResult,
    _list_indexed_metas,
    _matches_book,
    search_books,
)
from book_rag.rag.structure import load_structure, matches_section

# 模块级全局配置，由 get_config() 惰性初始化
_CONFIG: BookRagConfig | None = None


def get_config() -> BookRagConfig:
    """返回全局配置；首次调用时从环境变量构建。"""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = BookRagConfig.from_env()
    return _CONFIG


def format_search_results(results: list[SearchResult], query: str = "") -> str:
    """把检索结果格式化为 Markdown 文本（含书名、章节、相关度、原文片段）。"""
    if not results:
        return f"未找到与查询「{query}」相关的内容。" if query else "未找到相关内容。"

    lines = [f"找到 {len(results)} 条相关片段：", ""]
    for i, r in enumerate(results, start=1):
        lines.append(f"### {i}. {r.book} · {r.section}（相关度 {r.score:.4f}）")
        lines.append("")
        lines.extend(f"> {line}" for line in r.content.splitlines())
        lines.append("")
    return "\n".join(lines).rstrip()


def format_refresh_result(result: RefreshResult) -> str:
    """把增量同步统计格式化为文本。"""
    parts = [
        (
            f"索引同步完成：新建/重建 {len(result.built)} 本，"
            f"跳过 {len(result.skipped)} 本，删除 {len(result.deleted)} 本。"
        )
    ]
    if result.built:
        parts.append("新建/重建：" + "、".join(result.built))
    if result.skipped:
        parts.append("跳过：" + "、".join(result.skipped))
    if result.deleted:
        parts.append("删除：" + "、".join(result.deleted))
    return "\n".join(parts)


def search_book_knowledge(
    query: str,
    *,
    config: BookRagConfig,
    top_k: int = 5,
    book: str | None = None,
    search_all_languages: bool = False,
) -> str:
    """检索书籍知识并返回 Markdown 格式的原文片段。

    Args:
        query: 查询文本。
        config: 全局配置。
        top_k: 最多返回的结果条数。
        book: 可选书籍过滤，支持书名 / 路径的部分匹配。
        search_all_languages: 为 True 时跨语言搜索全部书籍。
    """
    results = search_books(
        query,
        config=config,
        top_k=top_k,
        book=book,
        search_all_languages=search_all_languages,
    )
    return format_search_results(results, query=query)


def list_indexed_books(*, config: BookRagConfig) -> list[str]:
    """列出已索引书籍，每项形如 ``"书名 (语言: zh, 块数: 123)"``。"""
    metas = _list_indexed_metas(config.index_dir)
    return [
        f"{meta.book_path} (语言: {meta.language}, 块数: {meta.chunk_count})"
        for meta in metas
    ]


def refresh_book_index(*, config: BookRagConfig) -> str:
    """执行一次增量同步并返回统计文本。"""
    return format_refresh_result(refresh_all_indexes(config))


def _find_matching_metas(config: BookRagConfig, book: str):
    """按 book 过滤参数（部分匹配）返回匹配书籍的元数据列表。"""
    return [m for m in _list_indexed_metas(config.index_dir) if _matches_book(m, book)]


def _format_toc_markdown(toc: dict) -> str:
    """把目录树渲染为 Markdown 缩进列表（含各级章节名）。"""
    lines: list[str] = []

    def walk(node: dict, depth: int) -> None:
        for child in node.get("children", []):
            lines.append("  " * depth + f"- {child['name']}")
            walk(child, depth + 1)

    walk(toc, 0)
    return "\n".join(lines)


def get_book_structure(book: str, *, config: BookRagConfig) -> str:
    """返回匹配书籍的目录树（Markdown 缩进列表），数据来自 toc.json，无 LLM。"""
    metas = _find_matching_metas(config, book)
    if not metas:
        return f"未找到匹配的书籍：{book}"

    parts: list[str] = []
    for meta in metas:
        idx_dir = get_book_index_dir(config.index_dir, Path(meta.book_path))
        toc = load_structure(idx_dir)
        if toc is None:
            parts.append(
                f"### {meta.book_path}\n\n该书索引尚未构建目录树（toc.json 缺失，请重建索引）。"
            )
            continue
        parts.append(f"### {meta.book_path} 的目录树\n\n{_format_toc_markdown(toc)}")
    return "\n\n".join(parts)


def search_within_section(
    query: str,
    book: str,
    section: str,
    *,
    config: BookRagConfig,
    top_k: int = 5,
) -> str:
    """在指定书籍的指定章节内精检索，返回 Markdown 格式的原文片段。"""
    results = search_books(
        query, config=config, top_k=top_k, book=book, section=section
    )
    return format_search_results(results, query=query)


def get_section_content(book: str, section: str, *, config: BookRagConfig) -> str:
    """按 section 拉取完整原文：聚合该章节下所有 chunk，按 position 排序拼接。"""
    metas = _find_matching_metas(config, book)
    if not metas:
        return f"未找到匹配的书籍：{book}"
    meta = metas[0]

    loaded = load_book_index(config.index_dir, Path(meta.book_path))
    if loaded is None:
        return f"书籍索引不存在：{meta.book_path}"
    vector_store, _ = loaded

    chunks = [
        doc
        for doc in vector_store.docstore._dict.values()
        if matches_section(doc.metadata.get("section", ""), section)
    ]
    if not chunks:
        return (
            f"未找到章节「{section}」对应的原文（书籍：{meta.book_path}）。"
            f"可先用 get_book_structure 查看目录树确认章节名。"
        )

    chunks.sort(key=lambda d: d.metadata.get("position", 0))
    lines = [f"### {meta.book_path} · {section}（完整原文，共 {len(chunks)} 段）", ""]
    lines.extend(chunk.page_content for chunk in chunks)
    return "\n".join(lines)


# 工具封装（create_mcp 内的同名 def）会遮蔽模块级实现，提前捕获引用
_search_impl = search_book_knowledge
_list_books_impl = list_indexed_books
_refresh_impl = refresh_book_index
_structure_impl = get_book_structure
_within_section_impl = search_within_section
_section_content_impl = get_section_content


def create_mcp(config: BookRagConfig | None = None) -> FastMCP:
    """创建并配置 FastMCP 服务实例。

    创建时先执行一次增量同步（服务启动即与书籍库保持一致），再注册三个工具。
    """
    config = config if config is not None else get_config()
    refresh_all_indexes(config)

    mcp = FastMCP("book-rag")

    @mcp.tool
    def search_book_knowledge(
        query: str,
        top_k: int = 5,
        book: str | None = None,
        search_all_languages: bool = False,
    ) -> str:
        """按内容检索书籍知识，返回书中相关原文片段（Markdown 格式）。

        Args:
            query: 查询文本。
            top_k: 最多返回的结果条数。
            book: 可选书籍过滤，支持书名 / 路径的部分匹配。
            search_all_languages: 为 True 时跨语言搜索全部书籍，
                默认只搜索与查询同语言的书籍。
        """
        return _search_impl(
            query,
            config=config,
            top_k=top_k,
            book=book,
            search_all_languages=search_all_languages,
        )

    @mcp.tool
    def list_indexed_books() -> list[str]:
        """列出所有已建立索引的书籍及其语言与块数。"""
        return _list_books_impl(config=config)

    @mcp.tool
    def refresh_book_index() -> str:
        """增量同步书籍库索引：新书建索引、变更重建、已删除的清理。"""
        return _refresh_impl(config=config)

    @mcp.tool
    def get_book_structure(book: str) -> str:
        """返回书籍的目录树（Markdown 缩进列表，含各级章节名），用于先定位章节再精检索。

        Args:
            book: 书籍过滤参数，支持书名 / 路径的部分匹配。
        """
        return _structure_impl(book, config=config)

    @mcp.tool
    def search_within_section(query: str, book: str, section: str, top_k: int = 5) -> str:
        """在指定书籍的指定章节内检索，返回相关原文片段（Markdown 格式）。

        Args:
            query: 查询文本。
            book: 书籍过滤参数，支持书名 / 路径的部分匹配。
            section: 章节过滤参数，章节路径前缀匹配（如「第一章」命中「第一章 起点」）。
            top_k: 最多返回的结果条数。
        """
        return _within_section_impl(query, book, section, config=config, top_k=top_k)

    @mcp.tool
    def get_section_content(book: str, section: str) -> str:
        """拉取指定书籍某章节的完整原文（按阅读顺序聚合，内容全部来自书籍本身）。

        Args:
            book: 书籍过滤参数，支持书名 / 路径的部分匹配。
            section: 章节路径（支持前缀匹配），可先用 get_book_structure 确认章节名。
        """
        return _section_content_impl(book, section, config=config)

    return mcp
