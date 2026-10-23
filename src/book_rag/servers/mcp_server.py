"""FastMCP 服务：把书籍知识检索暴露为 MCP 工具。

设计要点（见 docs/plans/book-rag-mvp-impl-brief.md 文件清单 10）：
- 全局配置由 ``get_config()`` 从环境变量惰性初始化，便于测试替换；
- ``create_mcp`` 在创建服务时调用一次 ``refresh_all_indexes`` 做增量同步，
  保证服务启动后索引与书籍库一致；
- 对外暴露九个工具：search_book_knowledge / list_indexed_books / refresh_book_index，
  结构层的 get_book_structure / search_within_section / get_section_content，
  以及 retrieve_concepts / get_book_summaries / search_by_topic；
- ``create_mcp`` 通过 ``instructions=PLAYBOOK`` 向接入 Agent 注入编排纪律
  （playbook，与 AGENTS.md 正文一致），指导 Agent 按「先定位章节再精检索」
  的顺序使用工具。

工具的检索 / 列举 / 刷新逻辑以模块级函数实现并显式接受 ``config``，
FastMCP 注册的只是薄封装——测试无需启动服务即可覆盖全部行为。
"""

from __future__ import annotations

import json
from pathlib import Path

from fastmcp import FastMCP

from book_rag.config import BookRagConfig
from book_rag.rag.concepts import Concept, load_concept_index
from book_rag.rag.embeddings import get_embeddings
from book_rag.rag.index import (
    RefreshResult,
    get_book_index_dir,
    load_section_snapshot,
    refresh_all_indexes,
)
from book_rag.rag.retriever import (
    SearchResult,
    _list_indexed_metas,
    _matches_book,
    _matches_genre,
    _matches_workspace,
    _to_similarity,
    search_books,
)
from book_rag.rag.structure import load_structure, matches_section
from book_rag.rag.summarizer import SUMMARY_VERSION

# 编排纪律（playbook）：通过 FastMCP instructions 注入接入 Agent，
# 与项目根目录 AGENTS.md 正文逐字一致；内容见 docs/plans/book-rag-playbook-agents-brief.md
PLAYBOOK = """\
你接入了一个「书籍知识检索」服务（Book-RAG），回答关于某本书的问题时遵循以下纪律：

1. 先 list_indexed_books 确认书已索引，再 get_book_structure 拿目录树，判断问题落在哪个章节。
2. 定位到章节后，优先 search_within_section 在该章节内精检索；只有范围不明时才用全局的 search_book_knowledge。
3. 需要完整上下文时，用 get_section_content 拉取章节原文；超长章按返回的 offset 续取直到章末（内容全部来自书籍本身）。
4. 「核心框架 / 方法论 / 心智模型 / 术语定义」类问题优先 retrieve_concepts；小说剧情可用 get_book_summaries 定位，摘要必须回到原文核对后引用。
5. 根据原文是否切题判断检索质量；RRF 融合分与 rerank 分不可比较，也不是相关概率，不套用统一数值阈值。结果明显不相关时，改写查询词或换检索范围再试，最多 2 次——这是纠正式检索，不是放弃。
6. 跨章节的「概念关系 / 推导 / 对比」题，拆成多个子查询分别检索后再综合。跨书题先用 books 列表与 group_by_book 分书取材，再各自 get_book_structure 定位和精检索，最后由接入 Agent 自行综合；服务端不合成答案。
7. 生成答案必须基于检索到的原文，并标注来源（source / section / position）；检索不到就明说「书中未检索到」，绝不编造、绝不用通识补全。"""

#: get_section_content 单次返回的正文上限（字）。章级切分保留超长章，
#: 工具层按此上限截断并要求用 offset 续取，避免一次吐出整章撑爆上下文。
SECTION_CONTENT_LIMIT = 15000

# 模块级全局配置，由 get_config() 惰性初始化
_CONFIG: BookRagConfig | None = None


def get_config() -> BookRagConfig:
    """返回全局配置；首次调用时从环境变量构建。"""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = BookRagConfig.from_env()
    return _CONFIG


def format_search_results(
    results: list[SearchResult], query: str = "", *, group_by_book: bool = False,
) -> str:
    """把检索结果格式化为 Markdown 文本（含书名、章节、相关度、原文片段）。"""
    if not results:
        return f"未找到与查询「{query}」相关的内容。" if query else "未找到相关内容。"
    if group_by_book:
        groups = {}
        for result in results:
            groups.setdefault(result.source, []).append(result)
        return "\n\n".join(
            f"## {source}\n\n{format_search_results(items, query)}"
            for source, items in groups.items()
        )

    lines = [f"找到 {len(results)} 条相关片段：", ""]
    for i, r in enumerate(results, start=1):
        meta_bits = [f"相关度（{r.score_kind} 排序分，非概率）{r.score:.4f}"]
        if r.workspaces:
            meta_bits.append("workspace: " + ", ".join(r.workspaces))
        if r.genre:
            meta_bits.append(f"genre: {r.genre}")
        lines.append(f"### {i}. {r.book} · {r.section}（{' · '.join(meta_bits)}）")
        lines.append(f"来源 source: {r.source} · section: {r.section} · position: {r.position}")
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
    if result.metadata_updated:
        parts.append("仅更新元数据（未重建向量）：" + "、".join(result.metadata_updated))
    return "\n".join(parts)


def search_book_knowledge(
    query: str,
    *,
    config: BookRagConfig,
    top_k: int = 5,
    book: str | None = None,
    books: list[str] | None = None,
    group_by_book: bool = False,
    search_all_languages: bool = False,
    workspace: str | None = None,
    genre: str | None = None,
) -> str:
    """检索书籍知识并返回 Markdown 格式的原文片段。

    Args:
        query: 查询文本。
        config: 全局配置。
        top_k: 最多返回的结果条数。
        book: 可选书籍过滤，支持书名 / 路径的部分匹配。
        search_all_languages: 为 True 时跨语言搜索全部书籍。
        workspace: 可选 workspace 过滤；未声明书属于 ``_unclassified``。
        genre: 可选体裁过滤（fiction / knowledge / mixed / unknown）。
    """
    results = search_books(
        query,
        config=config,
        top_k=top_k,
        book=book,
        books=books,
        search_all_languages=search_all_languages,
        workspace=workspace,
        genre=genre,
    )
    return format_search_results(results, query=query, group_by_book=group_by_book)


def _workspace_sort_key(meta) -> tuple:
    """workspace 分组排序键：按首个 workspace 排序，``_unclassified`` 放最后。"""
    primary = meta.workspaces[0] if meta.workspaces else "_unclassified"
    return (primary == "_unclassified", primary, meta.book_path)


def list_indexed_books(
    *,
    config: BookRagConfig,
    workspace: str | None = None,
    genre: str | None = None,
) -> list[str]:
    """按 workspace 分组列出已索引书籍。

    每项形如 ``"[fiction] library/x.md (语言: zh, 块数: 123, genre: fiction, tags: ...)"``；
    未在 ``workspaces.yaml`` 声明的书显式标为 ``_unclassified``（排在最后）。
    """
    metas = _list_indexed_metas(config.index_dir, config.library_path)
    if workspace:
        metas = [m for m in metas if _matches_workspace(m, workspace)]
    if genre:
        metas = [m for m in metas if _matches_genre(m, genre)]
    metas.sort(key=_workspace_sort_key)

    items: list[str] = []
    for meta in metas:
        workspaces = ",".join(meta.workspaces) if meta.workspaces else "_unclassified"
        tags = ", ".join(meta.tags) if meta.tags else "-"
        items.append(
            f"[{workspaces}] {meta.book_path} "
            f"(语言: {meta.language}, 块数: {meta.chunk_count}, "
            f"genre: {meta.genre}, tags: {tags})"
        )
    return items


def refresh_book_index(*, config: BookRagConfig) -> str:
    """执行一次增量同步并返回统计文本。"""
    return format_refresh_result(refresh_all_indexes(config))


def _find_matching_metas(config: BookRagConfig, book: str):
    """按 book 过滤参数（部分匹配）返回匹配书籍的元数据列表。"""
    return [
        m
        for m in _list_indexed_metas(config.index_dir, config.library_path)
        if _matches_book(m, book)
    ]


def search_by_topic(
    query: str, *, config: BookRagConfig, top_k: int = 3, per_book: int = 2,
) -> str:
    """按离线主题名/描述检索，返回按书分组的原文成员，不调用 LLM。"""
    from book_rag.rag.topics import LIMITATION, TopicIndexError, search_topic_members

    try:
        topics = search_topic_members(query, config=config, top_k=top_k, per_book=per_book)
    except TopicIndexError as exc:
        return str(exc)
    if not topics:
        return "主题名/描述中未检索到匹配项；请改写主题词。\n" + LIMITATION
    lines = [LIMITATION, "主题名与描述仅概括代表片段；簇内可能有边缘成员，不代表所有书具有同一观点。",
             "以下成员片段才是书籍原文。", ""]
    for topic in topics:
        lines.extend([f"## {topic['id']} · {topic['name']}（{topic['count']} 个成员）",
                      topic["description"], ""])
        lines.append("全簇来源分布：" + "；".join(
            f"{source} ({count})" for source, count in topic["book_counts"].items()
        ))
        groups = {}
        for member in topic["members"]:
            groups.setdefault(member["source"], []).append(member)
        for source, members in groups.items():
            lines.append(f"### {source}")
            for member in members:
                lines.append(f"source: {source} · section: {member['section']} · position: {member['position']}")
                lines.extend("> " + line for line in member["content"].splitlines())
                if member["truncated"]:
                    lines.append("（该 chunk 仅显示前1200字；请按章节获取更多原文。）")
                lines.append("")
    return "\n".join(lines)


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


def get_book_summaries(
    book: str, *, config: BookRagConfig, section: str | None = None,
    offset: int = 0, limit: int = 5,
) -> str:
    """只读离线摘要；返回生成内容与原文指针，阅读时不调用 LLM。"""
    metas = _find_matching_metas(config, book)
    if not metas:
        return f"未找到匹配的书籍：{book}"
    if len(metas) > 1:
        return "匹配多本书，请使用完整 source：\n" + "\n".join(m.book_path for m in metas)
    meta = metas[0]
    idx_dir = get_book_index_dir(config.index_dir, Path(meta.book_path))
    try:
        status = json.loads((idx_dir / "summary_status.json").read_text(encoding="utf-8"))
        if (not status.get("complete") or status.get("book_hash") != meta.book_hash
                or status.get("genre") != meta.genre
                or status.get("summary_version") != SUMMARY_VERSION):
            return f"摘要未完整构建或已过期，请配置 LLM 后运行 book-rag-mcp summarize：{meta.book_path}"
        summaries = json.loads((idx_dir / "summaries.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return f"摘要未构建，请配置 LLM 后运行 book-rag-mcp summarize：{meta.book_path}"
    if section:
        summaries = [s for s in summaries if matches_section(s["section"], section)]
    if not summaries:
        return "未找到该章节摘要；可先用 get_book_structure 确认章节路径。"
    offset, limit = max(0, offset), max(1, min(20, limit))
    lines = [f"### {meta.book_path} · genre: {meta.genre}",
             "离线 LLM 摘要用于定位章节；答案须核对下列指针的原文。", ""]
    for item in summaries[offset:offset + limit]:
        lines.extend([f"#### {item['section']}", item["summary"]])
        if item["genre"] == "fiction":
            lines.append("剧情：" + item["plot"])
            for key, label in (("characters", "人物"), ("turning_points", "转折"),
                               ("suspense", "悬念")):
                lines.append(label + "：" + ("；".join(item[key]) or "[未检测到]"))
            lines.extend("> " + quote.replace("\n", "\n> ") for quote in item["evidence"])
        else:
            for key, description in (("frameworks", "description"), ("terms", "definition")):
                lines.extend(v["name"] + "：" + v[description] for v in item[key])
            lines.extend(item["mental_models"])
        lines.extend(f"source: {p['source']} · section: {p['section']} · position: {p['position']}"
                     for p in item["pointers"])
        lines.append("")
    end = min(offset + limit, len(summaries))
    lines.append(f"摘要 {min(offset, len(summaries))}–{end} / {len(summaries)}")
    if end < len(summaries):
        lines.append(f"请用 offset={end} 续取。")
    return "\n".join(lines)


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


def get_section_content(
    book: str,
    section: str,
    *,
    config: BookRagConfig,
    offset: int = 0,
    limit: int = SECTION_CONTENT_LIMIT,
) -> str:
    """按 section 拉取原文：聚合该章节下所有 chunk，超长时截断并支持 offset 续取。

    章级切分下可能存在数万字的超长章（如 23,426 字的单章），一次返回整章会
    撑爆 Agent 上下文。因此正文超过 ``limit``（默认 15,000 字）时只返回
    ``[offset, offset+limit)`` 区间，并明确提示下一次续取的 offset。

    Args:
        book: 书籍过滤参数（部分匹配）。
        section: 章节路径（前缀匹配）。
        config: 全局配置。
        offset: 起始字符偏移（0 起），用于分页续取。
        limit: 单次返回的最大正文字数。
    """
    metas = _find_matching_metas(config, book)
    if not metas:
        return f"未找到匹配的书籍：{book}"
    meta = metas[0]

    sections = load_section_snapshot(
        get_book_index_dir(config.index_dir, Path(meta.book_path))
    )
    if sections is None:
        return f"章节原文快照未构建，请刷新索引：{meta.book_path}"

    chunks = [
        entry for entry in sections if matches_section(entry["path"], section)
    ]
    if not chunks:
        return (
            f"未找到章节「{section}」对应的原文（书籍：{meta.book_path}）。"
            f"可先用 get_book_structure 查看目录树确认章节名。"
        )

    chunks.sort(key=lambda entry: entry["position"])
    full_text = "\n".join(entry["text"] for entry in chunks)
    total = len(full_text)
    offset = max(0, int(offset))
    limit = max(1, min(SECTION_CONTENT_LIMIT, int(limit)))

    if offset >= total:
        return (
            f"### {meta.book_path} · {section}（原文共 {total} 字）\n\n"
            f"offset={offset} 已到达或超出本章末尾，无需继续续取。"
        )

    end = min(offset + limit, total)
    lines = [
        (
            f"### {meta.book_path} · {section}"
            f"（完整原文共 {total} 字，本次返回第 {offset + 1}–{end} 字）"
        ),
        "",
        full_text[offset:end],
    ]
    if end < total:
        lines.extend(
            [
                "",
                (
                    f"（内容已截断：本章共 {total} 字，单次上限 {limit} 字。"
                    f"如需后续内容，请用 offset={end} 续取。）"
                ),
            ]
        )
    elif offset > 0:
        lines.extend(["", "（已返回本章末尾。）"])
    return "\n".join(lines)


# 概念类型的中文标签（Markdown 输出用）
_KIND_LABELS = {"framework": "框架", "term": "术语", "mental_model": "心智模型"}


def _format_concept_hit(index: int, concept, score: float) -> str:
    """把一条命中概念渲染为 Markdown 块：名称/描述 + 类型 + 章节 + position + 相关度。"""
    if concept.kind in ("framework", "term") and "：" in concept.text:
        name, _, description = concept.text.partition("：")
    else:
        name, description = concept.text, ""
    label = _KIND_LABELS.get(concept.kind, concept.kind)
    lines = [
        (
            f"### {index}. {name}（{label}）· {concept.section} · position {concept.position}"
            f"（相关度 {score:.4f}）"
        )
    ]
    if description:
        lines.extend(["", description])
    lines.append(f"source: {concept.source} · section: {concept.section} · position: {concept.position}")
    return "\n".join(lines)


def retrieve_concepts(query: str, book: str, *, config: BookRagConfig, top_k: int = 5) -> str:
    """按概念 / 框架 / 术语 / 心智模型检索书籍，返回 Markdown 格式的概念条目。

    概念索引随索引构建离线生成（``concepts.faiss`` + ``concepts.json``），
    检索时不调 LLM；条目锚定章节与 position。

    Args:
        query: 查询文本。
        book: 书籍过滤参数，支持书名 / 路径的部分匹配。
        config: 全局配置。
        top_k: 最多返回的概念条数。
    """
    metas = _find_matching_metas(config, book)
    if not metas:
        return f"未找到匹配的书籍：{book}"
    meta = metas[0]

    idx_dir = get_book_index_dir(config.index_dir, Path(meta.book_path))
    if meta.genre != "knowledge":
        return f"概念索引仅对 knowledge 体裁构建；该书为 {meta.genre}，可用 get_book_summaries：{meta.book_path}"
    status_path = idx_dir / "summary_status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (not status.get("concepts_complete") or status.get("book_hash") != meta.book_hash
                or status.get("genre") != meta.genre):
            return f"概念索引未构建或已过期，请运行 book-rag-mcp summarize：{meta.book_path}"
        if status.get("concept_count") == 0:
            return f"书中未检索到概念条目：{meta.book_path}"
    if not (idx_dir / "concepts.json").is_file():
        return f"概念索引未构建，请配置 LLM 后运行 book-rag-mcp summarize：{meta.book_path}"
    loaded = load_concept_index(idx_dir, get_embeddings(meta.model_name))
    if loaded is None:
        return f"概念索引未构建（需配置 LLM 后重建索引）：{meta.book_path}"
    vector_store, concepts = loaded
    if not concepts:
        return f"该书暂无概念条目：{meta.book_path}"

    hits = vector_store.similarity_search_with_score(query, k=min(top_k, len(concepts)))
    lines = [f"找到 {len(hits)} 条相关概念：", ""]
    for i, (doc, distance) in enumerate(hits, start=1):
        lines.append(
            _format_concept_hit(
                i, Concept(text=doc.page_content, **doc.metadata), _to_similarity(distance)
            )
        )
        lines.append("")
    return "\n".join(lines).rstrip()


# 工具封装（create_mcp 内的同名 def）会遮蔽模块级实现，提前捕获引用
_search_impl = search_book_knowledge
_list_books_impl = list_indexed_books
_refresh_impl = refresh_book_index
_structure_impl = get_book_structure
_within_section_impl = search_within_section
_section_content_impl = get_section_content
_concepts_impl = retrieve_concepts
_summaries_impl = get_book_summaries
_topics_impl = search_by_topic


def create_mcp(config: BookRagConfig | None = None) -> FastMCP:
    """创建并配置 FastMCP 服务实例。

    创建时先执行一次增量同步（服务启动即与书籍库保持一致），再注册工具。
    """
    config = config if config is not None else get_config()
    refresh_all_indexes(config)

    mcp = FastMCP("book-rag", instructions=PLAYBOOK)

    @mcp.tool
    def search_book_knowledge(
        query: str,
        top_k: int = 5,
        book: str | None = None,
        books: list[str] | None = None,
        group_by_book: bool = False,
        search_all_languages: bool = False,
        workspace: str | None = None,
        genre: str | None = None,
    ) -> str:
        """按内容检索书籍知识，返回书中相关原文片段（Markdown 格式）。

        Args:
            query: 查询文本。
            top_k: 最多返回的结果条数。
            book: 可选书籍过滤，支持书名 / 路径的部分匹配。
            books: 显式多书列表，与 book 互斥；top_k 至少等于匹配的书籍数。
                在语言/workspace/体裁过滤后的书籍中预留每书一条，其余按相关度补齐。
            group_by_book: 按 source 分组展示原文，不做跨书综合。
            search_all_languages: 为 True 时跨语言搜索全部书籍，
                默认只搜索与查询同语言的书籍。
            workspace: 可选 workspace 过滤；未在 workspaces.yaml 声明的书
                属于 ``_unclassified``。
            genre: 可选体裁过滤（fiction / knowledge / mixed / unknown）。
        """
        return _search_impl(
            query,
            config=config,
            top_k=top_k,
            book=book,
            books=books,
            group_by_book=group_by_book,
            search_all_languages=search_all_languages,
            workspace=workspace,
            genre=genre,
        )

    @mcp.tool
    def list_indexed_books(
        workspace: str | None = None, genre: str | None = None
    ) -> list[str]:
        """按 workspace 分组列出已索引书籍、体裁与块数。

        Args:
            workspace: 可选 workspace 过滤；``_unclassified`` 表示未声明书。
            genre: 可选体裁过滤（fiction / knowledge / mixed / unknown）。
        """
        return _list_books_impl(config=config, workspace=workspace, genre=genre)

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
    def get_section_content(book: str, section: str, offset: int = 0) -> str:
        """拉取指定书籍某章节的原文（按阅读顺序聚合，内容全部来自书籍本身）。

        单次最多返回 15,000 字；超长章会被截断，返回末尾会提示 offset，
        用 ``offset=<返回值>`` 继续续取。

        Args:
            book: 书籍过滤参数，支持书名 / 路径的部分匹配。
            section: 章节路径（支持前缀匹配），可先用 get_book_structure 确认章节名。
            offset: 起始字符偏移（0 起），用于超长章续取；首次调用省略即可。
        """
        return _section_content_impl(book, section, config=config, offset=offset)

    @mcp.tool
    def search_by_topic(query: str, top_k: int = 3, per_book: int = 2) -> str:
        """按离线主题名/描述或 topic-ID 取原文，按书分组，每条附 source/section/position。

        top_k 是主题数（最多10），per_book 是每个主题内每本书的片段数（最多5）。
        只读，不调用 LLM/embedding。单层 KMeans 只产生单书内/混合主题，
        本期未做簇中心二次聚类，未实现真正的跨书主题对齐。
        """
        return _topics_impl(query, config=config, top_k=top_k, per_book=per_book)

    @mcp.tool
    def get_book_summaries(
        book: str, section: str | None = None, offset: int = 0, limit: int = 5,
    ) -> str:
        """读取离线章节摘要：小说剧情/人物/转折/悬念，知识类框架/术语/心智模型。

        返回 source/section/position 原文指针；摘要不是原文，应使用 get_section_content
        核对后引用。一次最多20章，按 offset 续取；读取时不调用 LLM。
        """
        return _summaries_impl(book, config=config, section=section, offset=offset, limit=limit)

    @mcp.tool
    def retrieve_concepts(query: str, book: str, top_k: int = 5) -> str:
        """按概念 / 框架 / 术语 / 心智模型检索书籍，返回锚定章节的概念条目与原文指针（Markdown）。

        概念索引随索引构建离线生成；未构建时（如未配置 LLM 的旧索引）返回提示，
        配置 LLM 后运行 book-rag-mcp summarize 可离线补齐，无需重建正文向量。

        Args:
            query: 查询文本（概念名 / 描述均可）。
            book: 书籍过滤参数，支持书名 / 路径的部分匹配。
            top_k: 最多返回的概念条数。
        """
        return _concepts_impl(query, book, config=config, top_k=top_k)

    return mcp
