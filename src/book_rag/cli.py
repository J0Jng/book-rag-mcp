"""命令行入口。

用法:
    book-rag-mcp                          # 启动 MCP 服务 (stdio，默认)
    book-rag-mcp --transport http         # HTTP-SSE 传输
    book-rag-mcp serve                    # 同默认行为
    book-rag-mcp index                    # 构建/刷新一次索引后退出
    book-rag-mcp web --port 8000          # 启动本地 Web 界面
    book-rag-mcp convert --source DIR     # 把 EPUB/TXT 归一为 Markdown（索引输入）
"""

from __future__ import annotations

import os

# 不强制 HF_HUB_OFFLINE：S3 的 reranker（BAAI/bge-reranker-v2-m3）首次使用需要
# 允许联网下载；是否离线由用户环境变量自行决定。这里仅静默 Windows 符号链接告警。
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import json
from pathlib import Path

import book_rag
from book_rag.config import BookRagConfig
from book_rag.convert import convert_library
from book_rag.rag.index import refresh_all_indexes
from book_rag.rag.summary_layer import enrich_indexes
from book_rag.servers.mcp_server import create_mcp, format_refresh_result


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="book-rag-mcp",
        description=book_rag.__doc__,
    )
    parser.add_argument(
        "--book", help="summarize 的书名或相对路径过滤（不传则处理全部已索引书籍）",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="MCP 传输方式（默认 stdio）",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Web 服务监听地址（默认 127.0.0.1，仅 action=web 生效）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Web 服务监听端口（默认 8000，仅 action=web 生效）",
    )
    parser.add_argument(
        "--source",
        help="convert 的源目录（默认：书籍库根目录）",
    )
    parser.add_argument(
        "--dest",
        help="convert 的 Markdown 输出目录（默认：<书籍库>/library）",
    )
    parser.add_argument(
        "--originals",
        help="convert 的原始文件归档目录（默认：<书籍库>/_originals）",
    )
    parser.add_argument(
        "--copy-originals",
        action="store_true",
        help="convert 时复制而非移动原始文件（默认移动）",
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="serve",
        choices=["serve", "index", "web", "convert", "summarize", "topics"],
        help=(
            "serve=启动 MCP 服务（默认）；index=构建/刷新索引后退出；"
            "web=启动本地 Web 界面；convert=把 EPUB/TXT 归一为 Markdown；"
            "summarize=离线补齐摘要与知识类概念；topics=复用现有向量聚类并命名主题"
        ),
    )
    return parser


def _run_convert(args: argparse.Namespace, config: BookRagConfig) -> int:
    """执行书籍归一，打印逐本统计；有任何一本书失败返回非零。"""
    source = Path(args.source) if args.source else config.library_path
    dest = Path(args.dest) if args.dest else config.library_path / "library"
    originals = Path(args.originals) if args.originals else config.library_path / "_originals"

    results = convert_library(
        source, dest, originals, move_originals=not args.copy_originals
    )
    for result in results:
        if result.error:
            print(f"❌ {Path(result.source).name}：{result.error}")
            continue
        print(
            f"✅ {Path(result.output).name}　章 {result.chapters}　"
            f"正文 {result.body_chars:,} 字　丢弃文档 {len(result.dropped_docs)} 个"
        )
    failed = sum(1 for r in results if r.error)
    print(f"\n转换完成：{len(results) - failed} 本成功，{failed} 本失败；输出目录 {dest}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BookRagConfig.from_env()

    if args.action == "topics":
        from book_rag.rag.llm import load_llm
        from book_rag.rag.topics import TopicIndexError, build_topics

        try:
            status = build_topics(config, load_llm())
        except TopicIndexError as exc:
            print(str(exc))
            return 1
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0 if status["complete"] else 1

    if args.action == "summarize":
        statuses = enrich_indexes(config, book=args.book)
        print(json.dumps(statuses, ensure_ascii=False, indent=2))
        return 0 if statuses and all(
            s["complete"] and (s["genre"] != "knowledge" or s["concepts_complete"])
            for s in statuses
        ) else 1

    if args.action == "index":
        result = refresh_all_indexes(config)
        print(format_refresh_result(result))
        return 0

    if args.action == "convert":
        return _run_convert(args, config)

    if args.action == "web":
        import uvicorn

        from book_rag.servers.web import create_web_app

        uvicorn.run(create_web_app(config), host=args.host, port=args.port)
        return 0

    # create_mcp 创建时已完成一次增量同步
    mcp = create_mcp(config)
    mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
