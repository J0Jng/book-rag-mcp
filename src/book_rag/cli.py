"""命令行入口。

用法:
    book-rag-mcp                          # 启动 MCP 服务 (stdio，默认)
    book-rag-mcp --transport http         # HTTP-SSE 传输
    book-rag-mcp serve                    # 同默认行为
    book-rag-mcp index                    # 构建/刷新一次索引后退出
"""

from __future__ import annotations

import argparse

import book_rag
from book_rag.config import BookRagConfig
from book_rag.rag.index import refresh_all_indexes
from book_rag.servers.mcp_server import create_mcp, format_refresh_result


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="book-rag-mcp",
        description=book_rag.__doc__,
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="MCP 传输方式（默认 stdio）",
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="serve",
        choices=["serve", "index"],
        help="serve=启动 MCP 服务（默认）；index=构建/刷新索引后退出",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BookRagConfig.from_env()

    if args.action == "index":
        result = refresh_all_indexes(config)
        print(format_refresh_result(result))
        return 0

    # create_mcp 创建时已完成一次增量同步
    mcp = create_mcp(config)
    mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
