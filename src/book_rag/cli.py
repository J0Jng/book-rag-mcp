"""CLI 入口：启动 MCP 服务（stdio 或 http）。

用法:
    book-rag-mcp                      # 默认 stdio 传输
    book-rag-mcp --transport http     # HTTP-SSE 传输
    book-rag-mcp --transport http --host 0.0.0.0 --port 8000
    book-rag-mcp index                # 仅构建一次索引后退出（便于调试）
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="book-rag-mcp", description="接入 AI Agent 的书籍 RAG MCP 服务")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="传输方式 (默认 stdio)",
    )
    parser.add_argument("--host", default=None, help="HTTP 传输监听地址 (默认取配置)")
    parser.add_argument("--port", type=int, default=None, help="HTTP 传输监听端口 (默认取配置)")
    parser.add_argument("--library", default=None, help="书籍库目录 (覆盖环境变量)")
    parser.add_argument(
        "action",
        nargs="?",
        default="serve",
        help="serve=启动服务 (默认)；index=构建一次索引后退出",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from book_rag.config import load_config
    from book_rag.rag.retriever import Retriever

    cfg = load_config()
    if args.library:
        cfg.library = __import__("pathlib").Path(args.library).resolve()
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    cfg.ensure_dirs()

    from book_rag.servers.mcp_server import create_server

    server = create_server(Retriever())

    if args.action == "index":
        stats = Retriever().index_library()
        print(f"索引完成: 扫描 {stats['scanned']} 本，索引 {stats['indexed_files']} 本，{stats['chunks']} 块", file=sys.stderr)
        return 0

    if args.transport == "http":
        host, port = cfg.host, cfg.port
        print(f"启动 HTTP-SSE 服务: http://{host}:{port}/sse", file=sys.stderr)
        server.run(transport="http", host=host, port=port)
    else:
        server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())