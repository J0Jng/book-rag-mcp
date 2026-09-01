"""CLI 入口（占位桩）。

计划用法:
    book-rag-mcp                         # 启动 MCP 服务 (stdio)
    book-rag-mcp --transport http        # HTTP-SSE 传输
    book-rag-mcp index                   # 构建一次索引后退出

当前仅为命令入口骨架，实际服务/索引逻辑后续迭代实现。
"""

from __future__ import annotations

import argparse
import sys

import book_rag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="book-rag-mcp",
        description=book_rag.__doc__,
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="传输方式 (默认 stdio)[占位]",
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="serve",
        help="serve=启动服务 (默认)；index=构建索引后退出 [占位]",
    )
    args = parser.parse_args(argv)

    # 占位：此处后续接入 MCP 服务启动 / 索引构建逻辑
    print(
        f"[book-rag-mcp] 占位入口：transport={args.transport} action={args.action} "
        f"(版本 {book_rag.__version__})。实现将在后续迭代填充。",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())