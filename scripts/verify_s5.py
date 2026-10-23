"""S5 真实多书覆盖、兼容查询、MCP 与 Web 分组验证（不调用 LLM）。"""

import asyncio
from dataclasses import asdict
from pathlib import Path

from fastmcp import Client
from starlette.testclient import TestClient

from book_rag.config import BookRagConfig
from book_rag.io.atomic import write_json_atomic
from book_rag.rag.retriever import search_books
from book_rag.servers.mcp_server import create_mcp, format_search_results
from book_rag.servers.web import create_web_app


async def main():
    config = BookRagConfig.from_env()
    books = ["library/投资最重要的事.md", "library/诺贝尔经济学奖经典收藏版.md"]
    query = "风险、不确定性与投资决策"
    report = {"books": books, "query": query, "cases": []}
    for enabled in (False, True):
        config.rerank_enabled = enabled
        hits = search_books(query, config=config, books=books, top_k=5)
        assert {hit.source for hit in hits} == set(books)
        assert all(hit.section and hit.position >= 0 for hit in hits)
        if enabled:
            assert all(hit.score_kind == "rerank" for hit in hits)
        markdown = format_search_results(hits, group_by_book=True)
        assert all("## " + book in markdown for book in books)
        old = search_books(query, config=config, book=books[0], top_k=5)
        single_list = search_books(query, config=config, books=[books[0]], top_k=5)
        assert old == single_list
        report["cases"].append({"rerank": enabled, "results": [asdict(h) for h in hits],
                                "markdown": markdown, "single_book_compatible": True})
    config.rerank_enabled = False
    async with Client(create_mcp(config)) as client:
        result = await client.call_tool("search_book_knowledge", {
            "query": query, "books": books, "group_by_book": True,
        })
        assert not result.is_error
        text = "\n".join(getattr(block, "text", "") for block in result.content)
        assert all("## " + book in text for book in books)
    with TestClient(create_web_app(config)) as client:
        response = client.post("/api/search", json={
            "query": query, "books": books, "group_by_book": True,
        })
        assert response.status_code == 200
        assert {group["source"] for group in response.json()["groups"]} == set(books)
    report["interfaces_passed"] = True
    write_json_atomic(Path("docs/reports/s5-evidence.json"), report)
    print("S5 real retrieval and interfaces passed", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
