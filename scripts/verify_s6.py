"""S6 全量聚类、向量零重算、只读检索和真实 MCP 验证。"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from fastmcp import Client
from verify_s4 import core_hashes

from book_rag.config import BookRagConfig
from book_rag.io.atomic import write_json_atomic
from book_rag.rag import embeddings, index
from book_rag.rag.llm import load_llm
from book_rag.rag.topics import (
    _load_documents,
    build_topics,
    corpus_fingerprint,
    search_topic_members,
)
from book_rag.servers.mcp_server import create_mcp


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    config = BookRagConfig.from_env()
    config.rerank_enabled = False
    before = core_hashes(config)

    def forbidden(*args, **kwargs):
        raise AssertionError("S6 called an embedding loader")

    embeddings.get_embeddings = forbidden
    embeddings.get_embeddings_for_language = forbidden
    index.get_embeddings = forbidden
    index.get_embeddings_for_language = forbidden
    start = time.perf_counter()
    if args.build:
        status = build_topics(config, load_llm())
        print(json.dumps(status, ensure_ascii=False), flush=True)
        assert status["complete"], status
    payload = json.loads((config.index_dir / "topics.json").read_text(encoding="utf-8"))
    fingerprint, metas = corpus_fingerprint(config)
    assert payload["corpus_fingerprint"] == fingerprint
    assert len(metas) == 5 and payload["chunk_count"] == 3257
    docs = _load_documents(config, metas)
    keys = []
    for topic in payload["topics"]:
        assert topic["name"] and topic["description"] and topic["representatives"]
        assert topic["count"] == len(topic["members"])
        for member in topic["members"]:
            key = (member["source"], member["position"])
            keys.append(key)
            assert docs[key].metadata["section"] == member["section"]
        assert set(topic["evidence_ids"]) <= {r["id"] for r in topic["representatives"]}
        hits = search_topic_members(topic["id"], config=config, top_k=1)
        assert hits and hits[0]["id"] == topic["id"]
        for member in hits[0]["members"]:
            assert docs[(member["source"], member["position"])].page_content.startswith(member["content"])
    assert len(keys) == len(set(keys)) == len(docs) == 3257
    async with Client(create_mcp(config)) as client:
        tools = {tool.name for tool in await client.list_tools()}
        assert len(tools) == 9 and "search_by_topic" in tools
        result = await client.call_tool("search_by_topic", {"query": payload["topics"][0]["id"], "top_k": 1})
        assert not result.is_error
        text = "\n".join(getattr(block, "text", "") for block in result.content)
        assert all(token in text for token in ("未实现真正的跨书主题对齐", "source:", "section:", "position:"))
    assert before == core_hashes(config)
    write_json_atomic(Path("docs/reports/s6-evidence.json"), {
        "passed": True, "core_unchanged": True, "embedding_loader_calls": 0,
        "sources": payload["sources"], "chunk_count": len(keys), "dimension": payload["dimension"],
        "selection": payload["selection"], "seconds": time.perf_counter() - start,
        "topics": [{k: v for k, v in t.items() if k != "members"} for t in payload["topics"]],
        "mcp_tools": sorted(tools), "limitation": payload["limitation"],
        "label_scope": payload["label_scope"],
    })
    print("S6 all 3257 chunks, zero re-embedding, topic retrieval and MCP passed", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
