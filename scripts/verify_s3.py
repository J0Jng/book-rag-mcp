"""S3 真实书库验收：完整性、检索对照、延迟、缓存、原文续取与接口冒烟。

默认校验正式索引，接口冒烟会执行应全部跳过的增量同步；--check-only 仅检查文件。
示例：python scripts/verify_s3.py --output docs/reports/s3-evidence.json
--build 默认在独立目录构建，始终禁用在线摘要；--index-dir 可显式指定目录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch


def save(path: Path, data: dict) -> None:
    """每个阶段持久化证据，中途失败保留已完成结果。"""
    from book_rag.io.atomic import write_json_atomic

    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, data)


def check_library(config) -> dict:
    """独立对照源书、manifest、FAISS、BM25 和章节快照。"""
    from book_rag.io.cleaning import clean_matter
    from book_rag.io.loader import compute_file_hash, discover_books, load_book
    from book_rag.rag.bm25 import USER_DICT_FILENAME, ensure_user_dict, load_bm25_index
    from book_rag.rag.index import get_book_index_dir, load_book_index, load_section_snapshot
    from book_rag.rag.language import DEFAULT_EMBEDDING_MODEL
    from book_rag.rag.metadata import load_catalog
    from book_rag.rag.splitter import split_book
    from book_rag.rag.structure import build_toc

    books = discover_books(config.library_path)
    keys = {p.relative_to(config.library_path).as_posix() for p in books}
    assert len(keys) == 5 and all(key.startswith("library/") and key.endswith(".md")
                                  for key in keys), keys
    manifest = json.loads((config.index_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == keys
    expected_dirs = {get_book_index_dir(config.index_dir, Path(key)).name for key in keys}
    actual_dirs = {p.name for p in config.index_dir.iterdir() if p.name.startswith("book-")}
    assert expected_dirs == actual_dirs
    ensure_user_dict(config.library_path / USER_DICT_FILENAME)
    catalog = load_catalog(config.library_path)
    details = []
    for source in books:
        key = source.relative_to(config.library_path).as_posix()
        idx_dir = get_book_index_dir(config.index_dir, Path(key))
        assert all((idx_dir / name).is_file() for name in (
            "meta.json", "index.faiss", "index.pkl", "bm25.json", "toc.json", "sections.json",
        ))
        loaded = load_book_index(config.index_dir, Path(key))
        assert loaded is not None
        store, meta = loaded
        metadata = catalog.resolve(key)
        assert (meta.workspaces, meta.genre, meta.tags) == (
            metadata.workspaces, metadata.genre, metadata.tags,
        )
        assert meta.model_name == DEFAULT_EMBEDDING_MODEL
        assert manifest[key]["hash"] == meta.book_hash == compute_file_hash(source)
        parsed, _ = clean_matter(load_book(source))
        documents = split_book(parsed.sections, config.chunk_size, config.chunk_overlap,
                               title=parsed.title, source=key)
        actual = sorted(store.docstore._dict.values(), key=lambda d: d.metadata["position"])
        assert [(d.page_content, d.metadata) for d in actual] == [
            (d.page_content, d.metadata) for d in documents
        ]
        assert meta.chunk_count == store.index.ntotal == len(documents)
        bm25 = load_bm25_index(idx_dir)
        assert bm25 is not None
        assert bm25.positions == [d.metadata["position"] for d in actual]
        assert load_section_snapshot(idx_dir) == [asdict(s) for s in parsed.sections]
        assert json.loads((idx_dir / "toc.json").read_text(encoding="utf-8")) == build_toc(
            parsed.sections
        )
        details.append({"source": key, "model": meta.model_name, "chunks": meta.chunk_count,
                        "sections": len(parsed.sections), "hash": meta.book_hash})
        print(f"CHECK {key}: {meta.chunk_count} chunks", flush=True)
    return {"passed": True, "books": details}


def smoke_interfaces(config) -> dict:
    """通过真实 MCP 客户端与 Web ASGI 客户端验证工具和管理入口。"""
    import asyncio

    from fastmcp import Client
    from starlette.testclient import TestClient

    from book_rag.servers.mcp_server import create_mcp
    from book_rag.servers.web import create_web_app

    async def check_mcp():
        server = create_mcp(config)
        async with Client(server) as client:
            tools = await client.list_tools()
            names = {tool.name for tool in tools}
            expected = {"list_indexed_books", "get_book_structure", "search_book_knowledge",
                        "search_within_section", "get_section_content", "retrieve_concepts",
                        "get_book_summaries",
                        "search_by_topic",
                        "refresh_book_index"}
            assert names == expected
            calls = [
                ("list_indexed_books", {}),
                ("get_book_structure", {"book": "工作、消费主义和新穷人"}),
                ("search_book_knowledge", {"query": "斯库特", "workspace": "fiction"}),
                ("search_within_section", {"query": "工作伦理", "book": "工作、消费主义和新穷人",
                                           "section": "第四章"}),
                ("get_section_content", {"book": "工作、消费主义和新穷人", "section": "第四章",
                                         "offset": 15000}),
                ("retrieve_concepts", {"query": "安全边际", "book": "投资最重要的事"}),
                ("get_book_summaries", {"book": "杀死一只知更鸟", "limit": 1}),
                ("refresh_book_index", {}),
            ]
            for name, arguments in calls:
                result = await client.call_tool(name, arguments)
                assert not result.is_error, name
                text = "\n".join(getattr(block, "text", "") for block in result.content)
                if name.startswith("search_"):
                    assert all(token in text for token in ("source:", "section:", "position:"))
                    if config.rerank_enabled:
                        assert "rerank 排序分" in text
                elif name == "get_section_content":
                    assert "23426" in text and "已返回本章末尾" in text
                elif name == "retrieve_concepts":
                    assert "概念索引未构建" in text or all(
                        token in text for token in ("source:", "section:", "position:")
                    )
                elif name == "get_book_summaries":
                    assert "摘要未" in text or all(
                        token in text for token in ("source:", "section:", "position:")
                    )
            return sorted(names)

    names = asyncio.run(check_mcp())
    with TestClient(create_web_app(config)) as client:
        assert client.get("/").status_code == 200
        books = client.get("/api/books").json()
        assert len(books) == 5
        available = client.get("/api/books/available").json()
        assert len(available) == 5 and all(item["indexed"] for item in available)
        workspaces = client.get("/api/workspaces").json()
        assert {row["id"]: row["book_count"] for row in workspaces} == {
            "fiction": 1, "finance": 2, "social": 1, "life": 1,
        }
        response = client.post("/api/search", json={"query": "斯库特", "workspace": "fiction"})
        assert response.status_code == 200
        results = response.json()["results"]
        kind = "rerank" if config.rerank_enabled else "rrf"
        assert results and all(item["score_kind"] == kind for item in results)
        job = client.post("/api/refresh").json()["job_id"]
        for _ in range(200):
            state = client.get(f"/api/refresh/{job}").json()
            if state["status"] != "running":
                break
            time.sleep(0.05)
        assert state["status"] == "done", state
        assert not state["result"]["built"] and len(state["result"]["skipped"]) == 5
    from book_rag.rag.reranker import get_reranker_runtime

    return {"passed": True, "mcp_tools": names, "web_book_count": len(books),
            "score_kind": kind, "reranker_runtime": get_reranker_runtime(),
            "refresh_rebuilt": 0, "refresh_skipped": 5}


def main() -> int:
    """执行验收；任一强制断言失败时非零退出，不把降级计为 rerank 成功。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("books"))
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/reports/s3-evidence.json"))
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-rerank", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.index_dir is None:
        args.index_dir = args.library / (".book_rag_index_s3" if args.build else ".book_rag_index")
    if args.repeats < 5:
        parser.error("正式验收至少重复五次")
    os.environ["HF_HUB_OFFLINE"] = "1"
    for name in ("BOOK_RAG_LLM_API_KEY", "BOOK_RAG_LLM_BASE_URL", "BOOK_RAG_LLM_MODEL",
                 "HERMES_CUSTOM_CUSTOM_API_KEY"):
        os.environ.pop(name, None)

    import torch

    from book_rag.config import BookRagConfig
    from book_rag.io.loader import load_book
    from book_rag.rag import index, retriever
    from book_rag.rag.embeddings import get_embeddings
    from book_rag.rag.language import DEFAULT_EMBEDDING_MODEL
    from book_rag.rag.reranker import (
        RERANK_MODEL_NAME,
        _get_cross_encoder,
        get_reranker_failure,
        get_reranker_runtime,
        rerank_scores,
    )
    from book_rag.servers import mcp_server

    config = BookRagConfig(library_path=args.library, index_dir=args.index_dir)
    evidence = {"environment": {"platform": platform.platform(), "python": platform.python_version(),
                                "torch": torch.__version__, "cuda": torch.cuda.is_available(),
                                "processor": platform.processor(),
                                "threads": torch.get_num_threads(), "cpu_count": os.cpu_count()},
                "index_dir": str(args.index_dir.resolve()), "repeats": args.repeats}
    save(args.output, evidence)
    if args.build:
        print("BUILD (online summaries disabled)", flush=True)
        evidence["refresh"] = asdict(index.refresh_all_indexes(config))
    started = time.perf_counter()
    embeddings = get_embeddings(DEFAULT_EMBEDDING_MODEL)
    evidence["embedding_load_seconds"] = time.perf_counter() - started
    evidence["integrity"] = check_library(config)
    save(args.output, evidence)
    if args.check_only:
        return 0
    if args.smoke_only:
        config.rerank_enabled = args.smoke_rerank
        evidence["interfaces"] = smoke_interfaces(config)
        save(args.output, evidence)
        return 0

    from huggingface_hub import try_to_load_from_cache

    weight_path = try_to_load_from_cache(RERANK_MODEL_NAME, "model.safetensors")
    if isinstance(weight_path, str):
        with Path(weight_path).open("rb") as weights:
            digest = hashlib.file_digest(weights, "sha256").hexdigest()
        evidence["reranker_artifact"] = {"sha256": digest,
                                         "revision": Path(weight_path).parent.name}
    started = time.perf_counter()
    assert _get_cross_encoder() is not None, get_reranker_failure()
    evidence["reranker_load_seconds"] = time.perf_counter() - started
    evidence["reranker_runtime"] = get_reranker_runtime()
    evidence["environment"]["threads"] = torch.get_num_threads()
    probe = rerank_scores("安全边际", ["价格显著低于价值形成安全边际。", "孩子们在镇上玩耍。"])
    assert probe is not None and probe[0] > probe[1], (probe, get_reranker_failure())
    evidence["reranker_probe"] = {"model": RERANK_MODEL_NAME, "scores": probe}
    print("MODELS ready", flush=True)
    save(args.output, evidence)

    book_texts = {p.name: p.read_text(encoding="utf-8")
                  for p in (args.library / "library").glob("*.md")}
    evidence["absent_original_examples"] = [term for term in (
        "逆向选择", "模范少数族裔", "阿蒂克斯",
    ) if not any(term in text for text in book_texts.values())]
    # 运行前固定题集；只验证原文是否出现查询词，不事后换题以掩盖失败。
    cases = [
        ("第二层次思维", "library/投资最重要的事.md"),
        ("安全边际", "library/投资最重要的事.md"),
        ("斯库特", "library/杀死一只知更鸟.md"),
        ("阿迪克斯", "library/杀死一只知更鸟.md"),
        ("有效市场假说", "library/投资最重要的事.md"),
        ("风险溢价", "library/投资最重要的事.md"),
        ("原型对话", "library/斯坦福大学人生设计课.md"),
        ("奥德赛计划", "library/斯坦福大学人生设计课.md"),
    ]
    evidence["queries"] = []
    for query, expected_source in cases:
        assert query in book_texts[Path(expected_source).name]
        all_hits = []
        vector = embeddings.embed_query(query)
        for meta in retriever._list_indexed_metas(config.index_dir, config.library_path):
            loaded = index.load_book_index_cached(config.index_dir, Path(meta.book_path))
            assert loaded is not None
            store = loaded[0]
            for doc, distance in store.similarity_search_with_score_by_vector(vector, k=5):
                all_hits.append({"source": doc.metadata["source"],
                                 "section": doc.metadata["section"],
                                 "position": doc.metadata["position"],
                                 "content": doc.page_content, "score": float(distance)})
        baseline = sorted(all_hits, key=lambda item: item["score"])[:5]
        record = {"query": query, "expected_source": expected_source,
                  "vector_only": baseline, "modes": {}}
        for enabled in (False, True):
            config.rerank_enabled = enabled
            index.clear_index_cache()
            loads = []
            real_load = index.FAISS.load_local

            def counted(*a, _loads=loads, _real=real_load, **kw):
                _loads.append(a[0])
                return _real(*a, **kw)

            rerank_calls = []

            def traced_rerank(q, texts, _calls=rerank_calls):
                _calls.append(len(texts))
                return rerank_scores(q, texts)

            with patch.object(index.FAISS, "load_local", counted), patch.object(
                retriever, "rerank_scores", traced_rerank
            ):
                start = time.perf_counter()
                results = retriever.search_books(query, config=config, top_k=5)
                cold = time.perf_counter() - start
                initial_loads = len(loads)
                assert initial_loads == 5
                timings = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    results = retriever.search_books(query, config=config, top_k=5)
                    timings.append(time.perf_counter() - start)
                assert len(loads) == initial_loads
            assert len(rerank_calls) == (args.repeats + 1 if enabled else 0)
            assert all(count <= 20 for count in rerank_calls)
            assert all(r.score_kind == ("rerank" if enabled else "rrf") for r in results)
            if enabled:
                assert get_reranker_failure() is None, get_reranker_failure()
            ranks = [i for i, r in enumerate(results, 1)
                     if r.source == expected_source and query in r.content]
            mode = "hybrid_rerank" if enabled else "hybrid"
            record["modes"][mode] = {"cold_index_seconds": cold, "warm_seconds": timings,
                                     "median_seconds": statistics.median(timings),
                                     "max_seconds": max(timings),
                                     "faiss_loads_first": initial_loads,
                                     "faiss_loads_repeated": len(loads) - initial_loads,
                                     "rerank_candidate_counts": rerank_calls,
                                     "first_literal_rank": min(ranks) if ranks else None,
                                     "literal_hits_top3": sum(rank <= 3 for rank in ranks),
                                     "results": [asdict(r) for r in results]}
            print(f"QUERY {query} {mode}: median={statistics.median(timings):.3f}s "
                  f"literal_ranks={ranks}", flush=True)
        baseline_ranks = [i for i, hit in enumerate(baseline, 1)
                          if hit["source"] == expected_source and query in hit["content"]]
        record["vector_first_literal_rank"] = min(baseline_ranks) if baseline_ranks else None
        record["vector_literal_hits_top3"] = sum(rank <= 3 for rank in baseline_ranks)
        evidence["queries"].append(record)
        save(args.output, evidence)

    book_key = "library/工作、消费主义和新穷人.md"
    original = next(s for s in load_book(args.library / book_key).sections if "第四章" in s.path)
    pages = []
    for offset in range(0, len(original.text), 15000):
        out = mcp_server.get_section_content(book_key, original.path, config=config, offset=offset)
        body = out.split("\n\n", 1)[1].rsplit("\n\n（", 1)[0]
        assert body == original.text[offset:offset + 15000]
        pages.append(body)
    assert "".join(pages) == original.text
    evidence["long_section"] = {"section": original.path, "chars": len(original.text),
                                "page_lengths": [len(page) for page in pages],
                                "exact_reconstruction": True}
    books = mcp_server.list_indexed_books(config=config)
    assert len(books) == 5
    assert "第四章" in mcp_server.get_book_structure(book_key, config=config)
    out = mcp_server.search_within_section("工作伦理", book_key, original.path, config=config)
    assert all(token in out for token in ("source:", "section:", "position:"))
    config.rerank_enabled = False
    for workspace in ("fiction", "finance", "social", "life"):
        results = retriever.search_books("人生与社会", config=config, workspace=workspace)
        assert results and all(workspace in item.workspaces for item in results)
    fiction = retriever.search_books("斯库特", config=config, genre="fiction")
    assert fiction and all(item.genre == "fiction" for item in fiction)
    evidence["tool_smoke"] = {"passed": True, "book_count": len(books),
                              "workspaces": ["fiction", "finance", "social", "life"]}
    evidence["interfaces"] = smoke_interfaces(config)
    evidence["acceptance"] = {
        "top3_all": all(row["modes"][mode]["first_literal_rank"] is not None and
                        row["modes"][mode]["first_literal_rank"] <= 3
                        for row in evidence["queries"] for mode in ("hybrid", "hybrid_rerank")),
        "latency_all": all(row["modes"][mode]["max_seconds"] <= budget
                           for row in evidence["queries"]
                           for mode, budget in (("hybrid", 1.5), ("hybrid_rerank", 5.0))),
        "improvement_over_vector": any(
            row["modes"][mode]["literal_hits_top3"] > row["vector_literal_hits_top3"]
            or (row["modes"][mode]["first_literal_rank"] or 99)
            < (row["vector_first_literal_rank"] or 99)
            for row in evidence["queries"] for mode in ("hybrid", "hybrid_rerank")
        ),
    }
    save(args.output, evidence)
    print(json.dumps(evidence["acceptance"], ensure_ascii=False), flush=True)
    return 0 if all(evidence["acceptance"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
