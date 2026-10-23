"""S5 显式多书范围、覆盖保障、一次全局重排和兼容接口。"""

import pytest
from langchain_core.embeddings import Embeddings
from starlette.testclient import TestClient

from book_rag.config import BookRagConfig
from book_rag.rag import index as index_mod
from book_rag.rag import retriever as retrieval
from book_rag.servers import mcp_server
from book_rag.servers.web import create_web_app


class StubEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


@pytest.fixture
def library(tmp_path, monkeypatch):
    books = tmp_path / "books"
    books.mkdir()
    (books / "a.md").write_text("# A\n" + "\n".join(
        f"## 第{i}章\nA {i} dominant content。" for i in range(30)
    ), encoding="utf-8")
    (books / "b.md").write_text("# B\n## 第一章\nB other source。", encoding="utf-8")
    monkeypatch.setattr(index_mod, "get_embeddings_for_language", lambda _: StubEmbeddings())
    monkeypatch.setattr(index_mod, "get_embeddings", lambda _: StubEmbeddings())
    monkeypatch.setattr(index_mod, "detect_language", lambda _: "zh")
    monkeypatch.setattr(retrieval, "detect_language", lambda _: "zh")
    config = BookRagConfig(library_path=books, index_dir=tmp_path / "index", rerank_enabled=True)
    index_mod.refresh_all_indexes(config)

    def vector_hits(store, query, k, section, query_vector):
        return [(doc, float(i if doc.metadata["source"] == "a.md" else 100))
                for i, doc in enumerate(index_mod._documents_by_position(store).values())][:k]

    monkeypatch.setattr(retrieval, "_vector_hits", vector_hits)
    monkeypatch.setattr(retrieval, "_bm25_hits", lambda *_: [])
    return config


def test_multibook_keeps_both_sources_inside_one_rerank_budget(library, monkeypatch):
    calls = []

    def rank(query, texts):
        calls.append(texts)
        return [100.0 if text.startswith("A ") else 0.0 for text in texts]

    monkeypatch.setattr(retrieval, "rerank_scores", rank)
    hits = retrieval.search_books("dominant", config=library, books=["a.md", "b.md"], top_k=2)
    assert {hit.source for hit in hits} == {"a.md", "b.md"}
    assert len(calls) == 1 and len(calls[0]) == 20
    assert all(hit.score_kind == "rerank" and hit.section and hit.position >= 0 for hit in hits)
    out = mcp_server.format_search_results(hits, group_by_book=True)
    assert "## a.md" in out and "## b.md" in out
    assert out.count("来源 source:") == 2


def test_single_string_and_single_list_are_equivalent(library, monkeypatch):
    monkeypatch.setattr(retrieval, "rerank_scores", lambda query, texts: list(range(len(texts))))
    single = retrieval.search_books("dominant", config=library, book="a.md", top_k=3)
    listed = retrieval.search_books("dominant", config=library, books=["a.md"], top_k=3)
    assert single == listed
    assert retrieval.search_books("dominant", config=library, books=[]) == []
    assert retrieval.search_books("dominant", config=library, books=["missing"]) == []


@pytest.mark.parametrize("kwargs", [{"book": "a", "books": ["b"]}, {"books": "a"},
                                   {"books": [""]}, {"books": [42]},
                                   {"books": ["a", "b"], "top_k": 1}])
def test_invalid_scope_is_explicit(library, kwargs):
    with pytest.raises(ValueError):
        retrieval.search_books("dominant", config=library, **kwargs)


def test_web_groups_preserve_legacy_results_and_validate_scope(library, monkeypatch):
    monkeypatch.setattr(retrieval, "rerank_scores", lambda query, texts: [1.] * len(texts))
    with TestClient(create_web_app(library)) as client:
        response = client.post("/api/search", json={
            "query": "dominant", "books": ["a.md", "b.md"], "group_by_book": True, "top_k": 2,
        })
        assert response.status_code == 200
        body = response.json()
        assert len(body["results"]) == 2 and {g["source"] for g in body["groups"]} == {"a.md", "b.md"}
        assert client.post("/api/search", json={
            "query": "dominant", "book": "a", "books": ["b"],
        }).status_code == 400
