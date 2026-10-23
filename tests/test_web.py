"""web 模块测试。

覆盖五类端点行为：
- ``GET /``：前端页面缺失时返回 500 与明确文案（前端 index.html 实现后转 200）；
- ``GET /api/books``：空索引返回 ``[]``，建索引后字段正确且不暴露 ``book_hash``；
- ``GET /api/books/available``：列出库内书籍及 ``indexed`` 状态；
- ``POST /api/search``：命中返结果且字段完整，空 query 返 400，``top_k`` 越界被 clamp；
- ``POST /api/refresh`` + ``GET /api/refresh/{job_id}``：轮询到 ``done`` 且统计完整，
  运行中重复 POST 幂等，未知 job_id 返 404。

为避免下载真实 HuggingFace 模型，复用 test_retriever.py 相同的基于 sha256 的
确定性假向量与语言检测 mock；全程不触碰真实模型。
"""

import hashlib
import math
import shutil
import time
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from starlette.testclient import TestClient

from book_rag.config import BookRagConfig
from book_rag.rag import index as index_mod
from book_rag.rag import retriever as retriever_mod
from book_rag.rag.index import RefreshResult, refresh_all_indexes
from book_rag.servers import web as web_mod

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_DIM = 16


class FakeHashEmbeddings(Embeddings):
    """基于 sha256 的确定性假 embedding，与 test_retriever.py 保持一致。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [b / 255.0 for b in (digest * 2)[:FAKE_DIM]]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BookRagConfig:
    """构造临时书籍库并 mock 掉模型与语言检测（不建索引）。"""
    library = tmp_path / "library"
    library.mkdir()
    shutil.copy(FIXTURES / "sample_book.md", library / "sample_book.md")
    shutil.copy(FIXTURES / "sample_book.txt", library / "sample_book.txt")

    # 索引构建侧（index 模块内部引用）
    monkeypatch.setattr(
        index_mod, "get_embeddings_for_language", lambda lang: FakeHashEmbeddings()
    )
    monkeypatch.setattr(index_mod, "get_embeddings", lambda name: FakeHashEmbeddings())
    monkeypatch.setattr(index_mod, "detect_language", lambda text: "zh-cn")
    # 检索侧（retriever 模块内部引用）
    monkeypatch.setattr(retriever_mod, "detect_language", lambda text: "zh-cn")

    return BookRagConfig(
        library_path=library, index_dir=tmp_path / "index", rerank_enabled=False
    )


@pytest.fixture
def indexed_config(config: BookRagConfig) -> BookRagConfig:
    """在 config 之上先建好两本书的索引。"""
    refresh_all_indexes(config)
    return config


@pytest.fixture
def client(config: BookRagConfig) -> TestClient:
    """为 config 创建 Web 应用客户端，并清空跨用例残留的后台 job 状态。"""
    web_mod._JOBS.clear()
    return TestClient(web_mod.create_web_app(config))


def _poll_job(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    """轮询刷新任务直到离开 running 状态，返回最终状态体。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/api/refresh/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] != "running":
            return body
        time.sleep(0.05)
    pytest.fail(f"刷新任务 {job_id} 超时未完成")


# ---------------------------------------------------------------------------
# GET /（前端页面）
# ---------------------------------------------------------------------------


def test_get_root_without_frontend_returns_500(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """前端 index.html 缺失（路径指向不存在的文件）时返回 500 与明确错误文案。"""
    monkeypatch.setattr(web_mod, "_INDEX_HTML_PATH", tmp_path / "missing" / "index.html")

    resp = client.get("/")

    assert resp.status_code == 500
    assert "index.html" in resp.json()["detail"]


def test_get_root_returns_html(client: TestClient):
    """前端实现后：GET / 返回 200 且 body 是 HTML 页面。"""
    resp = client.get("/")

    assert resp.status_code == 200
    assert "<html" in resp.text
    assert 'id="workspace-filter"' in resp.text
    assert "/api/workspaces" in resp.text


# ---------------------------------------------------------------------------
# GET /api/books
# ---------------------------------------------------------------------------


def test_api_books_empty_index(client: TestClient):
    """索引目录不存在时返回空列表，而不是抛异常。"""
    resp = client.get("/api/books")

    assert resp.status_code == 200
    assert resp.json() == []


def test_api_books_after_indexing(indexed_config: BookRagConfig):
    """建索引后返回两本书，字段为对外投影且不含内部 book_hash。"""
    web_mod._JOBS.clear()
    client = TestClient(web_mod.create_web_app(indexed_config))

    resp = client.get("/api/books")

    assert resp.status_code == 200
    books = resp.json()
    assert len(books) == 2
    expected_fields = {
        "book_path",
        "language",
        "chunk_count",
        "model_name",
        "updated_at",
        "workspaces",
        "genre",
        "tags",
    }
    names = set()
    for book in books:
        assert set(book) == expected_fields
        assert book["language"] == "zh"
        assert book["chunk_count"] > 0
        assert book["model_name"]
        assert book["updated_at"]
        assert book["workspaces"] == ["_unclassified"]
        assert book["genre"] == "unknown"
        assert book["tags"] == []
        names.add(book["book_path"])
    assert names == {"sample_book.md", "sample_book.txt"}


def test_api_books_workspace_and_genre_filters(indexed_config: BookRagConfig):
    """GET /api/books 支持 workspace / genre 过滤。"""
    (indexed_config.library_path / "workspaces.yaml").write_text(
        "workspaces:\n"
        "  fiction: {label: 小说, genre: fiction}\n"
        "  science: {label: 科学, genre: knowledge}\n"
        "books:\n"
        "  sample_book.md: {workspaces: [fiction], tags: [经典]}\n"
        "  sample_book.txt: {workspaces: [science]}\n",
        encoding="utf-8",
    )
    web_mod._JOBS.clear()
    client = TestClient(web_mod.create_web_app(indexed_config))

    fiction = client.get("/api/books", params={"workspace": "fiction"}).json()
    knowledge = client.get("/api/books", params={"genre": "knowledge"}).json()
    unclassified = client.get(
        "/api/books", params={"workspace": "_unclassified"}
    ).json()

    assert [b["book_path"] for b in fiction] == ["sample_book.md"]
    assert fiction[0]["tags"] == ["经典"]
    assert [b["book_path"] for b in knowledge] == ["sample_book.txt"]
    assert unclassified == []


def test_api_workspaces_lists_definitions_and_counts(indexed_config: BookRagConfig):
    """GET /api/workspaces 返回定义、label、genre、书籍数，并显式包含 _unclassified。"""
    (indexed_config.library_path / "workspaces.yaml").write_text(
        "workspaces:\n"
        "  fiction: {label: 小说, genre: fiction}\n"
        "  finance: {label: 金融投资, genre: knowledge}\n"
        "books:\n"
        "  sample_book.md: {workspaces: [fiction]}\n",
        encoding="utf-8",
    )
    web_mod._JOBS.clear()
    client = TestClient(web_mod.create_web_app(indexed_config))

    resp = client.get("/api/workspaces")

    assert resp.status_code == 200
    items = {item["id"]: item for item in resp.json()}
    assert items["fiction"] == {
        "id": "fiction",
        "label": "小说",
        "genre": "fiction",
        "book_count": 1,
    }
    assert items["finance"]["book_count"] == 0
    assert items["_unclassified"]["book_count"] == 1
    assert items["_unclassified"]["label"] == "未分类"


# ---------------------------------------------------------------------------
# GET /api/books/available
# ---------------------------------------------------------------------------


def test_api_books_available_before_indexing(client: TestClient, config: BookRagConfig):
    """未建索引时列出库内两本书，indexed 均为 False。"""
    resp = client.get("/api/books/available")

    assert resp.status_code == 200
    books = resp.json()
    assert len(books) == 2
    for book in books:
        assert set(book) == {"path", "indexed"}
        assert book["indexed"] is False
        assert book["path"] in {"sample_book.md", "sample_book.txt"}


def test_api_books_available_after_indexing(indexed_config: BookRagConfig):
    """建索引后同一批书籍的 indexed 变为 True。"""
    web_mod._JOBS.clear()
    client = TestClient(web_mod.create_web_app(indexed_config))

    resp = client.get("/api/books/available")

    assert resp.status_code == 200
    books = {b["path"]: b["indexed"] for b in resp.json()}
    assert books == {"sample_book.md": True, "sample_book.txt": True}


# ---------------------------------------------------------------------------
# POST /api/search
# ---------------------------------------------------------------------------


def test_api_search_returns_results_with_fields(
    indexed_config: BookRagConfig,
):
    """命中查询：返回非空 results，六字段完整且内容来自书中原文。"""
    web_mod._JOBS.clear()
    client = TestClient(web_mod.create_web_app(indexed_config))

    resp = client.post("/api/search", json={"query": "清晨的雾气"})

    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results
    originals = [
        (indexed_config.library_path / "sample_book.md").read_text(encoding="utf-8"),
        (indexed_config.library_path / "sample_book.txt").read_text(encoding="utf-8"),
    ]
    for r in results:
        assert set(r) == {
            "content",
            "book",
            "section",
            "source",
            "score",
        "score_kind",
            "position",
            "workspaces",
            "genre",
            "tags",
        }
        assert r["content"].strip()
        assert all(line in originals[0] or line in originals[1]
                   for line in r["content"].splitlines())
        assert r["score"] >= 0.0
        assert isinstance(r["position"], int)


def test_api_search_empty_query_returns_400(indexed_config: BookRagConfig):
    """query 缺失或空白时返回 400 与明确文案。"""
    web_mod._JOBS.clear()
    client = TestClient(web_mod.create_web_app(indexed_config))

    resp_missing = client.post("/api/search", json={})
    resp_blank = client.post("/api/search", json={"query": "   "})

    assert resp_missing.status_code == 400
    assert resp_blank.status_code == 400
    assert resp_missing.json()["detail"] == "query 不能为空"
    assert resp_blank.json()["detail"] == "query 不能为空"


def test_api_search_top_k_clamped(indexed_config: BookRagConfig):
    """top_k 越界时被 clamp 到 [1, 50]：0 视为 1，非法值不报 400。"""
    web_mod._JOBS.clear()
    client = TestClient(web_mod.create_web_app(indexed_config))

    resp_zero = client.post("/api/search", json={"query": "旅人", "top_k": 0})
    resp_invalid = client.post("/api/search", json={"query": "旅人", "top_k": "abc"})

    assert resp_zero.status_code == 200
    assert len(resp_zero.json()["results"]) == 1
    assert resp_invalid.status_code == 200


def test_api_search_workspace_and_genre_filters(indexed_config: BookRagConfig):
    """POST /api/search 的 workspace / genre 过滤生效并回传元数据。"""
    (indexed_config.library_path / "workspaces.yaml").write_text(
        "workspaces:\n"
        "  fiction: {label: 小说, genre: fiction}\n"
        "  science: {label: 科学, genre: knowledge}\n"
        "books:\n"
        "  sample_book.md: {workspaces: [fiction]}\n"
        "  sample_book.txt: {workspaces: [science]}\n",
        encoding="utf-8",
    )
    web_mod._JOBS.clear()
    client = TestClient(web_mod.create_web_app(indexed_config))

    fiction = client.post(
        "/api/search", json={"query": "旅人", "top_k": 10, "workspace": "fiction"}
    ).json()["results"]
    knowledge = client.post(
        "/api/search", json={"query": "旅人", "top_k": 10, "genre": "knowledge"}
    ).json()["results"]

    assert fiction and {r["source"] for r in fiction} == {"sample_book.md"}
    assert all(r["workspaces"] == ["fiction"] for r in fiction)
    assert all(r["genre"] == "fiction" for r in fiction)
    assert knowledge and {r["source"] for r in knowledge} == {"sample_book.txt"}
    assert all(r["genre"] == "knowledge" for r in knowledge)


# ---------------------------------------------------------------------------
# POST /api/refresh + GET /api/refresh/{job_id}
# ---------------------------------------------------------------------------


def test_refresh_job_lifecycle(client: TestClient):
    """提交刷新任务后轮询到 done，统计中包含新建的两本书。"""
    resp = client.post("/api/refresh")

    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert job_id

    body = _poll_job(client, job_id)
    assert body["status"] == "done"
    assert sorted(body["result"]["built"]) == ["sample_book.md", "sample_book.txt"]
    assert body["result"]["skipped"] == []
    assert body["result"]["deleted"] == []


def test_refresh_returns_same_job_id_while_running(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """已有 running 任务时重复 POST 返回同一 job_id（幂等，避免并发建索引）。"""

    def slow_refresh(cfg):
        time.sleep(0.3)
        return RefreshResult()

    monkeypatch.setattr(web_mod, "refresh_all_indexes", slow_refresh)

    first = client.post("/api/refresh").json()
    second = client.post("/api/refresh").json()

    assert first["job_id"] == second["job_id"]


def test_refresh_unknown_job_id_returns_404(client: TestClient):
    """查询不存在的 job_id 返回 404。"""
    resp = client.get("/api/refresh/not-a-real-job")

    assert resp.status_code == 404
    assert "detail" in resp.json()
