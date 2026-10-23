"""S6 向量复用、全量归属、恢复、失效与原文来源验证。"""

import json
from pathlib import Path

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from book_rag.config import BookRagConfig
from book_rag.rag.index import BookIndexMeta, get_book_index_dir
from book_rag.rag.llm import LLMError
from book_rag.rag.topics import (
    TopicIndexError,
    build_topics,
    corpus_fingerprint,
    search_topic_members,
)
from book_rag.servers.mcp_server import search_by_topic


class NeverEmbed(Embeddings):
    def embed_documents(self, texts):
        pytest.fail("S6 不得重新 embedding 文本")

    def embed_query(self, text):
        pytest.fail("主题标签搜索不应 embedding 查询")


class LabelLLM:
    def __init__(self, error=None):
        self.calls = 0
        self.error = error

    def complete_json(self, system, user, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        excerpts = json.loads(user)["representatives"]
        name = "风险管理" if "风险" in excerpts[0]["excerpt"] else "小说人物"
        return {"name": name, "description": "代表片段主要涉及给定原文主题", "evidence_ids": ["R1"]}


@pytest.fixture
def corpus(tmp_path):
    library = tmp_path / "books"
    library.mkdir()
    config = BookRagConfig(library_path=library, index_dir=tmp_path / "index")
    for source, texts, vectors in (
        ("a.md", ["风险管理原文一", "风险管理原文二", "风险管理原文三"], [[1, 0], [.95, .05], [.9, .1]]),
        ("b.md", ["小说人物原文一", "小说人物原文二", "小说人物原文三"], [[0, 1], [.05, .95], [.1, .9]]),
    ):
        folder = get_book_index_dir(config.index_dir, Path(source))
        folder.mkdir(parents=True)
        FAISS.from_embeddings(
            list(zip(texts, vectors, strict=True)), NeverEmbed(),
            metadatas=[{"source": source, "section": f"第一章 > {i}", "position": i}
                       for i in range(3)],
        ).save_local(str(folder))
        meta = BookIndexMeta(source, source, "zh", "same-model", 3, "now")
        (folder / "meta.json").write_text(json.dumps(meta.to_dict()), encoding="utf-8")
    return config


def test_all_vectors_clustered_once_without_embedding_and_grounded_search(corpus):
    before = corpus_fingerprint(corpus)[0]
    status = build_topics(corpus, LabelLLM(), candidates=(2, 3))
    assert status["complete"] and status["chunk_count"] == 6
    assert corpus_fingerprint(corpus)[0] == before
    payload = json.loads((corpus.index_dir / "topics.json").read_text(encoding="utf-8"))
    assert payload["selection"]["evaluations"] and payload["dimension"] == 2
    assert len({(m["source"], m["position"]) for t in payload["topics"] for m in t["members"]}) == 6
    assert all(t["name"] and t["description"] and t["representatives"] for t in payload["topics"])
    hits = search_topic_members("风险", config=corpus, top_k=1, per_book=1)
    assert hits and hits[0]["members"][0]["content"].startswith("风险管理原文")
    markdown = search_by_topic("风险", config=corpus, top_k=1)
    assert "未实现真正的跨书主题对齐" in markdown
    assert "source: a.md" in markdown and "section:" in markdown and "position:" in markdown


def test_missing_llm_keeps_checkpoint_and_resume_does_not_recluster(corpus, monkeypatch):
    assert not build_topics(corpus, candidates=(2, 3))["complete"]
    assert (corpus.index_dir / "topics.pending.json").is_file()
    assert not (corpus.index_dir / "topics.json").exists()
    from book_rag.rag import topics

    monkeypatch.setattr(topics, "_load_vectors", lambda *_: pytest.fail("应复用聚类断点"))
    llm = LabelLLM()
    assert build_topics(corpus, llm, candidates=(2, 3))["complete"]
    assert llm.calls == 2
    assert build_topics(corpus, llm, candidates=(2, 3))["complete"]
    assert llm.calls == 2


def test_changed_core_index_rejects_stale_topics(corpus):
    build_topics(corpus, LabelLLM(), candidates=(2, 3))
    folder = get_book_index_dir(corpus.index_dir, Path("a.md"))
    with (folder / "index.pkl").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(TopicIndexError, match="过期"):
        search_topic_members("风险", config=corpus)


def test_missing_topics_returns_actionable_hint(corpus):
    assert "book-rag-mcp topics" in search_by_topic("风险", config=corpus)


def test_unscoped_old_labels_are_rejected(corpus):
    build_topics(corpus, LabelLLM(), candidates=(2, 3))
    path = corpus.index_dir / "topics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("label_scope")
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert "旧版描述未限定范围" in search_by_topic("风险", config=corpus)


def test_quota_failure_stops_naming_and_does_not_publish(corpus):
    llm = LabelLLM(LLMError("quota", retryable=False, unavailable=True))
    status = build_topics(corpus, llm, candidates=(2, 3))
    assert not status["complete"] and llm.calls == 1
    assert not (corpus.index_dir / "topics.json").exists()


def test_invalid_evidence_does_not_publish_a_named_topic(corpus):
    class FabricatedLabel(LabelLLM):
        def complete_json(self, *args, **kwargs):
            return {"name": "不可靠主题", "description": "无原文", "evidence_ids": ["R999"]}

    assert not build_topics(corpus, FabricatedLabel(), candidates=(2, 3))["complete"]
    assert not (corpus.index_dir / "topics.json").exists()
