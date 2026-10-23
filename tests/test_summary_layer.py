"""S4 体裁路由、可恢复发布、真实指针与失败降级的行为验证。"""

import copy
import io
import json
import urllib.error
import urllib.request

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from book_rag.config import BookRagConfig
from book_rag.io.parsers import BookSection
from book_rag.rag import index as index_mod
from book_rag.rag.concepts import build_concept_entries
from book_rag.rag.llm import LLMError, SummarizerLLM
from book_rag.rag.summarizer import summarize_sections
from book_rag.rag.summary_layer import write_summary_layer
from book_rag.servers.mcp_server import get_book_summaries, retrieve_concepts

FICTION = {
    "summary": "小明回家。", "plot": "小明离开学校，走回家中。",
    "characters": ["小明"], "turning_points": [], "suspense": [],
    "evidence": ["小明回家。"],
}
KNOWLEDGE = {"summary": "风险需要管理。", "frameworks": [], "terms": [], "mental_models": []}


class StubLLM:
    def __init__(self, data=None, error=None):
        self.data = data or FICTION
        self.error = error
        self.calls = 0

    def complete_json(self, *args, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return copy.deepcopy(self.data)


class StubEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def inputs():
    sections = [BookSection("上部 > 第一章", 2, "小明回家。", 0),
                BookSection("下部 > 第一章", 2, "小明回家。", 1)]
    docs = [Document(page_content=s.text, metadata={
        "source": "novel.md", "section": s.path, "position": i * 8,
    }) for i, s in enumerate(sections)]
    return sections, docs


def write_layer(path, llm, *, genre="fiction"):
    sections, docs = inputs()
    return write_summary_layer(
        sections, docs, llm, idx_dir=path, source="novel.md", title="小说",
        genre=genre, book_hash="abc", embeddings_factory=lambda: StubEmbeddings(),
    )


def test_fiction_full_paths_and_chunk_positions():
    sections, docs = inputs()
    result = summarize_sections(sections, StubLLM(), genre="fiction", source="novel.md", documents=docs)
    assert [s["section"] for s in result] == [s.path for s in sections]
    assert result[1]["pointers"] == [{"source": "novel.md", "section": "下部 > 第一章", "position": 8}]
    assert all(s["genre"] == "fiction" and s["position_kind"] == "chunk" for s in result)
    assert build_concept_entries(result) == []


def test_typographical_whitespace_maps_back_to_exact_original():
    original = "西蒙·\u2009芬奇回家。\n他见到了小明。"
    data = copy.deepcopy(FICTION)
    data["characters"] = ["西蒙·芬奇"]
    data["evidence"] = ["西蒙·芬奇回家。他见到了小明。"]
    result = summarize_sections([BookSection("第一章", 1, original, 0)],
                                StubLLM(data), genre="fiction")
    assert result[0]["characters"] == ["西蒙·\u2009芬奇"]
    assert result[0]["evidence"] == [original]


def test_evidence_ids_are_resolved_to_original_and_unknown_ids_rejected():
    data = copy.deepcopy(FICTION)
    data["evidence"] = ["P0001"]
    sections = inputs()[0][:1]
    assert summarize_sections(sections, StubLLM(data), genre="fiction")[0]["evidence"] == ["小明回家。"]
    data["evidence"] = ["P9999"]
    assert summarize_sections(sections, StubLLM(data), genre="fiction") == []


@pytest.mark.parametrize("field,value", [("characters", ["不存在的人"]),
                                        ("evidence", ["编造的原文"]), ("evidence", []),
                                        ("plot", ""), ("characters", "小明")])
def test_fiction_rejects_ungrounded_or_malformed_output(field, value):
    data = copy.deepcopy(FICTION)
    data[field] = value
    llm = StubLLM(data)
    assert summarize_sections(inputs()[0][:1], llm, genre="fiction") == []
    assert llm.calls == 2


def test_partial_checkpoint_resumes_and_only_publishes_complete(tmp_path):
    class FailSecond(StubLLM):
        def complete_json(self, *args, **kwargs):
            if "下部" in args[1]:
                raise LLMError("offline", retryable=False, unavailable=True)
            return super().complete_json(*args, **kwargs)

    partial = write_layer(tmp_path, FailSecond())
    assert not partial["complete"] and partial["completed"] == 1
    assert (tmp_path / "summaries.pending.json").is_file()
    assert not (tmp_path / "summaries.json").exists()
    recovered = StubLLM()
    status = write_layer(tmp_path, recovered)
    assert status["complete"] and status["completed"] == 2
    assert recovered.calls == 1  # 首章断点不再付费调用
    assert not (tmp_path / "concepts.faiss").exists()
    assert write_layer(tmp_path, None)["complete"]  # 无 key 不抹掉已有完整结果


def test_corrupted_cached_content_is_regenerated():
    sections, docs = inputs()
    existing = summarize_sections(sections, StubLLM(), genre="fiction", documents=docs)
    existing[0]["evidence"] = ["伪造"]
    llm = StubLLM()
    result = summarize_sections(sections, llm, genre="fiction", documents=docs, existing=existing)
    assert len(result) == 2 and llm.calls == 1


def test_parallel_summary_order_and_resume():
    sections, docs = inputs()
    llm = StubLLM()
    progress = []
    result = summarize_sections(sections, llm, genre="fiction", documents=docs,
                                workers=4, on_progress=lambda items: progress.append(len(items)))
    assert [s["section"] for s in result] == [s.path for s in sections]
    assert progress[-1] == 2 and llm.calls == 2
    assert summarize_sections(sections, StubLLM(error=AssertionError("should not call")),
                              genre="fiction", documents=docs, workers=4, existing=result) == result


def test_unknown_genre_never_calls_llm(tmp_path, caplog):
    llm = StubLLM()
    status = write_layer(tmp_path, llm, genre="unknown")
    assert not status["complete"] and llm.calls == 0
    assert "摘要层跳过" in caplog.text


@pytest.mark.parametrize("error", [None, LLMError("network", unavailable=True)])
def test_core_index_survives_missing_key_and_network(tmp_path, monkeypatch, caplog, error):
    library = tmp_path / "books"
    library.mkdir()
    (library / "a.md").write_text("# 测试书\n## 第一章\n小明回家。", encoding="utf-8")
    (library / "workspaces.yaml").write_text(
        "workspaces:\n  f: {genre: fiction}\nbooks:\n  a.md: {workspaces: [f]}\n", encoding="utf-8",
    )
    monkeypatch.setattr(index_mod, "get_embeddings_for_language", lambda _: StubEmbeddings())
    monkeypatch.setattr(index_mod, "get_embeddings", lambda _: StubEmbeddings())
    monkeypatch.setattr(index_mod, "detect_language", lambda _: "zh")
    monkeypatch.setattr(index_mod, "load_llm", lambda: StubLLM(error=error) if error else None)
    config = BookRagConfig(library_path=library, index_dir=tmp_path / "index")
    meta = index_mod.build_book_index(library / "a.md", config)
    folder = index_mod.get_book_index_dir(config.index_dir, library.relative_to(library) / "a.md")
    assert meta.chunk_count > 0 and (folder / "index.faiss").is_file()
    assert not (folder / "summaries.json").exists()
    assert "摘要" in caplog.text


def test_quota_error_redacts_message_and_stops_requests(monkeypatch):
    calls = []

    def quota(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("https://example.com", 429, "quota", {}, io.BytesIO(
            b'{"error":{"code":"AccountQuotaExceeded","message":"secret test-key"}}',
        ))

    monkeypatch.setattr(urllib.request, "urlopen", quota)
    llm = SummarizerLLM("https://example.com", "test-key", "model")
    with pytest.raises(LLMError) as caught:
        llm.complete_json("sys", "user")
    assert not caught.value.retryable and caught.value.unavailable
    assert "test-key" not in str(caught.value) and "AccountQuotaExceeded" in str(caught.value)
    calls.clear()
    assert summarize_sections(inputs()[0], llm, genre="fiction") == []
    assert len(calls) == 1


def test_read_only_summary_pagination_staleness_and_fiction_hint(tmp_path, monkeypatch):
    from book_rag.servers import mcp_server

    config = BookRagConfig(library_path=tmp_path, index_dir=tmp_path)
    meta = index_mod.BookIndexMeta("novel.md", "abc", "zh", "unused", 16, "now", genre="fiction")
    monkeypatch.setattr(mcp_server, "_find_matching_metas", lambda *_: [meta])
    monkeypatch.setattr(mcp_server, "get_book_index_dir", lambda *_: tmp_path)
    write_layer(tmp_path, StubLLM())
    monkeypatch.setattr(mcp_server, "get_embeddings", lambda *_: pytest.fail("read should not embed"))
    out = get_book_summaries("novel", config=config, limit=1)
    assert "offset=1" in out and "source: novel.md" in out
    second = get_book_summaries("novel", config=config, offset=1)
    assert "position: 8" in second and "下部 > 第一章" in second
    assert "仅对 knowledge" in retrieve_concepts("query", "novel", config=config)
    meta.book_hash = "changed"
    assert "过期" in get_book_summaries("novel", config=config)
    # 读取不会改变已保存的摘要
    assert len(json.loads((tmp_path / "summaries.json").read_text(encoding="utf-8"))) == 2
