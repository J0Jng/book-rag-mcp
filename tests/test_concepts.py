"""概念层（concepts）测试：概念抽取 / 索引落盘加载 / build_book_index 集成 / MCP 工具。

覆盖五类行为：
- ``build_concept_entries``：从 summaries 抽出 framework / term / mental_model，
  text 拼接（「名称：描述」）、section / position（取 pointers[0]）/ source 正确，
  pointers 为空时 position 回落 0；
- ``build_concept_index`` / ``load_concept_index``：concepts.faiss / concepts.pkl /
  concepts.json 落盘与加载闭环，空列表不建索引，缺失 concepts.json 返回 None；
- ``build_book_index`` 集成：配置 LLM 时落盘 summaries.json 与概念索引，
  摘要层异常只降级不阻塞；无 LLM 时索引照常构建、不产生摘要层文件；
- ``retrieve_concepts``：命中返回含概念的 Markdown，索引未构建返回提示。

为避免下载真实 HuggingFace 模型，embedding 使用与 test_index.py 相同的
基于 sha256 的确定性假向量；LLM 用固定返回值的假对象，不触发真实网络。
"""

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from book_rag.config import BookRagConfig
from book_rag.rag import index as index_mod
from book_rag.rag.concepts import (
    Concept,
    build_concept_entries,
    build_concept_index,
    load_concept_index,
)
from book_rag.rag.index import build_book_index, get_book_index_dir
from book_rag.servers import mcp_server

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_DIM = 16


class FakeHashEmbeddings(Embeddings):
    """基于 sha256 的确定性假 embedding，与 test_index.py 保持一致。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [b / 255.0 for b in (digest * 2)[:FAKE_DIM]]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]


SUMMARY = {
    "summary": "旅人从起点出发。",
    "frameworks": [{"name": "旅程框架", "description": "起点到终点的结构"}],
    "terms": [{"name": "迷途", "definition": "森林深处没有路标的状态"}],
    "mental_models": ["迷路时沿溪水下行"],
    "section": "第一章 起点",
    "pointers": [{"section": "测试之书 > 第一章 起点", "position": 3}],
}


class FakeLLM:
    """固定返回同一份 summary 的假 LLM（summarize_sections 会附加真实 section/pointers）。"""

    def __init__(self, payload: dict | None = None):
        self._payload = payload if payload is not None else SUMMARY
        self.calls = 0

    def complete_json(self, system: str, user: str, **kwargs) -> dict:
        self.calls += 1
        return copy.deepcopy(self._payload)


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BookRagConfig:
    """构造临时书籍库 + 索引目录，并 mock 掉真实模型与语言检测（不建索引）。"""
    library = tmp_path / "library"
    library.mkdir()
    shutil.copy(FIXTURES / "sample_book.md", library / "sample_book.md")

    # 索引构建侧（index 模块内部引用）
    monkeypatch.setattr(
        index_mod, "get_embeddings_for_language", lambda lang: FakeHashEmbeddings()
    )
    monkeypatch.setattr(index_mod, "get_embeddings", lambda name: FakeHashEmbeddings())
    monkeypatch.setattr(index_mod, "detect_language", lambda text: "zh-cn")
    # MCP 工具侧加载概念索引用
    monkeypatch.setattr(mcp_server, "get_embeddings", lambda name: FakeHashEmbeddings())

    return BookRagConfig(library_path=library, index_dir=tmp_path / "index")


# ---------------------------------------------------------------------------
# build_concept_entries
# ---------------------------------------------------------------------------


def test_build_concept_entries_extracts_all_kinds():
    """frameworks / terms / mental_models 各自抽出 Concept，字段按 brief 拼接。"""
    concepts = build_concept_entries([SUMMARY], source="sample_book.md")

    kinds = [c.kind for c in concepts]
    assert kinds == ["framework", "term", "mental_model"]

    fw = concepts[0]
    assert fw.text == "旅程框架：起点到终点的结构"
    assert fw.section == "第一章 起点"
    assert fw.position == 3  # 取 pointers[0]["position"]
    assert fw.source == "sample_book.md"

    term = concepts[1]
    assert term.text == "迷途：森林深处没有路标的状态"
    assert term.kind == "term"

    mm = concepts[2]
    assert mm.text == "迷路时沿溪水下行"  # 心智模型直接用字符串
    assert mm.kind == "mental_model"
    assert mm.section == "第一章 起点"
    assert mm.position == 3


def test_build_concept_entries_empty_pointers_fall_back_to_zero():
    """pointers 为空 / 缺失时 position 回落 0；空列表返回空结果。"""
    no_pointers = {k: v for k, v in SUMMARY.items() if k != "pointers"}
    concepts = build_concept_entries([no_pointers], source="x.md")

    assert concepts
    assert all(c.position == 0 for c in concepts)

    assert build_concept_entries([]) == []


def test_build_concept_entries_filters_placeholder_entries():
    """框架/术语的 name 为空或含 [未检测到]、心智模型为空或含 [未检测到] 时跳过，
    有效条目保留。"""
    summary = {
        "section": "某章",
        "frameworks": [
            {"name": "[未检测到]", "description": "[未检测到]"},  # 过滤
            {"name": "有效框架", "description": "des"},  # 保留
        ],
        "terms": [
            {"name": "", "definition": "空名"},  # 过滤
            {"name": "有效术语", "definition": "def"},  # 保留
        ],
        "mental_models": [
            "[未检测到]",  # 过滤
            "",  # 过滤
            "正常心智模型",  # 保留
        ],
        "pointers": [{"section": "x", "position": 1}],
    }
    concepts = build_concept_entries([summary], source="a.md")

    assert [(c.kind, c.text) for c in concepts] == [
        ("framework", "有效框架：des"),
        ("term", "有效术语：def"),
        ("mental_model", "正常心智模型"),
    ]


def test_concept_dict_roundtrip():
    """to_dict / from_dict 互为逆操作。"""
    c = Concept(text="术语：定义", kind="term", section="第一章", position=2, source="a.md")

    assert Concept.from_dict(c.to_dict()) == c


# ---------------------------------------------------------------------------
# build_concept_index / load_concept_index
# ---------------------------------------------------------------------------


def test_build_and_load_concept_index_roundtrip(tmp_path: Path):
    """建索引落盘三个文件，load 还原出 (vector_store, list[Concept])，数量一致。"""
    concepts = build_concept_entries([SUMMARY], source="sample_book.md")

    out = build_concept_index(concepts, FakeHashEmbeddings(), tmp_path)

    assert out == tmp_path / "concepts.json"
    assert (tmp_path / "concepts.faiss").is_file()
    assert (tmp_path / "concepts.pkl").is_file()
    data = json.loads((tmp_path / "concepts.json").read_text(encoding="utf-8"))
    assert len(data) == len(concepts)
    assert data[0]["text"] == "旅程框架：起点到终点的结构"

    loaded = load_concept_index(tmp_path, FakeHashEmbeddings())
    assert loaded is not None
    vector_store, loaded_concepts = loaded
    assert loaded_concepts == concepts
    hits = vector_store.similarity_search_with_score(
        "旅程框架：起点到终点的结构", k=1
    )
    assert hits[0][0].page_content == "旅程框架：起点到终点的结构"
    assert hits[0][0].metadata["kind"] == "framework"


def test_build_concept_index_empty_returns_none_without_files(tmp_path: Path):
    """空概念列表不建索引、不落盘，返回 None。"""
    assert build_concept_index([], FakeHashEmbeddings(), tmp_path) is None
    assert not (tmp_path / "concepts.json").exists()
    assert not (tmp_path / "concepts.faiss").exists()


def test_load_concept_index_missing_returns_none(tmp_path: Path):
    """concepts.json 不存在时返回 None，而不是抛异常。"""
    assert load_concept_index(tmp_path, FakeHashEmbeddings()) is None


# ---------------------------------------------------------------------------
# build_book_index 集成（摘要 + 概念层）
# ---------------------------------------------------------------------------


def test_build_book_index_with_llm_writes_summaries_and_concepts(
    config: BookRagConfig, monkeypatch: pytest.MonkeyPatch
):
    """配置 LLM 时：summaries.json / concepts.json / concepts.faiss 落盘，
    原有 index.faiss / toc.json / meta.json 照常。"""
    monkeypatch.setattr(index_mod, "load_llm", lambda: FakeLLM())

    meta = build_book_index(config.library_path / "sample_book.md", config)

    idx_dir = get_book_index_dir(config.index_dir, Path(meta.book_path))
    assert (idx_dir / "index.faiss").is_file()
    assert (idx_dir / "toc.json").is_file()
    assert (idx_dir / "meta.json").is_file()

    summaries = json.loads((idx_dir / "summaries.json").read_text(encoding="utf-8"))
    assert summaries  # sample_book.md 有章节，LLM 假对象每章都返回固定摘要
    for item in summaries:
        assert item["section"]
        assert item["pointers"]

    concept_data = json.loads((idx_dir / "concepts.json").read_text(encoding="utf-8"))
    assert len(concept_data) == len(summaries) * 3  # 每章固定抽出 3 类概念各 1 条
    assert (idx_dir / "concepts.faiss").is_file()
    assert (idx_dir / "concepts.pkl").is_file()
    # source 是书籍相对路径
    assert all(c["source"] == "sample_book.md" for c in concept_data)


def test_build_book_index_summarizer_failure_degrades(
    config: BookRagConfig, monkeypatch: pytest.MonkeyPatch
):
    """摘要层抛异常时只跳过，索引照常构建，不产生摘要层文件。"""
    monkeypatch.setattr(index_mod, "load_llm", lambda: FakeLLM())

    def _boom(*args, **kwargs):
        raise RuntimeError("LLM 层炸了")

    monkeypatch.setattr(index_mod, "summarize_sections", _boom)

    meta = build_book_index(config.library_path / "sample_book.md", config)

    idx_dir = get_book_index_dir(config.index_dir, Path(meta.book_path))
    assert (idx_dir / "index.faiss").is_file()
    assert (idx_dir / "meta.json").is_file()
    assert not (idx_dir / "summaries.json").exists()
    assert not (idx_dir / "concepts.json").exists()


def test_build_book_index_without_llm_skips_summary_layer(
    config: BookRagConfig, monkeypatch: pytest.MonkeyPatch
):
    """无 LLM（load_llm 返回 None）：索引 / 目录树 / meta 照常，无摘要层文件，不抛异常。"""
    monkeypatch.setattr(index_mod, "load_llm", lambda: None)

    meta = build_book_index(config.library_path / "sample_book.md", config)

    idx_dir = get_book_index_dir(config.index_dir, Path(meta.book_path))
    assert (idx_dir / "index.faiss").is_file()
    assert (idx_dir / "toc.json").is_file()
    assert (idx_dir / "meta.json").is_file()
    assert not (idx_dir / "summaries.json").exists()
    assert not (idx_dir / "concepts.json").exists()
    assert not (idx_dir / "concepts.faiss").exists()


# ---------------------------------------------------------------------------
# retrieve_concepts（MCP 工具）
# ---------------------------------------------------------------------------


def test_retrieve_concepts_returns_markdown_hits(
    config: BookRagConfig, monkeypatch: pytest.MonkeyPatch
):
    """概念索引已构建：按精确概念文本查询命中，返回含概念与相关度的 Markdown。"""
    monkeypatch.setattr(index_mod, "load_llm", lambda: FakeLLM())
    build_book_index(config.library_path / "sample_book.md", config)

    out = mcp_server.retrieve_concepts(
        "旅程框架：起点到终点的结构", "sample_book.md", config=config
    )

    assert "旅程框架" in out
    assert "相关度" in out
    assert "第一章 起点" in out  # 概念锚定章节


def test_retrieve_concepts_missing_index_returns_hint(
    config: BookRagConfig, monkeypatch: pytest.MonkeyPatch
):
    """概念索引未构建：返回提示而非抛异常。"""
    monkeypatch.setattr(index_mod, "load_llm", lambda: None)
    build_book_index(config.library_path / "sample_book.md", config)

    out = mcp_server.retrieve_concepts("旅程框架", "sample_book.md", config=config)

    assert "概念索引未构建" in out


def test_retrieve_concepts_unknown_book(config: BookRagConfig):
    """无匹配书籍时返回提示，而不是抛异常。"""
    out = mcp_server.retrieve_concepts("任意查询", "不存在的书", config=config)

    assert "未找到匹配的书籍" in out
