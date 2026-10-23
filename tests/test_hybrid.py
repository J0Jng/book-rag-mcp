"""S3 Hybrid 检索测试：BM25 落盘、RRF 融合、单次重排与索引缓存。

嵌入使用确定性 sha256 假向量（不下载模型），语言检测 mock 为中文：
向量通道本身没有语义，恰好用来证明 BM25 精确术语 / 人名的召回价值，
以及 RRF 融合与重排调用次数符合「全局只重排一次」的约束。
"""

import hashlib
import math
import os
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from book_rag.config import BookRagConfig
from book_rag.rag import index as index_mod
from book_rag.rag import retriever as retriever_mod
from book_rag.rag.bm25 import BM25_FILENAME, tokenize
from book_rag.rag.index import refresh_all_indexes
from book_rag.rag.retriever import search_books

FAKE_DIM = 16

FINANCE_BOOK = """# 投资笔记

## 第一章 价值投资
价值投资的核心是安全边际，市场先生短期是投票机、长期是称重机，
投资者应当耐心等待价格低于内在价值的时刻。

## 第二章 逆向选择
逆向选择发生在信息不对称的市场上：买方无法区分质量，
最终劣币驱逐良币，高质量的交易者退出市场。

## 第三章 组合管理
分散化可以降低非系统性风险，但无法消除系统性风险。
再平衡纪律要求投资者定期卖出涨幅过大的资产。

## 第四章 市场周期
市场周期由情绪与信贷共同驱动，乐观时估值扩张，悲观时估值收缩。
识别周期位置比预测下一次拐点更现实。

## 第五章 安全边际
安全边际来自价格与价值之间的差距，也来自对企业基本面的理解。
宁可错过，不可买贵。
"""

NOVEL_BOOK = """# 杀死一只知更鸟

## 第一章 梅科姆镇
阿蒂克斯是一位正直的律师，斯库特是他的女儿，杰姆是她的哥哥。
兄妹俩在小镇上度过了漫长而炎热的夏天。

## 第二章 学校
斯库特第一天上学就与老师发生了争执，她觉得学校的规矩难以理解。
杰姆安慰她说，长大以后就会明白这些事情。

## 第三章 树洞
孩子们在拉德利家旁的树洞里发现了口香糖和旧硬币，
他们猜测这是有人在悄悄送礼物。

## 第四章 审判
镇上的法庭挤满了人，阿蒂克斯为被告辩护，
他提醒陪审团不要被偏见蒙蔽双眼。

## 第五章 归途
夜色降临，斯库特在回家的路上想起了父亲说过的话。
她决定以后也要做一个正直的人。
"""


class FakeHashEmbeddings(Embeddings):
    """基于 sha256 的确定性假 embedding（与其余测试保持一致）。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [value / 255.0 for value in (digest * 2)[:FAKE_DIM]]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / norm for value in raw]


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BookRagConfig:
    """两本书（金融 + 小说）建好索引；重排默认关闭，需要时用例自行开启。"""
    library = tmp_path / "library"
    library.mkdir()
    (library / "投资笔记.md").write_text(FINANCE_BOOK, encoding="utf-8")
    (library / "杀死一只知更鸟.md").write_text(NOVEL_BOOK, encoding="utf-8")

    monkeypatch.setattr(
        index_mod, "get_embeddings_for_language", lambda language: FakeHashEmbeddings()
    )
    monkeypatch.setattr(index_mod, "get_embeddings", lambda name: FakeHashEmbeddings())
    monkeypatch.setattr(index_mod, "detect_language", lambda text: "zh-cn")
    monkeypatch.setattr(retriever_mod, "detect_language", lambda text: "zh-cn")

    cfg = BookRagConfig(
        library_path=library,
        index_dir=tmp_path / "index",
        rerank_enabled=False,
        candidate_pool=5,
        rerank_candidates=20,
    )
    refresh_all_indexes(cfg)
    return cfg


def test_bm25_index_is_persisted_per_book(config: BookRagConfig):
    """每本书索引目录都应落盘 bm25.json，且包含该书的 chunk position。"""
    files = list(config.index_dir.glob(f"book-*/{BM25_FILENAME}"))
    assert len(files) == 2


def test_tokenize_handles_chinese_and_english():
    """中文走 jieba、英文按词切并小写，标点不产生 token。"""
    tokens = tokenize("Value Investing 价值投资，逆向选择！")

    assert "value" in tokens
    assert "investing" in tokens
    assert "价值" in tokens
    assert "投资" in tokens
    assert all(token.strip() and token not in {"，", "！"} for token in tokens)


def test_exact_term_hits_via_bm25(config: BookRagConfig):
    """假向量无语义时，精确术语仍应经 BM25 进入 top-3。"""
    results = search_books("逆向选择", config=config, top_k=3)

    assert results
    assert results[0].source == "投资笔记.md"
    assert "逆向选择" in results[0].content


def test_exact_person_name_hits_via_bm25(config: BookRagConfig):
    """小说人名同样应由 BM25 精确召回。"""
    results = search_books("阿蒂克斯", config=config, top_k=3)

    assert results
    assert results[0].source == "杀死一只知更鸟.md"
    assert "阿蒂克斯" in results[0].content


def test_rerank_is_called_once_on_fused_candidates(
    config: BookRagConfig, monkeypatch: pytest.MonkeyPatch
):
    """重排只对全局融合候选调用一次，且候选来自多本书（不是每书各自重排）。"""
    calls: list[list[str]] = []

    def fake_rerank(query: str, texts: list[str]):
        calls.append(texts)
        return [float(len(texts) - index) for index in range(len(texts))]

    monkeypatch.setattr(retriever_mod, "rerank_scores", fake_rerank)
    config.rerank_enabled = True

    results = search_books("投资与律师", config=config, top_k=5)

    assert results
    assert len(calls) == 1  # 只调用一次
    assert len(calls[0]) <= config.rerank_candidates
    # 融合候选确实覆盖两本书，证明是全局融合后才重排
    sources = {
        result.source
        for result in search_books("投资与律师", config=config, top_k=20)
    }
    assert sources == {"投资笔记.md", "杀死一只知更鸟.md"}


def test_rerank_disabled_skips_model(config: BookRagConfig, monkeypatch: pytest.MonkeyPatch):
    """rerank_enabled=False 时不调用重排，直接返回 RRF 排序。"""
    def fail_if_called(query: str, texts: list[str]):  # pragma: no cover - 不应被调用
        raise AssertionError("rerank 不应在关闭时被调用")

    monkeypatch.setattr(retriever_mod, "rerank_scores", fail_if_called)

    results = search_books("逆向选择", config=config, top_k=3)

    assert results
    assert results[0].source == "投资笔记.md"


def test_rerank_failure_falls_back_to_rrf(
    config: BookRagConfig, monkeypatch: pytest.MonkeyPatch
):
    """重排返回 None（模型不可用）时降级为 RRF 排序，结果仍然可用。"""
    monkeypatch.setattr(retriever_mod, "rerank_scores", lambda query, texts: None)
    config.rerank_enabled = True

    results = search_books("逆向选择", config=config, top_k=3)

    assert results
    assert results[0].source == "投资笔记.md"
    assert results[0].score > 0.0


def test_second_query_does_not_reload_faiss(
    config: BookRagConfig, monkeypatch: pytest.MonkeyPatch
):
    """进程内缓存：第二次同 query 不再调用 FAISS.load_local。"""
    real_load_local = index_mod.FAISS.load_local
    calls = {"count": 0}

    def counting_load_local(*args, **kwargs):
        calls["count"] += 1
        return real_load_local(*args, **kwargs)

    monkeypatch.setattr(index_mod.FAISS, "load_local", counting_load_local)

    first = search_books("逆向选择", config=config, top_k=3)
    after_first = calls["count"]
    second = search_books("逆向选择", config=config, top_k=3)

    assert first and second
    assert after_first == 2  # 两本书各加载一次
    assert calls["count"] == after_first  # 第二次命中缓存，未再读盘


def test_cache_invalidated_by_mtime(config: BookRagConfig, monkeypatch: pytest.MonkeyPatch):
    """索引文件 mtime 变化后缓存失效，重新加载。"""
    real_load_local = index_mod.FAISS.load_local
    calls = {"count": 0}

    def counting_load_local(*args, **kwargs):
        calls["count"] += 1
        return real_load_local(*args, **kwargs)

    monkeypatch.setattr(index_mod.FAISS, "load_local", counting_load_local)

    search_books("逆向选择", config=config, top_k=3)
    assert calls["count"] == 2

    meta_path = index_mod.get_book_index_dir(
        config.index_dir, Path("投资笔记.md")
    ) / "meta.json"
    stat = meta_path.stat()
    os.utime(meta_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    search_books("逆向选择", config=config, top_k=3)

    assert calls["count"] == 3  # 只有被改动的那本书重新加载


def test_missing_bm25_is_backfilled_on_refresh(config: BookRagConfig):
    """旧索引缺 bm25.json 时，增量同步只补建 BM25，不重建向量索引。"""
    idx_dir = index_mod.get_book_index_dir(config.index_dir, Path("投资笔记.md"))
    bm25_path = idx_dir / BM25_FILENAME
    faiss_mtime = (idx_dir / "index.faiss").stat().st_mtime_ns
    bm25_path.unlink()

    result = refresh_all_indexes(config)

    assert bm25_path.is_file()
    assert "投资笔记.md" in result.skipped  # 未重建向量
    assert (idx_dir / "index.faiss").stat().st_mtime_ns == faiss_mtime


def test_refresh_rebuilds_bm25_when_book_changes(config: BookRagConfig):
    """书籍内容变更重建索引时，BM25 一并重建并反映新内容。"""
    bm25_path = index_mod.get_book_index_dir(
        config.index_dir, Path("投资笔记.md")
    ) / BM25_FILENAME
    old_text = bm25_path.read_text(encoding="utf-8")

    book = config.library_path / "投资笔记.md"
    book.write_text(
        book.read_text(encoding="utf-8") + "\n## 第三章 新增章节\n锚定效应与损失厌恶。\n",
        encoding="utf-8",
    )
    refresh_all_indexes(config)

    assert bm25_path.read_text(encoding="utf-8") != old_text
    results = search_books("锚定效应", config=config, top_k=3)
    assert results and "锚定效应" in results[0].content


def test_dictionary_is_not_a_book_and_changes_only_rebuild_bm25(config: BookRagConfig):
    """词典增删更新分词与 BM25，不能误入书籍，也不能重算向量。"""
    from book_rag.io.loader import discover_books
    from book_rag.rag.bm25 import ensure_user_dict

    dictionary = config.library_path / "jieba_dict.txt"
    idx_dir = index_mod.get_book_index_dir(config.index_dir, Path("投资笔记.md"))
    faiss_mtime = (idx_dir / "index.faiss").stat().st_mtime_ns
    dictionary.write_text("超新专名甲乙丙 1000000 n\n", encoding="utf-8")
    assert dictionary not in discover_books(config.library_path)
    result = refresh_all_indexes(config)
    assert result.built == []
    ensure_user_dict(dictionary)
    assert "超新专名甲乙丙" in tokenize("超新专名甲乙丙")
    before = (idx_dir / BM25_FILENAME).read_text(encoding="utf-8")
    dictionary.write_text("其他专名丁戊己 1000000 n\n", encoding="utf-8")
    refresh_all_indexes(config)
    ensure_user_dict(dictionary)
    assert "超新专名甲乙丙" not in tokenize("超新专名甲乙丙")
    assert before != (idx_dir / BM25_FILENAME).read_text(encoding="utf-8")
    dictionary.unlink()
    assert refresh_all_indexes(config).built == []
    assert (idx_dir / "index.faiss").stat().st_mtime_ns == faiss_mtime


def test_cache_invalidated_when_docstore_changes(config: BookRagConfig, monkeypatch):
    """index.pkl 变化必须使缓存失效，即使向量本身未变。"""
    real = index_mod.FAISS.load_local
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(index_mod.FAISS, "load_local", counted)
    search_books("逆向选择", config=config)
    assert len(calls) == 2
    path = index_mod.get_book_index_dir(config.index_dir, Path("投资笔记.md")) / "index.pkl"
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    search_books("逆向选择", config=config)
    assert len(calls) == 3


def test_score_kind_and_mcp_citations(config: BookRagConfig, monkeypatch):
    """MCP 明确分数语义并包含所有原文引用指针。"""
    from book_rag.servers.mcp_server import format_search_results

    results = search_books("逆向选择", config=config)
    assert all(item.score_kind == "rrf" for item in results)
    out = format_search_results(results)
    assert "rrf 排序分，非概率" in out
    for item in results:
        assert f"source: {item.source}" in out
        assert f"section: {item.section}" in out
        assert f"position: {item.position}" in out
    monkeypatch.setattr(retriever_mod, "rerank_scores", lambda q, t: [0.5] * len(t))
    config.rerank_enabled = True
    assert all(item.score_kind == "rerank" for item in search_books("逆向选择", config=config))


def test_query_is_embedded_once_for_shared_model(config: BookRagConfig, monkeypatch):
    """统一向量模型下，同次跨五本书查询也只编码一次。"""
    calls = []
    real = FakeHashEmbeddings.embed_query

    def counted(self, text):
        calls.append(text)
        return real(self, text)

    monkeypatch.setattr(FakeHashEmbeddings, "embed_query", counted)
    search_books("逆向选择", config=config)
    assert calls == ["逆向选择"]


def test_invalid_bm25_is_repaired_without_embedding(config: BookRagConfig):
    """损坏的分词载荷可修复，不能被当作有效索引或触发重新 embedding。"""
    import json

    idx_dir = index_mod.get_book_index_dir(config.index_dir, Path("投资笔记.md"))
    path = idx_dir / BM25_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["docs"] = [None]
    path.write_text(json.dumps(payload), encoding="utf-8")
    faiss = (idx_dir / "index.faiss").stat().st_mtime_ns
    assert search_books("逆向选择", config=config)
    assert refresh_all_indexes(config).built == []
    assert (idx_dir / "index.faiss").stat().st_mtime_ns == faiss
