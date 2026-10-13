"""效果检验（快速层）：真实语言检测 → 真实语言过滤路由的行为验证。

与 test_retriever.py 的区别：那里 mock 了 ``detect_language``，只验证过滤逻辑
本身；这里**不 mock 语言检测**，用真实 ``langdetect`` + CJK 脚本启发式走完整
「query 语言检测 → 同语言书籍过滤」链路，验证真实输入下的路由行为。

仍使用 sha256 假向量（局部定义，沿用现有测试的重复模式），避免下载真实
embedding 模型——本文件只关心语言路由，不关心向量质量。
"""

import hashlib
import math
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from book_rag.config import BookRagConfig
from book_rag.rag import index as index_mod
from book_rag.rag.index import refresh_all_indexes
from book_rag.rag.language import detect_language
from book_rag.rag.retriever import search_books

FAKE_DIM = 16


class FakeHashEmbeddings(Embeddings):
    """基于 sha256 的确定性假 embedding，与 test_index/test_retriever 保持一致。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [b / 255.0 for b in (digest * 2)[:FAKE_DIM]]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]


GUIDE_BOOK = """# 人生指南

## 职业规划
人生设计是一种思考职业方向的方法，可以从兴趣、能力与价值三个维度出发，
逐步找到适合自己的发展路径。

## 读书方法
选书时先看目录与序言，判断这本书是否解决自己的问题；读书时边读边做笔记，
读完后用一页纸复述全书要点。

## 求职招聘
准备简历时突出与岗位匹配的项目经历，面试前要研究公司业务，
并用具体案例说明自己的能力。
"""


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BookRagConfig:
    """构造临时中文书籍库并建好索引；只 mock embedding 模型，语言检测保持真实。"""
    library = tmp_path / "library"
    library.mkdir()
    (library / "人生指南.md").write_text(GUIDE_BOOK, encoding="utf-8")

    # 索引构建侧与检索侧的模型加载均替换为假向量（index 模块内部引用）
    monkeypatch.setattr(
        index_mod, "get_embeddings_for_language", lambda lang: FakeHashEmbeddings()
    )
    monkeypatch.setattr(index_mod, "get_embeddings", lambda name: FakeHashEmbeddings())

    cfg = BookRagConfig(library_path=library, index_dir=tmp_path / "index")
    refresh_all_indexes(cfg)
    return cfg


def test_long_chinese_query_routes_to_zh_book(config: BookRagConfig):
    """长中文 query 经真实检测为 zh，默认路由到中文书且结果非空。"""
    results = search_books("人生设计的方法有哪些", config=config, top_k=3)

    assert results
    for r in results:
        assert r.source == "人生指南.md"


def test_short_chinese_query_not_misrouted(config: BookRagConfig):
    """短中文 query 不再被 langdetect 误判为 ko（依赖 CJK 脚本启发式修复）。"""
    results = search_books("招聘", config=config, top_k=3)

    assert results  # 修复前语言过滤会把中文书全部排除，返回空
    for r in results:
        assert r.source == "人生指南.md"


def test_empty_and_numeric_query_returns_unknown_gracefully(
    config: BookRagConfig,
):
    """空文本与纯数字 query 检测不抛异常，返回可预期的 unknown。"""
    assert detect_language("") == "unknown"
    assert detect_language("123456") == "unknown"

    # unknown 语言视为不过滤，检索照常工作、不抛异常
    results = search_books("123456", config=config, top_k=3)
    assert isinstance(results, list)


def test_mixed_language_query_no_crash(config: BookRagConfig):
    """中英混排 query 不抛异常；含汉字按启发式判为 zh，正常路由到中文书。"""
    results = search_books("How to 学习 Python 编程", config=config, top_k=5)

    assert isinstance(results, list)
    for r in results:
        assert r.source == "人生指南.md"
