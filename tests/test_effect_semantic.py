"""效果检验（集成层）：真实 bge-m3 的语义检索质量验证（Hybrid + 无重排）。

验证「语义检索真的 work」而非仅仅「不崩」：用换词改写的同义 query 检索
多主题中文书，断言 top-1 命中目标主题、返回内容逐字来自原文、meta 记录
真实模型名。语义测试统一用 ``search_all_languages=True`` 绕开语言过滤，
隔离「embedding 质量」与「语言路由」两个关注点（后者由快速层覆盖）。

运行方式：``pytest -m integration``（默认 addopts 排除本层）。
模型未缓存时自动 skip，绝不触发下载。
"""

import json

import pytest
from conftest import require_model

from book_rag.config import BookRagConfig
from book_rag.rag.index import refresh_all_indexes
from book_rag.rag.retriever import search_books

pytestmark = pytest.mark.integration

ZH_MODEL = "BAAI/bge-m3"
ZH_BOOK_NAME = "生活百科.md"

LIFE_BOOK = """# 生活百科

## 咖啡冲煮
手冲咖啡时，水温最好控制在九十摄氏度上下，研磨度要中等偏细，焖蒸三十秒后再缓慢注水，这样能充分萃取出咖啡的香气。

## 登山安全
登山前要查看天气预报，携带足够的饮水与干粮，在海拔较高的地方要注意预防高原反应，出现头晕时应立即下撤。

## 理财入门
理财的第一步是建立应急金，通常建议预留三到六个月的生活开支；在此基础上再考虑基金定投等中低风险的长期投资。

## 睡眠健康
成年人每晚应保证七到九小时的睡眠，睡前一小时避免使用发光的电子屏幕，保持卧室黑暗安静有助于入睡。
"""

# 换词改写的同义 query → 目标主题（section 路径中的关键字）
QUERY_TOPIC_CASES = [
    ("咖啡怎么冲泡才香？", "咖啡冲煮"),
    ("去爬山要注意什么？", "登山安全"),
    ("怎么开始攒钱投资？", "理财入门"),
    ("晚上睡不好怎么办？", "睡眠健康"),
]


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch):
    """集成测试强制离线：不联网、不发版本检查请求（模型必须已缓存）。"""
    require_model(ZH_MODEL)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> BookRagConfig:
    """在临时目录写一本多主题中文书，用真实模型构建索引（module 级只建一次）。"""
    library = tmp_path_factory.mktemp("library")
    (library / ZH_BOOK_NAME).write_text(LIFE_BOOK, encoding="utf-8")

    cfg = BookRagConfig(
        library_path=library, index_dir=library.parent / "index", rerank_enabled=False
    )
    refresh_all_indexes(cfg)
    return cfg


@pytest.mark.parametrize(("query", "topic"), QUERY_TOPIC_CASES)
def test_semantic_retrieval_hits_target_topic(
    config: BookRagConfig, query: str, topic: str
):
    """换词改写的同义 query 检索，top-1 应命中目标主题。"""
    results = search_books(query, config=config, search_all_languages=True, top_k=1)

    assert results, f"query {query!r} 应返回非空结果"
    assert topic in results[0].section, (
        f"query {query!r} 的 top-1 应命中 {topic!r}，实际 {results[0].section!r}"
    )


def test_retrieved_content_is_verbatim_from_source(config: BookRagConfig):
    """各 query 的 top-k 结果 content 应逐行来自《生活百科》原文。"""
    original = (config.library_path / ZH_BOOK_NAME).read_text(encoding="utf-8")

    for query, _topic in QUERY_TOPIC_CASES:
        results = search_books(
            query, config=config, search_all_languages=True, top_k=3
        )
        assert results
        for r in results:
            # splitter 可能把空行合并为单个换行，按行拆分后逐行比对
            assert all(line in original for line in r.content.splitlines()), (
                f"query {query!r} 的结果不在原文中：{r.content!r}"
            )


def test_meta_records_actual_model_name(config: BookRagConfig):
    """索引 meta 应记录真实使用的模型名与检测出的书籍语言。"""
    meta_files = list(config.index_dir.glob("book-*/meta.json"))
    assert len(meta_files) == 1

    meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
    assert meta["model_name"] == ZH_MODEL
    assert meta["language"] == "zh"
