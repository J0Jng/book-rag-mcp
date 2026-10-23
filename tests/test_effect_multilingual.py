"""效果检验（集成层）：单模型 bge-m3 下的跨语言检索验证。

S3 起所有语言统一使用 bge-m3（不再有语言专用模型）。本文件验证：中文书与
日文书都记录 bge-m3、默认按 query 语言过滤只搜中文书、``search_all_languages=True``
时两本书都参与检索。

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

EMBEDDING_MODEL = "BAAI/bge-m3"
ZH_BOOK_NAME = "生活百科.md"
JA_BOOK_NAME = "日本語テストブック.md"

ZH_BOOK = """# 生活百科

## 咖啡冲煮
手冲咖啡时，水温最好控制在九十摄氏度上下，研磨度要中等偏细，焖蒸三十秒后再缓慢注水，这样能充分萃取出咖啡的香气。

## 登山安全
登山前要查看天气预报，携带足够的饮水与干粮，在海拔较高的地方要注意预防高原反应，出现头晕时应立即下撤。

## 理财入门
理财的第一步是建立应急金，通常建议预留三到六个月的生活开支；在此基础上再考虑基金定投等中低风险的长期投资。

## 睡眠健康
成年人每晚应保证七到九小时的睡眠，睡前一小时避免使用发光的电子屏幕，保持卧室黑暗安静有助于入睡。
"""

# 日文示例语料（含假名 → 检测为 ja；embedding 与中文书同为 bge-m3）
JA_BOOK = """# 日本語テストブック

## 山登りの心得
これは山登りと健康に関する日本語のテスト文章です。山に登る前には天気予報を確認し、水と食料を十分に持っていきましょう。

## 健康と睡眠
これは山登りと健康に関する日本語のテスト文章です。毎晩七時間から九時間の睡眠をとり、寝る前は部屋を暗く静かに保つことが大切です。
"""


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch):
    """集成测试强制离线，且两个模型都必须已缓存（缺一即 skip）。"""
    require_model(EMBEDDING_MODEL)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> BookRagConfig:
    """写一本中文书 + 一本日文书，用真实模型分别构建索引。"""
    library = tmp_path_factory.mktemp("library")
    (library / ZH_BOOK_NAME).write_text(ZH_BOOK, encoding="utf-8")
    (library / JA_BOOK_NAME).write_text(JA_BOOK, encoding="utf-8")

    cfg = BookRagConfig(
        library_path=library, index_dir=library.parent / "index", rerank_enabled=False
    )
    refresh_all_indexes(cfg)
    return cfg


def _read_metas(config: BookRagConfig) -> dict[str, dict]:
    """按书籍相对路径读取各书的 meta.json。"""
    metas: dict[str, dict] = {}
    for meta_file in config.index_dir.glob("book-*/meta.json"):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        metas[meta["book_path"]] = meta
    return metas


def test_all_books_use_single_bge_m3_model(config: BookRagConfig):
    """中/日文书都使用统一的 bge-m3；语言仍作为元数据保留。"""
    metas = _read_metas(config)
    assert set(metas) == {ZH_BOOK_NAME, JA_BOOK_NAME}

    assert metas[JA_BOOK_NAME]["model_name"] == EMBEDDING_MODEL
    assert metas[JA_BOOK_NAME]["language"] == "ja"
    assert metas[ZH_BOOK_NAME]["model_name"] == EMBEDDING_MODEL
    assert metas[ZH_BOOK_NAME]["language"] == "zh"


def test_default_filters_by_query_language(config: BookRagConfig):
    """长中文 query 默认只搜中文书，日文书不应出现在结果中。"""
    results = search_books(
        "去爬山之前要注意什么？晚上睡不好应该怎么办？", config=config, top_k=5
    )

    assert results
    assert {r.source for r in results} == {ZH_BOOK_NAME}


def test_search_all_languages_returns_both(config: BookRagConfig):
    """search_all_languages=True 时，中文书与日文书都参与检索。"""
    results = search_books(
        "去爬山之前要注意什么？晚上睡不好应该怎么办？",
        config=config,
        top_k=10,
        search_all_languages=True,
    )

    assert {ZH_BOOK_NAME, JA_BOOK_NAME} <= {r.source for r in results}
