"""效果检验（集成层）：清洗后的索引里 front/back matter 是否真的不可检索。

这组题目走完整生产路径（parse → clean_matter → split → 真实 bge 向量化 → 检索），
验证两件事：
- 正向：推荐序 / 致谢的独有内容**检索不到**（已被清洗，不在索引里）；
- 反向：正文内容**照旧可检索**（清洗没有过度删除）。

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

ZH_MODEL = "BAAI/bge-large-zh-v1.5"
BOOK_NAME = "matter_book.md"

MATTER_BOOK = """# 推荐序

你的人生是工业品，还是艺术品？这段话只出现在推荐序里，由他人撰写。

# 第一章 咖啡冲煮

手冲咖啡时，水温最好控制在九十摄氏度上下，焖蒸三十秒后再缓慢注水。

# 第二章 登山安全

登山前要查看天气预报，携带足够的饮水与干粮，高原反应要及时下撤。

# 致谢

感谢尤金·科尔孙斯基与凯尔·威廉姆斯在写作过程中的帮助。
"""


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch):
    """集成测试强制离线：模型必须已缓存。"""
    require_model(ZH_MODEL)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> BookRagConfig:
    """写一本含 front/back matter 的书并构建真实索引（module 级只建一次）。"""
    library = tmp_path_factory.mktemp("matter_library")
    (library / BOOK_NAME).write_text(MATTER_BOOK, encoding="utf-8")

    cfg = BookRagConfig(library_path=library, index_dir=library.parent / "index")
    refresh_all_indexes(cfg)
    return cfg


def _sections_of(results) -> set[str]:
    """取出检索结果里出现过的 section 路径集合。"""
    return {r.section for r in results}


def test_q13_cleaning_report_recorded_on_disk(config: BookRagConfig):
    """【题 13】索引目录落下 cleaning_report.json，且恰好剥掉推荐序与致谢。"""
    report_path = next(config.index_dir.glob("book-*/cleaning_report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["aborted"] is False
    assert {item["path"] for item in report["removed"]} == {"推荐序", "致谢"}
    assert report["total_sections"] == 4
    assert report["kept_sections"] == 2


def test_q14_toc_has_no_front_or_back_matter(config: BookRagConfig):
    """【题 14】结构层（toc.json）顶层只应剩两章正文。"""
    toc_path = next(config.index_dir.glob("book-*/toc.json"))
    toc = json.loads(toc_path.read_text(encoding="utf-8"))

    top_names = [child["name"] for child in toc["children"]]
    assert top_names == ["第一章 咖啡冲煮", "第二章 登山安全"]


def test_q15_preface_content_is_not_retrievable(config: BookRagConfig):
    """【题 15】用推荐序的独有措辞提问 → 命中结果里不得出现「推荐序」。"""
    results = search_books(
        "你的人生是工业品还是艺术品？", config=config, search_all_languages=True, top_k=5
    )

    assert "推荐序" not in _sections_of(results)


def test_q16_acknowledgment_content_is_not_retrievable(config: BookRagConfig):
    """【题 16】用致谢人名的独有措辞提问 → 命中结果里不得出现「致谢」。"""
    results = search_books(
        "尤金·科尔孙斯基与凯尔·威廉姆斯帮助了什么？",
        config=config,
        search_all_languages=True,
        top_k=5,
    )

    assert "致谢" not in _sections_of(results)


def test_q17_body_content_still_retrievable(config: BookRagConfig):
    """【题 17】反方向验证：正文主题仍能被正确检索（清洗没有过度删除）。"""
    cases = [
        ("手冲咖啡水温怎么控制？", "咖啡冲煮"),
        ("登山前要准备什么？", "登山安全"),
    ]

    for query, topic in cases:
        results = search_books(
            query, config=config, search_all_languages=True, top_k=1
        )
        assert results, f"query {query!r} 应返回非空结果"
        assert topic in results[0].section, (
            f"query {query!r} 的 top-1 应命中 {topic!r}，实际 {results[0].section!r}"
        )
