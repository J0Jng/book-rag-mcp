"""metadata 模块测试：workspaces.yaml 解析、genre 推导与未声明书兜底。

覆盖 S2 验收口径：
- yaml 的 workspace 定义 / 书籍条目解析正确；
- 未声明书归入 ``_unclassified`` / ``unknown`` / ``[]``；
- 多 workspace 跨 genre 时归为 ``mixed``，显式 genre 可覆盖；
- 只写文件名（basename）时能唯一匹配；
- 文件缺失时全部按未分类处理；配置非法时显式报错。
"""

from pathlib import Path

import pytest

from book_rag.rag.metadata import (
    UNCLASSIFIED_WORKSPACE,
    UNKNOWN_GENRE,
    load_catalog,
    parse_catalog,
)


def _write_yaml(library: Path, text: str) -> None:
    (library / "workspaces.yaml").write_text(text, encoding="utf-8")


@pytest.fixture
def library(tmp_path: Path) -> Path:
    """临时书籍库根目录。"""
    path = tmp_path / "books"
    path.mkdir()
    return path


def test_parse_workspaces_and_books(library: Path):
    """workspace 定义与书籍条目的 label / workspaces / genre / tags 均正确解析。"""
    _write_yaml(
        library,
        """
workspaces:
  fiction:
    label: 小说
    genre: fiction
  finance: {label: 金融投资, genre: knowledge}
books:
  "library/杀死一只知更鸟.md":
    workspaces: [fiction]
    tags: [美国文学, 经典]
  "library/投资最重要的事.md":
    workspaces: [finance]
    tags: [投资]
""",
    )

    catalog = load_catalog(library)

    assert catalog.workspaces["fiction"].label == "小说"
    assert catalog.workspaces["fiction"].genre == "fiction"
    assert catalog.workspaces["finance"].label == "金融投资"

    novel = catalog.resolve("library/杀死一只知更鸟.md")
    assert novel.workspaces == ["fiction"]
    assert novel.genre == "fiction"
    assert novel.tags == ["美国文学", "经典"]

    finance = catalog.resolve("library/投资最重要的事.md")
    assert finance.workspaces == ["finance"]
    assert finance.genre == "knowledge"


def test_undeclared_book_is_unclassified(library: Path):
    """未在 yaml 声明的书 → [_unclassified] / unknown / []。"""
    _write_yaml(library, "workspaces:\n  fiction: {label: 小说, genre: fiction}\nbooks: {}\n")

    meta = load_catalog(library).resolve("library/某本未声明的书.md")

    assert meta.workspaces == [UNCLASSIFIED_WORKSPACE]
    assert meta.genre == UNKNOWN_GENRE
    assert meta.tags == []


def test_missing_yaml_marks_everything_unclassified(library: Path):
    """workspaces.yaml 不存在时返回空目录，任何书都按未分类处理。"""
    catalog = load_catalog(library)

    assert catalog.workspaces == {}
    assert catalog.books == {}
    meta = catalog.resolve("whatever.md")
    assert meta.workspaces == [UNCLASSIFIED_WORKSPACE]
    assert meta.genre == UNKNOWN_GENRE


def test_multi_workspace_across_genres_is_mixed(library: Path):
    """一本书属于不同 genre 的多个 workspace 时，genre 归为 mixed。"""
    _write_yaml(
        library,
        """
workspaces:
  fiction: {label: 小说, genre: fiction}
  social: {label: 社会学, genre: knowledge}
books:
  book.md:
    workspaces: [fiction, social]
""",
    )

    meta = load_catalog(library).resolve("book.md")

    assert meta.workspaces == ["fiction", "social"]
    assert meta.genre == "mixed"


def test_same_genre_workspaces_keep_genre(library: Path):
    """多个 workspace 的 genre 一致时沿用该 genre。"""
    _write_yaml(
        library,
        """
workspaces:
  finance: {label: 金融, genre: knowledge}
  social: {label: 社会, genre: knowledge}
books:
  book.md: {workspaces: [finance, social]}
""",
    )

    assert load_catalog(library).resolve("book.md").genre == "knowledge"


def test_explicit_book_genre_overrides_workspace(library: Path):
    """书条目上的 genre 优先于 workspace 推导值（如合集按 knowledge 处理）。"""
    _write_yaml(
        library,
        """
workspaces:
  fiction: {label: 小说, genre: fiction}
books:
  book.md:
    workspaces: [fiction]
    genre: knowledge
""",
    )

    meta = load_catalog(library).resolve("book.md")

    assert meta.workspaces == ["fiction"]
    assert meta.genre == "knowledge"


def test_basename_fallback_when_unique(library: Path):
    """yaml 只写文件名时，可唯一匹配到完整相对路径。"""
    _write_yaml(
        library,
        """
workspaces:
  fiction: {label: 小说, genre: fiction}
books:
  "杀死一只知更鸟.md": {workspaces: [fiction]}
""",
    )

    meta = load_catalog(library).resolve("library/杀死一只知更鸟.md")

    assert meta.workspaces == ["fiction"]


def test_duplicate_basename_requires_full_path(library: Path):
    """多个同名条目时不做 basename 猜测，避免错配。"""
    _write_yaml(
        library,
        """
workspaces:
  a: {label: A, genre: knowledge}
  b: {label: B, genre: knowledge}
books:
  "a/book.md": {workspaces: [a]}
  "b/book.md": {workspaces: [b]}
""",
    )

    catalog = load_catalog(library)

    assert catalog.resolve("a/book.md").workspaces == ["a"]
    assert catalog.resolve("b/book.md").workspaces == ["b"]
    assert catalog.resolve("c/book.md").workspaces == [UNCLASSIFIED_WORKSPACE]


def test_resolve_returns_copies(library: Path):
    """resolve 返回的列表是副本，修改它不会污染 catalog。"""
    _write_yaml(
        library,
        """
workspaces:
  fiction: {label: 小说, genre: fiction}
books:
  book.md: {workspaces: [fiction], tags: [经典]}
""",
    )

    catalog = load_catalog(library)
    first = catalog.resolve("book.md")
    first.workspaces.append("mutated")
    first.tags.append("mutated")

    second = catalog.resolve("book.md")
    assert second.workspaces == ["fiction"]
    assert second.tags == ["经典"]


def test_invalid_genre_raises(library: Path):
    """非法 genre 显式报错，避免静默归类。"""
    _write_yaml(
        library,
        "workspaces:\n  bad: {label: 坏, genre: poetry}\nbooks: {}\n",
    )

    with pytest.raises(ValueError, match="genre"):
        load_catalog(library)


def test_invalid_structure_raises():
    """顶层或字段类型非法时抛 TypeError。"""
    with pytest.raises(TypeError, match="顶层"):
        parse_catalog([], path=Path("workspaces.yaml"))
    with pytest.raises(TypeError, match="workspaces"):
        parse_catalog({"workspaces": ["bad"]}, path=Path("workspaces.yaml"))
    with pytest.raises(TypeError, match="books"):
        parse_catalog({"books": ["bad"]}, path=Path("workspaces.yaml"))
