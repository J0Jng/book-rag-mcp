"""书籍元数据：workspace / genre / tags 的解析与解析规则。

设计要点（见 docs/plans/book-rag-library-expansion-brief.md S2）：
- ``<书籍库根>/workspaces.yaml`` 是元数据的**唯一真源**；Markdown 正文不写 front matter；
- ``workspaces`` 是书单集合，可多对多；``genre`` 由所属 workspace 的 genre 推导，
  多个不同 genre 时归为 ``mixed``，也可在书条目上显式覆盖；
- 未在 yaml 声明的书一律归入保留 workspace ``_unclassified``、genre ``unknown``，
  由列表 / 检索显式暴露，禁止静默归类；
- 元数据与向量索引解耦：yaml 变更后无需重建 embedding，检索与管理接口在读取时
  实时解析（``refresh_all_indexes`` 也会把结果回写到 ``meta.json`` 便于审计）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import yaml

#: 未在 yaml 中声明的书籍归入的保留 workspace
UNCLASSIFIED_WORKSPACE = "_unclassified"

#: 无法确定体裁时的默认值
UNKNOWN_GENRE = "unknown"

#: 允许的 genre 取值
VALID_GENRES = ("fiction", "knowledge", "mixed", "unknown")

#: workspace 定义文件名（位于书籍库根目录）
WORKSPACES_FILENAME = "workspaces.yaml"


@dataclass
class WorkspaceDef:
    """一个 workspace 的定义。

    Attributes:
        id: workspace 标识（yaml 中的键），如 ``"fiction"``。
        label: 展示用中文名，缺省时退回 ``id``。
        genre: 该 workspace 的体裁，取值为 :data:`VALID_GENRES`。
    """

    id: str
    label: str
    genre: str = UNKNOWN_GENRE


@dataclass
class BookMetadata:
    """一本书的元数据。

    Attributes:
        workspaces: 所属 workspace 列表；未声明时为 ``["_unclassified"]``。
        genre: 体裁；由 workspace 推导或书条目显式指定。
        tags: 自由标签列表。
    """

    workspaces: list[str] = field(default_factory=lambda: [UNCLASSIFIED_WORKSPACE])
    genre: str = UNKNOWN_GENRE
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """序列化为可写入 JSON 的字典。"""
        return {
            "workspaces": list(self.workspaces),
            "genre": self.genre,
            "tags": list(self.tags),
        }


@dataclass
class WorkspaceCatalog:
    """``workspaces.yaml`` 的解析结果。

    Attributes:
        path: yaml 文件路径（文件缺失时仍保留预期路径，便于报错定位）。
        workspaces: workspace id → 定义。
        books: 归一化后的书籍相对路径 → 元数据。
    """

    path: Path
    workspaces: dict[str, WorkspaceDef] = field(default_factory=dict)
    books: dict[str, BookMetadata] = field(default_factory=dict)

    def resolve(self, book_path: str) -> BookMetadata:
        """解析一本书的元数据；未声明时返回 ``_unclassified`` 默认值。

        匹配顺序：
        1. 归一化后的相对路径精确匹配（如 ``library/杀死一只知更鸟.md``）；
        2. 精确匹配失败时，若 yaml 中只有一个条目的文件名（basename）相同，
           则按文件名匹配，方便只写书名、不写目录；
        3. 仍无匹配 → ``_unclassified`` / ``unknown`` / ``[]``。

        返回值中的列表均为副本，调用方可安全修改。
        """
        key = _normalize_book_key(book_path)
        entry = self.books.get(key)
        if entry is None:
            basename = PurePosixPath(key).name
            candidates = [
                meta for path, meta in self.books.items() if PurePosixPath(path).name == basename
            ]
            if len(candidates) == 1:
                entry = candidates[0]
        if entry is None:
            return BookMetadata()
        return BookMetadata(
            workspaces=list(entry.workspaces),
            genre=entry.genre,
            tags=list(entry.tags),
        )


def _normalize_book_key(key: str) -> str:
    """把书籍路径归一化为 POSIX 相对路径字符串。"""
    normalized = str(key).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def _as_str_list(value, *, where: str) -> list[str]:
    """把 yaml 值校验并转换为去重后的字符串列表（保序）。"""
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{where} 应为列表，实际为 {type(value).__name__}")
    items: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in items:
            items.append(text)
    return items


def _parse_genre(value, *, where: str) -> str:
    """校验 genre 取值；缺省为 ``unknown``。"""
    if value is None:
        return UNKNOWN_GENRE
    genre = str(value).strip().lower()
    if genre not in VALID_GENRES:
        raise ValueError(
            f"{where} 的 genre 非法：{value!r}（允许：{', '.join(VALID_GENRES)}）"
        )
    return genre


def _infer_genre(
    workspace_ids: list[str],
    definitions: dict[str, WorkspaceDef],
    override: str | None,
) -> str:
    """由所属 workspace 推导书的 genre；多个不同体裁时返回 ``mixed``。"""
    if override is not None:
        return override
    genres = {
        definitions[wid].genre
        for wid in workspace_ids
        if wid in definitions and definitions[wid].genre != UNKNOWN_GENRE
    }
    if not genres:
        return UNKNOWN_GENRE
    if len(genres) == 1:
        return next(iter(genres))
    return "mixed"


def parse_catalog(data: dict, *, path: Path) -> WorkspaceCatalog:
    """解析 ``workspaces.yaml`` 的 Python 字典结构。

    Raises:
        ValueError: 顶层结构、workspace 定义或书条目的字段类型 / 取值非法。
    """
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} 顶层应为映射（dict），实际为 {type(data).__name__}")

    raw_workspaces = data.get("workspaces") or {}
    if not isinstance(raw_workspaces, dict):
        raise TypeError(f"{path} 的 workspaces 应为映射（dict）")

    definitions: dict[str, WorkspaceDef] = {}
    for raw_id, raw_def in raw_workspaces.items():
        wid = str(raw_id).strip()
        if not wid:
            raise ValueError(f"{path} 中存在空 workspace id")
        raw_def = {} if raw_def is None else raw_def
        if not isinstance(raw_def, dict):
            raise TypeError(f"{path} 的 workspaces.{wid} 应为映射（dict）")
        definitions[wid] = WorkspaceDef(
            id=wid,
            label=str(raw_def.get("label") or wid),
            genre=_parse_genre(raw_def.get("genre"), where=f"workspaces.{wid}"),
        )

    raw_books = data.get("books") or {}
    if not isinstance(raw_books, dict):
        raise TypeError(f"{path} 的 books 应为映射（dict）")

    books: dict[str, BookMetadata] = {}
    for raw_key, raw_entry in raw_books.items():
        key = _normalize_book_key(raw_key)
        if not key:
            raise ValueError(f"{path} 中存在空书籍路径")
        raw_entry = {} if raw_entry is None else raw_entry
        if not isinstance(raw_entry, dict):
            raise TypeError(f"{path} 的 books[{key!r}] 应为映射（dict）")

        workspaces = _as_str_list(
            raw_entry.get("workspaces"), where=f"books[{key!r}].workspaces"
        )
        if not workspaces:
            workspaces = [UNCLASSIFIED_WORKSPACE]
        tags = _as_str_list(raw_entry.get("tags"), where=f"books[{key!r}].tags")
        override = (
            _parse_genre(raw_entry["genre"], where=f"books[{key!r}]")
            if "genre" in raw_entry
            else None
        )
        books[key] = BookMetadata(
            workspaces=workspaces,
            genre=_infer_genre(workspaces, definitions, override),
            tags=tags,
        )

    return WorkspaceCatalog(path=path, workspaces=definitions, books=books)


def load_catalog(library_path: Path) -> WorkspaceCatalog:
    """读取 ``<library_path>/workspaces.yaml``；文件缺失时返回空目录（全部未分类）。

    Raises:
        ValueError: yaml 语法错误或结构非法（配置问题应显式暴露，不静默吞掉）。
    """
    path = Path(library_path) / WORKSPACES_FILENAME
    if not path.is_file():
        return WorkspaceCatalog(path=path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} 不是合法的 YAML：{exc}") from exc
    return parse_catalog(data, path=path)
