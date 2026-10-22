"""结构层：从解析出的 BookSection 构建目录树（TOC）并落盘 / 加载。

设计要点（见 docs/plans/book-rag-structure-layer-impl-brief.md Ticket 1）：
- 纯结构化，无 LLM：目录树完全由解析器已有的 ``BookSection.path``（``" > "`` 分隔）
  与 ``level`` 聚合而成，不生成任何新文本；
- 落盘为索引子目录下的 ``toc.json``，随索引持久化，读取 O(1)；
- 所有条目锚定 section position，供上层定位原文，绝不编造。
"""

from __future__ import annotations

import json
from pathlib import Path

from book_rag.io.parsers import BookSection

#: TOC 在索引子目录中的文件名
TOC_FILENAME = "toc.json"


def _new_node(name: str, level: int) -> dict:
    """创建一个目录树节点（含叶子所需的全量键，便于 JSON 序列化统一）。"""
    return {"name": name, "level": level, "positions": [], "children": []}


def build_toc(sections: list[BookSection]) -> dict:
    """由 section 列表构造嵌套目录树。

    路径按 ``" > "`` 分隔逐级下钻；同一完整路径出现多次（如 EPUB 同一文档多段）
    时聚合为同一个叶子，``positions`` 按阅读顺序升序累积。

    Args:
        sections: 解析器输出的章节 / 段落列表，按阅读顺序排列。

    Returns:
        根节点字典：``{"name": "", "level": 0, "positions": [], "children": [...]}``，
        每个节点含 ``name`` / ``level`` / ``positions`` / ``children`` 四个键。
    """
    root = _new_node("", 0)
    for section in sections:
        parts = [p for p in (s.strip() for s in section.path.split(">")) if p]
        if not parts:
            continue
        node = root
        for part in parts:
            child = next((c for c in node["children"] if c["name"] == part), None)
            if child is None:
                child = _new_node(part, section.level)
                node["children"].append(child)
            node = child
        node["positions"].append(section.position)
    return root


def save_structure(idx_dir: Path, toc: dict) -> Path:
    """把目录树落盘为 ``idx_dir / toc.json``，返回落盘路径。"""
    idx_dir.mkdir(parents=True, exist_ok=True)
    path = idx_dir / TOC_FILENAME
    path.write_text(
        json.dumps(toc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def load_structure(idx_dir: Path) -> dict | None:
    """加载 ``idx_dir / toc.json``；文件不存在时返回 ``None``。"""
    path = idx_dir / TOC_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def matches_section(chunk_section: str, section: str) -> bool:
    """判断 chunk 的章节路径是否命中 section 过滤参数。

    解析器产出的 ``section`` 是完整路径链（如 ``"测试之书 > 第一章 起点"``），因此
    以下三种写法都视为命中，便于与目录树（渲染为各级章节名）配合使用：
    - 路径前缀：``"测试之书"``（命中其下所有子章节）；
    - 完整路径：``"测试之书 > 第一章 起点"``；
    - 路径中的单级章节名或其前缀：``"第一章 起点"`` / ``"第一章"``。

    两侧空白不影响匹配。
    """
    section = section.strip()
    full = chunk_section.strip()
    if not section or not full:
        return False
    if full.startswith(section):
        return True
    return any(part.strip().startswith(section) for part in full.split(">"))
