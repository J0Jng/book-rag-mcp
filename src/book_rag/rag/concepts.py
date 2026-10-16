"""概念索引：从章节摘要抽取概念条目并建立独立 FAISS 索引（``concepts.faiss``）。

概念条目（框架 / 术语 / 心智模型）全部锚定 ``section`` / ``position``，杜绝编造；
索引目录内使用独立文件名（``concepts.*``），不与 chunk 索引（``index.faiss``）冲突。

只放不依赖 index / retriever 的纯函数，保证依赖单向：index → concepts，无循环。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 概念索引在书籍索引子目录内的固定文件名
_CONCEPTS_JSON = "concepts.json"
_INDEX_NAME = "concepts"


@dataclass
class Concept:
    """一条可检索的概念条目。

    Attributes:
        text: 概念内容（框架/术语为「名称：描述」拼接，心智模型为原句）。
        kind: 概念类型，``"framework"`` / ``"term"`` / ``"mental_model"``。
        section: 概念来源的顶层章节名。
        position: 来源章节首个 pointer 的 position（无 pointer 时为 0）。
        source: 书籍相对路径（POSIX 形式）。
    """

    text: str
    kind: str
    section: str
    position: int
    source: str

    def to_dict(self) -> dict:
        """序列化为可写入 JSON 的字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Concept:
        """从 JSON 字典反序列化。"""
        return cls(**data)


def build_concept_entries(summaries: list[dict], *, source: str = "") -> list[Concept]:
    """从章节摘要抽出全部概念条目。

    每条 summary 的 ``frameworks`` / ``terms`` 各项拼接为「名称：描述」，
    ``mental_models`` 直接使用字符串；``section`` 取 summary 的章节名，
    ``position`` 取首个 pointer 的 position（pointers 为空则 0）。

    名称为空或含 ``[未检测到]`` 占位 token 的条目（LLM 未检出时的诚实标注）
    在写入 Concept 前跳过，避免污染概念索引；description 为占位但 name
    有效的条目仍保留（名称可检索）。
    """
    concepts: list[Concept] = []
    for summary in summaries:
        section = summary.get("section", "")
        pointers = summary.get("pointers") or []
        position = pointers[0].get("position", 0) if pointers else 0

        for fw in summary.get("frameworks") or []:
            name = (fw.get("name") or "").strip()
            if not name or "[未检测到]" in name:
                continue
            concepts.append(
                Concept(
                    text=f"{name}：{fw.get('description', '')}",
                    kind="framework",
                    section=section,
                    position=position,
                    source=source,
                )
            )
        for term in summary.get("terms") or []:
            name = (term.get("name") or "").strip()
            if not name or "[未检测到]" in name:
                continue
            concepts.append(
                Concept(
                    text=f"{name}：{term.get('definition', '')}",
                    kind="term",
                    section=section,
                    position=position,
                    source=source,
                )
            )
        for model in summary.get("mental_models") or []:
            text = (model or "").strip() if isinstance(model, str) else str(model or "")
            if not text or "[未检测到]" in text:
                continue
            concepts.append(
                Concept(
                    text=text,
                    kind="mental_model",
                    section=section,
                    position=position,
                    source=source,
                )
            )
    return concepts


def build_concept_index(
    concepts: list[Concept], embeddings, idx_dir: Path
) -> Path | None:
    """为概念条目建 FAISS 索引并落盘；空列表直接返回 ``None``（不建索引）。

    落盘三件：``concepts.faiss`` + ``concepts.pkl``（``save_local`` 的
    ``index_name="concepts"``）与 ``concepts.json``（概念元数据，加载侧据此还原）。

    Returns:
        ``concepts.json`` 的路径；``concepts`` 为空时 ``None``。
    """
    if not concepts:
        return None

    docs = [
        Document(
            page_content=c.text,
            metadata={
                "kind": c.kind,
                "section": c.section,
                "position": c.position,
                "source": c.source,
            },
        )
        for c in concepts
    ]
    FAISS.from_documents(docs, embeddings).save_local(str(idx_dir), index_name=_INDEX_NAME)

    json_path = idx_dir / _CONCEPTS_JSON
    json_path.write_text(
        json.dumps([c.to_dict() for c in concepts], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return json_path


def load_concept_index(idx_dir: Path, embeddings) -> tuple | None:
    """加载概念索引；``concepts.json`` 不存在时返回 ``None``（未构建）。

    Returns:
        ``(FAISS 索引, list[Concept])`` 元组。
    """
    json_path = idx_dir / _CONCEPTS_JSON
    if not json_path.is_file():
        return None

    concepts = [
        Concept.from_dict(item)
        for item in json.loads(json_path.read_text(encoding="utf-8"))
    ]
    vector_store = FAISS.load_local(
        str(idx_dir),
        embeddings,
        index_name=_INDEX_NAME,
        allow_dangerous_deserialization=True,
    )
    return vector_store, concepts
