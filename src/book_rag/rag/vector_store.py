"""FAISS 向量库：存嵌入向量 + 元数据（来源文件、块文本）。

持久化到磁盘（索引目录），可增量添加文档。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from book_rag.config import load_config
from book_rag.rag.embeddings import EmbeddingModel


@dataclass
class ScoredChunk:
    """检索结果块。"""

    text: str
    score: float
    source: str
    chunk_index: int


class FaissVectorStore:
    """基于 FAISS 的向量存储。

    - ``index.faiss``：向量索引（ID -> 向量）
    - ``meta.json``：按 ID 记录来源文件与文本
    默认归一化向量用余弦相似度（faiss.METRIC_INNER_PRODUCT）。
    """

    def __init__(self, model: EmbeddingModel, index_dir: Path | None = None) -> None:
        self.model = model
        cfg = load_config()
        self.index_dir = Path(index_dir) if index_dir else cfg.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index = None
        self._meta: dict[int, dict] = {}
        self._next_id = 0
        self._load()

    def _load(self) -> None:
        import faiss

        index_path = self.index_dir / "index.faiss"
        meta_path = self.index_dir / "meta.json"
        if index_path.exists() and meta_path.exists():
            self._index = faiss.read_index(str(index_path))
            self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._next_id = max((int(k) + 1 for k in self._meta), default=0)
        else:
            # 延迟创建索引：需要先知道维度
            self._meta = {}
            self._next_id = 0

    def _ensure_index(self) -> None:
        if self._index is None:
            import faiss

            d = self.model.dim
            self._index = faiss.IndexFlatIP(d)

    def add_texts(self, texts: list[str], source: str, metadata: dict | None = None) -> None:
        """添加一批文本块到向量库。

        参数:
            texts: 文本块列表。
            source: 来源标识（如文件相对路径）。
            metadata: 额外元数据（可选）。
        """
        if not texts:
            return
        self._ensure_index()
        vecs = self.model.embed(texts)
        start = self._next_id
        ids = np.arange(start, start + len(texts), dtype=np.int64)
        self._index.add_with_ids(vecs, ids)
        meta = metadata or {}
        for i, t in enumerate(texts):
            self._meta[int(ids[i])] = {"source": source, "text": t, **meta}
        self._next_id = start + len(texts)
        self._persist()

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        """按查询语义检索最相近的 k 个块。"""
        if self._index is None or self._index.ntotal == 0:
            return []
        q = self.model.embed_query(query)
        scores, ids = self._index.search(np.asarray([q], dtype=np.float32), k)
        results: list[ScoredChunk] = []
        for score, idx in zip(scores[0], ids[0], strict=False):
            if idx < 0:
                continue
            doc = self._meta.get(int(idx))
            if doc is None:
                continue
            results.append(
                ScoredChunk(
                    text=doc["text"],
                    score=float(score),
                    source=doc.get("source", ""),
                    chunk_index=int(doc.get("chunk_index", idx)),
                )
            )
        return results

    @property
    def size(self) -> int:
        return int(self._index.ntotal) if self._index is not None else 0

    def _persist(self) -> None:
        if self._index is None:
            return
        import faiss

        faiss.write_index(self._index, str(self.index_dir / "index.faiss"))
        (self.index_dir / "meta.json").write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=1), encoding="utf-8"
        )