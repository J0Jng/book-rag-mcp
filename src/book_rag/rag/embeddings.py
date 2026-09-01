"""本地嵌入模型封装。

用 sentence-transformers 加载本地/HuggingFace 嵌入模型，输出归一化向量。
默认中文友好的多语言模型。
"""

from __future__ import annotations

import numpy as np

from book_rag.config import DEFAULT_MODEL


class EmbeddingModel:
    """sentence-transformers 嵌入模型封装（懒加载）。"""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu", normalize: bool = True) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def dim(self) -> int:
        """嵌入向量维度（触发模型加载以探测）。"""
        self._lazy_load()
        return int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        """批量嵌入文本，返回归一化的 (n, dim) float32 数组。

        空输入返回 (0, dim) 空数组。
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._lazy_load()
        vecs = self._model.encode(list(texts), normalize_embeddings=self.normalize, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """嵌入单个查询，返回 (dim,) 向量。"""
        return self.embed([query])[0]