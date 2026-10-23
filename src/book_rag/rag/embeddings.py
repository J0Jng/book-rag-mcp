"""Embedding 实例管理：按模型名缓存。

S3 起所有书籍与 query 统一使用 ``BAAI/bge-m3``；``get_embeddings_for_language``
仅保留为兼容入口，实际恒返回同一多语言模型实例，语言只作为元数据。
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from book_rag.rag.language import choose_embedding_model

# 按模型名缓存已创建的 HuggingFaceEmbeddings 实例
_embedding_cache: dict[str, HuggingFaceEmbeddings] = {}


def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """取指定模型的 embedding 实例；缓存命中直接返回，否则新建并缓存。

    所有实例统一开启向量归一化（``normalize_embeddings=True``），
    使相似度计算等价于余弦相似度。

    Args:
        model_name: HuggingFace 模型名，如 ``"BAAI/bge-large-zh-v1.5"``。
    """
    instance = _embedding_cache.get(model_name)
    if instance is None:
        instance = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        _embedding_cache[model_name] = instance
    return instance


def get_embeddings_for_language(language: str) -> HuggingFaceEmbeddings:
    """兼容旧接口：按语言入口取 embedding，实际恒为统一的 bge-m3。

    Args:
        language: ISO 639-1 语言码，仅作元数据，不再影响模型选择。
    """
    return get_embeddings(choose_embedding_model(language))
