"""Embedding 实例管理：按模型名缓存，按语言路由。

书籍索引与查询检索共享同一实例，避免重复加载大模型；每本书记录
实际使用的模型名，检索时用同一模型编码 query，保证向量空间一致。
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
    """按语言选择专用 embedding 模型并返回其实例。

    Args:
        language: ISO 639-1 语言码（如 ``"zh"`` / ``"en"``），
            未知或其他语言由 :func:`choose_embedding_model` 回退到多语言模型。
    """
    return get_embeddings(choose_embedding_model(language))
