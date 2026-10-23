"""embeddings 模块测试。

全部 mock ``HuggingFaceEmbeddings``，避免测试下载真实 HuggingFace 模型。
"""

from unittest.mock import MagicMock, patch

import pytest

from book_rag.rag import embeddings


@pytest.fixture(autouse=True)
def _clear_embedding_cache():
    """每个测试前清空模块级缓存，避免跨测试互相污染。"""
    embeddings._embedding_cache.clear()
    yield
    embeddings._embedding_cache.clear()


def test_get_embeddings_returns_huggingface_embeddings():
    """get_embeddings 应构造 HuggingFaceEmbeddings，并配置 model_name 与归一化。"""
    with patch.object(embeddings, "HuggingFaceEmbeddings", MagicMock()) as mock_cls:
        result = embeddings.get_embeddings("BAAI/bge-large-zh-v1.5")

        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model_name"] == "BAAI/bge-large-zh-v1.5"
        assert kwargs["encode_kwargs"] == {"normalize_embeddings": True}
        assert result is mock_cls.return_value


def test_get_embeddings_cache_hit():
    """同一 model_name 两次调用应返回同一实例，且只构造一次。"""
    with patch.object(embeddings, "HuggingFaceEmbeddings", MagicMock()) as mock_cls:
        first = embeddings.get_embeddings("test-model-a")
        second = embeddings.get_embeddings("test-model-a")

        assert first is second
        assert mock_cls.call_count == 1


def test_get_embeddings_cache_by_model_name():
    """不同 model_name 应各自构造实例，互不共享缓存。"""
    with patch.object(embeddings, "HuggingFaceEmbeddings", MagicMock()) as mock_cls:
        # MagicMock 默认多次调用返回同一对象，这里让每次调用产生新实例
        mock_cls.side_effect = lambda **kwargs: MagicMock()
        a = embeddings.get_embeddings("test-model-a")
        b = embeddings.get_embeddings("test-model-b")

        assert a is not b
        assert mock_cls.call_count == 2


def test_get_embeddings_for_language_always_uses_bge_m3():
    """语言入口保留兼容，但任何语言都返回统一的 bge-m3。"""
    with patch.object(embeddings, "HuggingFaceEmbeddings", MagicMock()) as mock_cls:
        for language in ("zh", "en", "ja", "unknown"):
            embeddings.get_embeddings_for_language(language)

        assert mock_cls.call_count == 1  # 同一模型名只构造一次（实例缓存）
        assert mock_cls.call_args.kwargs["model_name"] == "BAAI/bge-m3"
