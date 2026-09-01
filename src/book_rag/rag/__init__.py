"""RAG 内核：文本切分、嵌入、向量库与检索。"""

from book_rag.rag.embeddings import EmbeddingModel
from book_rag.rag.retriever import Retriever
from book_rag.rag.splitter import split_text
from book_rag.rag.vector_store import FaissVectorStore

__all__ = [
    "EmbeddingModel",
    "Retriever",
    "split_text",
    "FaissVectorStore",
]