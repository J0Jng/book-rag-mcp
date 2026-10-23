"""BM25 关键词索引：分词、落盘与内存检索。

设计要点（见 docs/plans/book-rag-library-expansion-brief.md S3）：
- 每本书在索引子目录下额外落盘 ``bm25.json``（分词结果，不含正文），
  与 FAISS 索引同步构建、随书增量重建；
- 中文用 ``jieba`` 分词，英文/数字按词切分并转小写；可选加载
  ``<书籍库根>/jieba_dict.txt`` 用户词典补充小说人名与专有名词；
- 落盘只保存 ``position`` 与 token 列表，加载时用 ``rank_bm25.BM25Okapi``
  在内存中重建，避免 pickle 跨版本不兼容；
- 每次查询只对 query 分词，文档侧分词在建索引时一次性完成。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import warnings
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

with warnings.catch_warnings():
    # jieba 0.42 在 Python 3.12 下有已知的 SyntaxWarning（正则转义），与本项目无关
    warnings.simplefilter("ignore", SyntaxWarning)
    import jieba

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from book_rag.io.atomic import write_json_atomic

logger = logging.getLogger(__name__)

#: BM25 索引文件名（位于每本书的索引子目录）
BM25_FILENAME = "bm25.json"

#: 落盘格式版本，分词方案变化时递增，旧文件会被视为不可用并重建
BM25_VERSION = 2

#: 用户词典文件名（位于书籍库根目录，可选）
USER_DICT_FILENAME = "jieba_dict.txt"

_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[._'-][A-Za-z0-9]+)*")
_CJK_RUN_RE = re.compile(r"[㐀-䶿一-鿿]+")

# jieba 首次分词会打印构建前缀词典的日志，静默到 WARNING
jieba.setLogLevel(logging.WARNING)

_default_tokenizer = jieba.Tokenizer()
_tokenizers: dict[tuple[str, str], object] = {}
_tokenizer_lock = RLock()
_active_tokenizer = ContextVar("bm25_tokenizer", default=_default_tokenizer)
_active_fingerprint = ContextVar("bm25_dictionary_hash", default="none")


def dictionary_fingerprint() -> str:
    """返回当前调用上下文的用户词典指纹，用于缓存失效。"""
    return _active_fingerprint.get()


def ensure_user_dict(path: Path | None) -> str:
    """加载 jieba 用户词典（幂等）；文件不存在或格式非法时只记日志、不中断。

    用户词典用于补充小说人名 / 专业术语，让「阿蒂克斯」这类词保持整体 token，
    提升 BM25 精确命中率。
    """
    tokenizer = _default_tokenizer
    fingerprint = "none"
    try:
        if path is not None and Path(path).is_file():
            content = Path(path).read_bytes()
            fingerprint = hashlib.sha256(content).hexdigest()
            key = (str(Path(path).resolve()), fingerprint)
            with _tokenizer_lock:
                tokenizer = _tokenizers.get(key)
                if tokenizer is None:
                    import io

                    tokenizer = jieba.Tokenizer()
                    tokenizer.load_userdict(io.BytesIO(content))
                    _tokenizers[key] = tokenizer
    except Exception as exc:  # noqa: BLE001 - 用户词典是可选增强，坏了不应影响索引
        logger.warning("jieba 用户词典加载失败（已忽略）：%s（%s）", path, exc)
        tokenizer, fingerprint = _default_tokenizer, "none"
    _active_tokenizer.set(tokenizer)
    _active_fingerprint.set(fingerprint)
    return fingerprint


def tokenize(text: str, tokenizer=None) -> list[str]:
    """把文本切成 BM25 词项：中文走 jieba，英文/数字按词切并转小写。

    纯标点、空白不会产生 token；同一文本在任何时候分词结果稳定，
    保证索引期与查询期使用同一口径。
    """
    if not text:
        return []
    tokens: list[str] = [match.group(0).lower() for match in _ASCII_WORD_RE.finditer(text)]
    tokenizer = tokenizer if tokenizer is not None else _active_tokenizer.get()
    for match in _CJK_RUN_RE.finditer(text):
        tokens.extend(token for token in tokenizer.lcut(match.group(0)) if token.strip())
    return tokens


def build_bm25_payload(docs: list[Document]) -> dict:
    """把 Document 列表转成可落盘的 BM25 载荷（position + tokens）。"""
    return {
        "version": BM25_VERSION,
        "tokenizer": "jieba+ascii",
        "dictionary_hash": _active_fingerprint.get(),
        "docs": [
            {
                "position": int(doc.metadata.get("position", index)),
                "tokens": tokenize(doc.page_content),
            }
            for index, doc in enumerate(docs)
        ],
    }


def save_bm25_index(idx_dir: Path, docs: list[Document]) -> Path:
    """把一本书的 BM25 分词结果写入 ``<idx_dir>/bm25.json``，返回落盘路径。"""
    idx_dir.mkdir(parents=True, exist_ok=True)
    path = idx_dir / BM25_FILENAME
    write_json_atomic(path, build_bm25_payload(docs), indent=None)
    return path


@dataclass
class BM25Index:
    """一本书加载到内存的 BM25 索引。

    Attributes:
        bm25: ``rank_bm25.BM25Okapi`` 实例。
        positions: 与 BM25 语料下标一一对应的 chunk position 列表。
    """

    bm25: BM25Okapi
    positions: list[int]
    tokenizer: object = None

    def search(
        self,
        query: str,
        documents_by_position: dict[int, Document],
        *,
        k: int,
        section_matcher=None,
    ) -> list[tuple[Document, float]]:
        """返回 query 在该书内的 BM25 top-k 命中（分数 > 0）。

        Args:
            query: 查询文本。
            documents_by_position: position → Document 映射（来自 FAISS docstore）。
            k: 最多返回条数。
            section_matcher: 可选的章节过滤谓词，接收 Document 返回 bool。
        """
        tokens = tokenize(query, self.tokenizer)
        if not tokens or not self.positions:
            return []
        scores = self.bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[tuple[Document, float]] = []
        for index in order:
            score = float(scores[index])
            if score <= 0.0:
                break
            document = documents_by_position.get(self.positions[index])
            if document is None:
                continue
            if section_matcher is not None and not section_matcher(document):
                continue
            hits.append((document, score))
            if len(hits) >= k:
                break
        return hits


def load_bm25_index(idx_dir: Path) -> BM25Index | None:
    """加载 ``<idx_dir>/bm25.json``；缺失 / 版本不符 / 无有效文档时返回 None。"""
    path = idx_dir / BM25_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("BM25 索引读取失败，回退为纯向量检索：%s（%s）", path, exc)
        return None
    if payload.get("version") != BM25_VERSION:
        return None
    if payload.get("dictionary_hash") != _active_fingerprint.get():
        return None
    entries = payload.get("docs") or []
    if not isinstance(entries, list) or any(
        not isinstance(entry, dict)
        or not isinstance(entry.get("position"), int)
        or entry["position"] < 0
        or not isinstance(entry.get("tokens"), list)
        or any(not isinstance(token, str) for token in entry["tokens"])
        for entry in entries
    ):
        return None
    positions: list[int] = []
    corpus: list[list[str]] = []
    for entry in entries:
        positions.append(int(entry.get("position", -1)))
        corpus.append([str(token) for token in entry.get("tokens") or []])
    if not corpus or not any(corpus) or len(set(positions)) != len(positions):
        return None
    return BM25Index(
        bm25=BM25Okapi(corpus), positions=positions, tokenizer=_active_tokenizer.get()
    )
