"""语言检测与 embedding 模型选择。

S3 起向量模型统一为多语言模型 ``BAAI/bge-m3``：语言检测仍然保留，
但只作为**元数据**（展示、摘要 prompt、同语言书籍过滤），不再决定
embedding 模型——所有书与 query 共享同一向量空间。
"""

from __future__ import annotations

import re

from langdetect import DetectorFactory, LangDetectException, detect

# 固定随机种子，保证 langdetect 检测结果可复现
DetectorFactory.seed = 0

#: 全局唯一的 embedding 模型（多语言，跨书共享同一向量空间）
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

# CJK 脚本字符区间：langdetect 对短 CJK 文本误判率高，先按脚本启发式判定
_HANGUL_RE = re.compile(r"[가-힯ᄀ-ᇿ]")
_KANA_RE = re.compile(r"[぀-ヿ]")
_HAN_RE = re.compile(r"[一-鿿]")


def _detect_cjk_script(text: str) -> str | None:
    """按脚本特征判定 CJK 语言：假名→ja、谚文→ko、汉字→zh、否则 None。"""
    if _KANA_RE.search(text):
        return "ja"
    if _HANGUL_RE.search(text):
        return "ko"
    if _HAN_RE.search(text):
        return "zh"
    return None


def detect_language(text: str) -> str:
    """检测文本语言；空文本或检测失败时返回 "unknown"。"""
    if not text or not text.strip():
        return "unknown"
    cjk = _detect_cjk_script(text)
    if cjk:
        return cjk
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def normalize_language(lang: str) -> str:
    """把 zh-cn / zh-tw / en-us 等带区域的语言码归一化为主语言码。"""
    lang = (lang or "").strip().lower()
    if lang.startswith("zh"):
        return "zh"
    return lang.split("-", 1)[0]


def choose_embedding_model(language: str) -> str:
    """返回统一使用的 embedding 模型（恒为 bge-m3）。

    ``language`` 参数仅为兼容旧调用保留：S3 起语言不再决定模型，
    语言检测结果只作为元数据用于展示、摘要与同语言书籍过滤。
    """
    return DEFAULT_EMBEDDING_MODEL
