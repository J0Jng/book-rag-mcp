"""语言检测与 embedding 模型选择。

书籍索引与查询检索都需要按语言路由：索引时检测书籍主导语言，
检索时检测 query 语言，再用同一语言的专用 embedding 模型编码。
"""

from __future__ import annotations

import re

from langdetect import DetectorFactory, LangDetectException, detect

# 固定随机种子，保证 langdetect 检测结果可复现
DetectorFactory.seed = 0

# 归一化语言 -> 专用 embedding 模型
_EMBEDDING_MODEL_MAP: dict[str, str] = {
    "zh": "BAAI/bge-large-zh-v1.5",
    "en": "BAAI/bge-large-en-v1.5",
}
# 其他 / 混合语言的回退多语言模型
_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

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
    """按语言选择 embedding 模型；未知 / 其他语言回退到多语言模型 bge-m3。"""
    return _EMBEDDING_MODEL_MAP.get(normalize_language(language), _DEFAULT_EMBEDDING_MODEL)
