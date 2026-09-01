"""配置管理：书籍库路径、嵌入模型、向量库位置等。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 环境变量默认值
DEFAULT_LIBRARY = Path("./books")
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

# 支持的书籍格式
SUPPORTED_EXTS = {".pdf", ".epub", ".txt", ".md"}


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return default.resolve()


@dataclass
class Config:
    """Book-RAG 运行时配置。"""

    #: 书籍库根目录（扫描书籍）
    library: Path = field(default_factory=lambda: _env_path("BOOK_RAG_LIBRARY", DEFAULT_LIBRARY))
    #: 向量库/索引存储目录
    index_dir: Path = field(default_factory=lambda: _env_path("BOOK_RAG_INDEX_DIR", DEFAULT_LIBRARY / ".book_rag_index"))
    #: 嵌入模型名称（HuggingFace 上可加载的 sentence-transformers 模型）
    embed_model: str = field(default_factory=lambda: os.getenv("BOOK_RAG_EMBED_MODEL", DEFAULT_MODEL))
    #: 设备 (cpu / cuda / mps)
    device: str = field(default_factory=lambda: os.getenv("BOOK_RAG_DEVICE", "cpu"))

    #: 文本切分参数
    chunk_size: int = field(default_factory=lambda: int(os.getenv("BOOK_RAG_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("BOOK_RAG_CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP))))

    #: HTTP 传输参数（可选）
    host: str = field(default_factory=lambda: os.getenv("BOOK_RAG_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("BOOK_RAG_PORT", "8000")))

    def ensure_dirs(self) -> None:
        """确保书籍库与索引目录存在。"""
        self.library.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    """从环境变量构建默认配置。"""
    return Config()