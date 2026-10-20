"""全局配置模块。

集中管理书籍库路径、索引目录与检索参数，避免在源码中硬编码路径。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BookRagConfig:
    """Book-RAG 全局配置。"""

    library_path: Path  # 书籍库根目录
    index_dir: Path  # 索引输出目录（含 manifest.json 与各书籍索引子目录）
    chunk_size: int = 512
    chunk_overlap: int = 64
    default_top_k: int = 5
    clean_matter: bool = True  # 向量化前剥离 front/back matter（默认开启）

    @classmethod
    def from_env(cls) -> BookRagConfig:
        """从环境变量构建配置。

        - ``BOOK_RAG_LIBRARY``：书籍库目录（默认 ``./books``）。
        - ``BOOK_RAG_INDEX_DIR``：索引目录（默认 ``<library_path>/.book_rag_index``）。
        - ``BOOK_RAG_CLEAN_MATTER``：front/back matter 清洗开关；值为
          ``0 / false / off / no``（不区分大小写）时关闭，其余（含未设置）开启。
        """
        library_path = Path(os.environ.get("BOOK_RAG_LIBRARY", "books"))
        index_dir = Path(
            os.environ.get("BOOK_RAG_INDEX_DIR", str(library_path / ".book_rag_index"))
        )
        clean_matter = os.environ.get("BOOK_RAG_CLEAN_MATTER", "").strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        return cls(
            library_path=library_path,
            index_dir=index_dir,
            clean_matter=clean_matter,
        )
