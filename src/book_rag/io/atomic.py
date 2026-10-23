"""同目录 JSON 原子替换，兼容 Windows 读者/扫描器短暂占用文件。"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path


def replace_with_retry(source: Path, target: Path) -> None:
    """仅对 Windows 临时访问冲突有限重试；绝不先删除目标文件。"""
    for attempt in range(6):
        try:
            source.replace(target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 5:
                raise
            time.sleep(0.025 * 2 ** attempt)


def write_json_atomic(path: Path, data, *, indent: int | None = 2) -> None:
    """完整写入同目录唯一临时文件，替换失败时保留上一版目标。"""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
