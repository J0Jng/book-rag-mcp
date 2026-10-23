"""pytest 共享配置：效果检验集成测试的模型缓存 gate。

说明：``HF_HUB_OFFLINE`` 等 env 由 huggingface_hub 在**首次 import 时**读取并
固化为常量，必须在其被任何模块 import 之前设置——因此放在 conftest 顶部，
而不是只靠测试内的 autouse fixture（后者在 import 之后才生效，仅作冗余保险）。
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pathlib import Path

import pytest

try:
    from huggingface_hub import try_to_load_from_cache
except ImportError:  # pragma: no cover - 老版本 huggingface_hub 的兜底
    try_to_load_from_cache = None


def _snapshot_dir_exists(repo_id: str) -> bool:
    """回退检查：HF 缓存目录下该模型的 snapshots 是否存在非空目录。"""
    repo_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{repo_id.replace('/', '--')}"
        / "snapshots"
    )
    return repo_dir.is_dir() and any(repo_dir.iterdir())


def require_model(repo_id: str) -> None:
    """集成测试 gate：模型未在本地缓存时 skip，绝不触发 HuggingFace 下载。"""
    if try_to_load_from_cache is not None:
        try:
            cached = try_to_load_from_cache(repo_id, "config.json")
        except Exception:  # noqa: BLE001 - 缓存探测失败时退回目录检查
            cached = None
        if cached is not None:
            return
    if _snapshot_dir_exists(repo_id):
        return
    pytest.skip(
        f"模型 {repo_id} 未缓存，跳过集成测试（先跑一次 book-rag-mcp index 即可缓存）"
    )


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离 LLM 相关环境变量，防止有 key 的环境里 build_book_index 触发真实 LLM 调用。"""
    for name in (
        "BOOK_RAG_LLM_BASE_URL",
        "BOOK_RAG_LLM_API_KEY",
        "BOOK_RAG_LLM_MODEL",
        "BOOK_RAG_LLM_THINKING",
        "BOOK_RAG_SUMMARY_WORKERS",
        "HERMES_CUSTOM_CUSTOM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
