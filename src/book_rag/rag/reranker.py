"""Cross-encoder 重排：对 RRF 融合后的候选集做一次精排。

设计要点（见 docs/plans/book-rag-library-expansion-brief.md S3）：
- 模型固定为 ``BAAI/bge-reranker-v2-m3``，用 sentence-transformers 的
  ``CrossEncoder`` 懒加载；进程内只加载一次；
- 默认开启，``BOOK_RAG_RERANK=0`` 时由配置层直接跳过，不会加载模型；
- 模型下载 / 加载失败时**降级为关闭**并记录真实原因（只尝试一次，避免每次
  检索都卡在失败重试上），调用方回退到 RRF 排序；
- 只在融合后的全局候选集上调用一次，禁止每本书各自重排。
"""

from __future__ import annotations

import logging
import math
import os
from threading import RLock

logger = logging.getLogger(__name__)

#: 重排模型名（固定，不随语言路由变化）
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

#: 单条候选送入 cross-encoder 的最大字符数（控制 CPU 延迟）
MAX_CANDIDATE_CHARS = 1200

#: 模型最大序列长度（token）
MAX_SEQUENCE_LENGTH = 512

_cross_encoder = None
_failure_reason: str | None = None
_MODEL_LOCK = RLock()
_runtime: dict = {}


def get_reranker_runtime() -> dict:
    """返回实际推理设备、精度与线程数，供离线验收记录。"""
    return dict(_runtime)


def _configure_cpu_inference(encoder) -> dict:
    """CPU 用同一模型的动态 INT8；CUDA 保持原生精度，支持显式关闭。"""
    import torch

    model = getattr(encoder, "model", None)
    if model is None:
        return {}
    parameter = next(model.parameters())
    profile = {"device": str(parameter.device), "precision": str(parameter.dtype)}
    if parameter.device.type != "cpu":
        return profile
    threads = int(os.environ.get("BOOK_RAG_RERANK_CPU_THREADS", min(12, os.cpu_count() or 1)))
    if threads < 1:
        raise ValueError("BOOK_RAG_RERANK_CPU_THREADS 必须为正整数")
    torch.set_num_threads(threads)
    profile["threads"] = threads
    enabled = os.environ.get("BOOK_RAG_RERANK_INT8", "1").strip().lower() not in {
        "0", "false", "off", "no",
    }
    if enabled:
        torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
        )
        profile["precision"] = "dynamic-int8"
    logger.info("rerank 实际推理配置：%s", profile)
    return profile


def get_reranker_failure() -> str | None:
    """返回当前进程内的加载失败原因；未失败 / 未尝试时返回 None。"""
    return _failure_reason


def reset_reranker() -> None:
    """清空模型实例与失败状态（测试与重试场景使用）。"""
    global _cross_encoder, _failure_reason
    with _MODEL_LOCK:
        _cross_encoder = None
        _failure_reason = None
        _runtime.clear()


def _get_cross_encoder():
    """懒加载并缓存 CrossEncoder；失败时记录原因并返回 None。"""
    global _cross_encoder, _failure_reason
    if _failure_reason is not None:
        return None
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder
        from torch.nn import Identity

        encoder = CrossEncoder(RERANK_MODEL_NAME, max_length=MAX_SEQUENCE_LENGTH)
        # sentence-transformers 3.x 与新版分别使用这两个属性；强制 raw logits，
        # 避免默认 sigmoid 后在本模块再次 sigmoid，且不提高现有依赖下限。
        if hasattr(encoder, "activation_fn"):
            encoder.activation_fn = Identity()
        else:
            encoder.default_activation_function = Identity()
        profile = _configure_cpu_inference(encoder)
        _cross_encoder = encoder
        _runtime.update(profile)
    except Exception as exc:  # noqa: BLE001 - 模型不可用属可降级故障
        _failure_reason = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "rerank 模型 %s 加载失败，已降级为关闭：%s", RERANK_MODEL_NAME, _failure_reason
        )
        return None
    return _cross_encoder


def _rerank_scores(query: str, texts: list[str]) -> list[float] | None:
    """对 ``(query, text)`` 候选对打分，返回与 texts 等长的分数列表。

    分数经 sigmoid 映射到 ``(0, 1)``，越大越相关；模型不可用时返回 ``None``，
    由调用方回退到融合排序。文本按 :data:`MAX_CANDIDATE_CHARS` 截断以控制延迟。
    """
    if not texts:
        return []
    encoder = _get_cross_encoder()
    if encoder is None:
        return None
    pairs = [(query, text[:MAX_CANDIDATE_CHARS]) for text in texts]
    try:
        raw = encoder.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
        values = [float(score) for score in raw]
        if len(values) != len(texts) or not all(math.isfinite(score) for score in values):
            raise ValueError("rerank 返回了数量不符或非有限分数")
    except Exception as exc:  # noqa: BLE001 - 推理失败同样降级
        global _failure_reason
        _failure_reason = f"{type(exc).__name__}: {exc}"
        logger.warning("rerank 推理失败，已降级为关闭：%s", _failure_reason)
        return None
    return [
        1.0 / (1.0 + math.exp(-score)) if score >= 0
        else math.exp(score) / (1.0 + math.exp(score))
        for score in values
    ]


def rerank_scores(query: str, texts: list[str]) -> list[float] | None:
    """串行加载与推理；失败锁存到显式 reset，避免并发重复加载。"""
    with _MODEL_LOCK:
        return _rerank_scores(query, texts)
