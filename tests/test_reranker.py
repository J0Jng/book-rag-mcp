"""重排器边界测试，不加载或下载真实模型。"""

import pytest

from book_rag.rag import reranker


@pytest.fixture(autouse=True)
def reset():
    """隔离进程内模型与失败状态。"""
    reranker.reset_reranker()
    yield
    reranker.reset_reranker()


class Encoder:
    """记录推理次数并返回预设分数或错误。"""

    def __init__(self, scores=None, error=None):
        self.scores = scores
        self.error = error
        self.calls = 0

    def predict(self, pairs, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.scores


def test_inference_failure_latches_until_reset(monkeypatch):
    encoder = Encoder(error=RuntimeError("unavailable"))
    monkeypatch.setattr(reranker, "_cross_encoder", encoder)
    assert reranker.rerank_scores("q", ["a"]) is None
    assert reranker.rerank_scores("q", ["a"]) is None
    assert encoder.calls == 1
    assert "unavailable" in reranker.get_reranker_failure()
    reranker.reset_reranker()
    good = Encoder(scores=[0.0])
    monkeypatch.setattr(reranker, "_cross_encoder", good)
    assert reranker.rerank_scores("q", ["a"]) == [0.5]


def test_sigmoid_handles_extreme_logits(monkeypatch):
    monkeypatch.setattr(reranker, "_cross_encoder", Encoder(scores=[-1000.0, 1000.0]))
    assert reranker.rerank_scores("q", ["a", "b"]) == [0.0, 1.0]


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [], [0.0, 1.0]])
def test_invalid_scores_fall_back(monkeypatch, scores):
    monkeypatch.setattr(reranker, "_cross_encoder", Encoder(scores=scores))
    assert reranker.rerank_scores("q", ["a"]) is None
    assert reranker.get_reranker_failure()


def test_load_failure_is_attempted_only_once(monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = []

    def fail(*a, **kw):
        calls.append(1)
        raise OSError("weights missing")

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=fail))
    assert reranker.rerank_scores("q", ["a"]) is None
    assert reranker.rerank_scores("q", ["a"]) is None
    assert calls == [1]
    assert "weights missing" in reranker.get_reranker_failure()


@pytest.mark.parametrize("attribute", ["activation_fn", "default_activation_function"])
def test_activation_is_identity_across_supported_apis(monkeypatch, attribute):
    """新旧 API 均只做一次 sigmoid，不会把 0 logit 映射成 0.622。"""
    import sys
    from types import SimpleNamespace

    class CompatibleEncoder:
        def predict(self, pairs, **kwargs):
            return [getattr(self, attribute)(0.0)]

    encoder = CompatibleEncoder()
    setattr(encoder, attribute, lambda value: 0.5)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        CrossEncoder=lambda *a, **kw: encoder,
    ))
    assert reranker.rerank_scores("q", ["a"]) == [0.5]


@pytest.mark.parametrize("enabled", [False, True])
def test_cpu_precision_is_explicit_and_optional(monkeypatch, enabled):
    """验证真实 Linear 被量化，并可显式保留 FP32；不下载模型。"""
    import torch

    original_threads = torch.get_num_threads()
    encoder = type("LocalEncoder", (), {})()
    encoder.model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    monkeypatch.setenv("BOOK_RAG_RERANK_CPU_THREADS", "2")
    monkeypatch.setenv("BOOK_RAG_RERANK_INT8", "1" if enabled else "0")
    try:
        profile = reranker._configure_cpu_inference(encoder)
        assert profile["device"] == "cpu" and profile["threads"] == 2
        if enabled:
            assert profile["precision"] == "dynamic-int8"
            assert isinstance(encoder.model[0], torch.ao.nn.quantized.dynamic.Linear)
        else:
            assert profile["precision"] == "torch.float32"
            assert isinstance(encoder.model[0], torch.nn.Linear)
        assert torch.isfinite(encoder.model(torch.ones(1, 2))).all()
    finally:
        torch.set_num_threads(original_threads)
