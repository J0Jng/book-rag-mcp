"""LLM 封装测试。

全部 mock ``urllib.request.urlopen``，绝不触发真实网络 / 真实模型。
"""

import json
import urllib.request

import pytest

from book_rag.rag.llm import LLMError, LLMTruncatedError, SummarizerLLM, load_llm


class _FakeResponse:
    """最小假响应：支持 with 语法与 read()。"""

    def __init__(self, status=200, payload=None, raw=b""):
        self.status = status
        self._raw = raw if payload is None else json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._raw


def _chat_payload(content: str, finish_reason: str = "stop") -> dict:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ]
    }


# ---------------------------------------------------------------------------
# load_llm：env 配置解析（env 优先 / 回退 HERMES key / 缺失降级 None）
# ---------------------------------------------------------------------------

_LLM_ENV_VARS = (
    "BOOK_RAG_LLM_BASE_URL",
    "BOOK_RAG_LLM_API_KEY",
    "BOOK_RAG_LLM_MODEL",
    "HERMES_CUSTOM_CUSTOM_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch):
    for name in _LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_load_llm_book_rag_env_wins(monkeypatch):
    monkeypatch.setenv("BOOK_RAG_LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("BOOK_RAG_LLM_API_KEY", "brk-key")
    monkeypatch.setenv("BOOK_RAG_LLM_MODEL", "My-Model")
    monkeypatch.setenv("HERMES_CUSTOM_CUSTOM_API_KEY", "hermes-key")

    llm = load_llm()

    assert isinstance(llm, SummarizerLLM)
    assert llm.base_url == "https://example.com/v1"
    assert llm.api_key == "brk-key"
    assert llm.model == "My-Model"


def test_load_llm_falls_back_to_hermes_key(monkeypatch):
    monkeypatch.setenv("HERMES_CUSTOM_CUSTOM_API_KEY", "hermes-key")

    llm = load_llm()

    assert isinstance(llm, SummarizerLLM)
    assert llm.api_key == "hermes-key"
    # 未配置时用默认 base_url / model（火山方舟 Coding Plan + DeepSeek-V4-Pro）
    assert llm.base_url == "https://ark.cn-beijing.volces.com/api/coding/v3"
    assert llm.model == "DeepSeek-V4-Pro"


def test_load_llm_missing_key_returns_none(monkeypatch):
    assert load_llm() is None


# ---------------------------------------------------------------------------
# SummarizerLLM.complete_json：五类响应（mock urlopen）
# ---------------------------------------------------------------------------


def _make_llm() -> SummarizerLLM:
    return SummarizerLLM(
        base_url="https://example.com/v1",
        api_key="test-key",
        model="Test-Model",
        timeout=5.0,
    )


def test_complete_json_ok(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(payload=_chat_payload('{"summary": "ok"}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = _make_llm().complete_json("sys", "user", max_tokens=1234)

    assert result == {"summary": "ok"}
    assert captured["url"] == "https://example.com/v1/chat/completions"
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["authorization"] == "Bearer test-key"
    assert headers["content-type"] == "application/json"
    assert captured["body"]["model"] == "Test-Model"
    assert captured["body"]["max_tokens"] == 1234
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]
    assert "max_completion_tokens" not in captured["body"]


def test_deepseek_thinking_parameter_is_explicit_opt_in(monkeypatch):
    bodies = []

    def fake_urlopen(req, **kwargs):
        bodies.append(json.loads(req.data))
        return _FakeResponse(payload=_chat_payload('{"ok":true}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    _make_llm().complete_json("sys", "user")
    llm = _make_llm()
    llm.thinking = "disabled"
    llm.complete_json("sys", "user")
    assert "thinking" not in bodies[0]
    assert bodies[1]["thinking"] == {"type": "disabled"}


def test_complete_json_http_non_200(monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(status=500)
    )

    with pytest.raises(LLMError):
        _make_llm().complete_json("sys", "user")


def test_complete_json_truncated(monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(
            payload=_chat_payload('{"summary": "被截断', finish_reason="length")
        ),
    )

    with pytest.raises(LLMTruncatedError):
        _make_llm().complete_json("sys", "user")


def test_complete_json_empty_content(monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(payload=_chat_payload("")),
    )

    with pytest.raises(LLMError):
        _make_llm().complete_json("sys", "user")


def test_complete_json_bad_json(monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(payload=_chat_payload("不是 JSON")),
    )

    with pytest.raises(LLMError):
        _make_llm().complete_json("sys", "user")
