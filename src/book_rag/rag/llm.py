"""LLM 封装：stdlib ``urllib.request`` 裸 HTTP 调用 chat/completions。

刻意不引 langchain-openai / httpx / requests：langchain-openai 会把 ``max_tokens``
改名 ``max_completion_tokens``，在火山方舟端点上语义不同（推理 token 与输出 token
合并计预算），导致空输出。零新依赖。

单次调用、异常上抛；重试策略由调用方（summarizer）决定。
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

_DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
_DEFAULT_MODEL = "DeepSeek-V4-Pro"


class LLMError(Exception):
    """LLM 调用失败（HTTP 非 200 / 网络 / 响应结构异常 / 空 content / 坏 JSON）。"""

    def __init__(self, message: str, *, retryable: bool = True, unavailable: bool = False):
        super().__init__(message)
        self.retryable = retryable
        self.unavailable = unavailable


class LLMTruncatedError(LLMError):
    """输出因 ``max_tokens`` 截断（``finish_reason == "length"``），内容不可信。"""


class SummarizerLLM:
    """章节摘要用的最小 LLM 客户端。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        *,
        thinking: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.thinking = thinking

    def complete_json(self, system: str, user: str, *, max_tokens: int = 2000) -> dict:
        """单次 chat/completions 调用，返回解析后的 JSON dict。

        Raises:
            LLMTruncatedError: ``finish_reason == "length"``（截断，输出不可信）。
            LLMError: HTTP 非 200 / 网络错误 / 空 content / JSON 解析失败。
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # 注意：用 max_tokens 而非 max_completion_tokens（见模块 docstring）
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.thinking is not None:
            payload["thinking"] = {"type": self.thinking}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status = resp.status
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            # 仅保留错误码；提供方消息可能包含请求原文或凭据，不写日志。
            code = ""
            try:
                error = json.loads(exc.read(16384)).get("error", {})
                value = error.get("code", "")
                if isinstance(value, str) and value.replace("_", "").replace(".", "").isalnum():
                    code = value[:100]
            except (ValueError, TypeError, AttributeError, OSError):
                pass
            quota = "quota" in code.lower() or "insufficient" in code.lower()
            unavailable = exc.code in (401, 403, 429) or exc.code >= 500 or quota
            raise LLMError(
                f"LLM HTTP {exc.code}" + (f" ({code})" if code else ""),
                retryable=exc.code >= 500 and not quota,
                unavailable=unavailable,
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMError("LLM 网络不可用", unavailable=True) from exc
        except (TimeoutError, OSError) as exc:
            raise LLMError("LLM 连接超时或中断", unavailable=True) from exc
        if status != 200:
            raise LLMError(f"LLM HTTP {status}")

        try:
            payload = json.loads(raw.decode("utf-8"))
            choice = payload["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            UnicodeDecodeError,
        ) as exc:
            raise LLMError(f"LLM 响应结构异常: {exc}") from exc

        if finish_reason == "length":
            raise LLMTruncatedError("LLM 输出因 max_tokens 截断，结果不可信")
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM 返回空 content")

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM content 不是合法 JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMError("LLM content 不是 JSON object")
        return data


def load_llm() -> SummarizerLLM | None:
    """按环境变量构建 LLM 客户端；未配置 key 时返回 None（优雅降级），绝不 raise。

    优先 ``BOOK_RAG_LLM_*``；key 回退复用 Hermes 的 ``HERMES_CUSTOM_CUSTOM_API_KEY``。
    """
    base_url = os.environ.get("BOOK_RAG_LLM_BASE_URL") or _DEFAULT_BASE_URL
    api_key = os.environ.get("BOOK_RAG_LLM_API_KEY") or os.environ.get(
        "HERMES_CUSTOM_CUSTOM_API_KEY"
    )
    model = os.environ.get("BOOK_RAG_LLM_MODEL") or _DEFAULT_MODEL
    if not api_key:
        return None
    thinking = os.environ.get("BOOK_RAG_LLM_THINKING")
    if thinking not in ("enabled", "disabled", None, ""):
        logging.getLogger(__name__).warning("摘要跳过：BOOK_RAG_LLM_THINKING 配置无效")
        return None
    return SummarizerLLM(base_url, api_key, model, thinking=thinking or None)
