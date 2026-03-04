from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.core.config import settings

_NONE_LIKE = {"", "none", "null", "nil"}


def _clean_model_name(value: Any) -> str | None:
    """Normalize configured model names; treat empty/none-like values as missing."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _NONE_LIKE:
        return None
    return text


def _pick_model_name(*candidates: Any) -> str:
    for candidate in candidates:
        cleaned = _clean_model_name(candidate)
        if cleaned:
            return cleaned
    raise RuntimeError("未配置可用模型名，请检查 .env 的模型配置")


@dataclass
class LLMResponse:
    text: str


class BaseLLM:
    def invoke(self, _prompt: str) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


class MockLLM(BaseLLM):
    """离线可运行：不调用任何外部模型，用规则+随机生成可用输出。"""

    def invoke(self, prompt: str) -> LLMResponse:
        seed = sum(ord(c) for c in prompt) % 10_000
        rng = random.Random(seed)
        bullets = [
            "市场处于震荡期，建议控制仓位并分散行业暴露。",
            "优先选择流动性好、基本面稳健的龙头标的。",
            "关注估值与盈利预期的匹配，避免单一主题过度集中。",
            "对高波动标的设置较低权重，并强调止损纪律。",
        ]
        rng.shuffle(bullets)
        return LLMResponse(text="\n".join(f"- {b}" for b in bullets[:3]))


class LangChainLLM(BaseLLM):
    """用 LangChain Chat 模型包装成我们的 BaseLLM 接口"""

    def __init__(self, model: BaseChatModel):
        self.model = model

    @staticmethod
    def _is_soft_llm_error(err: Exception) -> bool:
        name = err.__class__.__name__.lower()
        text = str(err).lower()
        keys = (
            "timeout",
            "timed out",
            "readtimeout",
            "apitimeouterror",
            "ratelimit",
            "429",
            "engine_overloaded",
            "overloaded",
            "too many requests",
        )
        return any(k in name or k in text for k in keys)

    def invoke(self, prompt: str) -> LLMResponse:
        messages = [HumanMessage(content=prompt)]

        for attempt in range(3):
            try:
                resp = self.model.invoke(messages)
                text = getattr(resp, "content", str(resp))
                return LLMResponse(text=text)
            except Exception as err:
                if not self._is_soft_llm_error(err):
                    raise
                if attempt < 2:
                    time.sleep(2.0 * (attempt + 1))

        return LLMResponse(text="模型服务繁忙或超时，已触发本地回退策略。")


class N1NLLM(BaseLLM):
    """N1N 聚合网关直连，兼容非标准 OpenAI SDK 响应格式。"""

    def __init__(self, base_url: str, api_key: str, model: str):
        normalized = base_url.rstrip("/")
        self.api_base = normalized if normalized.endswith("/v1") else f"{normalized}/v1"
        self.api_key = api_key
        self.model = model

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            msg = first.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return str(msg["content"])
            if isinstance(first.get("text"), str):
                return str(first["text"])
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _parse_json_safe(resp: httpx.Response) -> dict[str, Any]:
        raw = (resp.text or "").strip()
        if not raw:
            raise RuntimeError(f"n1n 返回空响应: status={resp.status_code}")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as err:
            snippet = raw[:240].replace("\n", " ")
            raise RuntimeError(
                f"n1n 返回非JSON响应: status={resp.status_code}, body={snippet}"
            ) from err
        if not isinstance(data, dict):
            raise RuntimeError(f"n1n 响应JSON不是对象: type={type(data).__name__}")
        return data

    def invoke(self, prompt: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = httpx.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=90.0,
                )
                resp.raise_for_status()
                data = self._parse_json_safe(resp)
                return LLMResponse(text=self._extract_text(data))
            except Exception as err:
                last_err = err
                if attempt < 2:
                    time.sleep(1.2 * (attempt + 1))

        if last_err is not None:
            raise last_err
        return LLMResponse(text="模型服务异常")


def _make_openai_compatible(base_url: str, api_key: str, model: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0.2,
        timeout=90,
    )


def _make_ollama(base_url: str, model: str) -> BaseChatModel:
    from langchain_ollama import ChatOllama  # type: ignore[import-not-found]

    return ChatOllama(
        base_url=base_url,
        model=model,
        temperature=0.2,
        timeout=90,
    )


def get_llm(
    model_tag: str = "primary",
    provider_override: str | None = None,
    model_override: str | None = None,
) -> BaseLLM:
    provider = str(provider_override or getattr(settings, "llm_provider", "mock") or "mock").lower()
    secondary_provider = getattr(settings, "secondary_llm_provider", None)
    if not provider_override and model_tag == "secondary" and secondary_provider:
        provider = str(secondary_provider).lower()

    if provider == "mock":
        return MockLLM()

    if provider == "openai_compatible":
        if not settings.openai_base_url or not settings.openai_api_key:
            raise RuntimeError("openai_compatible 需要配置 OPENAI_BASE_URL 和 OPENAI_API_KEY")
        model_name = _pick_model_name(
            model_override,
            settings.secondary_model if model_tag == "secondary" else None,
            settings.openai_model,
        )
        return LangChainLLM(
            _make_openai_compatible(
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key,
                model=model_name,
            )
        )

    if provider == "qwen":
        api_key = settings.dashscope_api_key or settings.openai_api_key
        if not api_key:
            raise RuntimeError("qwen 需要配置 DASHSCOPE_API_KEY（或 OPENAI_API_KEY）")
        model_name = _pick_model_name(
            model_override,
            settings.secondary_model if model_tag == "secondary" else None,
            settings.dashscope_model,
        )
        return LangChainLLM(
            _make_openai_compatible(
                base_url=settings.dashscope_base_url,
                api_key=api_key,
                model=model_name,
            )
        )

    if provider == "kimi":
        api_key = settings.kimi_api_key or settings.openai_api_key
        if not api_key:
            raise RuntimeError("kimi 需要配置 KIMI_API_KEY（或 OPENAI_API_KEY）")
        model_name = _pick_model_name(
            model_override,
            settings.secondary_model if model_tag == "secondary" else None,
            settings.kimi_model,
        )
        return LangChainLLM(
            _make_openai_compatible(
                base_url=settings.kimi_base_url,
                api_key=api_key,
                model=model_name,
            )
        )

    if provider == "n1n":
        api_key = settings.n1n_api_key or settings.openai_api_key
        if not api_key:
            raise RuntimeError("n1n 需要配置 N1N_API_KEY（或 OPENAI_API_KEY）")
        model_name = _pick_model_name(
            model_override,
            settings.secondary_model if model_tag == "secondary" else None,
            settings.n1n_model,
        )
        return N1NLLM(base_url=settings.n1n_base_url, api_key=api_key, model=model_name)

    if provider == "ollama":
        model_name = _pick_model_name(
            model_override,
            settings.secondary_ollama_model if model_tag == "secondary" else None,
            settings.secondary_model if model_tag == "secondary" else None,
            settings.ollama_model,
        )
        return LangChainLLM(_make_ollama(base_url=settings.ollama_base_url, model=model_name))

    raise RuntimeError(f"不支持的 llm_provider: {provider}")


def safe_json_dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
